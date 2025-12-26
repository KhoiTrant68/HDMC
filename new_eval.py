import argparse
import math
import sys
import time
import glob
import warnings
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Importing the model (Keep your original import)
from models.hdmc import HDMC

# Try importing MS-SSIM
try:
    from pytorch_msssim import ms_ssim
except ImportError:
    print("Please install pytorch-msssim: pip install pytorch-msssim")
    sys.exit(1)

warnings.filterwarnings("ignore")


# --- Helper Classes & Functions (From Template) ---

class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().item()
        
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pad(x, p=128):
    """Padding function from template (Centers the image)"""
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    """Cropping function from template"""
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


def compute_psnr(org: torch.Tensor, rec: torch.Tensor, max_val: int = 255):
    """
    Standard PSNR calculation using rounded 0-255 values 
    (matches template logic for standardized evaluation).
    """
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    mse = (org - rec).pow(2).mean()
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * math.log10(max_val) - 10 * torch.log10(mse)


def compute_msssim_db(org, rec):
    """Calculates MS-SSIM and converts to dB scale."""
    val = ms_ssim(rec, org, data_range=1.0)
    # Convert to dB: -10 * log10(1 - msssim)
    return -10 * math.log10(max(1 - val.item(), 1e-10))


def load_image(filepath: Path):
    return Image.open(filepath).convert("RGB")


def img2torch(img: Image.Image):
    return transforms.ToTensor()(img).unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description="HDMC Integrated Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save outputs")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers (unused currently)")
    return parser.parse_args()


# --- Main Test Logic ---

def test(args):
    # Setup Device
    if args.cuda and torch.cuda.is_available():
        device = "cuda:0"
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"

    # Setup Paths
    data_path = Path(args.data)
    save_path = Path(args.save_path) if args.save_path else None
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)

    # Filter Images
    extensions = {".jpg", ".png", ".jpeg", ".bmp"}
    img_list = sorted([f for f in data_path.iterdir() if f.suffix.lower() in extensions])

    if not img_list:
        print(f"No images found in {data_path}")
        return

    # --- Load Model (HDMC) ---
    print(f"Loading model from {args.checkpoint}...")
    
    # Initialize HDMC (Ensure N/M match your training config)
    net = HDMC(N=192, M=320).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "state_dict" in checkpoint:
        state_dict = {
            k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
        }
    else:
        state_dict = checkpoint
    
    net.load_state_dict(state_dict)
    net.eval()

    # Update CDFs (Crucial for real compression in HDMC)
    print("Updating entropy bottlenecks (CDFs)...")
    net.update(force=True)

    # --- Initialize Meters ---
    bpp_meter = AverageMeter()
    psnr_meter = AverageMeter()
    msssim_meter = AverageMeter()
    y_bpp_meter = AverageMeter()
    z_bpp_meter = AverageMeter()
    enc_time_meter = AverageMeter()
    dec_time_meter = AverageMeter()

    print(f"Starting inference on {len(img_list)} images...")
    print("-" * 80)
    
    # Process Images
    for img_file in img_list:
        img = load_image(img_file)
        x = img2torch(img).to(device)
        
        # Original Dimensions
        h_orig, w_orig = x.size(2), x.size(3)
        num_pixels = h_orig * w_orig

        # Pad (using template logic)
        x_pad = pad(x, p=128) # 128 is standard for these transformers/CNNs
        
        # --- Encode ---
        if args.cuda: torch.cuda.synchronize()
        t_start = time.time()
        
        with torch.no_grad():
            out_enc = net.compress(x_pad)
        
        if args.cuda: torch.cuda.synchronize()
        t_enc = time.time() - t_start

        # --- Decode ---
        if args.cuda: torch.cuda.synchronize()
        t_start = time.time()
        
        with torch.no_grad():
            out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
        
        if args.cuda: torch.cuda.synchronize()
        t_dec = time.time() - t_start

        # --- Post-Process ---
        # Crop back to original size
        x_hat = crop(out_dec["x_hat"], (h_orig, w_orig))
        x_hat.clamp_(0, 1)

        # --- Calculate Metrics ---
        # 1. PSNR (Standardized)
        p_val = compute_psnr(x, x_hat).item()
        
        # 2. MS-SSIM (dB)
        m_val_db = compute_msssim_db(x, x_hat)

        # 3. Bitrate (Total, Y, Z)
        # HDMC strings structure: [[y_string], [z_string_1, z_string_2...]] usually
        # Assuming batch size 1 for evaluation
        y_bytes = len(out_enc["strings"][0][0])
        z_bytes = sum(len(s) for s in out_enc["strings"][1])
        
        total_bits = (y_bytes + z_bytes) * 8.0
        bpp_val = total_bits / num_pixels
        y_bpp_val = (y_bytes * 8.0) / num_pixels
        z_bpp_val = (z_bytes * 8.0) / num_pixels

        # Logging
        img_name = img_file.name
        print(
            f"{img_name:<20} | "
            f"Bpp: {bpp_val:.4f} (Y: {y_bpp_val:.4f}, Z: {z_bpp_val:.4f}) | "
            f"PSNR: {p_val:.2f} | "
            f"MS-SSIM (dB): {m_val_db:.2f} | "
            f"Enc: {t_enc*1000:.1f}ms | Dec: {t_dec*1000:.1f}ms"
        )

        # Update Meters
        bpp_meter.update(bpp_val)
        y_bpp_meter.update(y_bpp_val)
        z_bpp_meter.update(z_bpp_val)
        psnr_meter.update(p_val)
        msssim_meter.update(m_val_db)
        enc_time_meter.update(t_enc)
        dec_time_meter.update(t_dec)

        # Save Image (Optional)
        if save_path:
            save_image(x_hat, save_path / f"recon_{img_name}")

    # --- Final Summary ---
    print("-" * 80)
    print(f"Results ({len(img_list)} images):")
    print(f"\tAvg PSNR:      {psnr_meter.avg:.3f} dB")
    print(f"\tAvg MS-SSIM:   {msssim_meter.avg:.3f} dB")
    print(f"\tAvg Total Bpp: {bpp_meter.avg:.4f} bpp")
    print(f"\tAvg Y Bpp:     {y_bpp_meter.avg:.4f} bpp")
    print(f"\tAvg Z Bpp:     {z_bpp_meter.avg:.4f} bpp")
    print(f"\tAvg Enc Time:  {enc_time_meter.avg * 1000:.2f} ms")
    print(f"\tAvg Dec Time:  {dec_time_meter.avg * 1000:.2f} ms")
    print("-" * 80)


if __name__ == "__main__":
    args = parse_args()
    test(args)