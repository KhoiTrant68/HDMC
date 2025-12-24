import argparse
import struct
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Local Import
from models.hdmc import HDMC

warnings.filterwarnings("ignore")

# =========================================================
#  UTILS
# =========================================================


def pad(x, p=64):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p

    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top

    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )


def save_binary(strings, shape, filename):
    """
    Saves compressed bitstream to file.
    Structure:
    [H (2B)][W (2B)]
    [Num_Y_Strings (1B)]
    [Len_Y1 (4B)][Y1 Data]...
    [Num_Z_Strings (1B)]
    [Len_Z1 (4B)][Z1 Data]...
    """
    with open(filename, "wb") as f:
        # 1. Image Dimensions (Original)
        f.write(struct.pack(">H", shape[0]))
        f.write(struct.pack(">H", shape[1]))

        # 2. Y Strings (Likely 1 per slice/batch, but HPCM might have multiple)
        # strings[0] is the list of Y strings
        y_strings = strings[0]
        f.write(struct.pack(">B", len(y_strings)))  # Num Y strings
        for s in y_strings:
            f.write(struct.pack(">I", len(s)))
            f.write(s)

        # 3. Z Strings
        # strings[1] is the list of Z strings
        z_strings = strings[1]
        f.write(struct.pack(">B", len(z_strings)))  # Num Z strings
        for s in z_strings:
            f.write(struct.pack(">I", len(s)))
            f.write(s)


def read_binary(filename):
    """Reads bitstream from file."""
    with open(filename, "rb") as f:
        # 1. Dimensions
        h = struct.unpack(">H", f.read(2))[0]
        w = struct.unpack(">H", f.read(2))[0]

        # 2. Y Strings
        num_y = struct.unpack(">B", f.read(1))[0]
        y_strings = []
        for _ in range(num_y):
            l = struct.unpack(">I", f.read(4))[0]
            y_strings.append(f.read(l))

        # 3. Z Strings
        num_z = struct.unpack(">B", f.read(1))[0]
        z_strings = []
        for _ in range(num_z):
            l = struct.unpack(">I", f.read(4))[0]
            z_strings.append(f.read(l))

    return [y_strings, z_strings], (h, w)


# =========================================================
#  MAIN
# =========================================================


def parse_args(argv):
    parser = argparse.ArgumentParser(description="DCAE Inference")
    parser.add_argument(
        "--mode", type=str, choices=["compress", "decompress"], required=True
    )
    parser.add_argument("--input", type=str, required=True, help="Input file or folder")
    parser.add_argument("--output", type=str, required=True, help="Output folder")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    # Device
    if args.cuda and torch.cuda.is_available():
        device = "cuda:0"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"

    # Setup Paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load Model
    print(f"Loading model: {args.checkpoint}")
    net = HDMC(N=192, M=320).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)

    net.eval()
    net.update(force=True)  # Critical for ANS tables

    # File List
    if input_path.is_dir():
        if args.mode == "compress":
            files = sorted(
                [
                    f
                    for f in input_path.iterdir()
                    if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ]
            )
        else:
            files = sorted([f for f in input_path.iterdir() if f.suffix == ".bin"])
    else:
        files = [input_path]

    print(f"Processing {len(files)} files in {args.mode} mode...")

    # Transform
    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        for f in files:
            if args.mode == "compress":
                # --- COMPRESS ---
                img = Image.open(f).convert("RGB")
                x = to_tensor(img).unsqueeze(0).to(device)

                # Pad (Must match model stride, usually 64 or 128)
                x_padded, padding = pad(x, p=128)

                # Compress
                # Returns dictionary: {"strings": [[y],[z]], "shape": z_shape}
                out = net.compress(x_padded)

                # Save
                bin_name = output_path / (f.stem + ".bin")
                # We save original H,W to calculate padding later
                original_shape = (x.shape[2], x.shape[3])
                save_binary(out["strings"], original_shape, bin_name)

                # Calculate BPP
                num_pixels = x.shape[2] * x.shape[3]
                total_bytes = bin_name.stat().st_size
                bpp = (total_bytes * 8) / num_pixels
                print(f"Compressed {f.name}: {bpp:.4f} bpp")

            else:
                # --- DECOMPRESS ---
                strings, original_shape = read_binary(f)

                # Calculate Padding & Latent Shape
                h, w = original_shape
                # Re-calculate padded dimensions to determine Z shape
                new_h = (h + 127) // 128 * 128
                new_w = (w + 127) // 128 * 128

                # Z stride is 64 (16 * 4) for this architecture usually
                # Architecture: 4x downsample in g_a, then 4x downsample in h_a?
                # Check DCAE config: g_a stride 16, h_a stride 4 -> Total 64
                # Actually DCAE often has g_a stride 16.
                z_shape = (new_h // 64, new_w // 64)

                # Decompress
                out = net.decompress(strings, z_shape)

                # Crop
                # Calculate padding used
                pad_left = (new_w - w) // 2
                pad_right = new_w - w - pad_left
                pad_top = (new_h - h) // 2
                pad_bottom = new_h - h - pad_top
                padding = (pad_left, pad_right, pad_top, pad_bottom)

                x_hat = crop(out["x_hat"], padding).clamp(0, 1)

                save_name = output_path / (f.stem + ".png")
                save_image(x_hat, save_name)
                print(f"Decompressed: {save_name}")


if __name__ == "__main__":
    main(sys.argv[1:])
