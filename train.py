import argparse
import logging
import math
import os
import random
import shutil
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

# Accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

# CompressAI
from compressai.datasets import ImageFolder

# SOTA Utils
from timm.utils import ModelEmaV2
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid

from loss.loss import AverageMeter, RateDistortionLoss

# Local Modules
from models.hdmc import HDMC


# =========================================================
#  LOSS-FREE BALANCER (DeepSeek Logic)
# =========================================================
class LossFreeBalancer:
    """
    Implements Loss-Free Balancing for MoE.
    Logic: If an expert is underutilized (count < avg), we INCREASE its bias.
           If overloaded (count > avg), we DECREASE its bias.

    Update Rule: bias += update_rate * sign(avg_count - count)
    """

    def __init__(self, num_experts=4, update_rate=0.001):
        self.num_experts = num_experts
        self.update_rate = update_rate

    @torch.no_grad()
    def update_biases(self, model, router_data_tuple, accelerator=None):
        """
        Args:
            model: Unwrapped HDMC model.
            router_data_tuple: Tuple of (logits, indices) from forward pass.
            accelerator: For syncing counts across GPUs.
        """
        if not router_data_tuple:
            return

        for i, layer_data in enumerate(router_data_tuple):
            # layer_data is (logits, indices)
            if not (isinstance(layer_data, (tuple, list)) and len(layer_data) == 2):
                continue

            _, topk_indices = layer_data

            # 1. Count usage (Flatten batch and spatial dims)
            flat_indices = topk_indices.flatten()
            local_counts = torch.bincount(
                flat_indices, minlength=self.num_experts
            ).float()

            # 2. DDP Sync: Sum counts across all GPUs to get global load
            if accelerator and accelerator.num_processes > 1:
                expert_counts = accelerator.reduce(local_counts, reduction="sum")
            else:
                expert_counts = local_counts

            # 3. Calculate Error
            # Target is balanced load, so avg_count is the target.
            avg_count = expert_counts.mean()

            # Error = Target - Actual
            # Positive Error => Underloaded => Sign is +1 => Bias increases
            # Negative Error => Overloaded  => Sign is -1 => Bias decreases
            error = avg_count - expert_counts

            # 4. Find the correct module to update
            moe_module = self._get_module_by_index(model, i)

            if moe_module is not None and hasattr(moe_module, "expert_biases"):
                update_step = self.update_rate * torch.sign(error)

                # Ensure device compatibility
                device = moe_module.expert_biases.device

                # In-place update. expert_biases has requires_grad=False, so this is safe.
                moe_module.expert_biases.data.add_(update_step.to(device))

    def _get_module_by_index(self, model, index):
        """
        Maps the flat index from `all_logits` to the specific MoE module in HDMC.
        Forward order in HDMC:
        1. Standard Slices (dt_cross_attention list)
        2. Anchor (moe_anchor)
        3. Non-Anchor (moe_non_anchor)
        """
        num_standard = len(model.dt_cross_attention)

        if index < num_standard:
            return model.dt_cross_attention[index]
        elif index == num_standard:
            return model.moe_anchor
        elif index == num_standard + 1:
            return model.moe_non_anchor
        return None


# =========================================================
#  LOGGING & SETUP
# =========================================================


def setup_logger(log_dir):
    logger = logging.getLogger("Train")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def configure_optimizers(net, args):
    """Separates parameters for Main Optimizer (AdamW) and Aux Optimizer (Adam)."""
    params_dict = dict(net.named_parameters())
    params = {n: p for n, p in params_dict.items() if not n.endswith(".quantiles")}
    aux_params = {n: p for n, p in params_dict.items() if n.endswith(".quantiles")}

    # Main: AdamW for better convergence on Transformers/ConvNeXt
    optimizer = optim.AdamW(
        [p for p in params.values() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Aux: Standard Adam
    aux_optimizer = optim.Adam(
        [p for p in aux_params.values() if p.requires_grad],
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


# =========================================================
#  TRAINING LOOP
# =========================================================


def train_one_epoch(
    model,
    criterion,
    train_dataloader,
    optimizer,
    aux_optimizer,
    epoch,
    clip_max_norm,
    accelerator,
    logger,
    writer,
    global_step,
    balancer,
    ema_model,
    args,
):
    model.train()

    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    dist_meter = AverageMeter()

    # Log max/min bias occasionally to check if balancer is working
    bias_stats_logged = False

    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # 1. Forward Pass (Noise Injection Mode)
        out_net = model(batch, training_mode="noise")

        # 2. Compute Loss
        out_criterion = criterion(out_net, batch)
        loss = out_criterion["loss"]

        # 3. Backward Pass
        accelerator.backward(loss)

        if clip_max_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), clip_max_norm)

        optimizer.step()

        # 4. LOSS-FREE BALANCING UPDATE
        # Done outside of autograd, after optimizer step
        if "router_logits" in out_net and out_net["router_logits"]:
            unwrapped = accelerator.unwrap_model(model)
            balancer.update_biases(unwrapped, out_net["router_logits"], accelerator)

        # 5. Aux Loss (Entropy Bottleneck CDFs)
        aux_loss = model.aux_loss()
        accelerator.backward(aux_loss)
        aux_optimizer.step()

        # 6. Update EMA Model
        if ema_model is not None:
            ema_model.update(model)

        # 7. Logging
        loss_meter.update(loss.item())
        bpp_meter.update(out_criterion["bpp_loss"].item())
        dist_meter.update(out_criterion["mse_loss"].item())  # Raw MSE

        if i % 100 == 0 and accelerator.is_main_process:
            logger.info(
                f"Epoch [{epoch}][{i}/{len(train_dataloader)}] "
                f"Loss: {loss_meter.val:.4f} | Bpp: {bpp_meter.val:.4f} | MSE: {dist_meter.val:.6f}"
            )
            writer.add_scalar("Train/Loss", loss_meter.val, global_step)
            writer.add_scalar("Train/Bpp", bpp_meter.val, global_step)
            writer.add_scalar("Train/MSE", dist_meter.val, global_step)

            # Optional: Log bias stats to verify balancing
            if not bias_stats_logged:
                unwrapped = accelerator.unwrap_model(model)
                if hasattr(unwrapped.dt_cross_attention[0], "expert_biases"):
                    biases = unwrapped.dt_cross_attention[0].expert_biases
                    writer.add_scalar(
                        "Debug/Bias_Max", biases.max().item(), global_step
                    )
                    writer.add_scalar(
                        "Debug/Bias_Min", biases.min().item(), global_step
                    )
                    bias_stats_logged = True

        global_step += 1

    return global_step


def test_epoch(epoch, test_dataloader, model, criterion, accelerator, logger, writer):
    model.eval()
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()
    psnr_meter = AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # Test Mode: Use STE (Rounding) instead of Noise
            out_net = model(batch, training_mode="ste")
            out_criterion = criterion(out_net, batch)

            loss_meter.update(out_criterion["loss"].item())
            bpp_meter.update(out_criterion["bpp_loss"].item())
            mse_meter.update(out_criterion["mse_loss"].item())
            psnr_meter.update(out_criterion["psnr"].item())

            if i == 0 and accelerator.is_main_process:
                # Log Reconstruction Image
                x_hat = out_net["x_hat"].clamp(0, 1)
                # Concat Input and Output
                combined = torch.cat([batch[:4], x_hat[:4]], dim=0)
                grid = make_grid(combined, nrow=4, padding=2)
                writer.add_image("Val/Reconstruction", grid, epoch)

    if accelerator.is_main_process:
        logger.info(
            f"Test Epoch [{epoch}] "
            f"Loss: {loss_meter.avg:.4f} | Bpp: {bpp_meter.avg:.4f} | PSNR: {psnr_meter.avg:.2f} dB"
        )
        writer.add_scalar("Val/Loss", loss_meter.avg, epoch)
        writer.add_scalar("Val/Bpp", bpp_meter.avg, epoch)
        writer.add_scalar("Val/PSNR", psnr_meter.avg, epoch)

    return loss_meter.avg


def save_checkpoint(state, is_best, save_path, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(save_path, filename),
            os.path.join(save_path, "checkpoint_best.pth.tar"),
        )


# =========================================================
#  MAIN ENTRY
# =========================================================


def parse_args():
    parser = argparse.ArgumentParser(description="HDMC Training")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Path to ImageFolder root"
    )
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("-e", "--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=8)

    # Optimization
    parser.add_argument("--lr", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--aux-lr", dest="aux_learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--lmbda", type=float, default=0.0130, help="Lagrange multiplier"
    )
    parser.add_argument("--clip_max_norm", type=float, default=1.0)

    # Balancing
    parser.add_argument(
        "--update_rate", type=float, default=0.001, help="Bias update step size"
    )

    # Data / System
    parser.add_argument("--patch_size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1926)
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Resume from checkpoint"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1. Setup Accelerator (DDP, Mixed Precision handles automatically)
    accelerator = Accelerator(log_with="tensorboard", project_dir=args.save_path)

    # 2. Logging
    save_path = os.path.join(args.save_path, f"lambda_{args.lmbda}")
    os.makedirs(save_path, exist_ok=True)

    logger = None
    writer = None
    if accelerator.is_main_process:
        logger = setup_logger(save_path)
        writer = SummaryWriter(os.path.join(save_path, "tb"))
        logger.info(f"Training Config: {args}")

    # 3. Data Loading
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(args.patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="valid", transform=test_transforms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 4. Initialize Model
    net = HDMC(N=192, M=320)  # Must match HDMC.py definitions

    # Initialize EMA (Exponential Moving Average) - SOTA trick
    # decay=0.999 is standard for compression
    ema_model = ModelEmaV2(net, decay=0.999)

    # 5. Initialize Balancer
    balancer = LossFreeBalancer(
        num_experts=4, update_rate=args.update_rate  # Must match num_experts in HDMC.py
    )

    # 6. Optimizers & LR Scheduler
    optimizer, aux_optimizer = configure_optimizers(net, args)

    # Cosine Annealing (SOTA Scheduler)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 7. Criterion
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    # 8. Resume / Load Checkpoint
    start_epoch = 0
    best_loss = float("inf")

    if args.checkpoint:
        if accelerator.is_main_process:
            logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        net.load_state_dict(checkpoint["state_dict"])

        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "aux_optimizer" in checkpoint:
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])

    # 9. Prepare with Accelerator
    net, optimizer, aux_optimizer, train_dataloader, test_dataloader, lr_scheduler = (
        accelerator.prepare(
            net,
            optimizer,
            aux_optimizer,
            train_dataloader,
            test_dataloader,
            lr_scheduler,
        )
    )

    # Move EMA to device (Accelerate doesn't wrap EMA automatically)
    ema_model.module.to(accelerator.device)

    # 10. Training Loop
    global_step = 0
    for epoch in range(start_epoch, args.epochs):

        # Train
        global_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            accelerator,
            logger,
            writer,
            global_step,
            balancer,
            ema_model,
            args,
        )

        # Test (Use EMA model for validation if possible, here using Main for consistency)
        # To test EMA: accelerator.unwrap_model(ema_model.module)
        loss = test_epoch(
            epoch, test_dataloader, net, criterion, accelerator, logger, writer
        )

        lr_scheduler.step()

        # Save
        if accelerator.is_main_process:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": accelerator.unwrap_model(net).state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                save_path,
                filename="checkpoint.pth.tar",
            )

            # Save EMA weights separately (Use these for final inference!)
            torch.save(
                {"state_dict": ema_model.module.state_dict(), "epoch": epoch},
                os.path.join(save_path, "checkpoint_ema.pth.tar"),
            )

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
