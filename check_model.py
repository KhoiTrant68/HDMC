import time

import torch
from thop import profile

from models.hdmc import HDMC


# ------------------------------
# LayerNorm FLOP Hook (only count)
# ------------------------------
def layernorm_hook(module, input, output):
    # Do NOT manually add total_ops, THOP will handle it
    pass  # Leave empty; THOP automatically counts LayerNorm if hook is registered


def apply_layernorm_hook(model):
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.LayerNorm):
            try:
                from thop.vision.basic_hooks import register_hooks

                register_hooks["LayerNorm"] = layernorm_hook
            except:
                # Fallback: manually register hook (safe)
                m.register_forward_hook(layernorm_hook)


# ------------------------------
# Wrapper for dict output
# ------------------------------
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["x_hat"]


# ------------------------------
# Test function
# ------------------------------
def test_dcae_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    model = HDMC(N=192, M=320).to(device).eval()
    model.update(force=True)
    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    wrapped_model = Wrapper(model)
    apply_layernorm_hook(wrapped_model)

    # --- 1. Input/Output Check ---
    print("\n--- 1. Input/Output Check ---")
    print(f"Input shape: {input_tensor.shape}")
    out = wrapped_model(input_tensor)
    print(f"Output shape: {out.shape}")
    print(
        "✅ Shapes match!" if out.shape == input_tensor.shape else "❌ Shape Mismatch!"
    )

    # --- 2. Parameters Count ---
    print("\n--- 2. Parameters Count ---")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params / 1e6:.2f} Million")

    # --- 3. FLOPs / Complexity ---
    print("\n--- 3. FLOPs / Complexity ---")
    macs, _ = profile(wrapped_model, inputs=(input_tensor,), verbose=False)
    gflops = (macs * 2) / 1e9
    print(f"MACs: {macs / 1e9:.2f} G")
    print(f"GFLOPs (approx): {gflops:.2f} G")
    print("Note: Calculated on 256x256 resolution.")
    print("-----------------------------------------")


if __name__ == "__main__":
    test_dcae_model()
