import torch
from thop import profile

from models.hdmc import HDMC


# Wrapper to handle dict output
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["x_hat"]


# Layer-by-layer FLOPs counting hook
def flops_counter(module, input, output):
    flops = 0
    if isinstance(module, torch.nn.Conv2d):
        out_c, out_h, out_w = output.shape[1:]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
        flops = out_c * out_h * out_w * kernel_ops
    elif isinstance(module, torch.nn.Linear):
        flops = module.in_features * module.out_features
    elif isinstance(module, torch.nn.LayerNorm):
        flops = output.numel()
    elif "Attention" in module.__class__.__name__:
        # Rough approximation: Q*K^T + softmax + proj
        flops = output.numel() * 2
    else:
        return
    module.__flops__ = getattr(module, "__flops__", 0) + flops


# Attach hooks
def attach_hooks(model):
    handles = []
    for name, module in model.named_modules():
        handles.append(module.register_forward_hook(flops_counter))
    return handles


# Print top FLOPs layers
def print_top_flops(model, top_k=10):
    flops_list = []
    for name, module in model.named_modules():
        if hasattr(module, "__flops__"):
            flops_list.append((name, module.__class__.__name__, module.__flops__))
    flops_list.sort(key=lambda x: x[2], reverse=True)

    print("\n--- Top FLOPs Layers ---")
    print(f"{'Layer Name':<40} {'Type':<20} {'FLOPs (M)':>12}")
    print("-" * 75)
    for name, typ, flops in flops_list[:top_k]:
        print(f"{name:<40} {typ:<20} {flops/1e6:>12.3f}")
    print("-" * 75)
    total_flops = sum([f for _, _, f in flops_list])
    print(f"{'Total FLOPs (approx)':<62} {total_flops/1e9:>12.3f} G")


# ==============================
# Run the analysis
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HDMC(N=192, M=320).to(device).eval()
model.update(force=True)
input_tensor = torch.randn(1, 3, 256, 256).to(device)

wrapped_model = Wrapper(model)
handles = attach_hooks(wrapped_model)

# Forward pass
with torch.no_grad():
    _ = wrapped_model(input_tensor)

# Print top FLOPs layers
print_top_flops(wrapped_model)

# Remove hooks
for h in handles:
    h.remove()
