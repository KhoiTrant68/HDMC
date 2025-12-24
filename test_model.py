import time

import torch
from flops_profiler.profiler import get_model_profile

from models.hdmc import HDMC


# ================================================================
# Wrapper because HDMC returns dict
# ================================================================
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["x_hat"]


# ================================================================
# Measure Latency (ms)
# ================================================================
@torch.inference_mode()
def measure_latency(model, inp, warmup=10, runs=50):
    # Warm-up
    for _ in range(warmup):
        _ = model(inp)

    if inp.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for _ in range(runs):
        _ = model(inp)

    if inp.device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    return (end - start) * 1000 / runs


# ================================================================
# Main Evaluation
# ================================================================
def test_hdmc_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Model
    model = HDMC(N=192, M=320).to(device).eval()
    model.update(force=True)

    inp = torch.randn(1, 3, 256, 256).to(device)

    encoder = model.g_a
    decoder = model.g_s

    # ================================================================
    # FLOPs: Encoder
    # ================================================================
    enc_flops, enc_macs, enc_params = get_model_profile(
        model=encoder,
        args=(inp,),
        print_profile=False,
        detailed=False,
        as_string=False,
    )

    # Compute z for decoder
    with torch.no_grad():
        z = encoder(inp)

    # ================================================================
    # FLOPs: Decoder
    # ================================================================
    dec_flops, dec_macs, dec_params = get_model_profile(
        model=decoder,
        args=(z,),
        print_profile=False,
        detailed=False,
        as_string=False,
    )

    # ================================================================
    # Total Params
    # ================================================================
    total_params = sum(p.numel() for p in model.parameters())

    # ================================================================
    # Latency
    # ================================================================
    wrapped = Wrapper(model)
    total_latency = measure_latency(wrapped, inp)
    enc_latency = measure_latency(encoder, inp)
    dec_latency = measure_latency(decoder, z)

    # ================================================================
    # Print Result Row
    # ================================================================
    print("\n================= RESULT ROW =================")
    print("Model | Latency (ms)      |   GFLOPs           | Params")
    print("      | Tot   Enc   Dec   | Enc     Dec        |")
    print(
        f"HDMC  | "
        f"{total_latency:5.1f} {enc_latency:5.1f} {dec_latency:5.1f}  |  "
        f"{enc_flops/1e9:5.2f}  {dec_flops/1e9:5.2f}  |  "
        f"{total_params/1e6:.2f}M"
    )
    print("================================================\n")


if __name__ == "__main__":
    test_hdmc_model()
