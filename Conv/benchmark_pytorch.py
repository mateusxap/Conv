import torch
import torch.nn as nn
import time
import statistics
import os

# Match C++ OMP defaults — use all available cores
NUM_THREADS = os.cpu_count() or 6
torch.set_num_threads(NUM_THREADS)

# Sweep parameters (same as C++ benchmark_NCHWc_sweep)
HW_VALS   = [7, 14, 28, 56]
CIN_VALS  = [16, 32, 64, 128, 256]
COUT_VALS = [16, 32, 64, 128, 256]
N_BATCH   = 1
KH, KW    = 1, 1
N_ITERS   = 100
WARMUP    = 5

device = torch.device("cpu")

W = 100
print()
print("=" * W)
print(f"  PyTorch sweep benchmark: 1x1 conv, N={N_BATCH}")
print(f"  torch {torch.__version__}, threads={NUM_THREADS}")
print(f"  Iterations={N_ITERS}  Warmup={WARMUP}")
print(f"  Combined = 1 call with full Cout;  Sequential = 2 calls with Cout/2")
print("=" * W)
print(f"{'HW':<5}{'Cin':<5}{'Cout':<5} | {'pt_c (ms)':<12}{'pt_s (ms)':<12} | {'s/c':>6}")
print("-" * W)

for hw in HW_VALS:
    for ci in CIN_VALS:
        for co in COUT_VALS:
            co_half = co // 2
            inp = torch.randn(N_BATCH, ci, hw, hw, device=device)

            # Combined conv
            conv_full = nn.Conv2d(ci, co, (KH, KW), padding=0, stride=1, bias=False, device=device)
            # Sequential: two half-convs
            conv_h1 = nn.Conv2d(ci, co_half, (KH, KW), padding=0, stride=1, bias=False, device=device)
            conv_h2 = nn.Conv2d(ci, co_half, (KH, KW), padding=0, stride=1, bias=False, device=device)

            # Warmup
            with torch.no_grad():
                for _ in range(WARMUP):
                    _ = conv_full(inp)
                    _ = conv_h1(inp)
                    _ = conv_h2(inp)

            # Bench combined
            durs_c = []
            with torch.no_grad():
                for _ in range(N_ITERS):
                    torch.cpu.synchronize()
                    t0 = time.perf_counter()
                    _ = conv_full(inp)
                    torch.cpu.synchronize()
                    t1 = time.perf_counter()
                    durs_c.append((t1 - t0) * 1000)

            # Bench sequential
            durs_s = []
            with torch.no_grad():
                for _ in range(N_ITERS):
                    torch.cpu.synchronize()
                    t0 = time.perf_counter()
                    _ = conv_h1(inp)
                    _ = conv_h2(inp)
                    torch.cpu.synchronize()
                    t1 = time.perf_counter()
                    durs_s.append((t1 - t0) * 1000)

            med_c = statistics.median(durs_c)
            med_s = statistics.median(durs_s)
            ratio = med_s / med_c if med_c > 0 else 0

            print(f"{hw:<5}{ci:<5}{co:<5} | {med_c:<12.4f}{med_s:<12.4f} | {ratio:6.2f}")

print("=" * W)
