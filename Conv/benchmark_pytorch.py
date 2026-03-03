import torch
import torch.nn as nn
import time
import statistics
import os

torch.set_num_threads(8)

HW_VALS  = [7, 14, 28, 56]
CIN_VALS = [16, 32, 64, 128, 256]
N_BATCH  = 1
N_ITERS  = 100
WARMUP   = 5

# Kernel configs: (kh, kw, cout_list)
# padding = (kh-1)//2  => same spatial size as C++ (no padding = valid conv)
# We use padding=0 to match C++ conv_param (valid convolution)
KERNEL_CONFIGS = [
    (1, 1, [16, 32, 64, 128, 256],  [16, 32, 64, 128, 256, 512, 1024]),
    (3, 3, [16, 32, 64, 128, 256],  [512, 1024]),
    (3, 3, [128, 512],              [48]),
]

device = torch.device("cpu")

W = 100
print()
print("=" * W)
print("  PyTorch sweep benchmark: 1x1 and 3x3 conv (valid, no padding), N={}".format(N_BATCH))
print("  torch {}  threads={}".format(torch.__version__, torch.get_num_threads()))
print("  Iterations={}  Warmup={}".format(N_ITERS, WARMUP))
print("  Combined = 1 call with full Cout;  Sequential = 2 calls with Cout/2")
print("=" * W)
print("{:<5}{:<5}{:<5}{:<4}{:<4} | {:<12}{:<12} | {:>6}".format(
    "HW", "Cin", "Cout", "KH", "KW", "pt_c (ms)", "pt_s (ms)", "s/c"))
print("-" * W)

for kh, kw, cin_vals, cout_vals in KERNEL_CONFIGS:
    for hw in HW_VALS:
        h_out = hw - kh + 1
        w_out = hw - kw + 1
        for ci in cin_vals:
            for co in cout_vals:
                co_half = co // 2
                inp = torch.randn(N_BATCH, ci, hw, hw, device=device)

                conv_full = nn.Conv2d(ci, co,      (kh, kw), padding=0, bias=False, device=device)
                conv_h1   = nn.Conv2d(ci, co_half, (kh, kw), padding=0, bias=False, device=device)
                conv_h2   = nn.Conv2d(ci, co_half, (kh, kw), padding=0, bias=False, device=device)

                with torch.no_grad():
                    for _ in range(WARMUP):
                        conv_full(inp)
                        conv_h1(inp)
                        conv_h2(inp)

                durs_c = []
                with torch.no_grad():
                    for _ in range(N_ITERS):
                        torch.cpu.synchronize()
                        t0 = time.perf_counter()
                        conv_full(inp)
                        torch.cpu.synchronize()
                        durs_c.append((time.perf_counter() - t0) * 1000)

                durs_s = []
                with torch.no_grad():
                    for _ in range(N_ITERS):
                        torch.cpu.synchronize()
                        t0 = time.perf_counter()
                        conv_h1(inp)
                        conv_h2(inp)
                        torch.cpu.synchronize()
                        durs_s.append((time.perf_counter() - t0) * 1000)

                med_c = statistics.median(durs_c)
                med_s = statistics.median(durs_s)
                ratio = med_s / med_c if med_c > 0 else 0

                print("{:<5}{:<5}{:<5}{:<4}{:<4} | {:<12.4f}{:<12.4f} | {:6.2f}".format(
                    hw, ci, co, kh, kw, med_c, med_s, ratio))
        print("-" * W)

print("=" * W)
