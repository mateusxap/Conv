import torch
import torch.nn as nn
import time
import os

# Ограничение на 1 поток
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

print("torch.get_num_threads():", torch.get_num_threads())


# Параметры из Conv.cpp
N_BATCH = 4
H_DIM = 28
W_DIM = 28
C_IN_DIM = 64
C_OUT_1x1_DIM = 32
C_OUT_3x3_DIM = 48
N_ITERATIONS = 100
WARMUP_ITERATIONS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Входной тензор
input_tensor = torch.randn(N_BATCH, C_IN_DIM, H_DIM, W_DIM, device=device)

# Свертка 1x1
conv1x1 = nn.Conv2d(C_IN_DIM, C_OUT_1x1_DIM, kernel_size=1, padding=0, bias=False).to(device)
# Свертка 3x3 с паддингом "same"
conv3x3 = nn.Conv2d(C_IN_DIM, C_OUT_3x3_DIM, kernel_size=3, padding=1, bias=False).to(device)

# Прогрев (warmup)
with torch.no_grad():
    for _ in range(WARMUP_ITERATIONS):
        out1 = conv1x1(input_tensor)
        out2 = conv3x3(input_tensor)

# Измерение времени
torch.cuda.synchronize() if device.type == 'cuda' else None
t1 = 0.0
t2 = 0.0

with torch.no_grad():
    # 1x1
    for _ in range(N_ITERATIONS):
        start = time.perf_counter()
        out1 = conv1x1(input_tensor)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t1 += time.perf_counter() - start

    # 3x3
    for _ in range(N_ITERATIONS):
        start = time.perf_counter()
        out2 = conv3x3(input_tensor)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t2 += time.perf_counter() - start

print(f"Average time for 1x1 conv: {t1 * 1000 / N_ITERATIONS:.3f} ms")
print(f"Average time for 3x3 conv: {t2 * 1000 / N_ITERATIONS:.3f} ms")
print(f"Sum: {(t1 + t2) * 1000 / N_ITERATIONS:.3f} ms")

# torch.get_num_threads(): 1
# Using device: cpu
# Average time for 1x1 conv: 0.580 ms
# Average time for 3x3 conv: 3.696 ms
# Sum: 4.276 ms