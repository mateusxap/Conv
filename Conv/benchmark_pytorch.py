import torch
import torch.nn as nn
import time

# --- Параметры (должны совпадать с C++ кодом) ---
N_BATCH = 1
C_IN_DIM = 1024
H_DIM = 28
W_DIM = 28
C_OUT_DIM = 48 # Используем C_OUT_3x3_DIM из вашего main.cpp
KH = 3
KW = 3
N_ITERATIONS = 50
WARMUP_ITERATIONS = 10

print("--- PyTorch Benchmark ---")
print(f"Parameters: N={N_BATCH}, C_in={C_IN_DIM}, H={H_DIM}, W={W_DIM}, C_out={C_OUT_DIM}, Kernel={KH}x{KW}")

# Убедимся, что используем CPU
device = torch.device("cpu")

# --- Создание данных ---
# PyTorch использует формат NCHW по умолчанию
input_tensor = torch.randn(N_BATCH, C_IN_DIM, H_DIM, W_DIM, device=device)
# Веса для свертки в PyTorch имеют формат (out_channels, in_channels, kH, kW)
kernel_tensor = torch.randn(C_OUT_DIM, C_IN_DIM, KH, KW, device=device)

# Создаем слой свертки
# padding=0, stride=1 - это поведение "valid" свертки, как в вашем C++ коде
conv_layer = nn.Conv2d(in_channels=C_IN_DIM, out_channels=C_OUT_DIM, kernel_size=(KH, KW), padding=0, stride=1)

# Загружаем наши случайные веса в слой (не обязательно для замера, но для корректности)
conv_layer.weight.data = kernel_tensor
conv_layer.to(device)

# --- Прогрев ---
print("Warming up...")
with torch.no_grad(): # Отключаем расчет градиентов для чистоты замера
    for _ in range(WARMUP_ITERATIONS):
        _ = conv_layer(input_tensor)

# --- Замеры времени ---
print(f"Starting benchmarks ({N_ITERATIONS} iterations)...")
total_duration = 0.0
with torch.no_grad():
    for _ in range(N_ITERATIONS):
        # Важно: синхронизация нужна для CUDA, но для CPU она не вредит и является хорошей практикой
        torch.cpu.synchronize()
        start_time = time.perf_counter()
        
        output_tensor = conv_layer(input_tensor)
        
        torch.cpu.synchronize()
        end_time = time.perf_counter()
        
        total_duration += (end_time - start_time)

avg_duration_ms = (total_duration / N_ITERATIONS) * 1000
print(f"PyTorch Conv2d Average Time: {avg_duration_ms:.4f} ms")

# Проверка размеров выходного тензора
H_out = H_DIM - KH + 1
W_out = W_DIM - KW + 1
print(f"Output tensor shape: {output_tensor.shape}")
print(f"Expected shape: ({N_BATCH}, {C_OUT_DIM}, {H_out}, {W_out})")