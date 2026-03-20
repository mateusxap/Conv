# Conv — Оптимизированная свёртка (Convolution Benchmark)

Бенчмарк реализаций 1×1 свёрточных операций на C++ с AVX2/FMA-интринсиками и OpenMP-параллелизацией.
Сравниваются три варианта NCHWc-свёртки и PyTorch CPU.

## Требования

- **CMake** ≥ 3.10
- **C++17**-совместимый компилятор (GCC / Clang)
- **OpenMP**
- Поддержка AVX2/FMA (`-march=native`)
- **Python 3** + **PyTorch CPU** (для сравнения с PyTorch)

---

## Реализованные варианты

| Вариант | Описание |
|---------|----------|
| `w_c` | Базовый (SIMD-подсказка `#pragma omp simd`), внутренний цикл по `w_out` |
| `c_w` | Переставлены циклы `w_out` ↔ `c_in_inner` (хуже кэш-локальности) |
| `v3`  | AVX2 + FMA интринсики (`_mm256_fmadd_ps`), выровненный `accum_block`, unroll ×4 по `w_out` |

Тензорный формат входа: **NCHWc** (`[N, C_in/8, H, W, 8]`).  
Тензорный формат ядра: **OIHWio** (`[C_out/8, C_in/8, KH, KW, 8, 8]`).  
Тензорный формат выхода: **NHWC** (`[N, H, W, C_out]`).

---

## Сборка C++

```bash
mkdir -p build && cd build
cmake ../Conv
make -j$(nproc)
# Исполняемый файл: build/Conv
```

---

## Запуск бенчмарков

### 1. C++ sweep-бенчмарк (все три варианта)

Измеряет медианное время по 100 итерациям для Grid:
- HW ∈ {7, 14, 28, 56}
- C_in ∈ {16, 32, 64, 128, 256}
- C_out ∈ {16, 32, 64, 128, 256}

Каждая конфигурация тестируется в двух режимах:
- **Combined** — один вызов с полным `C_out`
- **Sequential** — два вызова с `C_out/2` каждый

```bash
cd build
./Conv
# Сохранить результаты для дальнейшего сравнения:
./Conv | tee /tmp/cpp_results.txt
```

---

### 2. Настройка Python-окружения (один раз)

```bash
cd /home/matthew/projects/Conv
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

### 3. PyTorch sweep-бенчмарк

Те же 100 конфигураций, те же combined/sequential режимы через `nn.Conv2d`.

```bash
cd /home/matthew/projects/Conv
source venv/bin/activate
python Conv/benchmark_pytorch.py
# Сохранить результаты:
python Conv/benchmark_pytorch.py | tee /tmp/pytorch_results.txt
```

---

### 4. Сравнительная таблица v3 vs PyTorch

После того как оба файла собраны, запустить скрипт сравнения:

```bash
source venv/bin/activate
python Conv/compare_results.py /tmp/cpp_results.txt /tmp/pytorch_results.txt
```

Скрипт выводит таблицу вида:

```
HW   Cin  Cout  |   v3_c   v3_s |   pt_c   pt_s |  pt/v3c pt/v3s |  win_c win_s
7    16   16    |  0.0025 0.0047 |  0.0226 0.0561 |    9.04  11.94 |     v3    v3
...
Summary:
Combined:    v3 wins 98,  PyTorch wins 2,  ~tie 0
Sequential:  v3 wins 98,  PyTorch wins 2,  ~tie 0
```

- `_c` — combined (один вызов, полный `C_out`)
- `_s` — sequential (два вызова по `C_out/2`)
- `pt/v3` > 1 означает что v3 быстрее PyTorch в `pt/v3` раз

---

## Краткие итоги (последний запуск)

| Конфигурация | v3 combined | PyTorch combined | Ускорение v3 |
|---|---|---|---|
| 7×7, Cin=16, Cout=16 | 0.0025 ms | 0.023 ms | **9×** |
| 28×28, Cin=256, Cout=256 | 0.24 ms | 0.40 ms | **1.65×** |
| 56×56, Cin=256, Cout=256 | 2.63 ms | 4.57 ms | **1.74×** |
| 14×14, Cin=256, Cout=256 | 0.073 ms | 0.21 ms | **2.9×** |

v3 быстрее PyTorch CPU **в 98 из 100** протестированных конфигураций.

---

## Структура проекта

| Файл | Описание |
|------|----------|
| `Conv/Conv.hpp` | Классы тензоров, константы, объявления функций, `ConvParams` |
| `Conv/Conv.cpp` | Реализация: `conv_optimized_w_c`, `conv_optimized_c_w`, `conv_optimized_v3`, + параметризованные `conv_param_*` |
| `Conv/main.cpp` | Точка входа: `benchmark_NCHWc_convs_googlenet()`, `benchmark_NCHWc_sweep()` |
| `Conv/benchmark_pytorch.py` | PyTorch sweep-бенчмарк (те же параметры что у C++) |
| `Conv/compare_results.py` | Скрипт сравнения вывода C++ и PyTorch |
| `Conv/CMakeLists.txt` | Конфигурация сборки |
