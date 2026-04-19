import tvm
from tvm import te

# 1. Задаем размеры
N, C, H, W = 1, 3, 8, 8      # Входной тензор (Батч, Каналы, Высота, Ширина)
K3, R3, S3 = 16, 3, 3        # Ядро для 3x3 (16 выходных каналов)
K1, R1, S1 = 16, 1, 1        # Ядро для 1x1 (16 выходных каналов)

# Вычисляем размер выхода (без паддинга, размер уменьшится из-за 3x3)
out_h = H - R3 + 1
out_w = W - S3 + 1

# 2. Создаем входные тензоры (Placeholders)
A = te.placeholder((N, C, H, W), name="A", dtype="float32")
F3 = te.placeholder((K3, C, R3, S3), name="F3", dtype="float32")
F1 = te.placeholder((K1, C, R1, S1), name="F1", dtype="float32")

# 3. Описываем математику (Compute)

# Оси редукции для 3x3
rc3 = te.reduce_axis((0, C), name="rc3")
ry3 = te.reduce_axis((0, R3), name="ry3")
rx3 = te.reduce_axis((0, S3), name="rx3")

# Свертка 3x3 (стандартная)
conv3x3 = te.compute(
    (N, K3, out_h, out_w),
    lambda n, k, h, w: te.sum(
        A[n, rc3, h + ry3, w + rx3] * F3[k, rc3, ry3, rx3],
        axis=[rc3, ry3, rx3]
    ),
    name="conv3x3"
)

# Ось редукции для 1x1 (только по каналам)
rc1 = te.reduce_axis((0, C), name="rc1")

# Свертка 1x1 (МАГИЯ ЗДЕСЬ: берем h+1 и w+1, чтобы попасть в центр окна 3x3)
conv1x1 = te.compute(
    (N, K1, out_h, out_w),
    lambda n, k, h, w: te.sum(
        A[n, rc1, h + 1, w + 1] * F1[k, rc1, 0, 0],
        axis=[rc1]
    ),
    name="conv1x1"
)

# Объединяем результаты (Конкатенация по оси каналов K)
joined_conv = te.compute(
    (N, K3 + K1, out_h, out_w),
    lambda n, k, h, w: tvm.tir.if_then_else(
        k < K3,
        conv3x3[n, k, h, w],
        conv1x1[n, k - K3, h, w]
    ),
    name="joined_conv"
)

# 4. Создаем расписание (Schedule)
s = te.create_schedule(joined_conv.op)

# Получаем оси итогового тензора: n, k, h, w
n, k, h, w = joined_conv.op.axis

# ИСПРАВЛЕНИЕ: Меняем порядок циклов! Сначала пиксели (h, w), потом каналы (k)
s[joined_conv].reorder(n, h, w, k)

# Теперь привязываем вычисления к циклу по ширине (w)
s[conv3x3].compute_at(s[joined_conv], w)
s[conv1x1].compute_at(s[joined_conv], w)

# 5. Генерируем и печатаем псевдо-C код (IR)
print("=== Сгенерированный код (IR) ===")
print(tvm.lower(s, [A, F3, F1, joined_conv], simple_mode=True))