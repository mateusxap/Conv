import tvm
from tvm import te

# 1. Задаем размеры
N, H, W, C = 1, 8, 8, 3      # Входной тензор NHWC (Батч, Высота, Ширина, Каналы)
R3, S3, K3 = 3, 3, 16        # Ядро 3x3 HWIO (Высота, Ширина, Входные, Выходные)
R1, S1, K1 = 1, 1, 16        # Ядро 1x1 HWIO

# Вычисляем размер выхода
out_h = H - R3 + 1
out_w = W - S3 + 1

# 2. Создаем входные тензоры (Placeholders) в форматах NHWC и HWIO
A = te.placeholder((N, H, W, C), name="A", dtype="float32")
F3 = te.placeholder((R3, S3, C, K3), name="F3", dtype="float32")
F1 = te.placeholder((R1, S1, C, K1), name="F1", dtype="float32")

# 3. Описываем математику (Compute)

# Оси редукции для 3x3
rc3 = te.reduce_axis((0, C), name="rc3")
ry3 = te.reduce_axis((0, R3), name="ry3")
rx3 = te.reduce_axis((0, S3), name="rx3")

# Свертка 3x3 (NHWC)
conv3x3 = te.compute(
    (N, out_h, out_w, K3), # Каналы K3 теперь в конце!
    lambda n, h, w, k: te.sum(
        A[n, h + ry3, w + rx3, rc3] * F3[ry3, rx3, rc3, k], # Индексы изменены под NHWC и HWIO
        axis=[ry3, rx3, rc3]
    ),
    name="conv3x3"
)

# Ось редукции для 1x1
rc1 = te.reduce_axis((0, C), name="rc1")

# Свертка 1x1 (NHWC, смещение в центр h+1, w+1)
conv1x1 = te.compute(
    (N, out_h, out_w, K1),
    lambda n, h, w, k: te.sum(
        A[n, h + 1, w + 1, rc1] * F1[0, 0, rc1, k],
        axis=[rc1]
    ),
    name="conv1x1"
)

# Объединяем результаты (Конкатенация по последней оси K)
joined_conv = te.compute(
    (N, out_h, out_w, K3 + K1),
    lambda n, h, w, k: tvm.tir.if_then_else(
        k < K3,
        conv3x3[n, h, w, k],
        conv1x1[n, h, w, k - K3]
    ),
    name="joined_conv"
)

# 4. Создаем расписание (Schedule)
s = te.create_schedule(joined_conv.op)

# Получаем оси итогового тензора: n, h, w, k
n, h, w, k = joined_conv.op.axis

# Привязываем вычисления к циклу по ширине (w)
s[conv3x3].compute_at(s[joined_conv], w)
s[conv1x1].compute_at(s[joined_conv], w)

# МАГИЯ СОВПАДЕНИЯ С C++:
# В твоем C++ коде внутренние циклы идут так: kh -> kw -> c_in -> c_out
# Давай заставим TVM сделать точно так же!
k_3 = conv3x3.op.axis[3] # Ось выходных каналов для 3x3
s[conv3x3].reorder(ry3, rx3, rc3, k_3)

k_1 = conv1x1.op.axis[3] # Ось выходных каналов для 1x1
s[conv1x1].reorder(rc1, k_1)

# 5. Генерируем и печатаем псевдо-C код (IR)
print("=== Сгенерированный код (IR) ===")
print(tvm.lower(s, [A, F3, F1, joined_conv], simple_mode=True))