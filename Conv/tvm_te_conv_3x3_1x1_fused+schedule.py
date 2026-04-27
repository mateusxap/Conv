import tvm
from tvm import te

# ==============================================================================
# 1. ФУНКЦИЯ МАТЕМАТИКИ (Compute)
# ==============================================================================
def compute_fused_conv(A, F3, F1):
    N, H, W, C = A.shape
    R3, S3, _, K3 = F3.shape
    R1, S1, _, K1 = F1.shape

    out_h = H - R3 + 1
    out_w = W - S3 + 1

    # 3x3
    rc3 = te.reduce_axis((0, C), name="rc3")
    ry3 = te.reduce_axis((0, R3), name="ry3")
    rx3 = te.reduce_axis((0, S3), name="rx3")
    conv3x3 = te.compute(
        (N, out_h, out_w, K3),
        lambda n, h, w, k: te.sum(A[n, h + ry3, w + rx3, rc3] * F3[ry3, rx3, rc3, k], axis=[ry3, rx3, rc3]),
        name="conv3x3"
    )

    # 1x1
    rc1 = te.reduce_axis((0, C), name="rc1")
    conv1x1 = te.compute(
        (N, out_h, out_w, K1),
        lambda n, h, w, k: te.sum(A[n, h + 1, w + 1, rc1] * F1[0, 0, rc1, k], axis=[rc1]),
        name="conv1x1"
    )

    # Склейка
    joined_conv = te.compute(
        (N, out_h, out_w, K3 + K1),
        lambda n, h, w, k: tvm.tir.if_then_else(k < K3, conv3x3[n, h, w, k], conv1x1[n, h, w, k - K3]),
        name="joined_conv"
    )
    
    return joined_conv, conv3x3, conv1x1

# ==============================================================================
# 2. ФУНКЦИЯ ШЕДУЛЕРА (Переделанная из TOPI)
# ==============================================================================
def schedule_fused_conv_nhwc(outs, conv3x3, conv1x1):
    """Create schedule for custom fused 1x1 and 3x3 conv2d_nhwc"""
    # 1. Базовая инициализация (Взято из TOPI)
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    joined_conv = outs[0]

    # Получаем оси итогового тензора
    n, h, w, k = joined_conv.op.axis

    # 2. НАША ЛОГИКА СЛИЯНИЯ (Без нее алгоритм сломается)
    s[conv3x3].compute_at(s[joined_conv], w)
    s[conv1x1].compute_at(s[joined_conv], w)

    # Достаем оси редукции программно и делаем reorder
    ry3, rx3, rc3 = conv3x3.op.reduce_axis
    k_3 = conv3x3.op.axis[3]
    s[conv3x3].reorder(ry3, rx3, rc3, k_3)

    rc1 = conv1x1.op.reduce_axis[0]
    k_1 = conv1x1.op.axis[3]
    s[conv1x1].reorder(rc1, k_1)

    # 3. АППАРАТНЫЕ ОПТИМИЗАЦИИ (Взято из TOPI)
    
    # А) Распараллеливание (fuse + parallel)
    # В TOPI было: fused = s[O].fuse(n, h, w); s[O].parallel(fused)
    fused_spatial = s[joined_conv].fuse(n, h, w)
    s[joined_conv].parallel(fused_spatial)

    # Б) Векторизация (vectorize)
    # В TOPI было: s[C].vectorize(c)
    s[joined_conv].vectorize(k)
    s[conv3x3].vectorize(k_3)
    s[conv1x1].vectorize(k_1)

    return s

# ==============================================================================
# 3. ЗАПУСК И ПРОВЕРКА
# ==============================================================================
# Создаем плейсхолдеры
A = te.placeholder((1, 8, 8, 3), name="A", dtype="float32")
F3 = te.placeholder((3, 3, 3, 16), name="F3", dtype="float32")
F1 = te.placeholder((1, 1, 3, 16), name="F1", dtype="float32")

# Вызываем математику
joined_conv, conv3x3, conv1x1 = compute_fused_conv(A, F3, F1)

# Вызываем наш новый шедулер
s = schedule_fused_conv_nhwc(joined_conv, conv3x3, conv1x1)

# Генерируем код
print("=== Сгенерированный код (IR) ===")
print(tvm.lower(s, [A, F3, F1, joined_conv], simple_mode=True))