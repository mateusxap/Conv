import time

import numpy as np
import tvm
from tvm import te, topi


def build_topi_conv2d_nhwc(N, H, W, C_in, KH, KW, C_out, target, name):
    """Build a TOPI conv2d using optimized x86 NHWC schedule."""
    inp = te.placeholder((N, H, W, C_in), name=f"{name}_inp", dtype="float32")
    weight = te.placeholder((KH, KW, C_in, C_out), name=f"{name}_weight", dtype="float32")

    conv = topi.nn.conv2d(
        inp,
        weight,
        1,
        padding=0,
        dilation=1,
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )

    with target:
        sched = topi.x86.schedule_conv2d_nhwc([conv])

    return tvm.build(sched, [inp, weight, conv], target=target, name=name)


def compute_fused_conv(A, F3, F1):
    """Compute graph for fused 3x3 + centered 1x1 on pre-padded NHWC input."""
    N, H, W, C = A.shape
    R3, S3, _, K3 = F3.shape
    _, _, _, K1 = F1.shape

    out_h = H - R3 + 1
    out_w = W - S3 + 1

    rc3 = te.reduce_axis((0, C), name="rc3")
    ry3 = te.reduce_axis((0, R3), name="ry3")
    rx3 = te.reduce_axis((0, S3), name="rx3")
    conv3x3 = te.compute(
        (N, out_h, out_w, K3),
        lambda n, h, w, k: te.sum(
            A[n, h + ry3, w + rx3, rc3] * F3[ry3, rx3, rc3, k],
            axis=[ry3, rx3, rc3],
        ),
        name="conv3x3",
    )

    rc1 = te.reduce_axis((0, C), name="rc1")
    conv1x1 = te.compute(
        (N, out_h, out_w, K1),
        lambda n, h, w, k: te.sum(
            A[n, h + 1, w + 1, rc1] * F1[0, 0, rc1, k],
            axis=[rc1],
        ),
        name="conv1x1",
    )

    joined_conv = te.compute(
        (N, out_h, out_w, K3 + K1),
        lambda n, h, w, k: tvm.tir.if_then_else(
            k < K3,
            conv3x3[n, h, w, k],
            conv1x1[n, h, w, k - K3],
        ),
        name="joined_conv",
    )

    return joined_conv, conv3x3, conv1x1


def schedule_fused_conv_nhwc(outs, conv3x3, conv1x1):
    """Schedule for custom fused 3x3+1x1 NHWC compute."""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    joined_conv = outs[0]

    n, h, w, k = joined_conv.op.axis

    s[conv3x3].compute_at(s[joined_conv], w)
    s[conv1x1].compute_at(s[joined_conv], w)

    ry3, rx3, rc3 = conv3x3.op.reduce_axis
    k_3 = conv3x3.op.axis[3]
    s[conv3x3].reorder(ry3, rx3, rc3, k_3)

    rc1 = conv1x1.op.reduce_axis[0]
    k_1 = conv1x1.op.axis[3]
    s[conv1x1].reorder(rc1, k_1)

    fused_spatial = s[joined_conv].fuse(n, h, w)
    s[joined_conv].parallel(fused_spatial)

    s[joined_conv].vectorize(k)
    s[conv3x3].vectorize(k_3)
    s[conv1x1].vectorize(k_1)

    return s


def build_custom_fused_conv_nhwc(N, H, W, C_in, Co_3x3, Co_1x1, target, name):
    """Build the custom fused kernel from tvm_te_conv_3x3_1x1_fused+schedule.py."""
    inp = te.placeholder((N, H, W, C_in), name=f"{name}_inp", dtype="float32")
    f3 = te.placeholder((3, 3, C_in, Co_3x3), name=f"{name}_f3", dtype="float32")
    f1 = te.placeholder((1, 1, C_in, Co_1x1), name=f"{name}_f1", dtype="float32")

    joined_conv, conv3x3, conv1x1 = compute_fused_conv(inp, f3, f1)
    sched = schedule_fused_conv_nhwc(joined_conv, conv3x3, conv1x1)
    return tvm.build(sched, [inp, f3, f1, joined_conv], target=target, name=name)


def evaluate_tvm_conv(N, C_in, H, W, Co_1x1, Co_3x3, iters, warmup):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.cpu(0)

    K_total = Co_1x1 + Co_3x3

    # Keep TOPI on pre-padded input + padding=0 to avoid extra padding-op overhead.
    func_merged = build_topi_conv2d_nhwc(
        N,
        H + 2,
        W + 2,
        C_in,
        3,
        3,
        K_total,
        target,
        "conv_merged",
    )
    func_1x1 = build_topi_conv2d_nhwc(N, H, W, C_in, 1, 1, Co_1x1, target, "conv_1x1")
    func_3x3 = build_topi_conv2d_nhwc(N, H + 2, W + 2, C_in, 3, 3, Co_3x3, target, "conv_3x3")
    func_custom_fused = build_custom_fused_conv_nhwc(
        N,
        H + 2,
        W + 2,
        C_in,
        Co_3x3,
        Co_1x1,
        target,
        "conv_fused_custom",
    )

    a_orig_np = np.random.uniform(-1.0, 1.0, size=(N, H, W, C_in)).astype("float32")
    a_pad_np = np.pad(a_orig_np, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant")

    f_merged_np = np.random.uniform(-1.0, 1.0, size=(3, 3, C_in, K_total)).astype("float32")
    f_1x1_np = np.random.uniform(-1.0, 1.0, size=(1, 1, C_in, Co_1x1)).astype("float32")
    f_3x3_np = np.random.uniform(-1.0, 1.0, size=(3, 3, C_in, Co_3x3)).astype("float32")

    a_pad_tvm = tvm.nd.array(a_pad_np, ctx)
    a_orig_tvm = tvm.nd.array(a_orig_np, ctx)
    f_merged_tvm = tvm.nd.array(f_merged_np, ctx)
    f_1x1_tvm = tvm.nd.array(f_1x1_np, ctx)
    f_3x3_tvm = tvm.nd.array(f_3x3_np, ctx)

    out_merged_tvm = tvm.nd.empty((N, H, W, K_total), dtype="float32", device=ctx)
    out_1x1_tvm = tvm.nd.empty((N, H, W, Co_1x1), dtype="float32", device=ctx)
    out_3x3_tvm = tvm.nd.empty((N, H, W, Co_3x3), dtype="float32", device=ctx)
    out_custom_tvm = tvm.nd.empty((N, H, W, K_total), dtype="float32", device=ctx)

    for _ in range(warmup):
        func_merged(a_pad_tvm, f_merged_tvm, out_merged_tvm)
        func_3x3(a_pad_tvm, f_3x3_tvm, out_3x3_tvm)
        func_1x1(a_orig_tvm, f_1x1_tvm, out_1x1_tvm)
        func_custom_fused(a_pad_tvm, f_3x3_tvm, f_1x1_tvm, out_custom_tvm)

    merged_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        func_merged(a_pad_tvm, f_merged_tvm, out_merged_tvm)
        t1 = time.perf_counter()
        merged_times.append((t1 - t0) * 1000)
    t_merged = sorted(merged_times)[iters // 2]

    seq_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        func_3x3(a_pad_tvm, f_3x3_tvm, out_3x3_tvm)
        func_1x1(a_orig_tvm, f_1x1_tvm, out_1x1_tvm)
        t1 = time.perf_counter()
        seq_times.append((t1 - t0) * 1000)
    t_seq = sorted(seq_times)[iters // 2]

    custom_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        func_custom_fused(a_pad_tvm, f_3x3_tvm, f_1x1_tvm, out_custom_tvm)
        t1 = time.perf_counter()
        custom_times.append((t1 - t0) * 1000)
    t_custom = sorted(custom_times)[iters // 2]

    return t_merged, t_seq, t_custom


def benchmark():
    SWEEP_ITERS = 100
    SWEEP_WARMUP = 10
    HW_vals = [7, 14, 28, 56]

    confs = [
        (16, 16, 16),
        (16, 32, 16),
        (32, 32, 32),
        (32, 64, 32),
        (64, 64, 64),
        (64, 128, 64),
        (128, 128, 128),
        (128, 256, 128),
        (256, 256, 256),
        (256, 512, 256),
        (512, 256, 256),
        (512, 512, 512),
    ]

    N = 1

    W_PAD = 138
    print()
    print("=" * W_PAD)
    print("  Sweep benchmark: TOPI x86 merged vs sequential (3x3 -> 1x1) vs fused_custom")
    print("  fused_custom: compute/schedule from tvm_te_conv_3x3_1x1_fused+schedule.py")
    print(f"  Iterations={SWEEP_ITERS}  Warmup={SWEEP_WARMUP}")
    print("=" * W_PAD)

    fmt_str = "{:<5} {:<6} {:<7} {:<7} | {:<11} {:<11} {:<16} {:<11} {:<13}"
    print(
        fmt_str.format(
            "HW",
            "Cin",
            "Co1x1",
            "Co3x3",
            "merged(ms)",
            "seq(ms)",
            "fused_custom(ms)",
            "seq/merged",
            "seq/fused_custom",
        )
    )
    print("-" * W_PAD)

    for hw in HW_vals:
        for cin, co_1x1, co_3x3 in confs:
            t_merged, t_seq, t_custom = evaluate_tvm_conv(
                N,
                cin,
                hw,
                hw,
                co_1x1,
                co_3x3,
                SWEEP_ITERS,
                SWEEP_WARMUP,
            )

            seq_ratio = t_seq / t_merged if t_merged > 0 else float("nan")
            custom_ratio = t_seq / t_custom if t_merged > 0 else float("nan")

            print(
                fmt_str.format(
                    hw,
                    cin,
                    co_1x1,
                    co_3x3,
                    f"{t_merged:.4f}",
                    f"{t_seq:.4f}",
                    f"{t_custom:.4f}",
                    f"{seq_ratio:.4f}",
                    f"{custom_ratio:.4f}",
                )
            )
        print("-" * W_PAD)


if __name__ == "__main__":
    benchmark()
