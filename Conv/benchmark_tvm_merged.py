import time

import numpy as np
import tvm
from tvm import relay, te, topi
from tvm.contrib import graph_executor


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


def build_relay_conv2d_nhwc(N, H, W, C_in, C_out, target, ctx):
    """Build a Relay NHWC/HWIO conv2d with internal padding=(1, 1)."""
    x = relay.var("x", shape=(N, H, W, C_in), dtype="float32")
    w = relay.var("w", shape=(3, 3, C_in, C_out), dtype="float32")

    y = relay.nn.conv2d(
        x,
        w,
        channels=C_out,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )

    mod = tvm.IRModule.from_expr(relay.Function([x, w], y))
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)

    return graph_executor.GraphModule(lib["default"](ctx))


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
    relay_merged = build_relay_conv2d_nhwc(N, H, W, C_in, K_total, target, ctx)

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

    relay_merged.set_input("x", a_orig_tvm)
    relay_merged.set_input("w", f_merged_tvm)

    for _ in range(warmup):
        func_merged(a_pad_tvm, f_merged_tvm, out_merged_tvm)
        func_1x1(a_orig_tvm, f_1x1_tvm, out_1x1_tvm)
        func_3x3(a_pad_tvm, f_3x3_tvm, out_3x3_tvm)
        relay_merged.run()

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
        func_1x1(a_orig_tvm, f_1x1_tvm, out_1x1_tvm)
        func_3x3(a_pad_tvm, f_3x3_tvm, out_3x3_tvm)
        t1 = time.perf_counter()
        seq_times.append((t1 - t0) * 1000)
    t_seq = sorted(seq_times)[iters // 2]

    relay_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        relay_merged.run()
        t1 = time.perf_counter()
        relay_times.append((t1 - t0) * 1000)
    t_relay = sorted(relay_times)[iters // 2]

    return t_merged, t_seq, t_relay


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
    print("  Sweep benchmark: TOPI x86 merged 1x1+3x3  vs  sequential 1x1+3x3")
    print("  Relay baseline: relay.nn.conv2d(..., kernel_size=(3,3), padding=(1,1))")
    print(f"  Iterations={SWEEP_ITERS}  Warmup={SWEEP_WARMUP}")
    print("=" * W_PAD)

    fmt_str = "{:<5} {:<6} {:<7} {:<7} | {:<11} {:<11} {:<11} {:<11} {:<11}"
    print(
        fmt_str.format(
            "HW",
            "Cin",
            "Co1x1",
            "Co3x3",
            "merged(ms)",
            "seq(ms)",
            "relay(ms)",
            "seq/merged",
            "relay/merged",
        )
    )
    print("-" * W_PAD)

    for hw in HW_vals:
        for cin, co_1x1, co_3x3 in confs:
            t_merged, t_seq, t_relay = evaluate_tvm_conv(
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
            relay_ratio = t_relay / t_merged if t_merged > 0 else float("nan")

            print(
                fmt_str.format(
                    hw,
                    cin,
                    co_1x1,
                    co_3x3,
                    f"{t_merged:.4f}",
                    f"{t_seq:.4f}",
                    f"{t_relay:.4f}",
                    f"{seq_ratio:.4f}",
                    f"{relay_ratio:.4f}",
                )
            )
        print("-" * W_PAD)


if __name__ == "__main__":
    benchmark()
