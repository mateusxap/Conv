import tvm
from tvm import te, topi
import torch
import torch.nn as nn
import numpy as np

# ============================================================
# Эталон: изолированный PyTorch Fire с весами features.3
# ============================================================
ref = torch.load("squeezenet_reference.pt", map_location="cpu", weights_only=False)
sd = ref["state_dict"]

class Fire(nn.Module):
    def __init__(self, inplanes, sp, e1, e3):
        super().__init__()
        self.squeeze   = nn.Conv2d(inplanes, sp, 1)
        self.expand1x1 = nn.Conv2d(sp, e1, 1)
        self.expand3x3 = nn.Conv2d(sp, e3, 3, padding=1)
    def forward(self, x):
        s = torch.relu(self.squeeze(x))
        return torch.cat([
            torch.relu(self.expand1x1(s)),
            torch.relu(self.expand3x3(s))
        ], 1)

# Параметры fire2 (features.3): C_in=64, squeeze=16, expand1x1=64, expand3x3=64
fire = Fire(64, 16, 64, 64)
for k in ["squeeze", "expand1x1", "expand3x3"]:
    getattr(fire, k).weight.data = sd[f"features.3.{k}.weight"].clone()
    getattr(fire, k).bias.data   = sd[f"features.3.{k}.bias"].clone()
fire.eval()

torch.manual_seed(42)
x_torch = torch.randn(1, 64, 55, 55)
with torch.no_grad():
    y_torch = fire(x_torch).numpy()
print("PyTorch Fire output:", y_torch.shape)

# ============================================================
# TVM Fire-модуль (squeeze отдельно, expand fused)
# ============================================================
N, H, W = 1, 55, 55
C_in, S = 64, 16
K1, K3  = 64, 64

def build_fire():
    A   = te.placeholder((N, H, W, C_in), name="A")
    Wsq = te.placeholder((1, 1, C_in, S), name="Wsq")
    Bsq = te.placeholder((S,),            name="Bsq")
    We1 = te.placeholder((1, 1, S, K1),   name="We1")
    Be1 = te.placeholder((K1,),           name="Be1")
    We3 = te.placeholder((3, 3, S, K3),   name="We3")
    Be3 = te.placeholder((K3,),           name="Be3")

    # ── SQUEEZE: 1x1 conv + bias + ReLU ──
    rc_sq = te.reduce_axis((0, C_in), name="rc_sq")
    sq_mm = te.compute((N, H, W, S),
        lambda n, h, w, s: te.sum(A[n, h, w, rc_sq] * Wsq[0, 0, rc_sq, s], axis=rc_sq),
        name="sq_mm")
    sq = te.compute((N, H, W, S),
        lambda n, h, w, s: tvm.te.max(sq_mm[n, h, w, s] + Bsq[s],
                                      tvm.tir.const(0.0, "float32")),
        name="sq")

    # ── PAD squeeze → (N, 57, 57, 16) — для 3x3 без padding'а в expand ──
    sq_pad = topi.nn.pad(sq, [0, 1, 1, 0], [0, 1, 1, 0], name="sq_pad")

    # ── EXPAND 3x3 (от padded squeeze) ──
    ry  = te.reduce_axis((0, 3), name="ry")
    rx  = te.reduce_axis((0, 3), name="rx")
    rc3 = te.reduce_axis((0, S), name="rc3")
    e3 = te.compute((N, H, W, K3),
        lambda n, h, w, k: te.sum(sq_pad[n, h+ry, w+rx, rc3] * We3[ry, rx, rc3, k],
                                  axis=[ry, rx, rc3]),
        name="e3")

    # ── EXPAND 1x1 (центр padded окна = h+1, w+1 — твой трюк) ──
    rc1 = te.reduce_axis((0, S), name="rc1")
    e1 = te.compute((N, H, W, K1),
        lambda n, h, w, k: te.sum(sq_pad[n, h+1, w+1, rc1] * We1[0, 0, rc1, k], axis=rc1),
        name="e1")

    # ── JOIN + bias + ReLU за один проход ──
    out = te.compute((N, H, W, K1 + K3),
        lambda n, h, w, k: tvm.te.max(
            tvm.tir.if_then_else(k < K1,
                                 e1[n, h, w, k]      + Be1[k],
                                 e3[n, h, w, k - K1] + Be3[k - K1]),
            tvm.tir.const(0.0, "float32")),
        name="out")

    # ── Шедулер: твой текущий + добавки для squeeze ──
    s = te.create_schedule(out.op)
    n_, h_, w_, k_ = out.op.axis
    s[e3].compute_at(s[out], w_)
    s[e1].compute_at(s[out], w_)

    ry_, rx_, rc3_ = e3.op.reduce_axis
    k3_ax = e3.op.axis[3]
    s[e3].reorder(ry_, rx_, rc3_, k3_ax)

    rc1_ = e1.op.reduce_axis[0]
    k1_ax = e1.op.axis[3]
    s[e1].reorder(rc1_, k1_ax)

    s[out].parallel(s[out].fuse(n_, h_, w_))
    s[out].vectorize(k_)
    s[e3].vectorize(k3_ax)
    s[e1].vectorize(k1_ax)

    # squeeze: sq_mm — материализованная стадия (reduce внутри),
    # sq и sq_pad — pointwise, инлайнятся в expand'ы
    n_mm, h_mm, w_mm, s_mm = sq_mm.op.axis
    s[sq_mm].parallel(s[sq_mm].fuse(n_mm, h_mm, w_mm))
    s[sq_mm].vectorize(s_mm)
    s[sq].compute_inline()
    s[sq_pad].compute_inline()

    return tvm.build(s, [A, Wsq, Bsq, We1, Be1, We3, Be3, out],
                     target="llvm", name="fire")

# ============================================================
# Конвертация форматов и запуск
# ============================================================
def nchw_to_nhwc(t): return t.permute(0, 2, 3, 1).contiguous().numpy().astype("float32")
def nhwc_to_nchw(a): return np.transpose(a, (0, 3, 1, 2))
def oihw_to_hwio(t): return t.permute(2, 3, 1, 0).contiguous().numpy().astype("float32")

a_np   = nchw_to_nhwc(x_torch)
wsq_np = oihw_to_hwio(sd["features.3.squeeze.weight"])
bsq_np = sd["features.3.squeeze.bias"].numpy().astype("float32")
we1_np = oihw_to_hwio(sd["features.3.expand1x1.weight"])
be1_np = sd["features.3.expand1x1.bias"].numpy().astype("float32")
we3_np = oihw_to_hwio(sd["features.3.expand3x3.weight"])
be3_np = sd["features.3.expand3x3.bias"].numpy().astype("float32")

fire_tvm = build_fire()
dev = tvm.cpu(0)
ndargs = [tvm.nd.array(t, dev)
          for t in (a_np, wsq_np, bsq_np, we1_np, be1_np, we3_np, be3_np)]
out_tvm = tvm.nd.empty((N, H, W, K1 + K3), "float32", dev)
fire_tvm(*ndargs, out_tvm)

y_tvm = nhwc_to_nchw(out_tvm.numpy())
print("TVM Fire output:    ", y_tvm.shape)

diff = np.abs(y_torch - y_tvm)
print(f"\nmax abs diff:  {diff.max():.6e}")
print(f"mean abs diff: {diff.mean():.6e}")
try:
    np.testing.assert_allclose(y_tvm, y_torch, rtol=1e-4, atol=1e-4)
    print("✅ TVM Fire совпадает с PyTorch Fire — fusion в Inception (a)-стиле работает!")
except AssertionError as e:
    print("❌ Расхождение:")
    print(str(e)[:500])

# Бенчмарк fused-Fire vs PyTorch CPU
import time
ev = fire_tvm.time_evaluator(fire_tvm.entry_name, dev, number=20, repeat=5)
t_tvm = np.median(ev(*ndargs, out_tvm).results) * 1000

# PyTorch один Fire-модуль
fire_torch_jit = torch.jit.trace(fire, x_torch)
fire_torch_jit(x_torch)  # прогрев
torch.set_num_threads(8)  # подстрой под свои ядра
ts = []
for _ in range(20):
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = fire_torch_jit(x_torch)
    ts.append(time.perf_counter() - t0)
t_pt = np.median(ts) * 1000

print(f"\n=== Fire2 (input 1x64x55x55) ===")
print(f"TVM (fused expand): {t_tvm:.3f} ms")
print(f"PyTorch JIT:        {t_pt:.3f} ms")