# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1, 8, 8, 3), "float32"), F3: T.Buffer((3, 3, 3, 16), "float32"), F1: T.Buffer((1, 1, 3, 16), "float32"), joined_conv: T.Buffer((1, 6, 6, 32), "float32")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
        conv3x3 = T.allocate([16], "float32", "global")
        conv1x1 = T.allocate([16], "float32", "global")
        for h, w in T.grid(6, 6):
            conv3x3_1 = T.Buffer((16,), data=conv3x3)
            for k_init in range(16):
                conv3x3_1[k_init] = T.float32(0)
            A_1 = T.Buffer((192,), data=A.data)
            for ry3, rx3, rc3, k in T.grid(3, 3, 3, 16):
                F3_1 = T.Buffer((432,), data=F3.data)
                conv3x3_1[k] = conv3x3_1[k] + A_1[h * 24 + ry3 * 24 + w * 3 + rx3 * 3 + rc3] * F3_1[ry3 * 144 + rx3 * 48 + rc3 * 16 + k]
            conv1x1_1 = T.Buffer((16,), data=conv1x1)
            for k_init in range(16):
                conv1x1_1[k_init] = T.float32(0)
            for rc1, k in T.grid(3, 16):
                F1_1 = T.Buffer((48,), data=F1.data)
                conv1x1_1[k] = conv1x1_1[k] + A_1[h * 24 + w * 3 + rc1 + 27] * F1_1[rc1 * 16 + k]
            for k in range(32):
                joined_conv_1 = T.Buffer((1152,), data=joined_conv.data)
                joined_conv_1[h * 192 + w * 32 + k] = T.if_then_else(k < 16, conv3x3_1[k], conv1x1_1[k - 16])