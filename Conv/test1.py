# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1, 3, 8, 8), "float32"), F3: T.Buffer((16, 3, 3, 3), "float32"), F1: T.Buffer((16, 3, 1, 1), "float32"), joined_conv: T.Buffer((1, 32, 6, 6), "float32")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
        conv3x3 = T.allocate([16], "float32", "global")
        conv1x1 = T.allocate([16], "float32", "global")
        for h, w in T.grid(6, 6):
            conv3x3_1 = T.Buffer((16,), data=conv3x3)
            A_1 = T.Buffer((192,), data=A.data)
            for k in range(16):
                conv3x3_1[k] = T.float32(0)
                for rc3, ry3, rx3 in T.grid(3, 3, 3):
                    F3_1 = T.Buffer((432,), data=F3.data)
                    conv3x3_1[k] = conv3x3_1[k] + A_1[rc3 * 64 + h * 8 + ry3 * 8 + w + rx3] * F3_1[k * 27 + rc3 * 9 + ry3 * 3 + rx3]
            conv1x1_1 = T.Buffer((16,), data=conv1x1)
            for k in range(16):
                conv1x1_1[k] = T.float32(0)
                for rc1 in range(3):
                    F1_1 = T.Buffer((48,), data=F1.data)
                    conv1x1_1[k] = conv1x1_1[k] + A_1[rc1 * 64 + h * 8 + w + 9] * F1_1[k * 3 + rc1]
            for k in range(32):
                joined_conv_1 = T.Buffer((1152,), data=joined_conv.data)
                joined_conv_1[k * 36 + h * 6 + w] = T.if_then_else(k < 16, conv3x3_1[k], conv1x1_1[k - 16])