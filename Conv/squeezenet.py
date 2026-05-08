import torch
import torchvision.models as models
import numpy as np

# 1. Загружаем предобученную SqueezeNet 1.1
model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
model.eval()

# 2. Готовим тестовый вход (детерминированный)
torch.manual_seed(0)
x = torch.randn(1, 3, 224, 224)

# 3. Прогоняем — это наш эталон
with torch.no_grad():
    logits_ref = model(x).numpy()
print("Logits shape:", logits_ref.shape)  # (1, 1000)
print("Top-5 indices:", logits_ref[0].argsort()[-5:][::-1])

# 4. Извлекаем выходы каждого Fire-модуля и их РАЗМЕРЫ
fire_outputs = {}
fire_dimensions = {} # Словарь для хранения HW, Cin, Co1x1, Co3x3
hooks = []

for name, module in model.named_modules():
    if "Fire" in type(module).__name__:
        # Достаем статические параметры каналов прямо из весов слоев Expand
        Cin = module.expand1x1.in_channels
        Co1x1 = module.expand1x1.out_channels
        Co3x3 = module.expand3x3.out_channels
        
        def make_hook(nm, c_in, c_o1, c_o3):
            def hook(_m, _inp, out):
                # _inp[0] - это входной тензор в Fire-модуль. 
                # Так как Squeeze (1x1) не меняет HW, размер HW для Expand будет таким же.
                HW = _inp[0].shape[2] # Берем высоту (предполагаем H == W)
                
                # Сохраняем эталонный выход
                fire_outputs[nm] = out.detach().numpy()
                # Сохраняем размеры для таблицы
                fire_dimensions[nm] = (HW, c_in, c_o1, c_o3)
            return hook
            
        hooks.append(module.register_forward_hook(make_hook(name, Cin, Co1x1, Co3x3)))

with torch.no_grad():
    _ = model(x)
for h in hooks: h.remove()

# ==============================================================================
# 4.5 ВЫВОД ТАБЛИЦЫ РАЗМЕРОВ (Как заказывали)
# ==============================================================================
print("\n" + "="*65)
print(f"{'Layer Name':<15} | {'HW':<5} {'Cin':<6} {'Co1x1':<7} {'Co3x3':<7}")
print("-" * 65)
for nm, dims in fire_dimensions.items():
    hw, cin, co1, co3 = dims
    print(f"{nm:<15} | {hw:<5} {cin:<6} {co1:<7} {co3:<7}")
print("=" * 65 + "\n")

# 5. Сохраняем веса всех слоёв (state_dict) — потом будем грузить в TVM
torch.save({
    "state_dict": model.state_dict(),
    "input_x":     x.numpy(),
    "logits_ref":  logits_ref,
    "fire_outputs": fire_outputs,
}, "squeezenet_reference.pt")

print("✅ Эталон сохранён в squeezenet_reference.pt")