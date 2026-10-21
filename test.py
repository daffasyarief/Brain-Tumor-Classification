import torch
from src.model import HybridModel

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = HybridModel(
    num_classes=4,
    pretrained=False
).to(device)

x = torch.randn(2, 3, 224, 224).to(device)
output = model(x)

print("Device:", device)
print("Output shape:", output.shape)
print(
    "Total parameters:",
    f"{sum(p.numel() for p in model.parameters()):,}"
)