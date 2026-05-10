import torch
import torch.nn as nn
from torch_mlir import fx

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv2D:
        # Input  : 1x3x80x60
        # Output : 1x8x76x56
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )

        self.relu = nn.ReLU()

        # MaxPool:
        # Input  : 1x8x76x56
        # Output : 1x8x38x18
        self.pool = nn.MaxPool2d(
            kernel_size=(2, 3),
            stride=(2, 3)
        )

        # Flattened size:
        # 8 * 38 * 18 = 5472
        self.fc = nn.Linear(5472, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten all except batch dimension
        x = torch.flatten(x, 1)

        # Binary classification logits
        x = self.fc(x)

        return x


model = SimpleCNN().eval()

example_input = torch.randn(1, 3, 80, 60)

mlir_module = fx.export_and_import(
    model,
    example_input,
    output_type="linalg-on-tensors"
)

print(mlir_module)
