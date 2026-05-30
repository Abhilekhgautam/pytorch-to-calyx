import allo
import torch
import torch.nn as nn


class SimpleFFNN(nn.Module):
    def __init__(self):
        super(SimpleFFNN, self).__init__()
        self.fc1 = nn.Linear(64, 48)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(48, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SimpleFFNN()
model.eval()

example_inputs = [torch.rand(1, 64)]

mod = allo.frontend.from_pytorch(model, target="mlir", example_inputs=example_inputs)

with open("fnn.mlir", "w") as f:
    f.write(str(mod.module))

print("MLIR saved to fnn.mlir")
