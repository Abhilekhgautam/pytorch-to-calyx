import torch
import torch.nn.functional as F
import torch.nn as nn

import allo

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = x + y
        x = F.relu(x)
        return x

model = Model()
model.eval()

example_inputs = [torch.rand(1, 3, 10, 10), torch.rand(1, 3, 10, 10)]

mod = allo.frontend.from_pytorch(model, target="mlir", example_inputs=example_inputs)

with open("fnn.mlir", "w") as f:
    f.write(str(mod.module))

print("MLIR saved to fnn.mlir")

