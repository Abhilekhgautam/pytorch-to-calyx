import torch
import torch.nn as nn
from torch_mlir import fx;

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
input = torch.randn(1, 64) 

mlir_module = fx.export_and_import(
    model,
    input,
    output_type="linalg-on-tensors"
)

print(mlir_module)
