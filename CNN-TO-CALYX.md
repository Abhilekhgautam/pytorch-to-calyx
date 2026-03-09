# CNN to Calyx

This continues from my work on [simple pytorch model](https://github.com/Abhilekhgautam/pytorch-to-calyx/blob/main/CNN-TO-CALYX.md). 

Below is a simple CNN model in pytorch:

```python
import torch
import torch.nn as nn

import allo

class TinyConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

model = TinyConvBlock()
dummy_input = [torch.randn(1, 3, 64, 64)]

mod = allo.frontend.from_pytorch(model, target="calyx", example_inputs=dummy_input)

with open("cnn-model.mlir", "w") as f:
    f.write(str(mod.module))

print("MLIR saved to cnn-model.mlir") 
```

We use allo to get the mlir code.

As expected, we get 4 function definitions - Conv2d, ReLU, MaxPool2d, and forward. Also we have a global memref `conv_weight` and `conv_bias` for weights and bias.

The pipeline is similar to what we did with the simple pytorch model.

We start by running mlir's `--lower-affine` pass to lower affines to scf.

We then need to flatten all `memrefs` because calyx needs all `memrefs` to be 1 dimensional.

We use `circt-opt` to flatten all memrefs using the pass `-flatten-memref`. Unfortunately there seem to be a bug with this version of circt-opt. I have sent a [PR](https://github.com/llvm/circt/pull/9615) that seems to fix this.

 We then run the `buffer-results-to-out-params` on the flattened memref as:

```
mlir-opt flattened-cnn.mlir \
 --symbol-privatize \
 --buffer-results-to-out-params \
 --canonicalize \
 --cse \
```

This generates `memref.copy` and bulk copies is not supported in calyx. So we need to run another
pass (a custom one this time) - `--memref_copy_to_affine_loops` which generates affine loops, which again needs to be lowered to scf using `-lower-affine` pass. 

Now, we can run the `lower-scf-to-calyx` pass from circt-opt to lower to calyx. 

```
circt-opt final.mlir --pass-pipeline="builtin.module(lower-scf-to-calyx{top-level-function=forward})"
```

But we get hit with illegal `arith.fptoui` error. Seems like `circt-opt` while lowering to calyx considers `arith.fptoui` as illegal.

I analyzed the generated mlir and found that `fptoui` is used to obtain a value that is used as an index. On further analysis, I found that replacing `arith.fptoui` with `arith.fptosi` is perfectly okay because the generated mlir has bunch of checks that ensures the obtained index is within the bounds. (I don't know yet If this is a generic solution, but perfectly fine in this case).

Now, if we again run the `lower-scf-to-calyx` pass we should we get. But there is another bug in `circt-opt` pass. circt-opt seem to crash most of the time when we have a global `memref` dense element. I have sent another [PR](https://github.com/llvm/circt/pull/9738).

We successfully lower to calyx. Wow.

I have to verify the generated calyx and then convert the calyx to verlog.
