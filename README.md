# Pytorch To Calyx

The main idea here is to convert a pytorch program into Calyx IR (IR for hardware generators). 

We follow a pipeline discussed in [this paper](https://arxiv.org/pdf/2512.06177).

 Here is the program:

```py
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

with open("model.mlir", "w") as f:
    f.write(str(mod.module))

print("MLIR saved to model.mlir")

```

Here, we use `allo` to convert our model to `mlir` and save the generated code into the file `model.mlir`.

Allo is the first stage in the pipeline, it simply gives us this mlir:

```mlir
module {
  func.func @relu4d_0(%arg0: memref<1x3x10x10xf32>) -> memref<1x3x10x10xf32> attributes {itypes = "_", otypes = "_"} {
    %alloc = memref.alloc() {name = "Z"} : memref<1x3x10x10xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 3 {
        affine.for %arg3 = 0 to 10 {
          affine.for %arg4 = 0 to 10 {
            %0 = affine.load %arg0[%arg1, %arg2, %arg3, %arg4] {from = "X"} : memref<1x3x10x10xf32>
            %cst = arith.constant 0.000000e+00 : f32
            %1 = arith.maximumf %cst, %0 : f32
            affine.store %1, %alloc[%arg1, %arg2, %arg3, %arg4] {to = "Z"} : memref<1x3x10x10xf32>
          } {loop_name = "w", pipeline_ii = 1 : ui32}
        } {loop_name = "h"}
      } {loop_name = "c"}
    } {loop_name = "n", op_name = "S_n_c_h_w_0"}
    return %alloc : memref<1x3x10x10xf32>
  }
  func.func @forward(%arg0: memref<1x3x10x10xf32>, %arg1: memref<1x3x10x10xf32>) -> memref<1x3x10x10xf32> attributes {itypes = "__", otypes = "_"} {
    %alloc = memref.alloc() {name = "add"} : memref<1x3x10x10xf32>
    linalg.add {op_name = "add_0"} ins(%arg0, %arg1 : memref<1x3x10x10xf32>, memref<1x3x10x10xf32>) outs(%alloc : memref<1x3x10x10xf32>)
    %0 = call @relu4d_0(%alloc) {name = "relu"} : (memref<1x3x10x10xf32>) -> memref<1x3x10x10xf32>
    return %0 : memref<1x3x10x10xf32>
  }
}
```

It generated `mlir` for two functions, `forward` and `relu`.

The forward function uses the op `add` from the `linalg` dialect to perfrom elementwise addition of the input tensors. The resultant tensor is passed to the `relu` function and the return value from the
relu function is returned from the forward function.

The generated `relu` function uses `affine` for loops to iterate over all the elements of the tensor. For every element in the tensor it checks if it greater than `0`, if it is , it places the element in the output tensor, else it places `0` in the output tensor. This is exactly what a `relu` function is expected to do.

Since we have mlir, we can use `mlir-opt` to perform any optimziation/transformation pass to the above code. For eg: In the above generated code, `%cst` is a constant value but being created for every iteration of the loop, we can perform loop invariant code motion to this.

mlir-opt has an option `-loop-invariant-code-motion` that does this for us. But this is not what we'll be looking at.

Since, we are targeting calyx, we need to subsequently lower to it. We'll first lower `linalg` to `affine` for which we'll use the flag `convert-linalg-to-affine-loops` to get:
```mlir
module {
  func.func @relu4d_0(%arg0: memref<1x3x10x10xf32>) -> memref<1x3x10x10xf32> attributes {itypes = "_", otypes = "_"} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "Z"} : memref<1x3x10x10xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 3 {
        affine.for %arg3 = 0 to 10 {
          affine.for %arg4 = 0 to 10 {
            %0 = affine.load %arg0[%arg1, %arg2, %arg3, %arg4] {from = "X"} : memref<1x3x10x10xf32>
            %1 = arith.maximumf %0, %cst : f32
            affine.store %1, %alloc[%arg1, %arg2, %arg3, %arg4] {to = "Z"} : memref<1x3x10x10xf32>
          } {loop_name = "w", pipeline_ii = 1 : ui32}
        } {loop_name = "h"}
      } {loop_name = "c"}
    } {loop_name = "n", op_name = "S_n_c_h_w_0"}
    return %alloc : memref<1x3x10x10xf32>
  }
  func.func @forward(%arg0: memref<1x3x10x10xf32>, %arg1: memref<1x3x10x10xf32>) -> memref<1x3x10x10xf32> attributes {itypes = "__", otypes = "_"} {
    %alloc = memref.alloc() {name = "add"} : memref<1x3x10x10xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 10 {
          affine.for %arg5 = 0 to 10 {
            %1 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x3x10x10xf32>
            %2 = affine.load %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x3x10x10xf32>
            %3 = arith.addf %1, %2 : f32
            affine.store %3, %alloc[%arg2, %arg3, %arg4, %arg5] : memref<1x3x10x10xf32>
          }
        }
      }
    }
    %0 = call @relu4d_0(%alloc) {name = "relu"} : (memref<1x3x10x10xf32>) -> memref<1x3x10x10xf32>
    return %0 : memref<1x3x10x10xf32>
  }
}
```

We'd also want to legalize our arithmetic operations which can be done with the `-arith-expand` command to get:
```mlir
module {
  func.func @relu4d_0(%arg0: memref<1x3x10x10xf32>) -> memref<1x3x10x10xf32> attributes {itypes = "_", otypes = "_"} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "Z"} : memref<1x3x10x10xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 3 {
        affine.for %arg3 = 0 to 10 {
          affine.for %arg4 = 0 to 10 {
            %0 = affine.load %arg0[%arg1, %arg2, %arg3, %arg4] {from = "X"} : memref<1x3x10x10xf32>
            %1 = arith.cmpf ugt, %0, %cst : f32
            %2 = arith.select %1, %0, %cst : f32
            %3 = arith.cmpf uno, %cst, %cst : f32
            %4 = arith.select %3, %cst, %2 : f32
            affine.store %4, %alloc[%arg1, %arg2, %arg3, %arg4] {to = "Z"} : memref<1x3x10x10xf32>
          } {loop_name = "w", pipeline_ii = 1 : ui32}
        } {loop_name = "h"}
      } {loop_name = "c"}
    } {loop_name = "n", op_name = "S_n_c_h_w_0"}
    return %alloc : memref<1x3x10x10xf32>
  }
  func.func @forward(%arg0: memref<1x3x10x10xf32>, %arg1: memref<1x3x10x10xf32>) -> memref<1x3x10x10xf32> attributes {itypes = "__", otypes = "_"} {
    %alloc = memref.alloc() {name = "add"} : memref<1x3x10x10xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 10 {
          affine.for %arg5 = 0 to 10 {
            %1 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x3x10x10xf32>
            %2 = affine.load %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x3x10x10xf32>
            %3 = arith.addf %1, %2 : f32
            affine.store %3, %alloc[%arg2, %arg3, %arg4, %arg5] : memref<1x3x10x10xf32>
          }
        }
      }
    }
    %0 = call @relu4d_0(%alloc) {name = "relu"} : (memref<1x3x10x10xf32>) -> memref<1x3x10x10xf32>
    return %0 : memref<1x3x10x10xf32>
  }
}
```
So far so good, but we have few more issues before we can lower it down to calyx. Since calyx is a hardware representation, we cannot have a function that returns a memref because memref are pointers and that is not what a hardware does. So we need to remove the return from the function. We can do so by taking an additional argument from the function which will be the output and hence we can return void from the function.

To do so we can use the option `-bufferize-results-to-out-params` with the option enabled for public function as well (typically this option only works for private function) to obtain this:

```mlir
module {
  func.func private @relu4d_0(%arg0: memref<1x3x10x10xf32>, %arg1: memref<1x3x10x10xf32>) attributes {itypes = "_", otypes = "_"} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "Z"} : memref<1x3x10x10xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 10 {
          affine.for %arg5 = 0 to 10 {
            %0 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] {from = "X"} : memref<1x3x10x10xf32>
            %1 = arith.cmpf ugt, %0, %cst : f32
            %2 = arith.select %1, %0, %cst : f32
            %3 = arith.cmpf uno, %cst, %cst : f32
            %4 = arith.select %3, %cst, %2 : f32
            affine.store %4, %alloc[%arg2, %arg3, %arg4, %arg5] {to = "Z"} : memref<1x3x10x10xf32>
          } {loop_name = "w", pipeline_ii = 1 : ui32}
        } {loop_name = "h"}
      } {loop_name = "c"}
    } {loop_name = "n", op_name = "S_n_c_h_w_0"}
    memref.copy %alloc, %arg1 : memref<1x3x10x10xf32> to memref<1x3x10x10xf32>
    return
  }
  func.func private @forward(%arg0: memref<1x3x10x10xf32>, %arg1: memref<1x3x10x10xf32>, %arg2: memref<1x3x10x10xf32>) attributes {itypes = "__", otypes = "_"} {
    %alloc = memref.alloc() {name = "add"} : memref<1x3x10x10xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 3 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %arg0[%arg3, %arg4, %arg5, %arg6] : memref<1x3x10x10xf32>
            %1 = affine.load %arg1[%arg3, %arg4, %arg5, %arg6] : memref<1x3x10x10xf32>
            %2 = arith.addf %0, %1 : f32
            affine.store %2, %alloc[%arg3, %arg4, %arg5, %arg6] : memref<1x3x10x10xf32>
          }
        }
      }
    }
    %alloc_0 = memref.alloc() : memref<1x3x10x10xf32>
    call @relu4d_0(%alloc, %alloc_0) : (memref<1x3x10x10xf32>, memref<1x3x10x10xf32>) -> ()
    memref.copy %alloc_0, %arg2 : memref<1x3x10x10xf32> to memref<1x3x10x10xf32>
    return
  }
}
```

Now, we can lower to scf (because we have a pass in `circt-opt` for `scf-to-calyx` transformation) using `lower-affine` to obtain:

```mlir
module {
  func.func private @relu4d_0(%arg0: memref<1x3x10x10xf32>, %arg1: memref<1x3x10x10xf32>) attributes {itypes = "_", otypes = "_"} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "Z"} : memref<1x3x10x10xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c1 step %c1_0 {
      %c0_1 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg3 = %c0_1 to %c3 step %c1_2 {
        %c0_3 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg4 = %c0_3 to %c10 step %c1_4 {
          %c0_5 = arith.constant 0 : index
          %c10_6 = arith.constant 10 : index
          %c1_7 = arith.constant 1 : index
          scf.for %arg5 = %c0_5 to %c10_6 step %c1_7 {
            %0 = memref.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x3x10x10xf32>
            %1 = arith.cmpf ugt, %0, %cst : f32
            %2 = arith.select %1, %0, %cst : f32
            %3 = arith.cmpf uno, %cst, %cst : f32
            %4 = arith.select %3, %cst, %2 : f32
            memref.store %4, %alloc[%arg2, %arg3, %arg4, %arg5] : memref<1x3x10x10xf32>
          }
        }
      }
    }
    memref.copy %alloc, %arg1 : memref<1x3x10x10xf32> to memref<1x3x10x10xf32>
    return
  }
  func.func private @forward(%arg0: memref<1x3x10x10xf32>, %arg1: memref<1x3x10x10xf32>, %arg2: memref<1x3x10x10xf32>) attributes {itypes = "__", otypes = "_"} {
    %alloc = memref.alloc() {name = "add"} : memref<1x3x10x10xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c1 step %c1_0 {
      %c0_2 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %c1_3 = arith.constant 1 : index
      scf.for %arg4 = %c0_2 to %c3 step %c1_3 {
        %c0_4 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg5 = %c0_4 to %c10 step %c1_5 {
          %c0_6 = arith.constant 0 : index
          %c10_7 = arith.constant 10 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg6 = %c0_6 to %c10_7 step %c1_8 {
            %0 = memref.load %arg0[%arg3, %arg4, %arg5, %arg6] : memref<1x3x10x10xf32>
            %1 = memref.load %arg1[%arg3, %arg4, %arg5, %arg6] : memref<1x3x10x10xf32>
            %2 = arith.addf %0, %1 : f32
            memref.store %2, %alloc[%arg3, %arg4, %arg5, %arg6] : memref<1x3x10x10xf32>
          }
        }
      }
    }
    %alloc_1 = memref.alloc() : memref<1x3x10x10xf32>
    call @relu4d_0(%alloc, %alloc_1) : (memref<1x3x10x10xf32>, memref<1x3x10x10xf32>) -> ()
    memref.copy %alloc_1, %arg2 : memref<1x3x10x10xf32> to memref<1x3x10x10xf32>
    return
  }
}
```

So far we have been using `mlir-opt`. But transformation to calyx is not standard so we'll need to use the `circt-opt` for lowering to calyx. `circt-opt` has an option `-lower-scf-to-calyx`. However this option needs to know a top level function (entry function) we can do so using following command (we set `forward` as the top level function because it is the only function we have defined.):

```
circt-opt <mlir-filename>  --pass-pipeline="builtin.module(lower-scf-to-calyx{top-level-function=forward})"
```
Unfortunately this doesn't work because we have copy operation and you cannot just tell a hardware to bulk copy it, we need to lower it to a loop. Luckily (because finally i can write some custom pass) we don't have such pass that converts `memref.copy` to `affine.for`. No problem I wrote it myself (was kinda fun and easy!). But it requires the memref to be flattened, so we'll first flatten the memref we have. This can be done with the help of `-flatten-memref` option in `circt-opt`.

```
circt-opt <mlir-filename> -flatten-memref
```

This gets us the following mlir:

```mlir
module {
  func.func private @relu4d_0(%arg0: memref<300xf32>, %arg1: memref<300xf32>) attributes {itypes = "_", otypes = "_"} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<300xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c1 step %c1_0 {
      %c0_1 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg3 = %c0_1 to %c3 step %c1_2 {
        %c0_3 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg4 = %c0_3 to %c10 step %c1_4 {
          %c0_5 = arith.constant 0 : index
          %c10_6 = arith.constant 10 : index
          %c1_7 = arith.constant 1 : index
          scf.for %arg5 = %c0_5 to %c10_6 step %c1_7 {
            %c300 = arith.constant 300 : index
            %0 = arith.muli %arg2, %c300 : index
            %1 = arith.addi %0, %arg3 : index
            %c100 = arith.constant 100 : index
            %2 = arith.muli %1, %c100 : index
            %3 = arith.addi %2, %arg4 : index
            %c10_8 = arith.constant 10 : index
            %4 = arith.muli %3, %c10_8 : index
            %5 = arith.addi %4, %arg5 : index
            %6 = memref.load %arg0[%5] : memref<300xf32>
            %7 = arith.cmpf ugt, %6, %cst : f32
            %8 = arith.select %7, %6, %cst : f32
            %9 = arith.cmpf uno, %cst, %cst : f32
            %10 = arith.select %9, %cst, %8 : f32
            %c300_9 = arith.constant 300 : index
            %11 = arith.muli %arg2, %c300_9 : index
            %12 = arith.addi %11, %arg3 : index
            %c100_10 = arith.constant 100 : index
            %13 = arith.muli %12, %c100_10 : index
            %14 = arith.addi %13, %arg4 : index
            %c10_11 = arith.constant 10 : index
            %15 = arith.muli %14, %c10_11 : index
            %16 = arith.addi %15, %arg5 : index
            memref.store %10, %alloc[%16] : memref<300xf32>
          }
        }
      }
    }
    memref.copy %alloc, %arg1 : memref<300xf32> to memref<300xf32>
    return
  }
  func.func private @forward(%arg0: memref<300xf32>, %arg1: memref<300xf32>, %arg2: memref<300xf32>) attributes {itypes = "__", otypes = "_"} {
    %alloc = memref.alloc() : memref<300xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c1 step %c1_0 {
      %c0_2 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %c1_3 = arith.constant 1 : index
      scf.for %arg4 = %c0_2 to %c3 step %c1_3 {
        %c0_4 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg5 = %c0_4 to %c10 step %c1_5 {
          %c0_6 = arith.constant 0 : index
          %c10_7 = arith.constant 10 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg6 = %c0_6 to %c10_7 step %c1_8 {
            %c300 = arith.constant 300 : index
            %0 = arith.muli %arg3, %c300 : index
            %1 = arith.addi %0, %arg4 : index
            %c100 = arith.constant 100 : index
            %2 = arith.muli %1, %c100 : index
            %3 = arith.addi %2, %arg5 : index
            %c10_9 = arith.constant 10 : index
            %4 = arith.muli %3, %c10_9 : index
            %5 = arith.addi %4, %arg6 : index
            %6 = memref.load %arg0[%5] : memref<300xf32>
            %c300_10 = arith.constant 300 : index
            %7 = arith.muli %arg3, %c300_10 : index
            %8 = arith.addi %7, %arg4 : index
            %c100_11 = arith.constant 100 : index
            %9 = arith.muli %8, %c100_11 : index
            %10 = arith.addi %9, %arg5 : index
            %c10_12 = arith.constant 10 : index
            %11 = arith.muli %10, %c10_12 : index
            %12 = arith.addi %11, %arg6 : index
            %13 = memref.load %arg1[%12] : memref<300xf32>
            %14 = arith.addf %6, %13 : f32
            %c300_13 = arith.constant 300 : index
            %15 = arith.muli %arg3, %c300_13 : index
            %16 = arith.addi %15, %arg4 : index
            %c100_14 = arith.constant 100 : index
            %17 = arith.muli %16, %c100_14 : index
            %18 = arith.addi %17, %arg5 : index
            %c10_15 = arith.constant 10 : index
            %19 = arith.muli %18, %c10_15 : index
            %20 = arith.addi %19, %arg6 : index
            memref.store %14, %alloc[%20] : memref<300xf32>
          }
        }
      }
    }
    %alloc_1 = memref.alloc() : memref<300xf32>
    call @relu4d_0(%alloc, %alloc_1) : (memref<300xf32>, memref<300xf32>) -> ()
    memref.copy %alloc_1, %arg2 : memref<300xf32> to memref<300xf32>
    return
  }
}
```

We still have the problematic `memref.cop` which we'll convert to `affine.for` first using our custom pass.

If you run the pass, we see `memref.copy` being converted to `affine.for` as:

```mlir
module {
  func.func private @relu4d_0(%arg0: memref<300xf32>, %arg1: memref<300xf32>) attributes {itypes = "_", otypes = "_"} {
    %c100 = arith.constant 100 : index
    %c10 = arith.constant 10 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<300xf32>
    scf.for %arg2 = %c0 to %c3 step %c1 {
      scf.for %arg3 = %c0 to %c10 step %c1 {
        scf.for %arg4 = %c0 to %c10 step %c1 {
          %0 = arith.muli %arg2, %c100 : index
          %1 = arith.addi %0, %arg3 : index
          %2 = arith.muli %1, %c10 : index
          %3 = arith.addi %2, %arg4 : index
          %4 = memref.load %arg0[%3] : memref<300xf32>
          %5 = arith.cmpf ugt, %4, %cst : f32
          %6 = arith.select %5, %4, %cst : f32
          %7 = arith.muli %arg2, %c100 : index
          %8 = arith.addi %7, %arg3 : index
          %9 = arith.muli %8, %c10 : index
          %10 = arith.addi %9, %arg4 : index
          memref.store %6, %alloc[%10] : memref<300xf32>
        }
      }
    }
    %c0_0 = arith.constant 0 : index
    %c300 = arith.constant 300 : index
    %c1_1 = arith.constant 1 : index
    scf.for %arg2 = %c0_0 to %c300 step %c1_1 {
      %0 = memref.load %alloc[%arg2] : memref<300xf32>
      memref.store %0, %arg1[%arg2] : memref<300xf32>
    }
    return
  }
  func.func private @forward(%arg0: memref<300xf32>, %arg1: memref<300xf32>, %arg2: memref<300xf32>) attributes {itypes = "__", otypes = "_"} {
    %c100 = arith.constant 100 : index
    %c10 = arith.constant 10 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<300xf32>
    scf.for %arg3 = %c0 to %c3 step %c1 {
      scf.for %arg4 = %c0 to %c10 step %c1 {
        scf.for %arg5 = %c0 to %c10 step %c1 {
          %0 = arith.muli %arg3, %c100 : index
          %1 = arith.addi %0, %arg4 : index
          %2 = arith.muli %1, %c10 : index
          %3 = arith.addi %2, %arg5 : index
          %4 = memref.load %arg0[%3] : memref<300xf32>
          %5 = arith.muli %arg3, %c100 : index
          %6 = arith.addi %5, %arg4 : index
          %7 = arith.muli %6, %c10 : index
          %8 = arith.addi %7, %arg5 : index
          %9 = memref.load %arg1[%8] : memref<300xf32>
          %10 = arith.addf %4, %9 : f32
          %11 = arith.muli %arg3, %c100 : index
          %12 = arith.addi %11, %arg4 : index
          %13 = arith.muli %12, %c10 : index
          %14 = arith.addi %13, %arg5 : index
          memref.store %10, %alloc[%14] : memref<300xf32>
        }
      }
    }
    %alloc_0 = memref.alloc() : memref<300xf32>
    call @relu4d_0(%alloc, %alloc_0) : (memref<300xf32>, memref<300xf32>) -> ()
    %c0_1 = arith.constant 0 : index
    %c300 = arith.constant 300 : index
    %c1_2 = arith.constant 1 : index
    scf.for %arg3 = %c0_1 to %c300 step %c1_2 {
      %0 = memref.load %alloc_0[%arg3] : memref<300xf32>
      memref.store %0, %arg2[%arg3] : memref<300xf32>
    }
    return
  }
}
```

We don't have any `memref.copy` now, so we should now lower it to `scf.for` and then to calyx (hopefully!). We'll again use `-lower-affine` from `mlir-opt` to get down to `scf.for` and then again use `circt-opt` to lower to calyx (we succeed this time) and here is what we get:

```calyx
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%done: i1 {done}) {
    %mem_4.addr0, %mem_4.clk, %mem_4.reset, %mem_4.content_en, %mem_4.write_en, %mem_4.write_data, %mem_4.read_data, %mem_4.done = calyx.seq_mem @mem_4 <[300] x 32> [9] {external = true} : i9, i1, i1, i1, i1, i32, i32, i1
    %mem_3.addr0, %mem_3.clk, %mem_3.reset, %mem_3.content_en, %mem_3.write_en, %mem_3.write_data, %mem_3.read_data, %mem_3.done = calyx.seq_mem @mem_3 <[300] x 32> [9] {external = true} : i9, i1, i1, i1, i1, i32, i32, i1
    %mem_2.addr0, %mem_2.clk, %mem_2.reset, %mem_2.content_en, %mem_2.write_en, %mem_2.write_data, %mem_2.read_data, %mem_2.done = calyx.seq_mem @mem_2 <[300] x 32> [9] {external = true} : i9, i1, i1, i1, i1, i32, i32, i1
    %mem_1.addr0, %mem_1.clk, %mem_1.reset, %mem_1.content_en, %mem_1.write_en, %mem_1.write_data, %mem_1.read_data, %mem_1.done = calyx.seq_mem @mem_1 <[300] x 32> [9] {external = true} : i9, i1, i1, i1, i1, i32, i32, i1
    %mem_0.addr0, %mem_0.clk, %mem_0.reset, %mem_0.content_en, %mem_0.write_en, %mem_0.write_data, %mem_0.read_data, %mem_0.done = calyx.seq_mem @mem_0 <[300] x 32> [9] {external = true} : i9, i1, i1, i1, i1, i32, i32, i1
    %forward_instance.clk, %forward_instance.reset, %forward_instance.go, %forward_instance.done = calyx.instance @forward_instance of @forward : i1, i1, i1, i1
    calyx.wires {
    }
    calyx.control {
      calyx.seq {
        calyx.seq {
          calyx.invoke @forward_instance[arg_mem_0 = mem_0, arg_mem_1 = mem_1, arg_mem_2 = mem_2, arg_mem_3 = mem_3, arg_mem_4 = mem_4]() -> ()
        }
      }
    }
  } {toplevel}
  calyx.component @relu4d_0(%clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%done: i1 {done}) {
    %c1_i32 = hw.constant 1 : i32
    %true = hw.constant true
    %false = hw.constant false
    %cst = calyx.constant @cst_0 <0.000000e+00 : f32> : i32
    %c0_i32 = hw.constant 0 : i32
    %c10_i32 = hw.constant 10 : i32
    %c100_i32 = hw.constant 100 : i32
    %std_slice_3.in, %std_slice_3.out = calyx.std_slice @std_slice_3 : i32, i9
    %std_slice_2.in, %std_slice_2.out = calyx.std_slice @std_slice_2 : i32, i9
    %std_slice_1.in, %std_slice_1.out = calyx.std_slice @std_slice_1 : i32, i9
    %std_slice_0.in, %std_slice_0.out = calyx.std_slice @std_slice_0 : i32, i9
    %std_add_7.left, %std_add_7.right, %std_add_7.out = calyx.std_add @std_add_7 : i32, i32, i32
    %load_0_reg.in, %load_0_reg.write_en, %load_0_reg.clk, %load_0_reg.reset, %load_0_reg.out, %load_0_reg.done = calyx.register @load_0_reg : i32, i1, i1, i1, i32, i1
    %std_add_6.left, %std_add_6.right, %std_add_6.out = calyx.std_add @std_add_6 : i32, i32, i32
    %std_add_5.left, %std_add_5.right, %std_add_5.out = calyx.std_add @std_add_5 : i32, i32, i32
    %std_add_4.left, %std_add_4.right, %std_add_4.out = calyx.std_add @std_add_4 : i32, i32, i32
    %std_add_3.left, %std_add_3.right, %std_add_3.out = calyx.std_add @std_add_3 : i32, i32, i32
    %muli_3_reg.in, %muli_3_reg.write_en, %muli_3_reg.clk, %muli_3_reg.reset, %muli_3_reg.out, %muli_3_reg.done = calyx.register @muli_3_reg : i32, i1, i1, i1, i32, i1
    %std_mult_pipe_3.clk, %std_mult_pipe_3.reset, %std_mult_pipe_3.go, %std_mult_pipe_3.left, %std_mult_pipe_3.right, %std_mult_pipe_3.out, %std_mult_pipe_3.done = calyx.std_mult_pipe @std_mult_pipe_3 : i1, i1, i1, i32, i32, i32, i1
    %std_add_2.left, %std_add_2.right, %std_add_2.out = calyx.std_add @std_add_2 : i32, i32, i32
    %muli_2_reg.in, %muli_2_reg.write_en, %muli_2_reg.clk, %muli_2_reg.reset, %muli_2_reg.out, %muli_2_reg.done = calyx.register @muli_2_reg : i32, i1, i1, i1, i32, i1
    %std_mult_pipe_2.clk, %std_mult_pipe_2.reset, %std_mult_pipe_2.go, %std_mult_pipe_2.left, %std_mult_pipe_2.right, %std_mult_pipe_2.out, %std_mult_pipe_2.done = calyx.std_mult_pipe @std_mult_pipe_2 : i1, i1, i1, i32, i32, i32, i1
    %std_mux_0.cond, %std_mux_0.tru, %std_mux_0.fal, %std_mux_0.out = calyx.std_mux @std_mux_0 : i1, i32, i32, i32
    %std_and_0.left, %std_and_0.right, %std_and_0.out = calyx.std_and @std_and_0 : i1, i1, i1
    %std_or_0.left, %std_or_0.right, %std_or_0.out = calyx.std_or @std_or_0 : i1, i1, i1
    %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
    %compare_port_0_reg.in, %compare_port_0_reg.write_en, %compare_port_0_reg.clk, %compare_port_0_reg.reset, %compare_port_0_reg.out, %compare_port_0_reg.done = calyx.register @compare_port_0_reg : i1, i1, i1, i1, i1, i1
    %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
    %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i32, i32, i1, i1, i1, i1, i1, i5, i1
    %std_add_1.left, %std_add_1.right, %std_add_1.out = calyx.std_add @std_add_1 : i32, i32, i32
    %muli_1_reg.in, %muli_1_reg.write_en, %muli_1_reg.clk, %muli_1_reg.reset, %muli_1_reg.out, %muli_1_reg.done = calyx.register @muli_1_reg : i32, i1, i1, i1, i32, i1
    %std_mult_pipe_1.clk, %std_mult_pipe_1.reset, %std_mult_pipe_1.go, %std_mult_pipe_1.left, %std_mult_pipe_1.right, %std_mult_pipe_1.out, %std_mult_pipe_1.done = calyx.std_mult_pipe @std_mult_pipe_1 : i1, i1, i1, i32, i32, i32, i1
    %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
    %muli_0_reg.in, %muli_0_reg.write_en, %muli_0_reg.clk, %muli_0_reg.reset, %muli_0_reg.out, %muli_0_reg.done = calyx.register @muli_0_reg : i32, i1, i1, i1, i32, i1
    %std_mult_pipe_0.clk, %std_mult_pipe_0.reset, %std_mult_pipe_0.go, %std_mult_pipe_0.left, %std_mult_pipe_0.right, %std_mult_pipe_0.out, %std_mult_pipe_0.done = calyx.std_mult_pipe @std_mult_pipe_0 : i1, i1, i1, i32, i32, i32, i1
    %mem_0.addr0, %mem_0.clk, %mem_0.reset, %mem_0.content_en, %mem_0.write_en, %mem_0.write_data, %mem_0.read_data, %mem_0.done = calyx.seq_mem @mem_0 <[300] x 32> [9] {external = true} : i9, i1, i1, i1, i1, i32, i32, i1
    %for_3_induction_var_reg.in, %for_3_induction_var_reg.write_en, %for_3_induction_var_reg.clk, %for_3_induction_var_reg.reset, %for_3_induction_var_reg.out, %for_3_induction_var_reg.done = calyx.register @for_3_induction_var_reg : i32, i1, i1, i1, i32, i1
    %for_2_induction_var_reg.in, %for_2_induction_var_reg.write_en, %for_2_induction_var_reg.clk, %for_2_induction_var_reg.reset, %for_2_induction_var_reg.out, %for_2_induction_var_reg.done = calyx.register @for_2_induction_var_reg : i32, i1, i1, i1, i32, i1
    %for_1_induction_var_reg.in, %for_1_induction_var_reg.write_en, %for_1_induction_var_reg.clk, %for_1_induction_var_reg.reset, %for_1_induction_var_reg.out, %for_1_induction_var_reg.done = calyx.register @for_1_induction_var_reg : i32, i1, i1, i1, i32, i1
    %for_0_induction_var_reg.in, %for_0_induction_var_reg.write_en, %for_0_induction_var_reg.clk, %for_0_induction_var_reg.reset, %for_0_induction_var_reg.out, %for_0_induction_var_reg.done = calyx.register @for_0_induction_var_reg : i32, i1, i1, i1, i32, i1
    %arg_mem_1.addr0, %arg_mem_1.clk, %arg_mem_1.reset, %arg_mem_1.content_en, %arg_mem_1.write_en, %arg_mem_1.write_data, %arg_mem_1.read_data, %arg_mem_1.done = calyx.seq_mem @arg_mem_1 <[300] x 32> [9] : i9, i1, i1, i1, i1, i32, i32, i1
    %arg_mem_0.addr0, %arg_mem_0.clk, %arg_mem_0.reset, %arg_mem_0.content_en, %arg_mem_0.write_en, %arg_mem_0.write_data, %arg_mem_0.read_data, %arg_mem_0.done = calyx.seq_mem @arg_mem_0 <[300] x 32> [9] : i9, i1, i1, i1, i1, i32, i32, i1
    calyx.wires {
      calyx.group @init_for_0_induction_var {
        calyx.assign %for_0_induction_var_reg.in = %c0_i32 : i32
        calyx.assign %for_0_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_0_induction_var_reg.done : i1
      }
      calyx.group @init_for_1_induction_var {
        calyx.assign %for_1_induction_var_reg.in = %c0_i32 : i32
        calyx.assign %for_1_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_1_induction_var_reg.done : i1
      }
      calyx.group @init_for_2_induction_var {
        calyx.assign %for_2_induction_var_reg.in = %c0_i32 : i32
        calyx.assign %for_2_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_2_induction_var_reg.done : i1
      }
      calyx.group @init_for_3_induction_var {
        calyx.assign %for_3_induction_var_reg.in = %c0_i32 : i32
        calyx.assign %for_3_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_3_induction_var_reg.done : i1
      }
      calyx.group @bb0_0 {
        calyx.assign %std_mult_pipe_0.left = %for_2_induction_var_reg.out : i32
        calyx.assign %std_mult_pipe_0.right = %c100_i32 : i32
        calyx.assign %muli_0_reg.in = %std_mult_pipe_0.out : i32
        calyx.assign %muli_0_reg.write_en = %std_mult_pipe_0.done : i1
        %0 = comb.xor %std_mult_pipe_0.done, %true : i1
        calyx.assign %std_mult_pipe_0.go = %0 ? %true : i1
        calyx.group_done %muli_0_reg.done : i1
      }
      calyx.group @bb0_2 {
        calyx.assign %std_mult_pipe_1.left = %std_add_0.out : i32
        calyx.assign %std_mult_pipe_1.right = %c10_i32 : i32
        calyx.assign %muli_1_reg.in = %std_mult_pipe_1.out : i32
        calyx.assign %muli_1_reg.write_en = %std_mult_pipe_1.done : i1
        %0 = comb.xor %std_mult_pipe_1.done, %true : i1
        calyx.assign %std_mult_pipe_1.go = %0 ? %true : i1
        calyx.assign %std_add_0.left = %muli_0_reg.out : i32
        calyx.assign %std_add_0.right = %for_1_induction_var_reg.out : i32
        calyx.group_done %muli_1_reg.done : i1
      }
      calyx.group @bb0_4 {
        calyx.assign %std_slice_3.in = %std_add_1.out : i32
        calyx.assign %arg_mem_0.addr0 = %std_slice_3.out : i9
        calyx.assign %arg_mem_0.content_en = %true : i1
        calyx.assign %arg_mem_0.write_en = %false : i1
        calyx.assign %std_add_1.left = %muli_1_reg.out : i32
        calyx.assign %std_add_1.right = %for_0_induction_var_reg.out : i32
        calyx.group_done %arg_mem_0.done : i1
      }
      calyx.group @bb0_5 {
        calyx.assign %std_compareFN_0.left = %arg_mem_0.read_data : i32
        calyx.assign %std_compareFN_0.right = %cst : i32
        calyx.assign %std_compareFN_0.signaling = %true : i1
        calyx.assign %compare_port_0_reg.write_en = %std_compareFN_0.done : i1
        calyx.assign %compare_port_0_reg.in = %std_compareFN_0.gt : i1
        calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
        calyx.assign %unordered_port_0_reg.in = %std_compareFN_0.unordered : i1
        calyx.assign %std_or_0.left = %compare_port_0_reg.out : i1
        calyx.assign %std_or_0.right = %unordered_port_0_reg.out : i1
        calyx.assign %std_and_0.left = %compare_port_0_reg.done : i1
        calyx.assign %std_and_0.right = %unordered_port_0_reg.done : i1
        calyx.assign %cmpf_0_reg.in = %std_or_0.out : i1
        calyx.assign %cmpf_0_reg.write_en = %std_and_0.out : i1
        %0 = comb.xor %std_compareFN_0.done, %true : i1
        calyx.assign %std_compareFN_0.go = %0 ? %true : i1
        calyx.group_done %cmpf_0_reg.done : i1
      }
      calyx.group @bb0_7 {
        calyx.assign %std_mult_pipe_2.left = %for_2_induction_var_reg.out : i32
        calyx.assign %std_mult_pipe_2.right = %c100_i32 : i32
        calyx.assign %muli_2_reg.in = %std_mult_pipe_2.out : i32
        calyx.assign %muli_2_reg.write_en = %std_mult_pipe_2.done : i1
        %0 = comb.xor %std_mult_pipe_2.done, %true : i1
        calyx.assign %std_mult_pipe_2.go = %0 ? %true : i1
        calyx.group_done %muli_2_reg.done : i1
      }
      calyx.group @bb0_9 {
        calyx.assign %std_mult_pipe_3.left = %std_add_2.out : i32
        calyx.assign %std_mult_pipe_3.right = %c10_i32 : i32
        calyx.assign %muli_3_reg.in = %std_mult_pipe_3.out : i32
        calyx.assign %muli_3_reg.write_en = %std_mult_pipe_3.done : i1
        %0 = comb.xor %std_mult_pipe_3.done, %true : i1
        calyx.assign %std_mult_pipe_3.go = %0 ? %true : i1
        calyx.assign %std_add_2.left = %muli_2_reg.out : i32
        calyx.assign %std_add_2.right = %for_1_induction_var_reg.out : i32
        calyx.group_done %muli_3_reg.done : i1
      }
      calyx.group @bb0_11 {
        calyx.assign %std_slice_2.in = %std_add_3.out : i32
        calyx.assign %mem_0.addr0 = %std_slice_2.out : i9
        calyx.assign %mem_0.write_data = %std_mux_0.out : i32
        calyx.assign %mem_0.write_en = %true : i1
        calyx.assign %mem_0.content_en = %true : i1
        calyx.assign %std_add_3.left = %muli_3_reg.out : i32
        calyx.assign %std_add_3.right = %for_0_induction_var_reg.out : i32
        calyx.assign %std_mux_0.cond = %cmpf_0_reg.out : i1
        calyx.assign %std_mux_0.tru = %arg_mem_0.read_data : i32
        calyx.assign %std_mux_0.fal = %cst : i32
        calyx.group_done %mem_0.done : i1
      }
      calyx.group @incr_for_0_induction_var {
        calyx.assign %std_add_4.left = %for_0_induction_var_reg.out : i32
        calyx.assign %std_add_4.right = %c1_i32 : i32
        calyx.assign %for_0_induction_var_reg.in = %std_add_4.out : i32
        calyx.assign %for_0_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_0_induction_var_reg.done : i1
      }
      calyx.group @incr_for_1_induction_var {
        calyx.assign %std_add_5.left = %for_1_induction_var_reg.out : i32
        calyx.assign %std_add_5.right = %c1_i32 : i32
        calyx.assign %for_1_induction_var_reg.in = %std_add_5.out : i32
        calyx.assign %for_1_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_1_induction_var_reg.done : i1
      }
      calyx.group @incr_for_2_induction_var {
        calyx.assign %std_add_6.left = %for_2_induction_var_reg.out : i32
        calyx.assign %std_add_6.right = %c1_i32 : i32
        calyx.assign %for_2_induction_var_reg.in = %std_add_6.out : i32
        calyx.assign %for_2_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_2_induction_var_reg.done : i1
      }
      calyx.group @bb0_12 {
        calyx.assign %std_slice_1.in = %for_3_induction_var_reg.out : i32
        calyx.assign %mem_0.addr0 = %std_slice_1.out : i9
        calyx.assign %mem_0.content_en = %true : i1
        calyx.assign %mem_0.write_en = %false : i1
        calyx.assign %load_0_reg.in = %mem_0.read_data : i32
        calyx.assign %load_0_reg.write_en = %mem_0.done : i1
        calyx.group_done %load_0_reg.done : i1
      }
      calyx.group @bb0_13 {
        calyx.assign %std_slice_0.in = %for_3_induction_var_reg.out : i32
        calyx.assign %arg_mem_1.addr0 = %std_slice_0.out : i9
        calyx.assign %arg_mem_1.write_data = %load_0_reg.out : i32
        calyx.assign %arg_mem_1.write_en = %true : i1
        calyx.assign %arg_mem_1.content_en = %true : i1
        calyx.group_done %arg_mem_1.done : i1
      }
      calyx.group @incr_for_3_induction_var {
        calyx.assign %std_add_7.left = %for_3_induction_var_reg.out : i32
        calyx.assign %std_add_7.right = %c1_i32 : i32
        calyx.assign %for_3_induction_var_reg.in = %std_add_7.out : i32
        calyx.assign %for_3_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_3_induction_var_reg.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        calyx.seq {
          calyx.par {
            calyx.enable @init_for_2_induction_var
          }
          calyx.repeat 3 {
            calyx.seq {
              calyx.par {
                calyx.enable @init_for_1_induction_var
              }
              calyx.repeat 10 {
                calyx.seq {
                  calyx.par {
                    calyx.enable @init_for_0_induction_var
                  }
                  calyx.repeat 10 {
                    calyx.seq {
                      calyx.seq {
                        calyx.enable @bb0_0
                        calyx.enable @bb0_2
                        calyx.enable @bb0_4
                        calyx.enable @bb0_5
                        calyx.enable @bb0_7
                        calyx.enable @bb0_9
                        calyx.enable @bb0_11
                      }
                      calyx.enable @incr_for_0_induction_var
                    }
                  }
                  calyx.enable @incr_for_1_induction_var
                }
              }
              calyx.enable @incr_for_2_induction_var
            }
          }
          calyx.par {
            calyx.enable @init_for_3_induction_var
          }
          calyx.repeat 300 {
            calyx.seq {
              calyx.seq {
                calyx.enable @bb0_12
                calyx.enable @bb0_13
              }
              calyx.enable @incr_for_3_induction_var
            }
          }
        }
      }
    }
  }
  calyx.component @forward(%clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%done: i1 {done}) {
    %c1_i32 = hw.constant 1 : i32
    %true = hw.constant true
    %false = hw.constant false
    %c0_i32 = hw.constant 0 : i32
    %c10_i32 = hw.constant 10 : i32
    %c100_i32 = hw.constant 100 : i32
    %std_slice_4.in, %std_slice_4.out = calyx.std_slice @std_slice_4 : i32, i9
    %std_slice_3.in, %std_slice_3.out = calyx.std_slice @std_slice_3 : i32, i9
    %std_slice_2.in, %std_slice_2.out = calyx.std_slice @std_slice_2 : i32, i9
    %std_slice_1.in, %std_slice_1.out = calyx.std_slice @std_slice_1 : i32, i9
    %std_slice_0.in, %std_slice_0.out = calyx.std_slice @std_slice_0 : i32, i9
    %std_add_9.left, %std_add_9.right, %std_add_9.out = calyx.std_add @std_add_9 : i32, i32, i32
    %std_add_8.left, %std_add_8.right, %std_add_8.out = calyx.std_add @std_add_8 : i32, i32, i32
    %std_add_7.left, %std_add_7.right, %std_add_7.out = calyx.std_add @std_add_7 : i32, i32, i32
    %std_add_6.left, %std_add_6.right, %std_add_6.out = calyx.std_add @std_add_6 : i32, i32, i32
    %std_add_5.left, %std_add_5.right, %std_add_5.out = calyx.std_add @std_add_5 : i32, i32, i32
    %muli_5_reg.in, %muli_5_reg.write_en, %muli_5_reg.clk, %muli_5_reg.reset, %muli_5_reg.out, %muli_5_reg.done = calyx.register @muli_5_reg : i32, i1, i1, i1, i32, i1
    %std_mult_pipe_5.clk, %std_mult_pipe_5.reset, %std_mult_pipe_5.go, %std_mult_pipe_5.left, %std_mult_pipe_5.right, %std_mult_pipe_5.out, %std_mult_pipe_5.done = calyx.std_mult_pipe @std_mult_pipe_5 : i1, i1, i1, i32, i32, i32, i1
    %std_add_4.left, %std_add_4.right, %std_add_4.out = calyx.std_add @std_add_4 : i32, i32, i32
    %muli_4_reg.in, %muli_4_reg.write_en, %muli_4_reg.clk, %muli_4_reg.reset, %muli_4_reg.out, %muli_4_reg.done = calyx.register @muli_4_reg : i32, i1, i1, i1, i32, i1
    %std_mult_pipe_4.clk, %std_mult_pipe_4.reset, %std_mult_pipe_4.go, %std_mult_pipe_4.left, %std_mult_pipe_4.right, %std_mult_pipe_4.out, %std_mult_pipe_4.done = calyx.std_mult_pipe @std_mult_pipe_4 : i1, i1, i1, i32, i32, i32, i1
    %addf_0_reg.in, %addf_0_reg.write_en, %addf_0_reg.clk, %addf_0_reg.reset, %addf_0_reg.out, %addf_0_reg.done = calyx.register @addf_0_reg : i32, i1, i1, i1, i32, i1
    %std_addFN_0.clk, %std_addFN_0.reset, %std_addFN_0.go, %std_addFN_0.control, %std_addFN_0.subOp, %std_addFN_0.left, %std_addFN_0.right, %std_addFN_0.roundingMode, %std_addFN_0.out, %std_addFN_0.exceptionalFlags, %std_addFN_0.done = calyx.ieee754.add @std_addFN_0 : i1, i1, i1, i1, i1, i32, i32, i3, i32, i5, i1
    %std_add_3.left, %std_add_3.right, %std_add_3.out = calyx.std_add @std_add_3 : i32, i32, i32
    %muli_3_reg.in, %muli_3_reg.write_en, %muli_3_reg.clk, %muli_3_reg.reset, %muli_3_reg.out, %muli_3_reg.done = calyx.register @muli_3_reg : i32, i1, i1, i1, i32, i1
    %std_mult_pipe_3.clk, %std_mult_pipe_3.reset, %std_mult_pipe_3.go, %std_mult_pipe_3.left, %std_mult_pipe_3.right, %std_mult_pipe_3.out, %std_mult_pipe_3.done = calyx.std_mult_pipe @std_mult_pipe_3 : i1, i1, i1, i32, i32, i32, i1
    %std_add_2.left, %std_add_2.right, %std_add_2.out = calyx.std_add @std_add_2 : i32, i32, i32
    %muli_2_reg.in, %muli_2_reg.write_en, %muli_2_reg.clk, %muli_2_reg.reset, %muli_2_reg.out, %muli_2_reg.done = calyx.register @muli_2_reg : i32, i1, i1, i1, i32, i1
    %std_mult_pipe_2.clk, %std_mult_pipe_2.reset, %std_mult_pipe_2.go, %std_mult_pipe_2.left, %std_mult_pipe_2.right, %std_mult_pipe_2.out, %std_mult_pipe_2.done = calyx.std_mult_pipe @std_mult_pipe_2 : i1, i1, i1, i32, i32, i32, i1
    %std_add_1.left, %std_add_1.right, %std_add_1.out = calyx.std_add @std_add_1 : i32, i32, i32
    %muli_1_reg.in, %muli_1_reg.write_en, %muli_1_reg.clk, %muli_1_reg.reset, %muli_1_reg.out, %muli_1_reg.done = calyx.register @muli_1_reg : i32, i1, i1, i1, i32, i1
    %std_mult_pipe_1.clk, %std_mult_pipe_1.reset, %std_mult_pipe_1.go, %std_mult_pipe_1.left, %std_mult_pipe_1.right, %std_mult_pipe_1.out, %std_mult_pipe_1.done = calyx.std_mult_pipe @std_mult_pipe_1 : i1, i1, i1, i32, i32, i32, i1
    %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
    %muli_0_reg.in, %muli_0_reg.write_en, %muli_0_reg.clk, %muli_0_reg.reset, %muli_0_reg.out, %muli_0_reg.done = calyx.register @muli_0_reg : i32, i1, i1, i1, i32, i1
    %std_mult_pipe_0.clk, %std_mult_pipe_0.reset, %std_mult_pipe_0.go, %std_mult_pipe_0.left, %std_mult_pipe_0.right, %std_mult_pipe_0.out, %std_mult_pipe_0.done = calyx.std_mult_pipe @std_mult_pipe_0 : i1, i1, i1, i32, i32, i32, i1
    %for_3_induction_var_reg.in, %for_3_induction_var_reg.write_en, %for_3_induction_var_reg.clk, %for_3_induction_var_reg.reset, %for_3_induction_var_reg.out, %for_3_induction_var_reg.done = calyx.register @for_3_induction_var_reg : i32, i1, i1, i1, i32, i1
    %for_2_induction_var_reg.in, %for_2_induction_var_reg.write_en, %for_2_induction_var_reg.clk, %for_2_induction_var_reg.reset, %for_2_induction_var_reg.out, %for_2_induction_var_reg.done = calyx.register @for_2_induction_var_reg : i32, i1, i1, i1, i32, i1
    %for_1_induction_var_reg.in, %for_1_induction_var_reg.write_en, %for_1_induction_var_reg.clk, %for_1_induction_var_reg.reset, %for_1_induction_var_reg.out, %for_1_induction_var_reg.done = calyx.register @for_1_induction_var_reg : i32, i1, i1, i1, i32, i1
    %for_0_induction_var_reg.in, %for_0_induction_var_reg.write_en, %for_0_induction_var_reg.clk, %for_0_induction_var_reg.reset, %for_0_induction_var_reg.out, %for_0_induction_var_reg.done = calyx.register @for_0_induction_var_reg : i32, i1, i1, i1, i32, i1
    %relu4d_0_instance.clk, %relu4d_0_instance.reset, %relu4d_0_instance.go, %relu4d_0_instance.done = calyx.instance @relu4d_0_instance of @relu4d_0 : i1, i1, i1, i1
    %arg_mem_4.addr0, %arg_mem_4.clk, %arg_mem_4.reset, %arg_mem_4.content_en, %arg_mem_4.write_en, %arg_mem_4.write_data, %arg_mem_4.read_data, %arg_mem_4.done = calyx.seq_mem @arg_mem_4 <[300] x 32> [9] : i9, i1, i1, i1, i1, i32, i32, i1
    %arg_mem_3.addr0, %arg_mem_3.clk, %arg_mem_3.reset, %arg_mem_3.content_en, %arg_mem_3.write_en, %arg_mem_3.write_data, %arg_mem_3.read_data, %arg_mem_3.done = calyx.seq_mem @arg_mem_3 <[300] x 32> [9] : i9, i1, i1, i1, i1, i32, i32, i1
    %arg_mem_2.addr0, %arg_mem_2.clk, %arg_mem_2.reset, %arg_mem_2.content_en, %arg_mem_2.write_en, %arg_mem_2.write_data, %arg_mem_2.read_data, %arg_mem_2.done = calyx.seq_mem @arg_mem_2 <[300] x 32> [9] : i9, i1, i1, i1, i1, i32, i32, i1
    %arg_mem_1.addr0, %arg_mem_1.clk, %arg_mem_1.reset, %arg_mem_1.content_en, %arg_mem_1.write_en, %arg_mem_1.write_data, %arg_mem_1.read_data, %arg_mem_1.done = calyx.seq_mem @arg_mem_1 <[300] x 32> [9] : i9, i1, i1, i1, i1, i32, i32, i1
    %arg_mem_0.addr0, %arg_mem_0.clk, %arg_mem_0.reset, %arg_mem_0.content_en, %arg_mem_0.write_en, %arg_mem_0.write_data, %arg_mem_0.read_data, %arg_mem_0.done = calyx.seq_mem @arg_mem_0 <[300] x 32> [9] : i9, i1, i1, i1, i1, i32, i32, i1
    calyx.wires {
      calyx.group @init_for_0_induction_var {
        calyx.assign %for_0_induction_var_reg.in = %c0_i32 : i32
        calyx.assign %for_0_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_0_induction_var_reg.done : i1
      }
      calyx.group @init_for_1_induction_var {
        calyx.assign %for_1_induction_var_reg.in = %c0_i32 : i32
        calyx.assign %for_1_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_1_induction_var_reg.done : i1
      }
      calyx.group @init_for_2_induction_var {
        calyx.assign %for_2_induction_var_reg.in = %c0_i32 : i32
        calyx.assign %for_2_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_2_induction_var_reg.done : i1
      }
      calyx.group @init_for_3_induction_var {
        calyx.assign %for_3_induction_var_reg.in = %c0_i32 : i32
        calyx.assign %for_3_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_3_induction_var_reg.done : i1
      }
      calyx.group @bb0_0 {
        calyx.assign %std_mult_pipe_0.left = %for_2_induction_var_reg.out : i32
        calyx.assign %std_mult_pipe_0.right = %c100_i32 : i32
        calyx.assign %muli_0_reg.in = %std_mult_pipe_0.out : i32
        calyx.assign %muli_0_reg.write_en = %std_mult_pipe_0.done : i1
        %0 = comb.xor %std_mult_pipe_0.done, %true : i1
        calyx.assign %std_mult_pipe_0.go = %0 ? %true : i1
        calyx.group_done %muli_0_reg.done : i1
      }
      calyx.group @bb0_2 {
        calyx.assign %std_mult_pipe_1.left = %std_add_0.out : i32
        calyx.assign %std_mult_pipe_1.right = %c10_i32 : i32
        calyx.assign %muli_1_reg.in = %std_mult_pipe_1.out : i32
        calyx.assign %muli_1_reg.write_en = %std_mult_pipe_1.done : i1
        %0 = comb.xor %std_mult_pipe_1.done, %true : i1
        calyx.assign %std_mult_pipe_1.go = %0 ? %true : i1
        calyx.assign %std_add_0.left = %muli_0_reg.out : i32
        calyx.assign %std_add_0.right = %for_1_induction_var_reg.out : i32
        calyx.group_done %muli_1_reg.done : i1
      }
      calyx.group @bb0_4 {
        calyx.assign %std_slice_4.in = %std_add_1.out : i32
        calyx.assign %arg_mem_0.addr0 = %std_slice_4.out : i9
        calyx.assign %arg_mem_0.content_en = %true : i1
        calyx.assign %arg_mem_0.write_en = %false : i1
        calyx.assign %std_add_1.left = %muli_1_reg.out : i32
        calyx.assign %std_add_1.right = %for_0_induction_var_reg.out : i32
        calyx.group_done %arg_mem_0.done : i1
      }
      calyx.group @bb0_5 {
        calyx.assign %std_mult_pipe_2.left = %for_2_induction_var_reg.out : i32
        calyx.assign %std_mult_pipe_2.right = %c100_i32 : i32
        calyx.assign %muli_2_reg.in = %std_mult_pipe_2.out : i32
        calyx.assign %muli_2_reg.write_en = %std_mult_pipe_2.done : i1
        %0 = comb.xor %std_mult_pipe_2.done, %true : i1
        calyx.assign %std_mult_pipe_2.go = %0 ? %true : i1
        calyx.group_done %muli_2_reg.done : i1
      }
      calyx.group @bb0_7 {
        calyx.assign %std_mult_pipe_3.left = %std_add_2.out : i32
        calyx.assign %std_mult_pipe_3.right = %c10_i32 : i32
        calyx.assign %muli_3_reg.in = %std_mult_pipe_3.out : i32
        calyx.assign %muli_3_reg.write_en = %std_mult_pipe_3.done : i1
        %0 = comb.xor %std_mult_pipe_3.done, %true : i1
        calyx.assign %std_mult_pipe_3.go = %0 ? %true : i1
        calyx.assign %std_add_2.left = %muli_2_reg.out : i32
        calyx.assign %std_add_2.right = %for_1_induction_var_reg.out : i32
        calyx.group_done %muli_3_reg.done : i1
      }
      calyx.group @bb0_9 {
        calyx.assign %std_slice_3.in = %std_add_3.out : i32
        calyx.assign %arg_mem_1.addr0 = %std_slice_3.out : i9
        calyx.assign %arg_mem_1.content_en = %true : i1
        calyx.assign %arg_mem_1.write_en = %false : i1
        calyx.assign %std_add_3.left = %muli_3_reg.out : i32
        calyx.assign %std_add_3.right = %for_0_induction_var_reg.out : i32
        calyx.group_done %arg_mem_1.done : i1
      }
      calyx.group @bb0_10 {
        calyx.assign %std_addFN_0.left = %arg_mem_0.read_data : i32
        calyx.assign %std_addFN_0.right = %arg_mem_1.read_data : i32
        calyx.assign %addf_0_reg.in = %std_addFN_0.out : i32
        calyx.assign %addf_0_reg.write_en = %std_addFN_0.done : i1
        %0 = comb.xor %std_addFN_0.done, %true : i1
        calyx.assign %std_addFN_0.go = %0 ? %true : i1
        calyx.assign %std_addFN_0.subOp = %false : i1
        calyx.group_done %addf_0_reg.done : i1
      }
      calyx.group @bb0_11 {
        calyx.assign %std_mult_pipe_4.left = %for_2_induction_var_reg.out : i32
        calyx.assign %std_mult_pipe_4.right = %c100_i32 : i32
        calyx.assign %muli_4_reg.in = %std_mult_pipe_4.out : i32
        calyx.assign %muli_4_reg.write_en = %std_mult_pipe_4.done : i1
        %0 = comb.xor %std_mult_pipe_4.done, %true : i1
        calyx.assign %std_mult_pipe_4.go = %0 ? %true : i1
        calyx.group_done %muli_4_reg.done : i1
      }
      calyx.group @bb0_13 {
        calyx.assign %std_mult_pipe_5.left = %std_add_4.out : i32
        calyx.assign %std_mult_pipe_5.right = %c10_i32 : i32
        calyx.assign %muli_5_reg.in = %std_mult_pipe_5.out : i32
        calyx.assign %muli_5_reg.write_en = %std_mult_pipe_5.done : i1
        %0 = comb.xor %std_mult_pipe_5.done, %true : i1
        calyx.assign %std_mult_pipe_5.go = %0 ? %true : i1
        calyx.assign %std_add_4.left = %muli_4_reg.out : i32
        calyx.assign %std_add_4.right = %for_1_induction_var_reg.out : i32
        calyx.group_done %muli_5_reg.done : i1
      }
      calyx.group @bb0_15 {
        calyx.assign %std_slice_2.in = %std_add_5.out : i32
        calyx.assign %arg_mem_3.addr0 = %std_slice_2.out : i9
        calyx.assign %arg_mem_3.write_data = %addf_0_reg.out : i32
        calyx.assign %arg_mem_3.write_en = %true : i1
        calyx.assign %arg_mem_3.content_en = %true : i1
        calyx.assign %std_add_5.left = %muli_5_reg.out : i32
        calyx.assign %std_add_5.right = %for_0_induction_var_reg.out : i32
        calyx.group_done %arg_mem_3.done : i1
      }
      calyx.group @incr_for_0_induction_var {
        calyx.assign %std_add_6.left = %for_0_induction_var_reg.out : i32
        calyx.assign %std_add_6.right = %c1_i32 : i32
        calyx.assign %for_0_induction_var_reg.in = %std_add_6.out : i32
        calyx.assign %for_0_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_0_induction_var_reg.done : i1
      }
      calyx.group @incr_for_1_induction_var {
        calyx.assign %std_add_7.left = %for_1_induction_var_reg.out : i32
        calyx.assign %std_add_7.right = %c1_i32 : i32
        calyx.assign %for_1_induction_var_reg.in = %std_add_7.out : i32
        calyx.assign %for_1_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_1_induction_var_reg.done : i1
      }
      calyx.group @incr_for_2_induction_var {
        calyx.assign %std_add_8.left = %for_2_induction_var_reg.out : i32
        calyx.assign %std_add_8.right = %c1_i32 : i32
        calyx.assign %for_2_induction_var_reg.in = %std_add_8.out : i32
        calyx.assign %for_2_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_2_induction_var_reg.done : i1
      }
      calyx.group @bb0_16 {
        calyx.assign %std_slice_1.in = %for_3_induction_var_reg.out : i32
        calyx.assign %arg_mem_4.addr0 = %std_slice_1.out : i9
        calyx.assign %arg_mem_4.content_en = %true : i1
        calyx.assign %arg_mem_4.write_en = %false : i1
        calyx.group_done %arg_mem_4.done : i1
      }
      calyx.group @bb0_17 {
        calyx.assign %std_slice_0.in = %for_3_induction_var_reg.out : i32
        calyx.assign %arg_mem_2.addr0 = %std_slice_0.out : i9
        calyx.assign %arg_mem_2.write_data = %arg_mem_4.read_data : i32
        calyx.assign %arg_mem_2.write_en = %true : i1
        calyx.assign %arg_mem_2.content_en = %true : i1
        calyx.group_done %arg_mem_2.done : i1
      }
      calyx.group @incr_for_3_induction_var {
        calyx.assign %std_add_9.left = %for_3_induction_var_reg.out : i32
        calyx.assign %std_add_9.right = %c1_i32 : i32
        calyx.assign %for_3_induction_var_reg.in = %std_add_9.out : i32
        calyx.assign %for_3_induction_var_reg.write_en = %true : i1
        calyx.group_done %for_3_induction_var_reg.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        calyx.seq {
          calyx.par {
            calyx.enable @init_for_2_induction_var
          }
          calyx.repeat 3 {
            calyx.seq {
              calyx.par {
                calyx.enable @init_for_1_induction_var
              }
              calyx.repeat 10 {
                calyx.seq {
                  calyx.par {
                    calyx.enable @init_for_0_induction_var
                  }
                  calyx.repeat 10 {
                    calyx.seq {
                      calyx.seq {
                        calyx.enable @bb0_0
                        calyx.enable @bb0_2
                        calyx.enable @bb0_4
                        calyx.enable @bb0_5
                        calyx.enable @bb0_7
                        calyx.enable @bb0_9
                        calyx.enable @bb0_10
                        calyx.enable @bb0_11
                        calyx.enable @bb0_13
                        calyx.enable @bb0_15
                      }
                      calyx.enable @incr_for_0_induction_var
                    }
                  }
                  calyx.enable @incr_for_1_induction_var
                }
              }
              calyx.enable @incr_for_2_induction_var
            }
          }
          calyx.seq {
            calyx.invoke @relu4d_0_instance[arg_mem_0 = arg_mem_3, arg_mem_1 = arg_mem_4]() -> ()
          }
          calyx.par {
            calyx.enable @init_for_3_induction_var
          }
          calyx.repeat 300 {
            calyx.seq {
              calyx.seq {
                calyx.enable @bb0_16
                calyx.enable @bb0_17
              }
              calyx.enable @incr_for_3_induction_var
            }
          }
        }
      }
    }
  }
}
```

Wow, this is it! Lovely.
