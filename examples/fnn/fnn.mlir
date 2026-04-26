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
