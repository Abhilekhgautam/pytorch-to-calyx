module {
  func.func private @relu4d_0(%arg0: memref<300xf32>, %arg1: memref<300xf32>) attributes {itypes = "_", otypes = "_"} {
    %c300 = arith.constant 300 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<300xf32>
    scf.for %arg2 = %c0 to %c300 step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<300xf32>
      %1 = arith.cmpf ugt, %0, %cst : f32
      %2 = arith.select %1, %0, %cst : f32
      memref.store %2, %alloc[%arg2] : memref<300xf32>
    }
    scf.for %arg2 = %c0 to %c300 step %c1 {
      %0 = memref.load %alloc[%arg2] : memref<300xf32>
      memref.store %0, %arg1[%arg2] : memref<300xf32>
    }
    return
  }
  func.func private @forward(%arg0: memref<300xf32>, %arg1: memref<300xf32>, %arg2: memref<300xf32>) attributes {itypes = "__", otypes = "_"} {
    %c300 = arith.constant 300 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<300xf32>
    scf.for %arg3 = %c0 to %c300 step %c1 {
      %0 = memref.load %arg0[%arg3] : memref<300xf32>
      %1 = memref.load %arg1[%arg3] : memref<300xf32>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %alloc[%arg3] : memref<300xf32>
    }
    %alloc_0 = memref.alloc() : memref<300xf32>
    call @relu4d_0(%alloc, %alloc_0) : (memref<300xf32>, memref<300xf32>) -> ()
    scf.for %arg3 = %c0 to %c300 step %c1 {
      %0 = memref.load %alloc_0[%arg3] : memref<300xf32>
      memref.store %0, %arg2[%arg3] : memref<300xf32>
    }
    return
  }
}

