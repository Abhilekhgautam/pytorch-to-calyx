#!/bin/bash

# Usage: ./compile.sh input.mlir output.futil

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

INPUT_FILE=$1
OUTPUT_FILE=$2

mlir-opt "$INPUT_FILE" \
    --empty-tensor-to-alloc-tensor \
    --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
    --drop-equivalent-buffer-results \
    --buffer-results-to-out-params="hoist-static-allocs" \
    -convert-linalg-to-affine-loops \
    --lower-affine \
    --canonicalize \
    --cse \
    -fold-memref-alias-ops --lower-affine | tee affine-lowered.mlir | \
~/pytorch-to-calyx/DenseResourceElement2DenseElement/build/DenseResourceElement2DenseElement --dense-resource-to-dense-element | tee before-flatten.mlir | \
~/os-contrib/circt/build/bin/circt-opt --flatten-memref | tee flattened.mlir | \
mlir-opt --symbol-privatize \
         --buffer-results-to-out-params \
         --canonicalize \
         --cse --loop-invariant-code-motion --canonicalize --cse |  \
~/pytorch-to-calyx/StrengthReduction/build/StrengthReduction --strength-reduce | tee reduced.mlir | \
~/pytorch-to-calyx/MemrefCopyToAffineLoop/build/MemrefCopyToAffineLoop \
    --memref_copy_to_affine \
    --lower-affine --canonicalize --cse --loop-invariant-code-motion | tee copied.mlir | \
mlir-opt --arith-expand -scf-for-to-while | tee expanded.mlir | \
~/os-contrib/circt/build/bin/circt-opt --lower-scf-to-calyx="top-level-function=main write-json=data" | tee cnn.calyx | \
~/os-contrib/circt/build/bin/circt-translate --export-calyx  | tee cnn.futil | \
~/os-contrib/calyx/target/debug/fud2 --from calyx -o "$OUTPUT_FILE"

echo "Success: Compiled $INPUT_FILE to $OUTPUT_FILE"
