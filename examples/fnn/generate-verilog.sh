#!/bin/bash

usage() {
    echo "Usage: $0 [-s] <input_filename> [output_filename]"
    echo "  -s    Skip strength reduction step"
    echo "  If output_filename is not provided, defaults to out.mlir"
    exit 1
}

SKIP_STRENGTH_REDUCE=false

while getopts "sh" opt; do
    case $opt in
        s)
            SKIP_STRENGTH_REDUCE=true
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            usage
            ;;
    esac
done

shift $((OPTIND-1))

if [ $# -eq 0 ]; then
    echo "Error: No input filename provided"
    usage
fi

INPUT_FILE="$1"
# Default to out.mlir if not provided
OUTPUT_FILE="${2:-out.mlir}"  
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Run the initial MLIR pipeline
if [ "$SKIP_STRENGTH_REDUCE" = true ]; then
    echo "Running pipeline WITHOUT strength reduction..."
    mlir-opt "$INPUT_FILE" \
        -convert-linalg-to-affine-loops \
        --lower-affine \
        --loop-invariant-code-motion \
        --canonicalize \
        --cse | \
    ~/os-contrib/circt/build/bin/circt-opt --flatten-memref | \
    mlir-opt --symbol-privatize \
             --buffer-results-to-out-params \
             --canonicalize \
             --cse | \
   ~/pytorch-to-calyx/MemrefCopyToAffineLoop/build/MemrefCopyToAffineLoop --memref_copy_to_affine --arith-expand --lower-affine| \
   ~/os-contrib/circt/build/bin/circt-opt --lower-scf-to-calyx="top-level-function=forward write-json=non-reduced-data" | \
   ~/os-contrib/circt/build/bin/circt-translate --export-calyx | \
   fud2 --from calyx -o "$OUTPUT_FILE"
else
    echo "Running pipeline WITH strength reduction..."
    mlir-opt "$INPUT_FILE" \
        -convert-linalg-to-affine-loops \
        --lower-affine \
        --loop-invariant-code-motion \
        --canonicalize \
        --cse | \
        ~/os-contrib/circt/build/bin/circt-opt --flatten-memref | \
        mlir-opt --symbol-privatize \
             --buffer-results-to-out-params \
             --canonicalize \
             --cse | \
        ~/pytorch-to-calyx/MemrefCopyToAffineLoop/build/MemrefCopyToAffineLoop --memref_copy_to_affine \
            --arith-expand \
             --lower-affine  \
             --loop-invariant-code-motion \
             --canonicalize \
             --cse | \
        ~/pytorch-to-calyx/StrengthReduction/build/StrengthReduction --strength-reduce --scf-for-to-while | \
        ~/os-contrib/circt/build/bin/circt-opt --lower-scf-to-calyx="top-level-function=forward write-json=reduced-data" | \
        ~/os-contrib/circt/build/bin/circt-translate --export-calyx | \
        fud2 --from calyx -o "$OUTPUT_FILE"

fi

if [ $? -eq 0 ]; then
    echo "Pipeline completed successfully. Output written to $OUTPUT_FILE"
else
    echo "Error: Pipeline failed"
    exit 1
fi
