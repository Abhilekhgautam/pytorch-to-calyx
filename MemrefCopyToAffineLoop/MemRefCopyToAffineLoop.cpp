//
// Created by void on 12/27/25.
//

#include <iostream>
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "MemRefCopyToAffineLoop.h"

using namespace mlir;

// Only works with flattened memref (rank = 1)
struct MemRefCopyToAffineLoopPattern : public OpRewritePattern<memref::CopyOp> {
    using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(memref::CopyOp op, PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto source = op.getSource();
        auto target  = op.getTarget();

        auto memrefType = mlir::cast<MemRefType>(source.getType());
        unsigned rank = memrefType.getRank();

        if (rank > 1) return failure();

        auto loopStart = 0;
        auto loopEnd = memrefType.getDimSize(0);

        auto affineLoop = rewriter.create<affine::AffineForOp>(loc, loopStart, loopEnd);

        rewriter.replaceOp(op, affineLoop);
        rewriter.setInsertionPointToStart(affineLoop.getBody());

        auto memLoad = rewriter.create<memref::LoadOp>(loc, source, affineLoop.getInductionVar());
        auto memStore = rewriter.create<memref::StoreOp>(loc, memLoad, target, affineLoop.getInductionVar());

        //rewriter.eraseOp(op);

        return success();
    }
};

void MemRefCopyToAffineLoopPass::runOnOperation(){

    RewritePatternSet patterns(&getContext());
    patterns.add<MemRefCopyToAffineLoopPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        signalPassFailure();
    }
}

llvm::StringRef MemRefCopyToAffineLoopPass::getArgument() const {return "memref_copy_to_affine";}
