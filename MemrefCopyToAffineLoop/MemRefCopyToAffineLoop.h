//
// Created by void on 12/27/25.
//

#ifndef MEMREFCOPYTOAFFINELOOP_MEMREFCOPYTOAFFINELOOP_H
#define MEMREFCOPYTOAFFINELOOP_MEMREFCOPYTOAFFINELOOP_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

struct MemRefCopyToAffineLoopPass : public mlir::PassWrapper<MemRefCopyToAffineLoopPass, mlir::OperationPass<mlir::func::FuncOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemRefCopyToAffineLoopPass)


    void runOnOperation() override;
    llvm::StringRef getArgument() const final;
};

#endif //MEMREFCOPYTOAFFINELOOP_MEMREFCOPYTOAFFINELOOP_H