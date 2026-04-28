#ifndef STRENGTH_REDUCTION_H
#define STRENGTH_REDUCTION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

struct SCFStrengthReductionPass
    : public mlir::PassWrapper<SCFStrengthReductionPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SCFStrengthReductionPass)

  void runOnOperation() override;
  llvm::StringRef getArgument() const final;
};

#endif // STRENGTH_REDUCTION_H
