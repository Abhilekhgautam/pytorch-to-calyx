#ifndef STRENGTH_REDUCTION_H
#define STRENGTH_REDUCTION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

struct StrengthReductionPass
    : public mlir::PassWrapper<StrengthReductionPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StrengthReductionPass)

  void runOnOperation() override;
  llvm::StringRef getArgument() const final;
};

#endif // STRENGTH_REDUCTION_H
