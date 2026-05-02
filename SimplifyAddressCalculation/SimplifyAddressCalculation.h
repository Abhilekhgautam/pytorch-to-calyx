#ifndef SIMPLIFY_ADDRESS_CALCULATION_H
#define SIMPLIFY_ADDRESS_CALCULATION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

struct SimplifyAddressCalculationPass
    : public mlir::PassWrapper<SimplifyAddressCalculationPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyAddressCalculationPass)

  void runOnOperation() override;
  llvm::StringRef getArgument() const final;
};

#endif // SIMPLIFY_ADDRESS_CALCULATION_H
