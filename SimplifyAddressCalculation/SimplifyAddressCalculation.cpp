#include "SimplifyAddressCalculation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "SimplifyAddressCalculation.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;

// Finds this pattern: (X / C) * C + (X % C) and replaces it with X.
struct MemRefIndexRewritePattern : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {

    // Get the operands of the add
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    // Check if rhs is the result of a mod operation.
    if (auto modOp =
            llvm::dyn_cast_if_present<arith::RemSIOp>(rhs.getDefiningOp())) {
      // Probable X: if the pattern is found, the index will be replaced with
      // lhs_mod
      auto lhs_mod = modOp.getLhs();
      // Probable C.
      auto rhs_mod = modOp.getRhs();

      // Check  if the lhs of add is a result of mul
      if (auto mulOp =
              llvm::dyn_cast_if_present<arith::MulIOp>(lhs.getDefiningOp())) {
        auto mul_lhs = mulOp.getLhs();
        // This must be C, must equal rhs_mod
        auto mul_rhs = mulOp.getRhs();

        if (mul_rhs == rhs_mod) {
          // Check if mul_lhs is result of div
          if (auto divOp = llvm::dyn_cast_if_present<arith::DivSIOp>(
                  mul_lhs.getDefiningOp())) {

            auto div_lhs = divOp.getLhs();
            auto div_rhs = divOp.getRhs();

            if (div_lhs == lhs_mod && div_rhs == rhs_mod) {
              // All checks passed, replace all uses with div_lhs
              rewriter.replaceAllOpUsesWith(op, div_lhs);
              return success();
            }
          }
        }
      }
    }

    return failure();
  }
};

void SimplifyAddressCalculationPass::runOnOperation() {

  RewritePatternSet patterns(&getContext());
  patterns.add<MemRefIndexRewritePattern>(&getContext());

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

llvm::StringRef SimplifyAddressCalculationPass::getArgument() const {
  return "simplify-address-calculation";
}
