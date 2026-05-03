#include "SimplifyAddressCalculation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "SimplifyAddressCalculation.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PDLPatternMatch.h.inc>
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

struct RemDivRewritePattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  // Find this pattern:
  // ========================
  // %0 = arith.remsi %arg4, %c56 : index
  // %1 = arith.divsi %arg4, %c56 : index
  // %2 = arith.remsi %1, %c76 : index
  // %3 = arith.divsi %1, %c76 : index
  // ========================
  // and add an iter arg to the scf::for
  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {

    auto indVar = forOp.getInductionVar();
    auto step = forOp.getStep();

    llvm::MapVector<int64_t, std::pair<arith::RemSIOp, arith::DivSIOp>>
        dimensionMap;

    auto body = forOp.getBody();

    for (auto &op : *body) {
      if (auto remOp = llvm::dyn_cast_if_present<arith::RemSIOp>(op)) {
        auto remLhs = remOp.getLhs();
        auto remRhs = remOp.getRhs();

        if (remLhs == indVar) {
          if (auto constOp = getConstantIntValue(remRhs); constOp.has_value()) {
            // TODO: maybe add to vector later.
            dimensionMap[*constOp].first = remOp;
          }
        } else {
          // Maybe the lhs is the result of division
          if (auto divResult =
                  llvm::dyn_cast_if_present<arith::DivSIOp>(remLhs)) {

            if (auto constOp = getConstantIntValue(remRhs);
                constOp.has_value()) {
              dimensionMap[*constOp].first = remOp;
            }
          }
        }
      } else if (auto divOp = llvm::dyn_cast_if_present<arith::DivSIOp>(op)) {
        auto divLhs = divOp.getLhs();
        auto divRhs = divOp.getRhs();

        if (divLhs == indVar) {
          if (auto constOp = getConstantIntValue(divRhs); constOp.has_value()) {
            dimensionMap[*constOp].second = divOp;
          }
        } else {
          // Maybe the the lhs is the result of previous div
          // Maybe we'll check its presence in the vector or better
          if (auto divResult =
                  llvm::dyn_cast_if_present<arith::DivSIOp>(divLhs)) {

            if (auto constOp = getConstantIntValue(divRhs);
                constOp.has_value()) {
              // TODO: maybe add to vector later
              dimensionMap[*constOp].second = divOp;
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
