#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "StrengthReduction.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
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

#include <tuple>

using namespace mlir;

mlir::TypedAttr multiplyAttrs(PatternRewriter &rewriter, mlir::TypedAttr opOne,
                              mlir::TypedAttr opTwo) {
  if (llvm::isa<IntegerAttr>(opOne)) {
    auto intOpOne = llvm::dyn_cast<IntegerAttr>(opOne);
    auto intOpTwo = llvm::dyn_cast<IntegerAttr>(opTwo);

    auto opOneValue = intOpOne.getValue();
    auto opTwoValue = intOpTwo.getValue();

    // let it truncate
    auto product = opOneValue * opTwoValue;

    return IntegerAttr::get(intOpOne.getType(), product);
  } else if (llvm::isa<FloatAttr>(opOne)) {

    auto intOpOne = llvm::dyn_cast<FloatAttr>(opOne);
    auto intOpTwo = llvm::dyn_cast<FloatAttr>(opTwo);

    auto opOneValue = intOpOne.getValue();
    auto opTwoValue = intOpTwo.getValue();

    // let it truncate
    auto product = opOneValue * opTwoValue;

    return FloatAttr::get(intOpOne.getType(), product);
  } else
    return {};
}

mlir::TypedAttr addAttrs(PatternRewriter &rewriter, mlir::TypedAttr opOne,
                         mlir::TypedAttr opTwo) {
  if (llvm::isa<IntegerAttr>(opOne)) {
    auto intOpOne = llvm::dyn_cast<IntegerAttr>(opOne);
    auto intOpTwo = llvm::dyn_cast<IntegerAttr>(opTwo);

    auto opOneValue = intOpOne.getValue();
    auto opTwoValue = intOpTwo.getValue();

    // let it truncate
    auto sum = opOneValue + opTwoValue;

    return IntegerAttr::get(intOpOne.getType(), sum);
  } else if (llvm::isa<FloatAttr>(opOne)) {

    auto intOpOne = llvm::dyn_cast<FloatAttr>(opOne);
    auto intOpTwo = llvm::dyn_cast<FloatAttr>(opTwo);

    auto opOneValue = intOpOne.getValue();
    auto opTwoValue = intOpTwo.getValue();

    // let it truncate
    auto sum = opOneValue + opTwoValue;

    return FloatAttr::get(intOpOne.getType(), sum);
  } else
    return {};
}

mlir::TypedAttr subAttrs(PatternRewriter &rewriter, mlir::TypedAttr opOne,
                         mlir::TypedAttr opTwo) {
  if (llvm::isa<IntegerAttr>(opOne)) {
    auto intOpOne = llvm::dyn_cast<IntegerAttr>(opOne);
    auto intOpTwo = llvm::dyn_cast<IntegerAttr>(opTwo);

    auto opOneValue = intOpOne.getValue();
    auto opTwoValue = intOpTwo.getValue();

    // let it truncate
    auto diff = opOneValue - opTwoValue;

    return IntegerAttr::get(intOpOne.getType(), diff);
  } else if (llvm::isa<FloatAttr>(opOne)) {

    auto intOpOne = llvm::dyn_cast<FloatAttr>(opOne);
    auto intOpTwo = llvm::dyn_cast<FloatAttr>(opTwo);

    auto opOneValue = intOpOne.getValue();
    auto opTwoValue = intOpTwo.getValue();

    // let it truncate
    auto diff = opOneValue - opTwoValue;

    return FloatAttr::get(intOpOne.getType(), diff);
  } else
    return {};
}

struct SCFLoopRewritePattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // iVar : <iVar, multiplicative factor, additive factor>
    llvm::MapVector<mlir::Value,
                    std::tuple<mlir::Value, mlir::TypedAttr, mlir::TypedAttr>>
        indVarMap;

    // scf.for always has one induction var, the iteration variable.
    auto iterationVar = op.getInductionVar();
    auto step = op.getConstantStep();

    // scf.for always has a constant step
    if (!step.has_value()) [[unlikely]]
      return failure();

    // FixMe: Assumes the step is always 1.
    indVarMap[iterationVar] = std::make_tuple(
        iterationVar, rewriter.getOneAttr(iterationVar.getType()),
        rewriter.getZeroAttr(iterationVar.getType()));

    auto body = op.getBody();

    for (auto &elt : *body) {

      if (llvm::isa<arith::IndexCastOp, arith::TruncIOp, arith::ExtSIOp,
                    arith::ExtUIOp>(elt)) {

        auto in = elt.getOperand(0);
        if (indVarMap.count(in)) {

          auto v = elt.getResult(0);
          auto out_type = v.getType();
          indVarMap[v] = std::make_tuple(v, rewriter.getOneAttr(out_type),
                                         rewriter.getZeroAttr(out_type));
        }
      }
      // check for k = j * b , where j is an inductionVar and b is a constant
      if (auto mulOp = llvm::dyn_cast<arith::MulIOp>(elt)) {
        auto lhs = mulOp.getLhs();
        auto rhs = mulOp.getRhs();

        // Check if lhs is an induction var
        if (indVarMap.count(lhs)) {
          // Check if rhs is an constant op
          if (auto c_op = llvm::dyn_cast_if_present<arith::ConstantOp>(
                  rhs.getDefiningOp())) {
            auto val = mulOp.getResult();
            auto attr = c_op.getValue();

            auto [baseVar, multiplicativeFactor, additiveFactor] =
                indVarMap[lhs];

            if (additiveFactor.getType() == attr.getType()) {
              auto mulAttr =
                  multiplyAttrs(rewriter, attr, multiplicativeFactor);
              auto addAttr = multiplyAttrs(rewriter, attr, additiveFactor);
              indVarMap[val] = std::make_tuple(baseVar, mulAttr, addAttr);
            }
            // Handle the case where the lhs is just a cast of induction var
            else {
              indVarMap[val] = std::make_tuple(
                  lhs, attr, rewriter.getZeroAttr(attr.getType()));
            }
          }
        }
      }
      // check for k = j + b
      else if (auto addOp = llvm::dyn_cast<arith::AddIOp>(elt)) {
        auto lhs = addOp.getLhs();
        auto rhs = addOp.getRhs();

        // Check if lhs is an induction var
        if (indVarMap.count(lhs)) {
          // Check if rhs is an constant op
          if (auto c_op = llvm::dyn_cast_if_present<arith::ConstantOp>(
                  rhs.getDefiningOp())) {
            auto val = addOp.getResult();
            auto attr = c_op.getValue();

            auto out_type = val.getType();

            auto [baseVar, multiplicativeFactor, additiveFactor] =
                indVarMap[lhs];

            if (multiplicativeFactor.getType() == attr.getType()) {
              auto addAttr = addAttrs(rewriter, attr, additiveFactor);
              indVarMap[val] =
                  std::make_tuple(baseVar, multiplicativeFactor, addAttr);
            }
            // Handles the cast from a induction var.
            else {
              indVarMap[val] = std::make_tuple(
                  lhs, rewriter.getOneAttr(attr.getType()), attr);
            }
          }
        }
      }

      // check for k = j - b
      else if (auto subOp = llvm::dyn_cast<arith::SubIOp>(elt)) {
        auto lhs = subOp.getLhs();
        auto rhs = subOp.getRhs();

        // Check if lhs is an induction var
        if (indVarMap.count(lhs)) {
          // Check if rhs is an constant op
          if (auto c_op = llvm::dyn_cast_if_present<arith::ConstantOp>(
                  rhs.getDefiningOp())) {
            auto val = subOp.getResult();
            auto attr = c_op.getValue();

            auto out_type = val.getType();

            indVarMap[val] =
                std::make_tuple(lhs, rewriter.getOneAttr(out_type), attr);
          }
        }
      }
    }

    for (const auto &[key, value] : indVarMap) {
      if (key) {
        if (auto mulOp =
                llvm::dyn_cast_if_present<arith::MulIOp>(key.getDefiningOp())) {
          auto resultType = mulOp.getType();

          auto additiveFactor = std::get<2>(value);

          auto newYieldValuesFn = [&](OpBuilder &rewriter, Location loc,
                                      ArrayRef<BlockArgument> newBbArgs)
              -> SmallVector<mlir::Value> {
            auto currentAcc = newBbArgs[0];

            auto resultType = currentAcc.getType();

            auto multiplicativeFactor = std::get<1>(value);

            // We Update the accumulator exactly before the yield.
            rewriter.setInsertionPoint(op.getBody()->getTerminator());

            auto additionFactor = arith::ConstantOp::create(
                rewriter, op.getLoc(), resultType, multiplicativeFactor);

            auto newAcc = arith::AddIOp::create(rewriter, loc, currentAcc,
                                                additionFactor);
            return {newAcc};
          };
          auto accumulator = arith::ConstantOp::create(
              rewriter, op.getLoc(), resultType, additiveFactor);

          rewriter.replaceAllOpUsesWith(mulOp, accumulator);
          rewriter.eraseOp(mulOp);

          return op.replaceWithAdditionalYields(rewriter, {accumulator}, true,
                                                newYieldValuesFn);
        }
      }
    }

    return failure();
  }
};

void StrengthReductionPass::runOnOperation() {

  RewritePatternSet patterns(&getContext());
  patterns.add<SCFLoopRewritePattern>(&getContext());

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

llvm::StringRef StrengthReductionPass::getArgument() const {
  return "strength-reduce";
}
