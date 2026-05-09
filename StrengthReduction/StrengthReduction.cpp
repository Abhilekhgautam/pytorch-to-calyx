#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "StrengthReduction.h"
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

#include <iostream>
#include <mutex>
#include <stdexcept>
#include <tuple>

using namespace mlir;

std::mutex printMutex;

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

using IndVarMap =
    llvm::MapVector<mlir::Value,
                    std::tuple<mlir::Value, mlir::TypedAttr, mlir::TypedAttr>>;
using RunTimeMap = llvm::DenseMap<mlir::Value, mlir::Value>;

/*
 * Handles the form : indvar, const_op
 *
 * The const_op can be either a constantOp or Block Argument.
 */
void handleDefaultForm(PatternRewriter &rewriter, scf::ForOp forOp,
                       Operation *op, mlir::Value lhs, mlir::Value rhs,
                       RunTimeMap &runtimeAddFactorMap, IndVarMap &indVarMap,
                       SmallVector<Operation *> &speculativeOps) {

  // Handle the Multiply Operation
  if (auto mulOp = llvm::dyn_cast_if_present<arith::MulIOp>(op)) {
    // Check if RHS is a constantOp.
    if (auto constOp =
            llvm::dyn_cast_if_present<arith::ConstantOp>(rhs.getDefiningOp())) {

      auto val = mulOp.getResult();
      auto attr = constOp.getValue();

      auto [baseVar, multiplicativeFactor, additiveFactor] = indVarMap[lhs];

      if (additiveFactor.getType() == attr.getType()) {
        if (baseVar != forOp.getInductionVar()) {

          auto mulAttr = multiplyAttrs(rewriter, attr, multiplicativeFactor);
          auto addAttr = multiplyAttrs(rewriter, attr, additiveFactor);
          indVarMap[val] = std::make_tuple(baseVar, mulAttr, addAttr);
        } else {

          indVarMap[val] =
              std::make_tuple(lhs, attr, rewriter.getZeroAttr(lhs.getType()));
        }
      }
      // Handle the case where the lhs is just a cast of induction var
      else {
        indVarMap[val] =
            std::make_tuple(lhs, attr, rewriter.getZeroAttr(attr.getType()));
      }

      if (runtimeAddFactorMap.contains(lhs)) {
        rewriter.setInsertionPoint(forOp);
        auto runtimeBase = runtimeAddFactorMap[lhs];
        auto resultType = mulOp.getResult().getType();

        auto mulFactor = arith::ConstantOp::create(rewriter, mulOp.getLoc(),
                                                   resultType, attr);
        speculativeOps.push_back(mulFactor);
        auto runtimeInitial = arith::MulIOp::create(rewriter, mulOp.getLoc(),
                                                    runtimeBase, mulFactor);

        speculativeOps.push_back(runtimeInitial);
        runtimeAddFactorMap[val] = runtimeInitial;
      }

    }
    // Check if RHS is a loop invariant.
    else if (forOp.isDefinedOutsideOfLoop(rhs)) {
      throw std::runtime_error("RHS of mulOp not a constant yikes!");
    }
  }
  // Handle the Addition Operation
  else if (auto addOp = llvm::dyn_cast_if_present<arith::AddIOp>(op)) {
    // Check if RHS is a Constant Literal
    if (auto constOp =
            llvm::dyn_cast_if_present<arith::ConstantOp>(rhs.getDefiningOp())) {

      auto val = addOp.getResult();
      auto attr = constOp.getValue();

      // if (runtimeAddFactorMap.count(lhs)) {
      //   rewriter.setInsertionPoint(op);
      //   auto runtimeBase = runtimeAddFactorMap[lhs];
      //   auto resultType = addOp.getResult().getType();

      //   auto mulFactor = arith::ConstantOp::create(rewriter, addOp.getLoc(),
      //                                              resultType, attr);
      //   auto runtimeInitial = arith::MulIOp::create(rewriter, addOp.getLoc(),
      //                                               runtimeBase, mulFactor);

      //   runtimeAddFactorMap[val] = runtimeInitial;
      // }

      auto [baseVar, multiplicativeFactor, additiveFactor] = indVarMap[lhs];

      if (additiveFactor.getType() == attr.getType()) {
        // auto mulAttr = multiplyAttrs(rewriter, attr, multiplicativeFactor);
        auto addAttr = addAttrs(rewriter, attr, additiveFactor);
        indVarMap[val] =
            std::make_tuple(baseVar, multiplicativeFactor, addAttr);
      }
      // Handle the case where the lhs is just a cast of induction var
      else {
        indVarMap[val] =
            std::make_tuple(lhs, rewriter.getOneAttr(attr.getType()), attr);
      }
    }
    // Check if RHS is a loop invariant
    else if (forOp.isDefinedOutsideOfLoop(rhs)) {

      auto [baseVar, mulFactor, addFactor] = indVarMap[lhs];

      indVarMap[addOp.getResult()] =
          std::make_tuple(baseVar, mulFactor, addFactor);

      if (runtimeAddFactorMap.contains(lhs)) [[likely]] {

        rewriter.setInsertionPoint(forOp);
        auto addFactor = runtimeAddFactorMap[lhs];

        auto newFactor =
            arith::AddIOp::create(rewriter, op->getLoc(), addFactor, rhs);

        speculativeOps.push_back(newFactor);

        runtimeAddFactorMap[addOp.getResult()] = newFactor;
      } else {
        runtimeAddFactorMap[addOp.getResult()] = rhs;
      }
    }
  }
}

void handleBothIndVar(Operation *op) {
  throw std::runtime_error("Unimplemented block called!");
}

void handleNoneIndVar(Operation *op, IndVarMap &indVarMap) { return; }

struct SCFLoopRewritePattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // iVar : <iVar, multiplicative factor, additive factor>
    llvm::MapVector<mlir::Value,
                    std::tuple<mlir::Value, mlir::TypedAttr, mlir::TypedAttr>>
        indVarMap;
    llvm::DenseMap<mlir::Value, mlir::Value> runtimeAddFactorMap;

    SmallVector<Operation *> speculativeOps;

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

    if (auto yieldOp =
            llvm::dyn_cast<scf::YieldOp>(op.getBody()->getTerminator())) {

      for (auto [iterArg, initialVal, yieldedVal] :
           llvm::zip(op.getRegionIterArgs(), op.getInitArgs(),
                     yieldOp.getOperands())) {

        if (auto addOp = llvm::dyn_cast_if_present<arith::AddIOp>(
                yieldedVal.getDefiningOp())) {

          auto lhs = addOp.getLhs();
          auto rhs = addOp.getRhs();

          if (lhs == iterArg) {
            if (auto constantOp =
                    llvm::dyn_cast<arith::ConstantOp>(rhs.getDefiningOp())) {

              if (auto initConstantOp =
                      llvm::dyn_cast_if_present<arith::ConstantOp>(
                          initialVal.getDefiningOp())) {

                indVarMap[lhs] = std::make_tuple(lhs, constantOp.getValue(),
                                                 initConstantOp.getValue());
              }
            }
          }
        }
      }
    }

    auto body = op.getBody();

    enum struct OpForm : uint8_t { Default, BothIndVar, NoneIndVar };

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
      else if (auto mulOp = llvm::dyn_cast<arith::MulIOp>(elt)) {

        OpForm form = OpForm::Default;

        auto lhs = mulOp.getLhs();
        auto rhs = mulOp.getRhs();

        // We'll always form on one of these form:
        // %result = %ind_var, %const_op
        // %result = %ind_var1, %ind_var2
        // %result = %const_op1, %const_op2

        // Swap to have lhs as induction Variable.
        if (indVarMap.contains(rhs) && !indVarMap.contains(lhs)) {
          std::swap(lhs, rhs);
        }
        // This is case 2. Perfectly okay.
        else if (indVarMap.contains(lhs) && indVarMap.contains(rhs)) {
          form = OpForm::BothIndVar;
        }
        // Both operand can be constant.
        else if (!indVarMap.contains(lhs) && !indVarMap.contains(rhs)) {
          form = OpForm::NoneIndVar;
        }

        switch (form) {
        case OpForm::Default:
          handleDefaultForm(rewriter, op, mulOp, lhs, rhs, runtimeAddFactorMap,
                            indVarMap, speculativeOps);
          break;
        case OpForm::BothIndVar:
          handleBothIndVar(mulOp);
          break;
        case OpForm::NoneIndVar:
          handleNoneIndVar(mulOp, indVarMap);
          break;
        }
      }
      // check for k = j + b
      else if (auto addOp = llvm::dyn_cast<arith::AddIOp>(elt)) {

        OpForm form = OpForm::Default;
        auto lhs = addOp.getLhs();
        auto rhs = addOp.getRhs();

        // We'll always form on one of these form:
        // %result = %ind_var, %const_op
        // %result = %ind_var1, %ind_var2
        // %result = %const_op1, %const_op2

        // Swap to have lhs as induction Variable.
        if (indVarMap.contains(rhs) && !indVarMap.contains(lhs)) {
          std::swap(lhs, rhs);
        }
        // This is case 2. Perfectly okay.
        else if (indVarMap.contains(lhs) && indVarMap.contains(rhs)) {
          form = OpForm::BothIndVar;
        }
        // Both operand can be constant.
        else if (!indVarMap.contains(lhs) && !indVarMap.contains(rhs)) {
          form = OpForm::NoneIndVar;
        }

        switch (form) {
        case OpForm::Default:
          handleDefaultForm(rewriter, op, addOp, lhs, rhs, runtimeAddFactorMap,
                            indVarMap, speculativeOps);
          break;
        case OpForm::BothIndVar:
          handleBothIndVar(addOp);
          break;
        case OpForm::NoneIndVar:
          handleNoneIndVar(addOp, indVarMap);
          break;
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
          rewriter.setInsertionPoint(op);
          mlir::Value accumulator = arith::ConstantOp::create(
              rewriter, op.getLoc(), resultType, additiveFactor);

          if (runtimeAddFactorMap.count(key)) {
            accumulator = arith::AddIOp::create(
                rewriter, op.getLoc(), accumulator, runtimeAddFactorMap[key]);
          }

          rewriter.replaceAllOpUsesWith(mulOp, accumulator);
          rewriter.eraseOp(mulOp);

          return op.replaceWithAdditionalYields(rewriter, {accumulator}, true,
                                                newYieldValuesFn);
        }
      }
    }

    // Erase all those ops:
    for (auto *op : llvm::reverse(speculativeOps)) {
      rewriter.eraseOp(op);
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
