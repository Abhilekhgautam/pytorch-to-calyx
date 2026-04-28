#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "StrengthReduction.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

#include <iostream>
#include <tuple>

using namespace mlir;

static std::mutex printMutex;

mlir::TypedAttr multiplyAttrs(OpBuilder builder, mlir::TypedAttr opOne,
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

mlir::TypedAttr addAttrs(OpBuilder &builder, mlir::TypedAttr opOne,
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

mlir::TypedAttr subAttrs(OpBuilder &builder, mlir::TypedAttr opOne,
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

//     for (const auto &[key, value] : indVarMap) {
//       if (key) {

//         {

//           std::lock_guard<std::mutex> lock(printMutex);
//           llvm::outs() << "indVar: \n";
//           key.dump();
//         }
//         if (auto mulOp =
//                 llvm::dyn_cast_if_present<arith::MulIOp>(key.getDefiningOp()))
//                 {
//           auto resultType = mulOp.getType();

//           auto additiveFactor = std::get<2>(value);

//           auto newYieldValuesFn = [&](OpBuilder &rewriter, Location loc,
//                                       ArrayRef<BlockArgument> newBbArgs)
//               -> SmallVector<mlir::Value> {
//             auto currentAcc = newBbArgs[0];

//             auto resultType = currentAcc.getType();

//             auto multiplicativeFactor = std::get<1>(value);

//             // We Update the accumulator exactly before the yield.
//             rewriter.setInsertionPoint(op.getBody()->getTerminator());

//             auto additionFactor = arith::ConstantOp::create(
//                 rewriter, op.getLoc(), resultType, multiplicativeFactor);

//             auto newAcc = arith::AddIOp::create(rewriter, loc, currentAcc,
//                                                 additionFactor);

//             auto additionResultType = additionFactor.getType();

//             return {newAcc};
//           };
//           auto accumulator = arith::ConstantOp::create(
//               rewriter, op.getLoc(), resultType, additiveFactor);

//           rewriter.replaceAllOpUsesWith(mulOp, accumulator);
//           rewriter.eraseOp(mulOp);

//           return op.replaceWithAdditionalYields(rewriter, {accumulator},
//           true,
//                                                 newYieldValuesFn);
//         }
//       }
//     }

//     return failure();
//   }
// };
using IndVarMap =
    llvm::MapVector<mlir::Value,
                    std::tuple<mlir::Value, mlir::TypedAttr, mlir::TypedAttr>>;

void processAddOp(arith::AddIOp addOp, OpBuilder &builder,
                  IndVarMap &indVarMap) {
  auto lhs = addOp.getLhs();
  auto rhs = addOp.getRhs();

  if (indVarMap.count(lhs) && indVarMap.count(rhs)) {
    {
      std::lock_guard<std::mutex> lock(printMutex);
      llvm::outs() << "Both lhs and rhs are induction var.\n";
      addOp.dump();
    }
    auto [baseVarL, mulFactorL, addFactorL] = indVarMap[lhs];
    auto [baseVarR, mulFactorR, addFactorR] = indVarMap[rhs];

    if (baseVarL == baseVarR) {
      auto combinedMul = addAttrs(builder, mulFactorL, mulFactorR);
      auto combinedAdd = addAttrs(builder, addFactorL, addFactorR);

      indVarMap[addOp.getResult()] =
          std::make_tuple(baseVarL, combinedMul, combinedAdd);
    } else {

      auto parent = addOp->getParentOfType<scf::ForOp>();
      bool isLoopInvariant = parent.isDefinedOutsideOfLoop(lhs);

      if (isLoopInvariant) {
        {
          std::lock_guard<std::mutex> lock(printMutex);
          llvm::outs() << "Adding to induction Var:\n";
          addOp.getResult().dump();
        }
        indVarMap[addOp.getResult()] =
            std::make_tuple(baseVarR, mulFactorR, addFactorL);
      } else {
        std::cout << "Haiteri not a loop invariant.\n";
      }
    }
  } else {
    if (indVarMap.count(rhs)) {
      if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(lhs)) {
        {

          std::lock_guard<std::mutex> lock(printMutex);
          llvm::outs() << "LHS not a indVar but the rhs is.\n";
          addOp.dump();
        }
      }
    }
  }
  // Check if lhs is an induction var
  if (indVarMap.count(lhs)) {
    //  Check if rhs is an constant op
    if (auto c_op =
            llvm::dyn_cast_if_present<arith::ConstantOp>(rhs.getDefiningOp())) {
      auto val = addOp.getResult();
      auto attr = c_op.getValue();

      auto out_type = val.getType();

      auto [baseVar, multiplicativeFactor, additiveFactor] = indVarMap[lhs];

      if (multiplicativeFactor.getType() == attr.getType()) {
        auto addAttr = addAttrs(builder, attr, additiveFactor);
        indVarMap[val] =
            std::make_tuple(baseVar, multiplicativeFactor, addAttr);
      }
      // Handles the cast from a induction var.
      else {
        indVarMap[val] =
            std::make_tuple(lhs, builder.getOneAttr(attr.getType()), attr);
      }
    }
  }
}

llvm::LogicalResult processMul(arith::MulIOp mulOp, OpBuilder &builder,
                               IndVarMap &indVarMap) {

  auto lhs = mulOp.getLhs();
  auto rhs = mulOp.getRhs();

  // Check if lhs is an induction var
  if (indVarMap.count(lhs)) {
    //  Check if rhs is an constant op
    if (auto c_op =
            llvm::dyn_cast_if_present<arith::ConstantOp>(rhs.getDefiningOp())) {
      auto val = mulOp.getResult();
      auto attr = c_op.getValue();

      auto [baseVar, multiplicativeFactor, additiveFactor] = indVarMap[lhs];

      if (additiveFactor.getType() == attr.getType()) {
        auto mulAttr = multiplyAttrs(builder, attr, multiplicativeFactor);
        auto addAttr = multiplyAttrs(builder, attr, additiveFactor);
        indVarMap[val] = std::make_tuple(baseVar, mulAttr, addAttr);
      }
      // Handle the case where the lhs is just a cast of induction
      else {
        indVarMap[val] =
            std::make_tuple(lhs, attr, builder.getZeroAttr(attr.getType()));
      }

      {

        std::lock_guard<std::mutex> lock(printMutex);

        std::cout << "Strength reduction. Here we go.\n";
        llvm::outs() << "Strength reduction opportunity: ";
        mulOp.print(llvm::outs());
        llvm::outs() << "\n";
      }
    }
  } else {
    std::cout << "LHS not induction var MulOp.\n";
  }
  return failure();
}

llvm::LogicalResult processLoop(scf::ForOp forOp, OpBuilder &builder,
                                IndVarMap &indVarMap) {
  IndVarMap localMap = indVarMap;

  // scf.for always has one induction var, the iteration variable.
  auto iterationVar = forOp.getInductionVar();
  auto lowerBound = forOp.getLowerBound();
  auto step = forOp.getConstantStep();

  // scf.for always has a constant step
  if (!step.has_value()) [[unlikely]]
    return failure();

  // FixMe: Assumes the step is always 1.
  localMap[iterationVar] =
      std::make_tuple(iterationVar, builder.getOneAttr(iterationVar.getType()),
                      builder.getZeroAttr(iterationVar.getType()));

  auto iterArgs = forOp.getRegionIterArgs();

  if (auto yieldOp =
          llvm::dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {

    for (auto [iterArg, initialVal, yieldedVal] :
         llvm::zip(forOp.getRegionIterArgs(), forOp.getInitArgs(),
                   yieldOp.getOperands())) {

      if (auto addOp =
              llvm::dyn_cast<arith::AddIOp>(yieldedVal.getDefiningOp())) {
        auto lhs = addOp.getLhs();
        auto rhs = addOp.getRhs();

        if (lhs == iterArg) {
          if (auto constantOp =
                  llvm::dyn_cast<arith::ConstantOp>(rhs.getDefiningOp())) {
            localMap[lhs] =
                std::make_tuple(lhs, builder.getOneAttr(addOp.getType()),
                                constantOp.getValue());
          }
        }
      }
    }
  }

  auto body = forOp.getBody();

  for (auto &elt : *body) {

    if (auto innerFor = llvm::dyn_cast<scf::ForOp>(elt)) {
      auto retVal = processLoop(innerFor, builder, localMap);
    }

    if (llvm::isa<arith::IndexCastOp, arith::TruncIOp, arith::ExtSIOp,
                  arith::ExtUIOp>(elt)) {

      auto in = elt.getOperand(0);
      if (indVarMap.count(in)) {

        auto v = elt.getResult(0);
        auto out_type = v.getType();
        localMap[v] = std::make_tuple(v, builder.getOneAttr(out_type),
                                      builder.getZeroAttr(out_type));
      }
    }
    // check for k = j * b , where j is an inductionVar and b is a constant
    if (auto mulOp = llvm::dyn_cast<arith::MulIOp>(elt)) {
      auto retVal = processMul(mulOp, builder, localMap);
    }
    // check for k = j + b
    else if (auto addOp = llvm::dyn_cast<arith::AddIOp>(elt)) {
      processAddOp(addOp, builder, localMap);
    }
  }
  return failure();
}

void SCFStrengthReductionPass::runOnOperation() {

  auto op = getOperation();

  OpBuilder rewriter(op.getContext());

  auto bodyOps = op.getBody().getOps();

  // iVar : <iVar, multiplicative factor, additive factor>
  llvm::MapVector<mlir::Value,
                  std::tuple<mlir::Value, mlir::TypedAttr, mlir::TypedAttr>>
      indVarMap;

  for (auto &op : bodyOps) {
    if (auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
      auto retVal = processLoop(forOp, rewriter, indVarMap);
    }
  }
}

llvm::StringRef SCFStrengthReductionPass::getArgument() const {
  return "strength-reduce";
}
