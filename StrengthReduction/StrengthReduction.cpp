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
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <iostream>
#include <tuple>

using namespace mlir;

// const std::unordered_set<std::string_view> cast_ext_ops = {
//     "arith.index_cast", "arith.extf",   "arith.extsi",        "arith.extui",
//     "arith.fptoui",     "arith.fptosi", "arith.index_castui",
//     "arith.trunci"};

struct SCFLoopRewritePattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // iVar : <iVar, multiplicative factor, additive factor>
    llvm::DenseMap<mlir::Value, std::tuple<mlir::Value, int64_t, int64_t>>
        indVarMap;

    // scf.for always has one induction var, the iteration variable.
    auto iterationVar = op.getInductionVar();

    indVarMap[iterationVar] = std::make_tuple(iterationVar, 1, 1);

    auto users = iterationVar.getUsers();

    for (const auto user : users) {
      // Iteration Variable is an index. The cast must be any of Index Cast.
      if (auto val = llvm::dyn_cast<arith::IndexCastOp>(user)) {

        auto v = val.getOut();
        indVarMap[v] = std::make_tuple(v, 1, 1);
      }
    }

    auto body = op.getBody();

    for (auto &elt : *body) {
      if (llvm::isa<arith::IndexCastOp, arith::TruncIOp, arith::ExtSIOp,
                    arith::ExtUIOp>(elt)) {
        auto in = elt.getOperand(0);

        if (indVarMap.count(in)) {
          auto out = elt.getResult(0);
          indVarMap[out] = indVarMap[in];
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
            if (auto int_op = llvm::dyn_cast_if_present<arith::ConstantIntOp>(
                    rhs.getDefiningOp())) {
              auto const_val = int_op.value();
              indVarMap[val] = std::make_tuple(lhs, const_val, 0);
            } else if (auto index_op =
                           llvm::dyn_cast_if_present<arith::ConstantIndexOp>(
                               rhs.getDefiningOp())) {
              auto const_val = index_op.value();
              indVarMap[val] = std::make_tuple(lhs, const_val, 0);
            }
          } else {
            std::cout << "RHS not a constant op\n";
            rhs.dump();
          }
        } else {
          std::cout << "LHS not an induction var\n";
          lhs.dump();
          mulOp.dump();
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
            if (auto int_op = llvm::dyn_cast_if_present<arith::ConstantIntOp>(
                    rhs.getDefiningOp())) {
              auto const_val = int_op.value();
              indVarMap[val] = std::make_tuple(lhs, 1, const_val);
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
            if (auto int_op = llvm::dyn_cast_if_present<arith::ConstantIntOp>(
                    rhs.getDefiningOp())) {
              auto const_val = int_op.value();
              indVarMap[val] = std::make_tuple(lhs, 1, const_val);
            }
          }
        }
      }

      // check if any induction var is used in memref.store
      if (auto storeOp = llvm::dyn_cast<memref::StoreOp>(elt)) {
        auto storing_value = storeOp.getValueToStore();

        if (indVarMap.count(storing_value)) {
          auto store_value = storeOp.getMemRef();
          indVarMap[store_value] = indVarMap[storing_value];
        }
      }
    }

    for (const auto &[key, value] : indVarMap) {
      if (key) {
        if (auto mulOp =
                llvm::dyn_cast_if_present<arith::MulIOp>(key.getDefiningOp())) {
          auto resultType = mulOp.getType();

          auto attr = mlir::IntegerAttr::get(resultType, std::get<2>(value));

          auto newYieldValuesFn = [&](OpBuilder &rewriter, Location loc,
                                      ArrayRef<BlockArgument> newBbArgs)
              -> SmallVector<mlir::Value> {
            auto currentAcc = newBbArgs[0];

            auto resultType = currentAcc.getType();

            auto attr = mlir::IntegerAttr::get(resultType, std::get<1>(value));

            // We Update the accumulator exactly before the yield.
            rewriter.setInsertionPoint(op.getBody()->getTerminator());

            auto additionFactor = arith::ConstantOp::create(
                rewriter, op.getLoc(), resultType, attr);

            auto newAcc = arith::AddIOp::create(rewriter, loc, currentAcc,
                                                additionFactor);
            return {newAcc};
          };
          auto accumulator = arith::ConstantOp::create(rewriter, op.getLoc(),
                                                       resultType, attr);

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
