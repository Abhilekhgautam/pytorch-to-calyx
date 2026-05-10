#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

class DenseResourceElementAttrToDenseElementAttrPass
    : public mlir::PassWrapper<DenseResourceElementAttrToDenseElementAttrPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      DenseResourceElementAttrToDenseElementAttrPass)

  void runOnOperation();
  llvm::StringRef getArgument() const;
};

struct DenseResourceRewriter : public OpRewritePattern<memref::GlobalOp> {
  using OpRewritePattern<memref::GlobalOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::GlobalOp globalOp,
                                PatternRewriter &rewriter) const override {
    auto memrefType = globalOp.getType();

    // Only work on 1D stuff because circt-opt flatten-memref handles multi
    // dimensions.
    if (memrefType.getShape().size() != 1)
      return failure();

    auto denseResource = llvm::dyn_cast_or_null<DenseResourceElementsAttr>(
        globalOp.getInitialValueAttr());

    if (!denseResource)
      return failure();

    auto denseResourceData = denseResource.getData();
    auto denseResourceDataType = denseResource.getType();

    auto newDenseAttr = DenseElementsAttr::getFromRawBuffer(
        denseResourceDataType, denseResourceData);
    rewriter.modifyOpInPlace(
        globalOp, [&]() { globalOp.setInitialValueAttr(newDenseAttr); });
    return success();
  }
};

namespace mlir {
void registerStrengthReductionPass() {
  PassRegistration<DenseResourceElementAttrToDenseElementAttrPass>();
}
} // namespace mlir

void DenseResourceElementAttrToDenseElementAttrPass::runOnOperation() {

  RewritePatternSet patterns(&getContext());
  patterns.add<DenseResourceRewriter>(&getContext());

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
llvm::StringRef
DenseResourceElementAttrToDenseElementAttrPass::getArgument() const {
  return "dense-resource-to-dense-element";
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::registerStrengthReductionPass();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "DenseResourceElement2DenseElement", registry));
}
