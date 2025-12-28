#include <mlir/Pass/PassRegistry.h>

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "MemRefCopyToAffineLoop.h"

namespace mlir {
    void registerMemRefCopyToAffineLoopPass() {
        PassRegistration<MemRefCopyToAffineLoopPass>();
    }
}

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllPasses();

    mlir::registerMemRefCopyToAffineLoopPass();

    return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "MemRefCopyToAffineLoopPass", registry));
}