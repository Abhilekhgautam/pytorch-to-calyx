#include <mlir/Pass/PassRegistry.h>

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "StrengthReduction.h"

namespace mlir {
void registerStrengthReductionPass() {
  PassRegistration<SCFStrengthReductionPass>();
}
} // namespace mlir

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::registerStrengthReductionPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "StrengthReductionPass", registry));
}
