cmake_minimum_required(VERSION 4.0)
project(SimplifyAddressCalculation)

set(CMAKE_CXX_STANDARD 20)

find_package(MLIR REQUIRED CONFIG)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

add_executable(SimplifyAddressCalculation main.cpp
        SimplifyAddressCalculation.h
        SimplifyAddressCalculation.cpp)

target_compile_options(SimplifyAddressCalculation PRIVATE -fno-rtti)

target_link_libraries(SimplifyAddressCalculation
        PRIVATE
        MLIRIR
        MLIRSupport
        MLIRPass

        MLIRTransforms
        MLIRMemRefDialect
        MLIRSCFDialect
        MLIRFuncDialect
        MLIRAffineDialect

        MLIRRegisterAllDialects
        MLIRRegisterAllPasses
        MLIRRegisterAllExtensions

        MLIROptLib
        MLIRRewrite
)
