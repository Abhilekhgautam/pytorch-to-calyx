cmake_minimum_required(VERSION 4.0)
project(StrengthReduction)

set(CMAKE_CXX_STANDARD 20)

find_package(MLIR REQUIRED CONFIG)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

add_executable(StrengthReduction main.cpp
        StrengthReduction.h
        StrengthReduction.cpp)

target_compile_options(StrengthReduction PRIVATE -fno-rtti)

target_link_libraries(StrengthReduction
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
