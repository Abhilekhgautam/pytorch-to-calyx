cmake_minimum_required(VERSION 4.0)
project(DenseResourceElement2DenseElement)

set(CMAKE_CXX_STANDARD 20)

find_package(MLIR REQUIRED CONFIG)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

add_executable(DenseResourceElement2DenseElement main.cpp)
target_compile_options(DenseResourceElement2DenseElement  PRIVATE -fno-rtti)

target_link_libraries(DenseResourceElement2DenseElement PRIVATE
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
