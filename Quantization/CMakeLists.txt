cmake_minimum_required(VERSION 4.0)
project(Quantization)

set(CMAKE_CXX_STANDARD 20)

find_package(MLIR REQUIRED CONFIG)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories("/home/void/json/include")
include_directories(${LLVM_INCLUDE_DIRS})

add_executable(Quantization main.cpp)

target_compile_options(Quantization PRIVATE -fno-rtti)

target_link_libraries(Quantization
        PRIVATE
        MLIRIR
        MLIRSupport
        MLIRPass

        MLIRTransforms
        MLIRMemRefDialect
        MLIRFuncDialect
        MLIRAffineDialect

        MLIRRegisterAllDialects
        MLIRRegisterAllPasses
        MLIRRegisterAllExtensions

        MLIROptLib
        MLIRRewrite
)