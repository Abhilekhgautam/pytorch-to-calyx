cmake_minimum_required(VERSION 4.0)
project(MemrefCopyToAffineLoop)

set(CMAKE_CXX_STANDARD 20)

find_package(MLIR REQUIRED CONFIG)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

add_executable(MemrefCopyToAffineLoop main.cpp
        MemRefCopyToAffineLoop.h
        MemRefCopyToAffineLoop.cpp)

target_compile_options(MemrefCopyToAffineLoop PRIVATE -fno-rtti)

target_link_libraries(MemrefCopyToAffineLoop
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