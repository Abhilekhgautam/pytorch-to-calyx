//
// Created by void on 6/27/26.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <ranges>

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Pass/Pass.h>
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Transforms/DialectConversion.h"

#include <nlohmann/json.hpp>

const std::string conv_bias_bin_path = "/home/void/quantization-test/conv_bias_i32.bin";
const std::string conv_weight_bin_path = "/home/void/quantization-test/conv_weight_i8.bin";

const std::string fc_bias_bin_path = "/home/void/quantization-test/fc_bias_i32.bin";
const std::string fc_weight_bin_path = "/home/void/quantization-test/fc_weight_i8.bin";

const std::string scale_json_path = "/home/void/quantization-test/quantization.json";

struct RequantizeData {
  int64_t multiplier;
  int64_t shift;
};

RequantizeData calculateMultiplierAndShift(double in_scale, double out_scale, double weight_scale) {
  const double scale = (in_scale * weight_scale) / out_scale;

  constexpr int64_t shift = 31;
  const auto multiplier = static_cast<int64_t>(std::round(scale * (1ULL << shift)));

  return {multiplier, shift};
}

std::string readJsonFile(const std::string& filename) {
  std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error("Couldn't open file " + filename);
  }

  std::stringstream buffer;
  buffer << file.rdbuf();

  if (!file) {
    throw std::runtime_error("Couldn't read the file");
  }
  return buffer.str();
}

// Helper to load binary file into a vector
template <typename T>
std::vector<T> loadBinaryData(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Couldn't open the binary file " + filename);
  }
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<T> data(size / sizeof(T));
  file.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

class QuantizationPass
    : public mlir::PassWrapper<QuantizationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      QuantizationPass)

  void runOnOperation() override;
  llvm::StringRef getArgument() const override;
};

class QuantizedTypeConverter : public mlir::TypeConverter {
public:
  QuantizedTypeConverter() {
    addConversion([](mlir::Type type) { return type; });

    addConversion([&](mlir::RankedTensorType type) -> mlir::Type {
        if (type.getElementType().isF32()) {
            return mlir::RankedTensorType::get(type.getShape(), mlir::IntegerType::get(type.getContext(), 8));
        }
        return type;
    });
  }
};

struct CollapseShapeTypeSwap : public mlir::OpConversionPattern<mlir::tensor::CollapseShapeOp> {
  using mlir::OpConversionPattern<mlir::tensor::CollapseShapeOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::tensor::CollapseShapeOp collapseShapeOp, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {

    const auto loc = collapseShapeOp.getLoc();
    const auto result_type = collapseShapeOp.getResultType();

    auto new_result_type = mlir::RankedTensorType::get(result_type.getShape(), rewriter.getI8Type());

    const auto new_collapse_shape = mlir::tensor::CollapseShapeOp::create(rewriter, loc, new_result_type, adaptor.getOperands(), collapseShapeOp->getAttrs());

    rewriter.replaceOp(collapseShapeOp, new_collapse_shape->getResult(0));
    return mlir::success();
  }
};

struct TransposeTypeSwap : public mlir::OpConversionPattern<mlir::linalg::TransposeOp> {
  using mlir::OpConversionPattern<mlir::linalg::TransposeOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::TransposeOp transposeOp, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {

    const auto loc = transposeOp.getLoc();

    llvm::SmallVector<mlir::Type> quantizedTypes;

    for (const auto resultTypes = transposeOp.getResultTypes(); const auto& type: resultTypes) {
      auto t = llvm::dyn_cast<mlir::RankedTensorType>(type);
      quantizedTypes.push_back(mlir::RankedTensorType::get(t.getShape(), rewriter.getI8Type()));
    }

    const auto new_transpose_op = mlir::linalg::TransposeOp::create(rewriter, loc, adaptor.getInput(), adaptor.getOperands()[0], adaptor.getPermutation());

    rewriter.replaceOp(transposeOp, new_transpose_op->getResult(0));
    return mlir::success();
  }
};

struct MaxPoolQuantizer : public mlir::OpConversionPattern<mlir::linalg::PoolingNchwMaxOp> {
  using mlir::OpConversionPattern<mlir::linalg::PoolingNchwMaxOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::PoolingNchwMaxOp maxPoolingOp, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {

    const auto loc = maxPoolingOp.getLoc();

    const auto input_image = adaptor.getInputs()[0];

    const auto empty_shape = llvm::dyn_cast<mlir::RankedTensorType>(adaptor.getInputs()[1].getType()).getShape();
    const auto empty_tensor = mlir::tensor::EmptyOp::create(rewriter, loc, empty_shape, rewriter.getI8Type());

    const auto resultTypes = maxPoolingOp.getResultTypes();

    // -127
    const auto i8MinConst = mlir::arith::ConstantOp::create(rewriter, loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(-127));

    const auto output_shape = llvm::dyn_cast<mlir::RankedTensorType>(maxPoolingOp.getResult(0).getType()).getShape();
    const auto new_output = mlir::tensor::EmptyOp::create(rewriter, loc, output_shape, rewriter.getI8Type());

    const auto new_fill_op = mlir::linalg::FillOp::create(rewriter, loc, mlir::ValueRange{i8MinConst->getResult(0)}, new_output->getResult(0));

    llvm::SmallVector<mlir::Type> quantizedTypes;

    const auto new_max_pooling = mlir::linalg::PoolingNchwMaxOp::create(rewriter, loc, mlir::ValueRange{input_image, empty_tensor->getResult(0)}, mlir::ValueRange{new_fill_op->getResult(0)});
    new_max_pooling->setAttrs(maxPoolingOp->getAttrs());

    rewriter.replaceOp(maxPoolingOp, new_max_pooling);

    if (const auto oldFill = maxPoolingOp.getOutputs()[0].getDefiningOp()) {
      if (oldFill->use_empty() || oldFill->hasOneUse()) {

        const auto oldOutEmpty = oldFill->getOperand(1).getDefiningOp();
        const auto oldInfConst = oldFill->getOperand(0).getDefiningOp();

        // Erase the fill
        rewriter.eraseOp(oldFill);

        // Erase the things feeding the fill
        if (oldOutEmpty && (oldOutEmpty->use_empty() || oldOutEmpty->hasOneUse())) rewriter.eraseOp(oldOutEmpty);
        if (oldInfConst && (oldInfConst->use_empty() || oldInfConst->hasOneUse())) rewriter.eraseOp(oldInfConst);
      }
    }

    // Cleanup the old Window Tensor (empty)
    if (const auto oldWindow = maxPoolingOp.getInputs()[1].getDefiningOp()) {
      if (oldWindow->use_empty() || oldWindow->hasOneUse()) rewriter.eraseOp(oldWindow);
    }

    return mlir::success();
  }
};

struct MatMulQuantizer : public mlir::OpConversionPattern<mlir::linalg::MatmulOp> {
  using mlir::OpConversionPattern<mlir::linalg::MatmulOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::MatmulOp mulOp, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override{
    const auto loc = mulOp.getLoc();

    // this is a collapsed value
    const auto quantized_input = adaptor.getInputs()[0];

    const auto i10_1352TensorType = mlir::RankedTensorType::get({mlir::ArrayRef<int64_t>{10, 1352}}, mlir::IntegerType::get(getContext(), 8));
    const auto qfc_weight = loadBinaryData<int8_t>(fc_weight_bin_path);

    const auto qfc_weight_array_ref = llvm::ArrayRef<char>(
       reinterpret_cast<const char *>(qfc_weight.data()),
       qfc_weight.size() * sizeof(int8_t)
    );

    const auto fc_weight_dense_elt_attr = mlir::DenseElementsAttr::getFromRawBuffer(i10_1352TensorType, qfc_weight_array_ref);

    const auto fc_weight_const = mlir::arith::ConstantOp::create(rewriter, loc, i10_1352TensorType , fc_weight_dense_elt_attr);

    const auto transposed_type = mlir::RankedTensorType::get({mlir::ArrayRef<int64_t>{1352, 10}}, mlir::IntegerType::get(getContext(), 8));

    const auto empty_tensor = mlir::tensor::EmptyOp::create(rewriter, loc, llvm::ArrayRef<int64_t>{1352, 10}, rewriter.getI8Type());

    const auto transposeOp = mlir::linalg::TransposeOp::create(rewriter, loc, fc_weight_const->getResult(0), empty_tensor->getResult(0), rewriter.getDenseI64ArrayAttr({1, 0}));

    const auto qfc_bias = loadBinaryData<int32_t>(fc_bias_bin_path);

    const auto i10TensorType = mlir::RankedTensorType::get({mlir::ArrayRef<int64_t>{10}}, mlir::IntegerType::get(getContext(), 32));

    auto qfc_bias_array_ref = llvm::ArrayRef<char>(
      reinterpret_cast<const char*>(qfc_bias.data()),
      qfc_bias.size() * sizeof(int32_t)
    );

    const auto fc_bias_dense_elt_attr = mlir::DenseElementsAttr::getFromRawBuffer(i10TensorType, qfc_bias_array_ref);

    const auto fc_bias_const = mlir::arith::ConstantOp::create(rewriter, loc, i10TensorType, fc_bias_dense_elt_attr);

    const auto accumType = mlir::RankedTensorType::get({mlir::ArrayRef<int64_t>{1, 10}}, mlir::IntegerType::get(getContext(), 32));

    const auto accumEmptyTensor = mlir::tensor::EmptyOp::create(rewriter, loc, llvm::ArrayRef<int64_t>{1, 10}, rewriter.getI32Type());

    const auto broadCastOp = mlir::linalg::BroadcastOp::create(rewriter, loc, fc_bias_const->getResult(0), accumEmptyTensor->getResult(0), rewriter.getDenseI64ArrayAttr({0}));

    auto quantizedMatmulOp = mlir::linalg::MatmulOp::create(rewriter, loc, {quantized_input, transposeOp->getResult(0)}, {broadCastOp->getResult(0)});

    // 7. Truncate to i8
    const auto i8OutputType = mlir::RankedTensorType::get({1, 10}, rewriter.getI8Type());

    const std::string json = readJsonFile(scale_json_path);

    nlohmann::json json_code = nlohmann::json::parse(json);

    const double in_scale = json_code["activation_scales"]["fc"]["in_scale"];
    const double fc_weight_scale = json_code["weight_scales"]["fc.weight"];
    const double out_scale = json_code["activation_scales"]["fc"]["out_scale"];

    const auto [multiplier, shift] = calculateMultiplierAndShift(in_scale, out_scale, fc_weight_scale);

    auto emptyI8Tensor = mlir::tensor::EmptyOp::create(rewriter, loc, i8OutputType.getShape(), rewriter.getI8Type());

    llvm::SmallVector<mlir::Attribute> iteratorTypes(i8OutputType.getShape().size(),
      mlir::linalg::IteratorTypeAttr::get(getContext(), mlir::utils::IteratorType::parallel));

    auto test = rewriter.getArrayAttr(iteratorTypes);

    auto requantize_block = mlir::linalg::GenericOp::create(
      rewriter,
      loc,
      i8OutputType,
      mlir::ValueRange{quantizedMatmulOp->getResult(0)},
      mlir::ValueRange{emptyI8Tensor->getResult(0)},
      rewriter.getAffineMapArrayAttr({rewriter.getMultiDimIdentityMap(i8OutputType.getShape().size()), rewriter.getMultiDimIdentityMap(i8OutputType.getShape().size())}),
      test,
      nullptr, nullptr,
      [&](mlir::OpBuilder& builder, mlir::Location location, mlir::ValueRange args) {
        auto pixel_i64 = mlir::arith::ExtSIOp::create(builder, location, builder.getI64Type(), args[0]);

        auto multConst = mlir::arith::ConstantOp::create(builder, location, builder.getI64Type(), builder.getI64IntegerAttr(multiplier));
        auto shiftConst = mlir::arith::ConstantOp::create(builder, location, builder.getI64Type(), builder.getI64IntegerAttr(shift));

        auto multiplied = mlir::arith::MulIOp::create(builder, location, pixel_i64, multConst);
        auto shifted = mlir::arith::ShRSIOp::create(builder, location, multiplied, shiftConst);

        auto downcast_i32 = mlir::arith::TruncIOp::create(builder, location, builder.getI32Type(), shifted);

        // Clamp
        auto c127 = mlir::arith::ConstantOp::create(builder, location, builder.getI32Type(), builder.getI32IntegerAttr(127));
        auto c_128 = mlir::arith::ConstantOp::create(builder, location, builder.getI32Type(), builder.getI32IntegerAttr(-127));
        auto clamped_max = mlir::arith::MinSIOp::create(builder, location, downcast_i32, c127);
        auto clamped_min = mlir::arith::MaxSIOp::create(builder, location, clamped_max, c_128);

        auto final_i8 = mlir::arith::TruncIOp::create(builder, location, builder.getI8Type(), clamped_min);

        mlir::linalg::YieldOp::create(builder, loc, final_i8.getResult());
      }
      );

    rewriter.replaceOp(mulOp, requantize_block.getResult(0));


    if (auto oldMatmulResult = mulOp.getResult(0); oldMatmulResult.hasOneUse()) {
      auto oldBiasAddGeneric = *oldMatmulResult.getUsers().begin();
      rewriter.replaceOp(oldBiasAddGeneric, requantize_block.getResult(0));
    }

    // Erase the old Matmul (and downstream things are cleaned up automatically!)
    //rewriter.eraseOp(mulOp);


    if (auto oldTranspose = mulOp.getInputs()[1].getDefiningOp(); oldTranspose && oldTranspose->use_empty()) rewriter.eraseOp(oldTranspose);
    return llvm::success();

  }
};

struct Conv2DQuantizer : public mlir::OpConversionPattern<mlir::linalg::Conv2DNchwFchwOp> {
  using mlir::OpConversionPattern<mlir::linalg::Conv2DNchwFchwOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::Conv2DNchwFchwOp convOp, OpAdaptor adaptor,
                                mlir::ConversionPatternRewriter &rewriter) const override {

    const auto loc = convOp.getLoc();

    // conv.weight setup
    const auto i8_1_3_3TensorType = mlir::RankedTensorType::get(mlir::ArrayRef<int64_t>{8, 1, 3, 3}, mlir::IntegerType::get(convOp.getContext(), 8));
    const auto qconv_weight = loadBinaryData<int8_t>(conv_weight_bin_path);

    const auto qconv_weight_array_ref = llvm::ArrayRef<char>(
         reinterpret_cast<const char *>(qconv_weight.data()),
         qconv_weight.size() * sizeof(int8_t)
      );

    const auto conv_weight_dense_elt_attr = mlir::DenseElementsAttr::getFromRawBuffer(i8_1_3_3TensorType, qconv_weight_array_ref);

    // conv.bias setup
    const auto qconv_bias = loadBinaryData<int32_t>(conv_bias_bin_path);
    const auto i8TensorType = mlir::RankedTensorType::get({mlir::ArrayRef<int64_t>{8}}, mlir::IntegerType::get(getContext(), 32));

    const auto qconv_bias_array_ref = llvm::ArrayRef<char>(
      reinterpret_cast<const char*>(qconv_bias.data()),
      qconv_bias.size() * sizeof(int32_t)
    );

    const auto conv_bias_dense_elt_attr = mlir::DenseElementsAttr::getFromRawBuffer(i8TensorType, qconv_bias_array_ref);

    const auto quantized_conv_bias = mlir::arith::ConstantOp::create(rewriter, loc, i8TensorType, conv_bias_dense_elt_attr);

    const auto quantized_weight_i8 = mlir::arith::ConstantOp::create(rewriter, loc, i8_1_3_3TensorType, conv_weight_dense_elt_attr);
    const auto quantized_input_image = adaptor.getInputs()[0];

    const auto output_type = llvm::dyn_cast_if_present<mlir::RankedTensorType>(convOp.getResult(0).getType());
    const auto output_shape = output_type.getShape();

    const auto i32AccumType = mlir::RankedTensorType::get(output_shape, rewriter.getI32Type());
    const auto emptyTensor = mlir::tensor::EmptyOp::create(rewriter, loc, output_shape, rewriter.getI32Type());

    const auto new_broadcastOp = mlir::linalg::BroadcastOp::create(rewriter, loc, quantized_conv_bias->getResult(0), emptyTensor->getResult(0), rewriter.getDenseI64ArrayAttr({0, 2, 3}));

    const auto quantized_conv2d = mlir::linalg::Conv2DNchwFchwOp::create(rewriter, loc, {quantized_input_image, quantized_weight_i8->getResults()[0]}, new_broadcastOp->getResult(0));

    quantized_conv2d->setAttrs(convOp->getAttrs());

    // Requantize

    const std::string json = readJsonFile(scale_json_path);

    nlohmann::json json_code = nlohmann::json::parse(json);

    const double in_scale = json_code["activation_scales"]["conv"]["in_scale"];
    const double conv_weight_scale = json_code["weight_scales"]["conv.weight"];
    const double out_scale = json_code["activation_scales"]["conv"]["out_scale"];

    const auto requantize_data = calculateMultiplierAndShift(in_scale, out_scale, conv_weight_scale);

    auto emptyI8Tensor = mlir::tensor::EmptyOp::create(rewriter, loc, output_shape, rewriter.getI8Type());

    const auto i8OutputType = mlir::RankedTensorType::get(output_shape, rewriter.getI8Type());

    llvm::SmallVector<mlir::Attribute> iteratorTypes(output_shape.size(),
      mlir::linalg::IteratorTypeAttr::get(getContext(), mlir::utils::IteratorType::parallel));

    auto test = rewriter.getArrayAttr(iteratorTypes);

    auto requantize_block = mlir::linalg::GenericOp::create(
      rewriter,
      loc,
      i8OutputType,
      mlir::ValueRange{quantized_conv2d->getResult(0)},
      mlir::ValueRange{emptyI8Tensor->getResult(0)},
      rewriter.getAffineMapArrayAttr({rewriter.getMultiDimIdentityMap(output_shape.size()), rewriter.getMultiDimIdentityMap(output_shape.size())}),
      test,
      nullptr, nullptr,
      [&](mlir::OpBuilder& builder, mlir::Location location, mlir::ValueRange args) {
        auto pixel_i64 = mlir::arith::ExtSIOp::create(builder, location, builder.getI64Type(), args[0]);

        auto multConst = mlir::arith::ConstantOp::create(builder, location, builder.getI64Type(), builder.getI64IntegerAttr(requantize_data.multiplier));
        auto shiftConst = mlir::arith::ConstantOp::create(builder, location, builder.getI64Type(), builder.getI64IntegerAttr(requantize_data.shift));

        auto multiplied = mlir::arith::MulIOp::create(builder, location, pixel_i64, multConst);
        auto shifted = mlir::arith::ShRSIOp::create(builder, location, multiplied, shiftConst);

        auto downcast_i32 = mlir::arith::TruncIOp::create(builder, location, builder.getI32Type(), shifted);

        // Clamp
        auto c127 = mlir::arith::ConstantOp::create(builder, location, builder.getI32Type(), builder.getI32IntegerAttr(127));
        auto c_128 = mlir::arith::ConstantOp::create(builder, location, builder.getI32Type(), builder.getI32IntegerAttr(-127));
        auto clamped_max = mlir::arith::MinSIOp::create(builder, location, downcast_i32, c127);
        auto clamped_min = mlir::arith::MaxSIOp::create(builder, location, clamped_max, c_128);

        auto final_i8 = mlir::arith::TruncIOp::create(builder, location, builder.getI8Type(), clamped_min);

        mlir::linalg::YieldOp::create(builder, loc, final_i8.getResult());
      }
      );

    rewriter.replaceOp(convOp, requantize_block.getResult(0));

    if (const auto oldBroadcast = convOp.getOutputs()[0].getDefiningOp(); oldBroadcast) {
      if (oldBroadcast->use_empty()) rewriter.eraseOp(oldBroadcast);
      if (const auto empty = oldBroadcast->getOperand(1).getDefiningOp()) {
        if (empty->use_empty()) rewriter.eraseOp(empty);
      }
      if (const auto bConst = oldBroadcast->getOperand(0).getDefiningOp()) {
        if (bConst->use_empty()) rewriter.eraseOp(bConst);
      }
    }
    if (const auto wConst = convOp.getInputs()[1].getDefiningOp()) rewriter.eraseOp(wConst);
    return llvm::success();
  }
};

struct ReturnOpConverter : public mlir::OpConversionPattern<mlir::func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::func::ReturnOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    // Replaces the old f32 return with a new return that takes the converted i8 tensors!
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, adaptor.getOperands());
    return mlir::success();
  }
};

struct ReluQuantizer : public mlir::OpConversionPattern<mlir::linalg::GenericOp> {
  using mlir::OpConversionPattern<mlir::linalg::GenericOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp genericOp, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {

    if (const auto inputs = genericOp.getInputs(); inputs.size() != 1) return llvm::failure();

    const auto loc = genericOp.getLoc();
    const auto input_i8 = adaptor.getInputs()[0];

    const auto resultShape = llvm::dyn_cast<mlir::RankedTensorType>(genericOp.getResult(0).getType()).getShape();

    const auto newOutputType = mlir::RankedTensorType::get(resultShape, rewriter.getI8Type());

    const auto newOutput = mlir::tensor::EmptyOp::create(rewriter, loc, resultShape, rewriter.getI8Type());

    auto newGenericOp = mlir::linalg::GenericOp::create(
      rewriter,
      loc,
      newOutputType,
      mlir::ValueRange{input_i8},
      mlir::ValueRange{newOutput->getResult(0)},
      genericOp.getIndexingMapsAttr(),
      genericOp.getIteratorTypesAttr(),
      genericOp.getDocAttr(),
      genericOp.getLibraryCallAttr(),
      [&](mlir::OpBuilder& builder, mlir::Location location, mlir::ValueRange ips) {

        const auto i8Zero = mlir::arith::ConstantOp::create(builder, location, rewriter.getI8Type(), rewriter.getI8IntegerAttr(0));

        const auto i8Cmp = mlir::arith::CmpIOp::create(builder, location, mlir::arith::CmpIPredicate::sgt, ips[0], i8Zero->getResult(0));

        const auto selOp = mlir::arith::SelectOp::create(builder, location, rewriter.getI8Type(), i8Cmp->getResult(0), ips[0], i8Zero->getResult(0));

        const auto yieldOp = mlir::linalg::YieldOp::create(builder, location, selOp->getResult(0));
      });

    rewriter.replaceOp(genericOp, newGenericOp);

    if (const auto oldEmpty = genericOp.getOutputs()[0].getDefiningOp()) {
      if (oldEmpty->use_empty()) rewriter.eraseOp(oldEmpty);
    }

    return llvm::success();
  }
};

llvm::StringRef QuantizationPass::getArgument() const {
  return "quantize";
}

bool hasF32Type(mlir::Operation *op) {
  auto isOrContainsF32 = [](mlir::Type type) {
    // Check if it's a scalar f32
    if (type.isF32()) return true;

    // Check if it's a Tensor (or MemRef) containing f32
    if (auto shapedType = llvm::dyn_cast<mlir::ShapedType>(type)) {
      return shapedType.getElementType().isF32();
    }
    return false;
  };

  // Check all operand types
  for (mlir::Type type : op->getOperandTypes()) {
    if (isOrContainsF32(type)) return true;
  }
  // Check all result types
  for (mlir::Type type : op->getResultTypes()) {
    if (isOrContainsF32(type)) return true;
  }

  return false;
}

void QuantizationPass::runOnOperation() {
  QuantizedTypeConverter typeConverter;

  mlir::ConversionTarget target(getContext());
  mlir::RewritePatternSet patterns(&getContext());

  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);

  target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
  });

  target.addDynamicallyLegalOp<mlir::func::ReturnOp>([&](mlir::func::ReturnOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
  });

  target.addDynamicallyLegalOp<mlir::linalg::Conv2DNchwFchwOp>(
      [](mlir::Operation *op) {
          return !hasF32Type(op);
      });

  target.addDynamicallyLegalOp<mlir::tensor::CollapseShapeOp,
                             mlir::linalg::PoolingNchwMaxOp,
                             mlir::linalg::GenericOp,
                             mlir::linalg::MatmulOp>([](mlir::Operation *op) {
    return !hasF32Type(op);
});

  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<mlir::tensor::TensorDialect>();
  target.addLegalOp<mlir::linalg::BroadcastOp, mlir::linalg::YieldOp, mlir::linalg::TransposeOp, mlir::linalg::FillOp>();

  patterns.add<Conv2DQuantizer>(typeConverter, &getContext());
  patterns.add<ReturnOpConverter>(typeConverter, &getContext());
  patterns.add<CollapseShapeTypeSwap>(typeConverter, &getContext());
  patterns.add<MaxPoolQuantizer>(typeConverter, &getContext());
  patterns.add<MatMulQuantizer>(typeConverter, &getContext());
  patterns.add<ReluQuantizer>(typeConverter, &getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    llvm::errs() << "\n--- IR STATE AT FAILURE ---\n";
    getOperation()->dump(); // Prints the MLIR right before it crashes!
    llvm::errs() << "---------------------------\n\n";
    signalPassFailure();
  }
}

namespace mlir {
  void registerQuantizationPass() {
    PassRegistration<QuantizationPass>();
  }
}

int main(int argc, char** argv){

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::registerQuantizationPass();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "quantize", registry));
}