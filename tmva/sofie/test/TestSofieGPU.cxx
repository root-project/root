#include "TMVA/ROperator.hxx"
#include "TMVA/ROperator_Gemm.hxx"
// For the licensing terms see $ROOTSYS/LICENSE.
#include "TMVA/RGPUModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>

using namespace TMVA::Experimental::SOFIE;
namespace {
std::string Varint(std::uint64_t v)
{
   std::string s;
   do {
      s += char((v & 127) | (v >= 128 ? 128 : 0));
      v >>= 7;
   } while (v);
   return s;
}
std::string Number(int field, std::uint64_t n)
{
   return Varint(field * 8) + Varint(n);
}
std::string Bytes(int field, const std::string &s)
{
   return Varint(field * 8 + 2) + Varint(s.size()) + s;
}
std::string TensorInfo(const char *name, int type, int width)
{
   auto shape = Bytes(1, Bytes(2, "batch")) + Bytes(1, Number(1, width));
   return Bytes(1, name) + Bytes(2, Bytes(1, Number(1, type) + Bytes(2, shape)));
}
std::string TinyModel(bool half, const std::string &domain = "", const std::string &op = "Gemm")
{
   const int type = half ? 10 : 1;
   std::string raw = half ? std::string("\0\x3c\0\x40", 4) : std::string("\0\0\x80\x3f\0\0\0\x40", 8);
   auto weight = Number(1, 2) + Number(1, 1) + Number(2, type) + Bytes(8, "w") + Bytes(9, raw);
   auto node = Bytes(1, "x") + Bytes(1, "w") + Bytes(2, "y") + Bytes(4, op) + Bytes(7, domain);
   auto graph =
      Bytes(1, node) + Bytes(5, weight) + Bytes(11, TensorInfo("x", type, 2)) + Bytes(12, TensorInfo("y", type, 1));
   return Number(1, 8) + Bytes(7, graph);
}
} // namespace
TEST(SofieGPU, ParseFloatAndHalf)
{
   RModelParser_ONNX parser;
   for (bool half : {false, true}) {
      std::istringstream input(TinyModel(half));
      auto model = parser.ParseGPU(input);
      EXPECT_EQ(model.InputSize(), 2u);
      EXPECT_EQ(model.OutputSize(), 1u);
      EXPECT_EQ(model.GetPrecision(), half ? RGPUModel::Precision::Float16 : RGPUModel::Precision::Float32);
      EXPECT_EQ(model.WorkspaceSize(3) % model.WorkspaceAlignment(), 0u);
      const auto cuda = model.GenerateSource(RGPUModel::Backend::CUDA);
      const auto hip = model.GenerateSource(RGPUModel::Backend::HIP);
      EXPECT_NE(cuda.find("cuda_runtime.h"), std::string::npos);
      EXPECT_NE(hip.find("hip/hip_runtime.h"), std::string::npos);
      EXPECT_EQ(cuda.find("TMVA/"), std::string::npos);
      EXPECT_EQ(hip.find("Malloc"), std::string::npos);
      EXPECT_THROW(model.CreateSession(nullptr, 0, 3), std::logic_error);
   }
}
TEST(SofieGPU, RejectInvalidGraphs)
{
   RModelParser_ONNX parser;
   for (const auto &bytes : {TinyModel(false, "custom"), TinyModel(false, "", "Conv"), std::string("invalid")}) {
      std::istringstream input(bytes);
      EXPECT_THROW(parser.ParseGPU(input), std::runtime_error);
   }
}
TEST(SofieGPU, CapacityAndShapes)
{
   RGPUModel::Layer layer{2, 1, {1.f, 2.f}, {0.f}};
   RGPUModel model(RGPUModel::Precision::Float32, {layer});
   EXPECT_THROW(model.WorkspaceSize(0), std::invalid_argument);
   EXPECT_THROW(model.WorkspaceSize(std::numeric_limits<std::size_t>::max()), std::overflow_error);
   RGPUModel fixed(RGPUModel::Precision::Float32, {layer}, 4);
   EXPECT_THROW(fixed.WorkspaceSize(3), std::invalid_argument);
   EXPECT_GT(fixed.WorkspaceSize(4), 0u);
   layer.weights.pop_back();
   EXPECT_THROW((RGPUModel(RGPUModel::Precision::Float32, {layer})), std::invalid_argument);
}

TEST(SofieGPU, UsesSharedOperatorRegistry)
{
   RModelParser_ONNX parser;
   bool called = false;
   parser.RegisterOperator("Gemm", [&](RModelParser_ONNX &, const onnx::NodeProto &) -> std::unique_ptr<ROperator> {
      called = true;
      throw std::runtime_error("custom registry entry");
   });
   std::istringstream input(TinyModel(false));
   EXPECT_THROW(parser.ParseGPU(input), std::runtime_error);
   EXPECT_TRUE(called);
}

TEST(SofieGPU, LowerExistingRModel)
{
   RModel model;
   model.AddInputTensorInfo("x", ETensorType::FLOAT, std::vector<size_t>{1, 2});
   model.AddInputTensorName("x");
   float weights[] = {1.f, 2.f};
   model.AddInitializedTensor("w", ETensorType::FLOAT, {2, 1}, weights);
   model.AddOperator(std::make_unique<ROperator_Gemm<float>>(1.f, 1.f, 0, 0, "x", "w", "y"));
   model.AddOutputTensorNameList({"y"});
   auto gpu = model.MakeGPUModel(RGPUModel::Precision::Float32);
   EXPECT_EQ(gpu.InputSize(), 2u);
   EXPECT_EQ(gpu.OutputSize(), 1u);
}
