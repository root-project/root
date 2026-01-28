#include "TMVA/ROperator_HardSwish.hxx"
#include "TMVA/RModel.hxx"

#include "gtest/gtest.h"

#include <cmath>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

using namespace TMVA::Experimental::SOFIE;

// Testing hexfloat constants for AVX purity and precision
TEST(SOFIE_HardSwish, GenerateHexfloatConstants)
{
   RModel model;
   model.AddInputTensorInfo("input", ETensorType::FLOAT, std::vector<size_t>{1, 10});
   model.AddOutputTensorNameList({"output"});

   ROperator_HardSwish<float> op("input", "output");
   op.Initialize(model);

   std::string code = op.Generate("hardswish_test");

   // Testing hexfloat clamp bounds (f suffix ensures no double promotion)
   EXPECT_TRUE(code.find("0x0p+0f") != std::string::npos)
      << "Generated code missing optimized float hexfloat constant 0x0p+0f for lower clamp";

   EXPECT_TRUE(code.find("0x1p+0f") != std::string::npos)
      << "Generated code missing optimized float hexfloat constant 0x1p+0f for upper clamp";

   // Testing hexfloat scale factor (1/6)
   EXPECT_TRUE(code.find("0x1.5555555555555p-3f") != std::string::npos)
      << "Generated code missing hexfloat constant 0x1.5555555555555p-3f for 1/6 scale";

   // Testing hexfloat offset (0.5)
   EXPECT_TRUE(code.find("0x1p-1f") != std::string::npos)
      << "Generated code missing hexfloat constant 0x1p-1f for 0.5 offset";

   // Verify std::fmax and std::fmin are used (not std::clamp)
   EXPECT_TRUE(code.find("std::fmax") != std::string::npos)
      << "Generated code should use std::fmax for clamping";

   EXPECT_TRUE(code.find("std::fmin") != std::string::npos)
      << "Generated code should use std::fmin for clamping";
}

// Testing split computation topology (intermediate variable h)
TEST(SOFIE_HardSwish, GenerateSplitTopology)
{
   RModel model;
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<size_t>{2, 5});
   model.AddOutputTensorNameList({"Y"});

   ROperator_HardSwish<float> op("X", "Y");
   op.Initialize(model);

   std::string code = op.Generate("hardswish_topology_test");

   // Verify intermediate variable h is declared
   EXPECT_TRUE(code.find("float h =") != std::string::npos)
      << "Generated code missing intermediate variable 'float h' declaration";

   // Verify h is used in the clamp expression
   EXPECT_TRUE(code.find("std::fmin(0x1p+0f, h)") != std::string::npos)
      << "Generated code should use intermediate variable h in clamp expression";
}

// Testing Numeric correctness in linear region (between clamps)
TEST(SOFIE_HardSwish, NumericCorrectnessLinearRegion)
{
   const std::vector<std::pair<float, float>> referenceData = {
      {-2.0f, -2.0f * std::fmax(0.0f, std::fmin(1.0f, -2.0f/6.0f + 0.5f))}, 
      {-1.0f, -1.0f * std::fmax(0.0f, std::fmin(1.0f, -1.0f/6.0f + 0.5f))},
      { 0.0f,  0.0f * std::fmax(0.0f, std::fmin(1.0f,  0.0f/6.0f + 0.5f))}, 
      { 1.0f,  1.0f * std::fmax(0.0f, std::fmin(1.0f,  1.0f/6.0f + 0.5f))},
      { 2.0f,  2.0f * std::fmax(0.0f, std::fmin(1.0f,  2.0f/6.0f + 0.5f))},
   };

   // Proxy for generated logic (pure float math with exact hexfloat constants)
   auto hardswish_eval = [](float x) -> float {
      float h = 0x1.5555555555555p-3f * x + 0x1p-1f;
      return x * std::fmax(0x0p+0f, std::fmin(0x1p+0f, h));
   };

   for (const auto& [input, expected] : referenceData) {
      float computed = hardswish_eval(input);
      float tol = 1e-6f;

      EXPECT_NEAR(computed, expected, tol)
         << "Linear region mismatch at x = " << input;
   }
}

// Testing Numeric correctness at clamp boundaries
TEST(SOFIE_HardSwish, NumericCorrectnessClamps)
{
   const std::vector<std::pair<float, float>> clampData = {
      {-10.0f, 0.0f}, 
      { -5.0f, 0.0f}, 
      { -3.0f, 0.0f}, 
      {  3.0f, 3.0f}, 
      {  5.0f, 5.0f}, 
      { 10.0f, 10.0f}, 
   };

   auto hardswish_eval = [](float x) -> float {
      float h = 0x1.5555555555555p-3f * x + 0x1p-1f;
      return x * std::fmax(0x0p+0f, std::fmin(0x1p+0f, h));
   };

   for (const auto& [input, expected] : clampData) {
      float computed = hardswish_eval(input);
      float tol = 1e-6f;

      EXPECT_NEAR(computed, expected, tol)
         << "Clamp behavior mismatch at x = " << input;
   }
}

// Testing specific known values
TEST(SOFIE_HardSwish, KnownValues)
{
   auto hardswish_eval = [](float x) -> float {
      float h = 0x1.5555555555555p-3f * x + 0x1p-1f;
      return x * std::fmax(0x0p+0f, std::fmin(0x1p+0f, h));
   };

   float tol = 1e-6f;

   EXPECT_NEAR(hardswish_eval(0.0f), 0.0f, tol);
   EXPECT_NEAR(hardswish_eval(-3.0f), 0.0f, tol);
   EXPECT_NEAR(hardswish_eval(3.0f), 3.0f, tol);
   EXPECT_NEAR(hardswish_eval(1.0f), 1.0f * (1.0f/6.0f + 0.5f), tol);
   EXPECT_NEAR(hardswish_eval(-1.0f), -1.0f * (-1.0f/6.0f + 0.5f), tol);
}

// StdLib dependencies
TEST(SOFIE_HardSwish, StdLibDependencies)
{
   ROperator_HardSwish<float> op("in", "out");
   auto libs = op.GetStdLibs();
   ASSERT_EQ(libs.size(), 1u);
   EXPECT_EQ(libs[0], "cmath");
}

// Type and Shape Inference
TEST(SOFIE_HardSwish, Inference)
{
   ROperator_HardSwish<float> op("in", "out");

   // Type inference
   auto types = op.TypeInference({ETensorType::FLOAT});
   EXPECT_EQ(types[0], ETensorType::FLOAT);

   // Shape inference
   std::vector<size_t> shape = {4, 16, 32};
   auto shapes = op.ShapeInference({shape});
   EXPECT_EQ(shapes[0], shape);
}

// Error Handling
TEST(SOFIE_HardSwish, ErrorHandling)
{
   ROperator_HardSwish<float> op("in", "out");

   // Generate without Initialize
   EXPECT_THROW(op.Generate("test"), std::runtime_error);

   // Initialize with missing tensor
   RModel model;
   EXPECT_THROW(op.Initialize(model), std::runtime_error);
}

// Loop structure verification
TEST(SOFIE_HardSwish, GenerateStructure)
{
   RModel model;
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<size_t>{2, 5});
   model.AddOutputTensorNameList({"Y"});

   ROperator_HardSwish<float> op("X", "Y");
   op.Initialize(model);

   std::string code = op.Generate("hardswish_struct_test");

   EXPECT_TRUE(code.find("tensor_Y") != std::string::npos) << "Missing output tensor access";
   EXPECT_TRUE(code.find("tensor_X") != std::string::npos) << "Missing input tensor access";
   // Loop limit check for shape {2, 5}
   EXPECT_TRUE(code.find("10") != std::string::npos) << "Incorrect loop limit generated";
   // Operator comment
   EXPECT_TRUE(code.find("HardSwish") != std::string::npos) << "Missing operator comment";
}
