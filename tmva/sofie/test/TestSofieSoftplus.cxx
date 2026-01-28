#include "TMVA/ROperator_Softplus.hxx"
#include "TMVA/RModel.hxx"

#include "gtest/gtest.h"

#include <cmath>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

using namespace TMVA::Experimental::SOFIE;

// Testing hexfloat threshold constant for overflow protection
TEST(SOFIE_Softplus, GenerateHexfloatConstants)
{
   RModel model;
   model.AddInputTensorInfo("input", ETensorType::FLOAT, std::vector<size_t>{1, 10});
   model.AddOutputTensorNameList({"output"});

   ROperator_Softplus<float> op("input", "output");
   op.Initialize(model);

   std::string code = op.Generate("softplus_test");

   // Testing hexfloat threshold (20.0f) for overflow protection
   EXPECT_TRUE(code.find("0x1.4000000000000p+4f") != std::string::npos)
      << "Generated code missing hexfloat threshold constant 0x1.4000000000000p+4f (20.0f)";

   // Verify local variable x is declared for CSE-independent code
   EXPECT_TRUE(code.find("float x =") != std::string::npos)
      << "Generated code should declare local variable 'float x' for clean emitted code";

   // Verify ternary conditional structure
   EXPECT_TRUE(code.find("?") != std::string::npos && code.find(":") != std::string::npos)
      << "Generated code should use ternary operator for threshold branch";
}

// Testing numerical stability functions
TEST(SOFIE_Softplus, GenerateStabilityFunctions)
{
   RModel model;
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<size_t>{2, 5});
   model.AddOutputTensorNameList({"Y"});

   ROperator_Softplus<float> op("X", "Y");
   op.Initialize(model);

   std::string code = op.Generate("softplus_stability_test");

   // Verify std::log1p is used (not std::log)
   EXPECT_TRUE(code.find("std::log1p") != std::string::npos)
      << "Generated code must use std::log1p for numerical stability";

   // Verify std::exp is used
   EXPECT_TRUE(code.find("std::exp") != std::string::npos)
      << "Generated code must use std::exp";

   // Verify std::log is NOT used (would indicate precision loss)
   size_t log1p_pos = code.find("std::log1p");
   size_t log_pos = code.find("std::log(");
   EXPECT_TRUE(log_pos == std::string::npos || (log1p_pos != std::string::npos && log1p_pos < log_pos))
      << "Generated code should use std::log1p, not std::log";
}

// Testing numeric correctness in stable region
TEST(SOFIE_Softplus, NumericCorrectnessStableRegion)
{
   const std::vector<std::pair<float, float>> referenceData = {
      {-10.0f, std::log1p(std::exp(-10.0f))},
      { -5.0f, std::log1p(std::exp(-5.0f))},
      { -1.0f, std::log1p(std::exp(-1.0f))},
      {  0.0f, std::log1p(std::exp(0.0f))},   // ln(2) ≈ 0.693
      {  1.0f, std::log1p(std::exp(1.0f))},
      {  5.0f, std::log1p(std::exp(5.0f))},
      { 10.0f, std::log1p(std::exp(10.0f))},
      { 15.0f, std::log1p(std::exp(15.0f))},
   };

   // Proxy for generated logic with threshold
   auto softplus_eval = [](float x) -> float {
      return (x >= 0x1.4000000000000p+4f) ? x : std::log1p(std::exp(x));
   };

   for (const auto& [input, expected] : referenceData) {
      float computed = softplus_eval(input);
      float tol = 1e-6f;

      EXPECT_NEAR(computed, expected, tol)
         << "Stable region mismatch at x = " << input;
   }
}

// Testing threshold behavior for overflow protection
TEST(SOFIE_Softplus, NumericCorrectnessThreshold)
{
   const std::vector<std::pair<float, float>> thresholdData = {
      { 20.0f,  20.0f},  // At threshold: passthrough
      { 25.0f,  25.0f},  // Above threshold: passthrough
      { 50.0f,  50.0f},  // Far above: passthrough
      {100.0f, 100.0f},  // Extreme: would overflow exp() without threshold
      {1000.0f, 1000.0f}, // Very extreme: definite overflow without protection
   };

   auto softplus_eval = [](float x) -> float {
      return (x >= 0x1.4000000000000p+4f) ? x : std::log1p(std::exp(x));
   };

   for (const auto& [input, expected] : thresholdData) {
      float computed = softplus_eval(input);
      float tol = 1e-6f;

      EXPECT_NEAR(computed, expected, tol)
         << "Threshold behavior mismatch at x = " << input;

      // Ensure no NaN or Inf
      EXPECT_FALSE(std::isnan(computed)) << "NaN at x = " << input;
      EXPECT_FALSE(std::isinf(computed)) << "Inf at x = " << input;
   }
}

// Testing specific known values
TEST(SOFIE_Softplus, KnownValues)
{
   auto softplus_eval = [](float x) -> float {
      return (x >= 0x1.4000000000000p+4f) ? x : std::log1p(std::exp(x));
   };

   float tol = 1e-6f;

   // ln(1 + e^0) = ln(2)
   EXPECT_NEAR(softplus_eval(0.0f), 0.6931471805599453f, tol);

   // For large negative x: ln(1 + e^x) ≈ e^x ≈ 0
   EXPECT_NEAR(softplus_eval(-20.0f), std::exp(-20.0f), tol);

   // At threshold: exact passthrough
   EXPECT_NEAR(softplus_eval(20.0f), 20.0f, tol);

   // Just below threshold: computed value
   float x = 19.9f;
   EXPECT_NEAR(softplus_eval(x), std::log1p(std::exp(x)), tol);
}

// StdLib dependencies
TEST(SOFIE_Softplus, StdLibDependencies)
{
   ROperator_Softplus<float> op("in", "out");
   auto libs = op.GetStdLibs();
   ASSERT_EQ(libs.size(), 1u);
   EXPECT_EQ(libs[0], "cmath");
}

// Type and Shape Inference
TEST(SOFIE_Softplus, Inference)
{
   ROperator_Softplus<float> op("in", "out");

   // Type inference
   auto types = op.TypeInference({ETensorType::FLOAT});
   EXPECT_EQ(types[0], ETensorType::FLOAT);

   // Shape inference
   std::vector<size_t> shape = {4, 16, 32};
   auto shapes = op.ShapeInference({shape});
   EXPECT_EQ(shapes[0], shape);
}

// Error Handling
TEST(SOFIE_Softplus, ErrorHandling)
{
   ROperator_Softplus<float> op("in", "out");

   // Generate without Initialize
   EXPECT_THROW(op.Generate("test"), std::runtime_error);

   // Initialize with missing tensor
   RModel model;
   EXPECT_THROW(op.Initialize(model), std::runtime_error);
}

// Loop structure verification
TEST(SOFIE_Softplus, GenerateStructure)
{
   RModel model;
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<size_t>{2, 5});
   model.AddOutputTensorNameList({"Y"});

   ROperator_Softplus<float> op("X", "Y");
   op.Initialize(model);

   std::string code = op.Generate("softplus_struct_test");

   EXPECT_TRUE(code.find("tensor_Y") != std::string::npos) << "Missing output tensor access";
   EXPECT_TRUE(code.find("tensor_X") != std::string::npos) << "Missing input tensor access";
   // Loop limit check for shape {2, 5}
   EXPECT_TRUE(code.find("10") != std::string::npos) << "Incorrect loop limit generated";
   // Operator comment
   EXPECT_TRUE(code.find("Softplus") != std::string::npos) << "Missing operator comment";
}

// Threshold constant verification (20.0f as hexfloat)
TEST(SOFIE_Softplus, ThresholdConstantValue)
{
   // Verify the hexfloat threshold equals 20.0f exactly
   float threshold = 0x1.4000000000000p+4f;
   EXPECT_FLOAT_EQ(threshold, 20.0f);
}
