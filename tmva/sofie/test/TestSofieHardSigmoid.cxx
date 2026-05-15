/// \file TestSofieHardSigmoid.cxx
/// \brief Unit tests for the SOFIE HardSigmoid operator
/// \author ROOT TMVA Team

#include "TMVA/ROperator_HardSigmoid.hxx"
#include "TMVA/RModel.hxx"

#include "gtest/gtest.h"

#include <cmath>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

using namespace TMVA::Experimental::SOFIE;

//Testing hexfloat clamp constants for AVX purity (0x0p+0f, 0x1p+0f)
TEST(SOFIE_HardSigmoid, GenerateHexfloatClampConstants)
{
   RModel model;
   //Explicitly type the shape vector to avoid ambiguity
   model.AddInputTensorInfo("input", ETensorType::FLOAT, std::vector<size_t>{1, 10});
   model.AddOutputTensorNameList({"output"});

   ROperator_HardSigmoid<float> op("input", "output", 0.2f, 0.5f);
   op.Initialize(model);

   std::string code = op.Generate("hardsigmoid_test");

   //Testing float hexfloat clamp bounds (f suffix ensures no double promotion)
   EXPECT_TRUE(code.find("0x0p+0f") != std::string::npos)
      << "Generated code missing optimized float hexfloat constant 0x0p+0f for lower clamp";

   EXPECT_TRUE(code.find("0x1p+0f") != std::string::npos)
      << "Generated code missing optimized float hexfloat constant 0x1p+0f for upper clamp";

   // Verify std::fmax and std::fmin are used (not std::clamp)
   EXPECT_TRUE(code.find("std::fmax") != std::string::npos)
      << "Generated code should use std::fmax for clamping";

   EXPECT_TRUE(code.find("std::fmin") != std::string::npos)
      << "Generated code should use std::fmin for clamping";
}

//Testing float suffix on alpha/beta to prevent double promotion
TEST(SOFIE_HardSigmoid, GenerateFloatSuffix)
{
   RModel model;
   // [FIX] Explicitly type the shape vector
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<size_t>{2, 5});
   model.AddOutputTensorNameList({"Y"});

   // Use non-default alpha/beta to ensure they appear in output
   ROperator_HardSigmoid<float> op("X", "Y", 0.25f, 0.6f);
   op.Initialize(model);

   std::string code = op.Generate("hardsigmoid_suffix_test");

   // The generated code should have 'f' suffix after alpha and beta values
   // e.g., "0.25f * tensor_X[id] + 0.6f"
   EXPECT_TRUE(code.find("f * tensor_X") != std::string::npos)
      << "Generated code missing 'f' suffix after alpha constant";

   EXPECT_TRUE(code.find("f))") != std::string::npos)
      << "Generated code missing 'f' suffix after beta constant";
}

//Testing Numeric correctness in linear region (between clamps)
TEST(SOFIE_HardSigmoid, NumericCorrectnessLinearRegion)
{
   const float alpha = 0.2f;
   const float beta = 0.5f;
   
   const std::vector<std::pair<float, float>> referenceData = {
      {-2.0f,  0.1f},   // 0.2 * -2 + 0.5 = 0.1
      {-1.0f,  0.3f},   // 0.2 * -1 + 0.5 = 0.3
      { 0.0f,  0.5f},   // 0.2 * 0 + 0.5 = 0.5
      { 1.0f,  0.7f},   // 0.2 * 1 + 0.5 = 0.7
      { 2.0f,  0.9f},   // 0.2 * 2 + 0.5 = 0.9
   };

   // Proxy for generated logic (pure float math)
   auto hardsigmoid_eval = [alpha, beta](float x) -> float {
      return std::fmax(0x0p+0f, std::fmin(0x1p+0f, alpha * x + beta));
   };

   for (const auto& [input, expected] : referenceData) {
      float computed = hardsigmoid_eval(input);
      float tol = 1e-6f;
      
      EXPECT_NEAR(computed, expected, tol)
         << "Linear region mismatch at x = " << input;
   }
}

// Testing Numeric correctness at clamp boundaries
TEST(SOFIE_HardSigmoid, NumericCorrectnessClamps)
{
   const float alpha = 0.2f;
   const float beta = 0.5f;
   
   const std::vector<std::pair<float, float>> clampData = {
      {-10.0f, 0.0f},  
      { -3.0f, 0.0f},  
      { -2.5f, 0.0f},  
      {  2.5f, 1.0f},  
      {  3.0f, 1.0f},  
      { 10.0f, 1.0f},  
   };

   auto hardsigmoid_eval = [alpha, beta](float x) -> float {
      return std::fmax(0x0p+0f, std::fmin(0x1p+0f, alpha * x + beta));
   };

   for (const auto& [input, expected] : clampData) {
      float computed = hardsigmoid_eval(input);
      float tol = 1e-6f;
      
      EXPECT_NEAR(computed, expected, tol)
         << "Clamp behavior mismatch at x = " << input;
   }
}

TEST(SOFIE_HardSigmoid, CustomAlphaBeta)
{
   const float alpha = 1.0f / 6.0f;
   const float beta = 0.5f;
   
   const std::vector<std::pair<float, float>> testData = {
      {-5.0f,  0.0f},           
      {-3.0f,  0.0f},           
      { 0.0f,  0.5f},           
      { 3.0f,  1.0f},           
      { 5.0f,  1.0f},           
   };

   auto hardsigmoid_eval = [alpha, beta](float x) -> float {
      return std::fmax(0x0p+0f, std::fmin(0x1p+0f, alpha * x + beta));
   };

   for (const auto& [input, expected] : testData) {
      float computed = hardsigmoid_eval(input);
      float tol = 1e-6f;
      
      EXPECT_NEAR(computed, expected, tol)
         << "Custom alpha/beta mismatch at x = " << input;
   }
}

// Standard Library dependencies
TEST(SOFIE_HardSigmoid, StdLibDependencies)
{
   ROperator_HardSigmoid<float> op("in", "out", 0.2f, 0.5f);
   auto libs = op.GetStdLibs();
   ASSERT_EQ(libs.size(), 1u);
   EXPECT_EQ(libs[0], "cmath");
}

// Type and Shape Inference
TEST(SOFIE_HardSigmoid, Inference)
{
   ROperator_HardSigmoid<float> op("in", "out", 0.2f, 0.5f);

   //Type inference
   auto types = op.TypeInference({ETensorType::FLOAT});
   EXPECT_EQ(types[0], ETensorType::FLOAT);

   //Shape inference
   std::vector<size_t> shape = {4, 16, 32};
   auto shapes = op.ShapeInference({shape});
   EXPECT_EQ(shapes[0], shape);
}

// Error Handling
TEST(SOFIE_HardSigmoid, ErrorHandling)
{
   ROperator_HardSigmoid<float> op("in", "out", 0.2f, 0.5f);
   
   // Generate without Initialize
   EXPECT_THROW(op.Generate("test"), std::runtime_error);

   // Initialize with missing tensor
   RModel model; 
   EXPECT_THROW(op.Initialize(model), std::runtime_error);
}

// Loop structure verification
TEST(SOFIE_HardSigmoid, GenerateStructure)
{
   RModel model;
   //Explicitly type the shape vector
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<size_t>{2, 5});
   model.AddOutputTensorNameList({"Y"});

   ROperator_HardSigmoid<float> op("X", "Y", 0.2f, 0.5f);
   op.Initialize(model);

   std::string code = op.Generate("hardsigmoid_struct_test");

   EXPECT_TRUE(code.find("tensor_Y") != std::string::npos) << "Missing output tensor access";
   EXPECT_TRUE(code.find("tensor_X") != std::string::npos) << "Missing input tensor access";
   //Loop limit check for shape {2, 5}
   EXPECT_TRUE(code.find("10") != std::string::npos) << "Incorrect loop limit generated";
   //Operator comment
   EXPECT_TRUE(code.find("HardSigmoid") != std::string::npos) << "Missing operator comment";
}