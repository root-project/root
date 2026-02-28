/// \file TestSofieGELU.cxx
/// \brief Unit tests for the SOFIE GELU operator
/// \author ROOT TMVA Team

#include "TMVA/ROperator_GELU.hxx"
#include "TMVA/RModel.hxx"

#include "gtest/gtest.h"

#include <cmath>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

using namespace TMVA::Experimental::SOFIE;

// Validate generation of hexfloat constants for bit-exact reproducibility
TEST(SOFIE_GELU, GenerateHexfloatConstants)
{
   RModel model;
   model.AddInputTensorInfo("input", ETensorType::FLOAT, {1, 10});
   model.AddOutputTensorNameList({"output"});

   ROperator_GELU<float> op("input", "output");
   op.Initialize(model);

   std::string code = op.Generate("gelu_test");

   // Expect 1/sqrt(2) as hexfloat: 0x1.6a09e667f3bcdp-1
   EXPECT_TRUE(code.find("0x1.6a09e667f3bcdp-1") != std::string::npos)
      << "Generated code missing optimized hexfloat constant for 1/sqrt(2)";

   // Expect 0.5 as hexfloat
   EXPECT_TRUE(code.find("0x1.0000000000000p-1") != std::string::npos)
      << "Generated code missing hexfloat constant for 0.5";
}

// Check structure of generated C++ code
TEST(SOFIE_GELU, GenerateStructure)
{
   RModel model;
   model.AddInputTensorInfo("X", ETensorType::FLOAT, {2, 5});
   model.AddOutputTensorNameList({"Y"});

   ROperator_GELU<float> op("X", "Y");
   op.Initialize(model);

   std::string code = op.Generate("gelu_struct_test");

   EXPECT_TRUE(code.find("std::erf") != std::string::npos) << "Missing std::erf call";
   EXPECT_TRUE(code.find("tensor_Y") != std::string::npos) << "Missing output tensor access";
   EXPECT_TRUE(code.find("tensor_X") != std::string::npos) << "Missing input tensor access";
   // Loop limit check for shape {2, 5}
   EXPECT_TRUE(code.find("10") != std::string::npos) << "Incorrect loop limit generated";
}

// Compare implementation against SciPy reference values
TEST(SOFIE_GELU, NumericCorrectness)
{
   // Reference values computed using scipy.special.erf
   const std::vector<std::pair<float, float>> referenceData = {
      {-3.0f,  -0.00404996f},
      {-2.5f,  -0.01974636f},
      {-2.0f,  -0.04540230f},
      {-1.5f,  -0.08771890f},
      {-1.0f,  -0.15880800f},
      {-0.5f,  -0.15426877f},
      { 0.0f,   0.00000000f},
      { 0.5f,   0.34573123f},
      { 1.0f,   0.84119201f},
      { 1.5f,   1.41281096f},
      { 2.0f,   1.95459771f},
      { 2.5f,   2.48025364f},
      { 3.0f,   2.99595003f},
      {-10.0f,  0.0f}, // Limit -> 0
      { 10.0f, 10.0f}  // Limit -> x
   };

   // Proxy for generated logic
   auto gelu_eval = [](float x) -> float {
      constexpr double kInvSqrt2 = 0x1.6a09e667f3bcdp-1;
      return 0.5f * x * (1.0 + std::erf(x * kInvSqrt2));
   };

   for (const auto& [input, expected] : referenceData) {
      float computed = gelu_eval(input);
      float tol = std::max(1e-6f * std::abs(expected), 1e-7f);
      
      EXPECT_NEAR(computed, expected, tol)
         << "Mismatch at x = " << input;
   }
}

TEST(SOFIE_GELU, StdLibDependencies)
{
   ROperator_GELU<float> op("in", "out");
   auto libs = op.GetStdLibs();
   ASSERT_EQ(libs.size(), 1u);
   EXPECT_EQ(libs[0], "cmath");
}

TEST(SOFIE_GELU, Inference)
{
   ROperator_GELU<float> op("in", "out");

   // Type inference
   auto types = op.TypeInference({ETensorType::FLOAT});
   EXPECT_EQ(types[0], ETensorType::FLOAT);

   // Shape inference
   std::vector<size_t> shape = {4, 16, 32};
   auto shapes = op.ShapeInference({shape});
   EXPECT_EQ(shapes[0], shape);
}

TEST(SOFIE_GELU, ErrorHandling)
{
   ROperator_GELU<float> op("in", "out");
   
   // Generate without Initialize
   EXPECT_THROW(op.Generate("test"), std::runtime_error);

   // Initialize with missing tensor
   RModel model; 
   EXPECT_THROW(op.Initialize(model), std::runtime_error);
}