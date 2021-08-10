#include "Linear_16_FromROOT.hxx"
#include "input_models/references/Linear_16.ref.hxx"

#include "Linear_32_FromROOT.hxx"
#include "input_models/references/Linear_32.ref.hxx"

#include "Linear_64_FromROOT.hxx"
#include "input_models/references/Linear_64.ref.hxx"

#include "gtest/gtest.h"

constexpr float DEFAULT_TOLERANCE = 1e-6f;

TEST(ROOT, Linear16)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(1600);
   std::fill_n(input.data(), input.size(), 1.0f);
   std::vector<float> output = TMVA_SOFIE_Linear_16::infer(input.data());

   // Testing the actual and expected output sizes
   EXPECT_EQ(output.size(), sizeof(Linear_16_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = Linear_16_ExpectedOutput::all_ones;

   // Testing the actual and expected output values
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ROOT, Linear32)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   
   // Preparing the standard all-ones input
   std::vector<float> input(3200);
   std::fill_n(input.data(), input.size(), 1.0f);
   std::vector<float> output = TMVA_SOFIE_Linear_32::infer(input.data());

   // Testing the actual and expected output sizes
   EXPECT_EQ(output.size(), sizeof(Linear_32_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = Linear_32_ExpectedOutput::all_ones;

   // Testing the actual and expected output values
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ROOT, Linear64)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   
   // Preparing the standard all-ones input
   std::vector<float> input(6400);
   std::fill_n(input.data(), input.size(), 1.0f);
   std::vector<float> output = TMVA_SOFIE_Linear_64::infer(input.data());

   // Testing the actual and expected output values
   EXPECT_EQ(output.size(), sizeof(Linear_64_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = Linear_64_ExpectedOutput::all_ones;

   // Testing the actual and expected output values
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}
