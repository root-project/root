#include <numeric>

#include "Linear_16_FromROOT.hxx"
#include "input_models/references/Linear_16.ref.hxx"

#include "Linear_32_FromROOT.hxx"
#include "input_models/references/Linear_32.ref.hxx"

#include "Linear_64_FromROOT.hxx"
#include "input_models/references/Linear_64.ref.hxx"

#include "ConvWithPadding_FromROOT.hxx"
#include "input_models/references/ConvWithPadding.ref.hxx"

#include "ConvWithoutPadding_FromROOT.hxx"
#include "input_models/references/ConvWithoutPadding.ref.hxx"

#include "ConvWithAutopadSameLower_FromROOT.hxx"
#include "input_models/references/ConvWithAutopadSameLower.ref.hxx"

#include "ConvWithStridesPadding_FromROOT.hxx"
#include "input_models/references/ConvWithStridesPadding.ref.hxx"

#include "ConvWithStridesNoPadding_FromROOT.hxx"
#include "input_models/references/ConvWithStridesNoPadding.ref.hxx"

#include "ConvWithAsymmetricPadding_FromROOT.hxx"
#include "input_models/references/ConvWithAsymmetricPadding.ref.hxx"

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


TEST(ROOT, ConvWithPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(25);
   std::iota(input.begin(), input.end(), 0.0f);
   std::vector<float> output = TMVA_SOFIE_ConvWithPadding::infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvWithPadding_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = ConvWithPadding_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ROOT, ConvWithoutPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(25);
   std::iota(input.begin(), input.end(), 0.0f);
   std::vector<float> output = TMVA_SOFIE_ConvWithoutPadding::infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvWithoutPadding_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = ConvWithoutPadding_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ROOT, ConvWithAutopadSameLower)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(25);
   std::iota(input.begin(), input.end(), 0.0f);
   std::vector<float> output = TMVA_SOFIE_ConvWithAutopadSameLower::infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvWithAutopadSameLower_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = ConvWithAutopadSameLower_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ROOT, ConvWithStridesPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(35);
   std::iota(input.begin(), input.end(), 0.0f);
   std::vector<float> output = TMVA_SOFIE_ConvWithStridesPadding::infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvWithStridesPadding_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = ConvWithStridesPadding_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ROOT, ConvWithStridesNoPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(35);
   std::iota(input.begin(), input.end(), 0.0f);
   std::vector<float> output = TMVA_SOFIE_ConvWithStridesNoPadding::infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvWithStridesNoPadding_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = ConvWithStridesNoPadding_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ROOT, ConvWithAsymmetricPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(35);
   std::iota(input.begin(), input.end(), 0.0f);
   std::vector<float> output = TMVA_SOFIE_ConvWithAsymmetricPadding::infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvWithAsymmetricPadding_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = ConvWithAsymmetricPadding_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}
