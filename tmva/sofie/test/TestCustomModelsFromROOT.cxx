constexpr auto modelHeaderSuffix = "_FromROOT.hxx";
constexpr auto modelDataSuffix = "_FromROOT.root";
#include "test_helpers.h"

#include "input_models/references/Linear_16.ref.hxx"
// #include "input_models/references/Linear_32.ref.hxx"
// #include "input_models/references/Linear_64.ref.hxx"
#include "input_models/references/LinearWithSelu.ref.hxx"
#include "input_models/references/LinearWithSigmoid.ref.hxx"
#include "input_models/references/ConvWithPadding.ref.hxx"
#include "input_models/references/ConvWithoutPadding.ref.hxx"
#include "input_models/references/ConvWithAutopadSameLower.ref.hxx"
#include "input_models/references/ConvWithStridesPadding.ref.hxx"
#include "input_models/references/ConvWithStridesNoPadding.ref.hxx"
#include "input_models/references/ConvWithAsymmetricPadding.ref.hxx"
#include "input_models/references/RNNBatchwise.ref.hxx"
#include "input_models/references/RNNBidirectional.ref.hxx"
#include "input_models/references/RNNBidirectionalBatchwise.ref.hxx"
#include "input_models/references/RNNDefaults.ref.hxx"
#include "input_models/references/RNNSeqLength.ref.hxx"
#include "input_models/references/RNNSequence.ref.hxx"
#include "input_models/references/RNNSequenceBatchwise.ref.hxx"
#include "input_models/references/LSTMBatchwise.ref.hxx"
#include "input_models/references/LSTMBidirectional.ref.hxx"
#include "input_models/references/LSTMDefaults.ref.hxx"
#include "input_models/references/LSTMInitialBias.ref.hxx"
#include "input_models/references/LSTMPeepholes.ref.hxx"
#include "input_models/references/GRUBatchwise.ref.hxx"
#include "input_models/references/GRUBidirectional.ref.hxx"
#include "input_models/references/GRUDefaults.ref.hxx"
#include "input_models/references/GRUInitialBias.ref.hxx"
#include "input_models/references/GRUSeqLength.ref.hxx"
#include "input_models/references/RangeFloat.ref.hxx"
#include "input_models/references/RangeInt.ref.hxx"

#include "gtest/gtest.h"

TEST(ROOT, Linear16)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(1600);
   std::fill_n(input.data(), input.size(), 1.0f);

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Linear_16", input);

   // Testing the actual and expected output sizes
   EXPECT_EQ(output.size(), sizeof(Linear_16_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = Linear_16_ExpectedOutput::all_ones;

   // Testing the actual and expected output values
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

/*
TEST(ROOT, Linear32)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(3200);
   std::fill_n(input.data(), input.size(), 1.0f);
   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Linear_32", input);

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
   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Linear_64", input);

   // Testing the actual and expected output values
   EXPECT_EQ(output.size(), sizeof(Linear_64_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = Linear_64_ExpectedOutput::all_ones;

   // Testing the actual and expected output values
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}
*/

TEST(ROOT, LinearWithSelu)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(48);
   std::fill_n(input.data(), input.size(), 1.0f);
   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "LinearWithSelu", input);

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(LinearWithSelu_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = LinearWithSelu_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ROOT, LinearWithSigmoid)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(48);
   std::fill_n(input.data(), input.size(), 1.0f);
   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "LinearWithSigmoid", input);


   // Checking output size
   EXPECT_EQ(output.size(), sizeof(LinearWithSigmoid_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = LinearWithSigmoid_ExpectedOutput::all_ones;

   // Checking every output value, one by one
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
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<float>, "ConvWithPadding", input);

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
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<float>, "ConvWithoutPadding", input);

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
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<float>, "ConvWithAutopadSameLower", input);

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
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<float>, "ConvWithStridesPadding", input);

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
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<float>, "ConvWithStridesNoPadding", input);

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvWithStridesNoPadding_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = ConvWithStridesNoPadding_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

// disabled
TEST(DISABLED_ROOT, ConvWithAsymmetricPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(35);
   std::iota(input.begin(), input.end(), 0.0f);
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<float>, "ConvWithAsymmetricPadding", input);

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvWithAsymmetricPadding_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = ConvWithAsymmetricPadding_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ROOT, RNNBatchwise)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6);
   std::iota(input.begin(), input.end(), 1.0f);
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "RNNBatchwise", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(RNNBatchwise_ExpectedOutput::all_ones) / sizeof(float));
   EXPECT_EQ(output_yh.size(), sizeof(RNNBatchwise_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = RNNBatchwise_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct[i]), TOLERANCE);
      EXPECT_LE(std::abs(output_yh[i] - correct[i]), TOLERANCE);
   }
}

TEST(ROOT, RNNBidirectional)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input({0.,    0.01, 0.02, 0.03, 0.04, 0.05,
                             0.06, 0.07, 0.08, 0.09, 0.1,  0.11,
                             0.12, 0.13, 0.14, 0.15, 0.16, 0.17});
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "RNNBidirectional", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(RNNBidirectional_ExpectedOutput::all_ones_y) / sizeof(float));

   float *correct_y = RNNBidirectional_ExpectedOutput::all_ones_y;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct_y[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(RNNBidirectional_ExpectedOutput::all_ones_yh) / sizeof(float));

   float *correct_yh = RNNBidirectional_ExpectedOutput::all_ones_yh;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct_yh[i]), TOLERANCE);
   }
}

TEST(ROOT, RNNBidirectionalBatchwise)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input({
      0,    0.01, 0.06, 0.07, 0.12, 0.13,
      0.02, 0.03, 0.08, 0.09, 0.14, 0.15,
      0.04, 0.05, 0.1,  0.11, 0.16, 0.17});
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "RNNBidirectionalBatchwise", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(RNNBidirectionalBatchwise_ExpectedOutput::all_ones_y) / sizeof(float));

   float *correct_y = RNNBidirectionalBatchwise_ExpectedOutput::all_ones_y;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct_y[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(RNNBidirectionalBatchwise_ExpectedOutput::all_ones_yh) / sizeof(float));

   float *correct_yh = RNNBidirectionalBatchwise_ExpectedOutput::all_ones_yh;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct_yh[i]), TOLERANCE);
   }
}

TEST(ROOT, RNNDefaults)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(9);
   std::iota(input.begin(), input.end(), 1.0f);
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "RNNDefaults", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(RNNDefaults_ExpectedOutput::all_ones_y) / sizeof(float));

   float *correct_y = RNNDefaults_ExpectedOutput::all_ones_y;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct_y[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(RNNDefaults_ExpectedOutput::all_ones_yh) / sizeof(float));

   float *correct_yh = RNNDefaults_ExpectedOutput::all_ones_yh;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct_yh[i]), TOLERANCE);
   }
}

TEST(ROOT, RNNSeqLength)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(18);
   std::iota(input.begin(), input.end(), 1.0f);
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "RNNSeqLength", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(RNNSeqLength_ExpectedOutput::all_ones_y) / sizeof(float));

   float *correct_y = RNNSeqLength_ExpectedOutput::all_ones_y;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct_y[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(RNNSeqLength_ExpectedOutput::all_ones_yh) / sizeof(float));

   float *correct_yh = RNNSeqLength_ExpectedOutput::all_ones_yh;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct_yh[i]), TOLERANCE);
   }
}

TEST(ROOT, RNNSequence)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input({
      0.01,   -0.01,   0.08,    0.09,  0.001,
      0.09,   -0.7,   -0.35,    0.0,   0.001,
      0.16,   -0.19,   0.003,   0.0,   0.0001,
      0.05,   -0.09,   0.013,   0.5,   0.005,
      0.2,    -0.05,   0.062,  -0.04, -0.04,
      0.0,     0.0,    0.0,     0.0,   0.0,
      0.06,    0.087,  0.01,    0.3,  -0.001,
      0.0,     0.0,    0.0,     0.0,   0.0,
      0.0,     0.0,    0.0,     0.0,   0.0});
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "RNNSequence", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(RNNSequence_ExpectedOutput::all_ones_y) / sizeof(float));

   float *correct_y = RNNSequence_ExpectedOutput::all_ones_y;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct_y[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(RNNSequence_ExpectedOutput::all_ones_yh) / sizeof(float));

   float *correct_yh = RNNSequence_ExpectedOutput::all_ones_yh;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct_yh[i]), TOLERANCE);
   }
}

TEST(ROOT, RNNSequenceBatchwise)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input({
      0.01,  -0.01,   0.08,   0.09,  0.001,
      0.05,  -0.09,   0.013,  0.5,   0.005,
      0.06,   0.087,  0.01,   0.3,  -0.001,
      0.09,   -0.7,  -0.35,   0.0,   0.001,
      0.2,    -0.05,  0.062, -0.04, -0.04,
      0.0,     0.0,   0.0,    0.0,   0.0,
      0.16,  -0.19,   0.003,  0.0,   0.0001,
      0.0,     0.0,   0.0,    0.0,   0.0,
      0.0,     0.0,   0.0,    0.0,   0.0});
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "RNNSequenceBatchwise", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(RNNSequenceBatchwise_ExpectedOutput::all_ones_y) / sizeof(float));

   float *correct_y = RNNSequenceBatchwise_ExpectedOutput::all_ones_y;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct_y[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(RNNSequenceBatchwise_ExpectedOutput::all_ones_yh) / sizeof(float));

   float *correct_yh = RNNSequenceBatchwise_ExpectedOutput::all_ones_yh;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct_yh[i]), TOLERANCE);
   }
}

TEST(ROOT, LSTMBatchwise)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6);
   std::iota(input.begin(), input.end(), 1.0f);
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "LSTMBatchwise", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(LSTMBatchwise_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = LSTMBatchwise_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(LSTMBatchwise_ExpectedOutput::all_ones) / sizeof(float));

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct[i]), TOLERANCE);
   }
}

TEST(ROOT, LSTMBidirectional)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6);
   std::iota(input.begin(), input.end(), 1.0f);
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "LSTMBidirectional", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];
   std::vector<float> output_yc = output[2];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(LSTMBidirectional_ExpectedOutput::all_ones_y) / sizeof(float));

   float *correct_y = LSTMBidirectional_ExpectedOutput::all_ones_y;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct_y[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(LSTMBidirectional_ExpectedOutput::all_ones_yh) / sizeof(float));

   float *correct_yh = LSTMBidirectional_ExpectedOutput::all_ones_yh;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct_yh[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yc.size(), sizeof(LSTMBidirectional_ExpectedOutput::all_ones_yc) / sizeof(float));

   float *correct_yc = LSTMBidirectional_ExpectedOutput::all_ones_yc;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yc[i] - correct_yc[i]), TOLERANCE);
   }
}

TEST(ROOT, LSTMDefaults)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6);
   std::iota(input.begin(), input.end(), 1.0f);
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "LSTMDefaults", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(LSTMDefaults_ExpectedOutput::all_ones_y) / sizeof(float));

   float *correct_y = LSTMDefaults_ExpectedOutput::all_ones_y;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct_y[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(LSTMDefaults_ExpectedOutput::all_ones_yh) / sizeof(float));

   float *correct_yh = LSTMDefaults_ExpectedOutput::all_ones_yh;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct_yh[i]), TOLERANCE);
   }
}

TEST(ROOT, LSTMInitialBias)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(9);
   std::iota(input.begin(), input.end(), 1.0f);
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "LSTMInitialBias", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(LSTMInitialBias_ExpectedOutput::all_ones_y) / sizeof(float));

   float *correct_y = LSTMInitialBias_ExpectedOutput::all_ones_y;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct_y[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(LSTMInitialBias_ExpectedOutput::all_ones_yh) / sizeof(float));

   float *correct_yh = LSTMInitialBias_ExpectedOutput::all_ones_yh;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct_yh[i]), TOLERANCE);
   }
}

TEST(ROOT, LSTMPeepholes)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(8);
   std::iota(input.begin(), input.end(), 1.0f);
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "LSTMPeepholes", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(LSTMPeepholes_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = LSTMPeepholes_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(LSTMPeepholes_ExpectedOutput::all_ones) / sizeof(float));

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct[i]), TOLERANCE);
   }
}

TEST(ROOT, GRUBatchwise)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6);
   std::iota(input.begin(), input.end(), 1.0f);
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "GRUBatchwise", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(GRUBatchwise_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = GRUBatchwise_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(GRUBatchwise_ExpectedOutput::all_ones) / sizeof(float));

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct[i]), TOLERANCE);
   }
}

TEST(ROOT, GRUBidirectional)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6);
   std::iota(input.begin(), input.end(), 1.0f);
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "GRUBidirectional", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(GRUBidirectional_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = GRUBidirectional_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(GRUBidirectional_ExpectedOutput::all_ones) / sizeof(float));

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct[i]), TOLERANCE);
   }
}

TEST(ROOT, GRUDefaults)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6);
   std::iota(input.begin(), input.end(), 1.0f);
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "GRUDefaults", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(GRUDefaults_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = GRUDefaults_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(GRUDefaults_ExpectedOutput::all_ones) / sizeof(float));

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct[i]), TOLERANCE);
   }
}

TEST(ROOT, GRUInitialBias)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(9);
   std::iota(input.begin(), input.end(), 1.0f);
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "GRUInitialBias", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(GRUInitialBias_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = GRUInitialBias_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(GRUInitialBias_ExpectedOutput::all_ones) / sizeof(float));

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct[i]), TOLERANCE);
   }
}

TEST(ROOT, GRUSeqLength)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(18);
   std::iota(input.begin(), input.end(), 1.0f);
   ASSERT_INCLUDE_AND_RUN_NO_SESSION(std::vector<std::vector<float>>, "GRUSeqLength", input);
   std::vector<float> output_y = output[0];
   std::vector<float> output_yh = output[1];

   // Checking output size
   EXPECT_EQ(output_y.size(), sizeof(GRUSeqLength_ExpectedOutput::all_ones_y) / sizeof(float));

   float *correct_y = GRUSeqLength_ExpectedOutput::all_ones_y;

   // Checking every output value, one by one
   for (size_t i = 0; i < output_y.size(); ++i) {
      EXPECT_LE(std::abs(output_y[i] - correct_y[i]), TOLERANCE);
   }

   // Checking output size
   EXPECT_EQ(output_yh.size(), sizeof(GRUSeqLength_ExpectedOutput::all_ones_yh) / sizeof(float));

   float *correct_yh = GRUSeqLength_ExpectedOutput::all_ones_yh;

   // Checking every output value, one by one
   for (size_t i = 0; i < output_yh.size(); ++i) {
      EXPECT_LE(std::abs(output_yh[i] - correct_yh[i]), TOLERANCE);
   }
}

TEST(ROOT, RangeFloat) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // inputs
   std::vector<float> start{1.};
   std::vector<float> limit{10.};
   std::vector<float> delta{2.};
   ASSERT_INCLUDE_AND_RUN_SESSION_ARGS(std::vector<float>, "RangeFloat", "\"\", 5", start, limit, delta);

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(RangeFloat_ExpectedOutput::outputs) / sizeof(float));

   float* correct = RangeFloat_ExpectedOutput::outputs;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ROOT, RangeInt) {
   // inputs
   std::vector<int64_t> start{1};
   std::vector<int64_t> limit{10};
   std::vector<int64_t> delta{2};
   ASSERT_INCLUDE_AND_RUN_SESSION_ARGS(std::vector<int64_t>, "RangeInt", "\"\", 5", start, limit, delta);

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(RangeInt_ExpectedOutput::outputs) / sizeof(int64_t));

   int64_t* correct = RangeInt_ExpectedOutput::outputs;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_EQ(output[i], correct[i]);
   }
}
