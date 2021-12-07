#include <numeric>

#include "Linear_16_FromROOT.hxx"
#include "input_models/references/Linear_16.ref.hxx"

#include "Linear_32_FromROOT.hxx"
#include "input_models/references/Linear_32.ref.hxx"

#include "Linear_64_FromROOT.hxx"
#include "input_models/references/Linear_64.ref.hxx"

#include "LinearWithSelu_FromROOT.hxx"
#include "input_models/references/LinearWithSelu.ref.hxx"

#include "LinearWithSigmoid_FromROOT.hxx"
#include "input_models/references/LinearWithSigmoid.ref.hxx"

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

#include "RNNBatchwise_FromROOT.hxx"
#include "input_models/references/RNNBatchwise.ref.hxx"

#include "RNNBidirectional_FromROOT.hxx"
#include "input_models/references/RNNBidirectional.ref.hxx"

#include "RNNBidirectionalBatchwise_FromROOT.hxx"
#include "input_models/references/RNNBidirectionalBatchwise.ref.hxx"

#include "RNNDefaults_FromROOT.hxx"
#include "input_models/references/RNNDefaults.ref.hxx"

#include "RNNSeqLength_FromROOT.hxx"
#include "input_models/references/RNNSeqLength.ref.hxx"

#include "RNNSequence_FromROOT.hxx"
#include "input_models/references/RNNSequence.ref.hxx"

#include "RNNSequenceBatchwise_FromROOT.hxx"
#include "input_models/references/RNNSequenceBatchwise.ref.hxx"

#include "LSTMBatchwise_FromROOT.hxx"
#include "input_models/references/LSTMBatchwise.ref.hxx"

#include "LSTMBidirectional_FromROOT.hxx"
#include "input_models/references/LSTMBidirectional.ref.hxx"

#include "LSTMDefaults_FromROOT.hxx"
#include "input_models/references/LSTMDefaults.ref.hxx"

#include "LSTMInitialBias_FromROOT.hxx"
#include "input_models/references/LSTMInitialBias.ref.hxx"

#include "LSTMPeepholes_FromROOT.hxx"
#include "input_models/references/LSTMPeepholes.ref.hxx"

#include "GRUBatchwise_FromROOT.hxx"
#include "input_models/references/GRUBatchwise.ref.hxx"

#include "GRUBidirectional_FromROOT.hxx"
#include "input_models/references/GRUBidirectional.ref.hxx"

#include "GRUDefaults_FromROOT.hxx"
#include "input_models/references/GRUDefaults.ref.hxx"

#include "GRUInitialBias_FromROOT.hxx"
#include "input_models/references/GRUInitialBias.ref.hxx"

#include "GRUSeqLength_FromROOT.hxx"
#include "input_models/references/GRUSeqLength.ref.hxx"

#include "gtest/gtest.h"

constexpr float DEFAULT_TOLERANCE = 1e-3f;

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


TEST(ROOT, LinearWithSelu)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(48);
   std::fill_n(input.data(), input.size(), 1.0f);
   std::vector<float> output = TMVA_SOFIE_LinearWithSelu::infer(input.data());

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
   std::vector<float> output = TMVA_SOFIE_LinearWithSigmoid::infer(input.data());

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

// disabled
TEST(DISABLED_ROOT, ConvWithAsymmetricPadding)
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

TEST(ROOT, RNNBatchwise)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6);
   std::iota(input.begin(), input.end(), 1.0f);
   std::vector<std::vector<float>> output = TMVA_SOFIE_RNNBatchwise::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_RNNBidirectional::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_RNNBidirectionalBatchwise::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_RNNDefaults::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_RNNSeqLength::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_RNNSequence::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_RNNSequenceBatchwise::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_LSTMBatchwise::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_LSTMBidirectional::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_LSTMDefaults::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_LSTMInitialBias::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_LSTMPeepholes::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_GRUBatchwise::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_GRUBidirectional::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_GRUDefaults::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_GRUInitialBias::infer(input.data());
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
   std::vector<std::vector<float>> output = TMVA_SOFIE_GRUSeqLength::infer(input.data());
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
