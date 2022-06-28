#include <numeric>

#include "Linear_16_FromONNX.hxx"
#include "input_models/references/Linear_16.ref.hxx"

#include "Linear_32_FromONNX.hxx"
#include "input_models/references/Linear_32.ref.hxx"

#include "Linear_64_FromONNX.hxx"
#include "input_models/references/Linear_64.ref.hxx"

#include "LinearWithSelu_FromONNX.hxx"
#include "input_models/references/LinearWithSelu.ref.hxx"

#include "LinearWithSigmoid_FromONNX.hxx"
#include "input_models/references/LinearWithSigmoid.ref.hxx"

#include "ConvWithPadding_FromONNX.hxx"
#include "input_models/references/ConvWithPadding.ref.hxx"

#include "ConvWithoutPadding_FromONNX.hxx"
#include "input_models/references/ConvWithoutPadding.ref.hxx"

#include "ConvWithAutopadSameLower_FromONNX.hxx"
#include "input_models/references/ConvWithAutopadSameLower.ref.hxx"

#include "ConvWithStridesPadding_FromONNX.hxx"
#include "input_models/references/ConvWithStridesPadding.ref.hxx"

#include "ConvWithStridesNoPadding_FromONNX.hxx"
#include "input_models/references/ConvWithStridesNoPadding.ref.hxx"

#include "ConvWithAsymmetricPadding_FromONNX.hxx"
#include "input_models/references/ConvWithAsymmetricPadding.ref.hxx"

#include "MaxPool1d_FromONNX.hxx"
#include "input_models/references/MaxPool1d.ref.hxx"

#include "MaxPool2d_FromONNX.hxx"
#include "input_models/references/MaxPool2d.ref.hxx"

#include "MaxPool3d_FromONNX.hxx"
#include "input_models/references/MaxPool3d.ref.hxx"

#include "AvgPool_FromONNX.hxx"
#include "input_models/references/AvgPool.ref.hxx"

#include "RNNBatchwise_FromONNX.hxx"
#include "input_models/references/RNNBatchwise.ref.hxx"

#include "RNNBidirectional_FromONNX.hxx"
#include "input_models/references/RNNBidirectional.ref.hxx"

#include "RNNBidirectionalBatchwise_FromONNX.hxx"
#include "input_models/references/RNNBidirectionalBatchwise.ref.hxx"

#include "RNNDefaults_FromONNX.hxx"
#include "input_models/references/RNNDefaults.ref.hxx"

#include "RNNSeqLength_FromONNX.hxx"
#include "input_models/references/RNNSeqLength.ref.hxx"

#include "RNNSequence_FromONNX.hxx"
#include "input_models/references/RNNSequence.ref.hxx"

#include "RNNSequenceBatchwise_FromONNX.hxx"
#include "input_models/references/RNNSequenceBatchwise.ref.hxx"

#include "LSTMBatchwise_FromONNX.hxx"
#include "input_models/references/LSTMBatchwise.ref.hxx"

#include "LSTMBidirectional_FromONNX.hxx"
#include "input_models/references/LSTMBidirectional.ref.hxx"

#include "LSTMDefaults_FromONNX.hxx"
#include "input_models/references/LSTMDefaults.ref.hxx"

#include "LSTMInitialBias_FromONNX.hxx"
#include "input_models/references/LSTMInitialBias.ref.hxx"

#include "LSTMPeepholes_FromONNX.hxx"
#include "input_models/references/LSTMPeepholes.ref.hxx"

#include "GRUBatchwise_FromONNX.hxx"
#include "input_models/references/GRUBatchwise.ref.hxx"

#include "GRUBidirectional_FromONNX.hxx"
#include "input_models/references/GRUBidirectional.ref.hxx"

#include "GRUDefaults_FromONNX.hxx"
#include "input_models/references/GRUDefaults.ref.hxx"

#include "GRUInitialBias_FromONNX.hxx"
#include "input_models/references/GRUInitialBias.ref.hxx"

#include "GRUSeqLength_FromONNX.hxx"
#include "input_models/references/GRUSeqLength.ref.hxx"

#include "gtest/gtest.h"

constexpr float DEFAULT_TOLERANCE = 1e-3f;

TEST(ONNX, Linear16)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(1600);
   std::fill_n(input.data(), input.size(), 1.0f);
   TMVA_SOFIE_Linear_16::Session s("Linear_16_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Linear_16_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = Linear_16_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ONNX, Linear32)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(3200);
   std::fill_n(input.data(), input.size(), 1.0f);
   TMVA_SOFIE_Linear_32::Session s("Linear_32_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Linear_32_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = Linear_32_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ONNX, Linear64)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6400);
   std::fill_n(input.data(), input.size(), 1.0f);
   TMVA_SOFIE_Linear_64::Session s("Linear_64_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Linear_64_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = Linear_64_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ONNX, LinearWithSelu)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(48);
   std::fill_n(input.data(), input.size(), 1.0f);
   TMVA_SOFIE_LinearWithSelu::Session s("LinearWithSelu_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(LinearWithSelu_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = LinearWithSelu_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ONNX, LinearWithSigmoid)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(48);
   std::fill_n(input.data(), input.size(), 1.0f);
   TMVA_SOFIE_LinearWithSigmoid::Session s("LinearWithSigmoid_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(LinearWithSigmoid_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = LinearWithSigmoid_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ONNX, ConvWithPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(25);
   std::iota(input.begin(), input.end(), 0.0f);
   TMVA_SOFIE_ConvWithPadding::Session s("ConvWithPadding_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvWithPadding_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = ConvWithPadding_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ONNX, ConvWithoutPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(25);
   std::iota(input.begin(), input.end(), 0.0f);
   TMVA_SOFIE_ConvWithoutPadding::Session s("ConvWithoutPadding_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvWithoutPadding_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = ConvWithoutPadding_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ONNX, ConvWithAutopadSameLower)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(25);
   std::iota(input.begin(), input.end(), 0.0f);
   TMVA_SOFIE_ConvWithAutopadSameLower::Session s("ConvWithAutopadSameLower_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvWithAutopadSameLower_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = ConvWithAutopadSameLower_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ONNX, ConvWithStridesPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(35);
   std::iota(input.begin(), input.end(), 0.0f);
   TMVA_SOFIE_ConvWithStridesPadding::Session s("ConvWithStridesPadding_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvWithStridesPadding_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = ConvWithStridesPadding_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


TEST(ONNX, ConvWithStridesNoPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(35);
   std::iota(input.begin(), input.end(), 0.0f);
   TMVA_SOFIE_ConvWithStridesNoPadding::Session s("ConvWithStridesNoPadding_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvWithStridesNoPadding_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = ConvWithStridesNoPadding_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}


// Disables test (asymmetric padding not supported)
TEST(DISABLED_ONNX, ConvWithAsymmetricPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(35);
   std::iota(input.begin(), input.end(), 0.0f);
   TMVA_SOFIE_ConvWithAsymmetricPadding::Session s("ConvWithAsymmetricPadding_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvWithAsymmetricPadding_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = ConvWithAsymmetricPadding_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, MaxPool1d){
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard  input
   std::vector<float> input({0.0907,  0.1029,  0.8143,  1.4497, -0.7785,  0.3825, -0.3764,
           1.5785, -0.0835,  0.1622,
          1.5867,  0.9823, -0.8821,  0.4439, -0.1378, -0.2273, -0.0198,
          -2.0230,  0.0905,  0.6674,
         -1.4290, -1.3100, -0.9439, -0.0833, -0.1919,  0.6886,  0.9389,
          -1.2914, -1.3584, -2.0341,
         -0.3269,  0.1704,  1.1776,  1.3972, -1.8874, -1.5334,  1.1541,
           0.3011,  0.6569, -2.3504,
          0.4033,  0.1142,  2.2846, -1.3948, -0.8573,  0.5756, -1.0864,
           0.2283,  0.8947,  1.7627,
         -0.1657,  0.0649, -1.6066,  0.4162, -1.1525, -0.8184,  1.1324,
          -1.1086,  0.1061,  1.0071});
   
   TMVA_SOFIE_MaxPool1d::Session s("MaxPool1d_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());
   // Checking output size
   EXPECT_EQ(output.size(), sizeof(MaxPool1d_ExpectedOutput::output) / sizeof(float));
   
   float *correct = MaxPool1d_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }

}

TEST(ONNX, MaxPool2d){
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard  input
   std::vector<float> input({
      0.6266,  0.1656,  0.2753, -0.4558, -1.4592,  0.9285, -1.3410,
            1.3223, -0.5936, -1.3648,
          -0.2989,  0.5901, -0.8845, -0.0433,  0.8314, -1.7159, -0.5765,
            0.8678,  1.0257,  0.7847,
          -0.3421, -1.2364, -0.5805,  0.4421,  1.2184,  0.5043,  1.6823,
           -1.0483, -2.2798, -1.8927,
           0.7716,  0.0405,  0.3121, -0.3011, -0.3266, -1.9660,  1.0837,
            0.2317,  0.9084, -0.3285,
          -0.9398, -0.2065, -0.9499, -0.9739, -0.1288, -0.1375, -1.2612,
            0.8810,  0.8506,  0.4455
   });
   
   TMVA_SOFIE_MaxPool2d::Session s("MaxPool2d_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());
   // Checking output size
   EXPECT_EQ(output.size(), sizeof(MaxPool2d_ExpectedOutput::output) / sizeof(float));
   
   float *correct = MaxPool2d_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }

}

TEST(ONNX, MaxPool3d){
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard  input
   std::vector<float> input({
      -2.6496,  1.0476, -0.5153,
            0.3771,  0.4129, -0.3077,
           -0.8717, -0.8040, -0.3525,

          -0.1765, -0.3364,  0.8737,
           -0.2381, -0.8297,  0.4666,
            0.6984, -0.6760,  0.6298,

           1.3833,  0.1101,  0.2039,
           -0.5477,  0.2341,  0.9181,
            0.3842,  0.2428,  1.7924
   });
   
   TMVA_SOFIE_MaxPool3d::Session s("MaxPool3d_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());
   // Checking output size
   EXPECT_EQ(output.size(), sizeof(MaxPool3d_ExpectedOutput::output) / sizeof(float));
   
   float *correct = MaxPool3d_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }

}

TEST(ONNX, AvgPool){
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard  input
   std::vector<float> input({
      0.4764, -0.1976,  1.6506, -0.2421,  0.6412,  1.9985,  0.3938,
            0.1347,  0.2204, -0.7503,
           0.2139,  0.7285, -0.0210, -0.4585, -1.5333, -0.4772,  0.5560,
            0.6323, -2.5372,  1.4906,
          -1.1062, -0.9703,  0.2366, -0.9184,  0.3014,  0.7985, -0.6841,
           -2.2854, -2.7728, -1.2806,
          -1.0947, -0.5990, -0.3033, -1.9042, -0.5403,  0.2332,  0.9215,
           -0.1549,  0.0557, -0.5567,
          -1.4971,  0.5386, -0.2922,  0.4860, -0.3973, -0.4624,  0.4514,
            0.2385,  0.3783, -1.0500
   });
   
   TMVA_SOFIE_AvgPool::Session s("AvgPool_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());
   // Checking output size
   EXPECT_EQ(output.size(), sizeof(AvgPool_ExpectedOutput::output) / sizeof(float));
   
   float *correct = AvgPool_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }

}

TEST(ONNX, RNNBatchwise)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6);
   std::iota(input.begin(), input.end(), 1.0f);
   TMVA_SOFIE_RNNBatchwise::Session s("RNNBatchwise_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

TEST(ONNX, RNNBidirectional)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input({0.,    0.01, 0.02, 0.03, 0.04, 0.05,
                             0.06, 0.07, 0.08, 0.09, 0.1,  0.11,
                             0.12, 0.13, 0.14, 0.15, 0.16, 0.17});
   TMVA_SOFIE_RNNBidirectional::Session s("RNNBidirectional_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

TEST(ONNX, RNNBidirectionalBatchwise)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input({
      0,    0.01, 0.06, 0.07, 0.12, 0.13,
      0.02, 0.03, 0.08, 0.09, 0.14, 0.15,
      0.04, 0.05, 0.1,  0.11, 0.16, 0.17});
   TMVA_SOFIE_RNNBidirectionalBatchwise::Session s("RNNBidirectionalBatchwise_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

TEST(ONNX, RNNDefaults)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(9);
   std::iota(input.begin(), input.end(), 1.0f);
   TMVA_SOFIE_RNNDefaults::Session s("RNNDefaults_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

TEST(ONNX, RNNSeqLength)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(18);
   std::iota(input.begin(), input.end(), 1.0f);
   TMVA_SOFIE_RNNSeqLength::Session s("RNNSeqLength_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

TEST(ONNX, RNNSequence)
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
   TMVA_SOFIE_RNNSequence::Session s("RNNSequence_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

TEST(ONNX, RNNSequenceBatchwise)
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
   TMVA_SOFIE_RNNSequenceBatchwise::Session s("RNNSequenceBatchwise_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

TEST(ONNX, LSTMBatchwise)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6);
   std::iota(input.begin(), input.end(), 1.0f);
   TMVA_SOFIE_LSTMBatchwise::Session s("LSTMBatchwise_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

TEST(ONNX, LSTMBidirectional)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6);
   std::iota(input.begin(), input.end(), 1.0f);
   TMVA_SOFIE_LSTMBidirectional::Session s("LSTMBidirectional_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

TEST(ONNX, LSTMDefaults)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6);
   std::iota(input.begin(), input.end(), 1.0f);
   TMVA_SOFIE_LSTMDefaults::Session s("LSTMDefaults_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

TEST(ONNX, LSTMInitialBias)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(9);
   std::iota(input.begin(), input.end(), 1.0f);
   TMVA_SOFIE_LSTMInitialBias::Session s("LSTMInitialBias_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

TEST(ONNX, LSTMPeepholes)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(8);
   std::iota(input.begin(), input.end(), 1.0f);
   TMVA_SOFIE_LSTMPeepholes::Session s("LSTMPeepholes_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

// GRU tests
TEST(ONNX, GRUBatchwise)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6);
   std::iota(input.begin(), input.end(), 1.0f);
   TMVA_SOFIE_GRUBatchwise::Session s("GRUBatchwise_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

TEST(ONNX, GRUBidirectional)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6);
   std::iota(input.begin(), input.end(), 1.0f);
   TMVA_SOFIE_GRUBidirectional::Session s("GRUBidirectional_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

TEST(ONNX, GRUDefaults)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(6);
   std::iota(input.begin(), input.end(), 1.0f);
   TMVA_SOFIE_GRUDefaults::Session s("GRUDefaults_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

TEST(ONNX, GRUInitialBias)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(9);
   std::iota(input.begin(), input.end(), 1.0f);
   TMVA_SOFIE_GRUInitialBias::Session s("GRUInitialBias_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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

TEST(ONNX, GRUSeqLength)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(18);
   std::iota(input.begin(), input.end(), 1.0f);
   TMVA_SOFIE_GRUSeqLength::Session s("GRUSeqLength_FromONNX.dat");
   std::vector<std::vector<float>> output = s.infer(input.data());
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
