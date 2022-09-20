#include <numeric>

#include "Linear_16_FromONNX.hxx"
#include "input_models/references/Linear_16.ref.hxx"

#include "Linear_32_FromONNX.hxx"
#include "input_models/references/Linear_32.ref.hxx"

#include "Linear_64_FromONNX.hxx"
#include "input_models/references/Linear_64.ref.hxx"

#include "LinearWithSelu_FromONNX.hxx"
#include "input_models/references/LinearWithSelu.ref.hxx"

#include "Sub_FromONNX.hxx"
#include "input_models/references/Sub.ref.hxx"

#include "Add_FromONNX.hxx"
#include "input_models/references/Add.ref.hxx"

#include "Add_broadcast_FromONNX.hxx"
#include "input_models/references/Add_broadcast.ref.hxx"

#include "Mul_FromONNX.hxx"
#include "input_models/references/Mul.ref.hxx"

#include "Div_FromONNX.hxx"
#include "input_models/references/Div.ref.hxx"

#include "Neg_FromONNX.hxx"
#include "input_models/references/Neg.ref.hxx"

#include "Cast_FromONNX.hxx"
#include "input_models/references/Cast.ref.hxx"

#include "ReduceMean_FromONNX.hxx"
#include "input_models/references/ReduceMean.ref.hxx"

#include "ReduceProd_FromONNX.hxx"
#include "input_models/references/ReduceProd.ref.hxx"

#include "Shape_FromONNX.hxx"
#include "input_models/references/Shape.ref.hxx"

#include "LinearWithLeakyRelu_FromONNX.hxx"
#include "input_models/references/LinearWithLeakyRelu.ref.hxx"

#include "Tanh_FromONNX.hxx"
#include "input_models/references/Tanh.ref.hxx"

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

#include "Max_FromONNX.hxx"
#include "input_models/references/Max.ref.hxx"

#include "AvgPool_FromONNX.hxx"
#include "input_models/references/AvgPool.ref.hxx"

#include "Pow_FromONNX.hxx"
#include "input_models/references/Pow.ref.hxx"

#include "Pow_broadcast_FromONNX.hxx"
#include "input_models/references/Pow_broadcast.ref.hxx"

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

#include "Softmax1d_FromONNX.hxx"
#include "input_models/references/Softmax1d.ref.hxx"

#include "Softmax2d_FromONNX.hxx"
#include "input_models/references/Softmax2d.ref.hxx"

#include "Softmax3d_FromONNX.hxx"
#include "input_models/references/Softmax3d.ref.hxx"

#include "Softmax4d_FromONNX.hxx"
#include "input_models/references/Softmax4d.ref.hxx"

#include "ConvTranspose1d_FromONNX.hxx"
#include "input_models/references/ConvTranspose1d.ref.hxx"

#include "ConvTranspose2d_FromONNX.hxx"
#include "input_models/references/ConvTranspose2d.ref.hxx"

#include "ConvTranspose3d_FromONNX.hxx"
#include "input_models/references/ConvTranspose3d.ref.hxx"

#include "ConvTransposeBias2d_FromONNX.hxx"
#include "input_models/references/ConvTransposeBias2d.ref.hxx"

#include "ConvTransposeBias2dBatched_FromONNX.hxx"
#include "input_models/references/ConvTransposeBias2dBatched.ref.hxx"

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

TEST(ONNX, Sub)
   {
      constexpr float TOLERANCE = DEFAULT_TOLERANCE;

      // Preparing the standard input
      std::vector<float> input1({
         1, 2
      });
      std::vector<float> input2({
         0, 1
      });
      TMVA_SOFIE_Sub::Session s("Sub_FromONNX.dat");

      std::vector<float> output = s.infer(input2.data(),input1.data());

      // Checking output size
      EXPECT_EQ(output.size(), sizeof(Sub_ExpectedOutput::outputs) / sizeof(float));

      float *correct = Sub_ExpectedOutput::outputs;

      // Checking every output value, one by one
      for (size_t i = 0; i < output.size(); ++i) {
         EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
      }
   }

TEST(ONNX, Add_broadcast)
   {
      constexpr float TOLERANCE = DEFAULT_TOLERANCE;

      // Preparing the standard input
      std::vector<float> input1({
         1, 2, 3,
         3, 4, 5
      });
      std::vector<float> input2({
         5, 6, 7,
         8, 9, 10
      });
      TMVA_SOFIE_Add_broadcast::Session s("Add_broadcast_FromONNX.dat");

      std::vector<float> output = s.infer(input2.data(),input1.data());

      // Checking output size
      EXPECT_EQ(output.size(), sizeof(Add_broadcast_ExpectedOutput::outputs) / sizeof(float));

      float *correct = Add_broadcast_ExpectedOutput::outputs;

      // Checking every output value, one by one
      for (size_t i = 0; i < output.size(); ++i) {
         EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
      }
   }

TEST(ONNX, Add)
   {
      constexpr float TOLERANCE = DEFAULT_TOLERANCE;

      // Preparing the standard input
      std::vector<float> input1({
         1, 2
      });
      std::vector<float> input2({
         0, 1
      });
      TMVA_SOFIE_Add::Session s("Add_FromONNX.dat");

      std::vector<float> output = s.infer(input1.data(),input2.data());

      // Checking output size
      EXPECT_EQ(output.size(), sizeof(Add_ExpectedOutput::outputs) / sizeof(float));

      float *correct = Add_ExpectedOutput::outputs;

      // Checking every output value, one by one
      for (size_t i = 0; i < output.size(); ++i) {
         EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
      }
   }

TEST(ONNX, Mul)
   {
      constexpr float TOLERANCE = DEFAULT_TOLERANCE;

      // Preparing the standard input
      std::vector<float> input1({
         1, 2
      });
      std::vector<float> input2({
         0, 1
      });
      TMVA_SOFIE_Mul::Session s("Mul_FromONNX.dat");

      std::vector<float> output = s.infer(input1.data(),input2.data());

      // Checking output size
      EXPECT_EQ(output.size(), sizeof(Mul_ExpectedOutput::outputs) / sizeof(float));

      float *correct = Mul_ExpectedOutput::outputs;

      // Checking every output value, one by one
      for (size_t i = 0; i < output.size(); ++i) {
         EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
      }
   }

TEST(ONNX, Div)
   {
      constexpr float TOLERANCE = DEFAULT_TOLERANCE;

      // Preparing the standard input
      std::vector<float> input1({
         4, 2
      });
      std::vector<float> input2({
         2, 2
      });
      TMVA_SOFIE_Div::Session s("Div_FromONNX.dat");

      std::vector<float> output = s.infer(input2.data(),input1.data());

      // Checking output size
      EXPECT_EQ(output.size(), sizeof(Div_ExpectedOutput::outputs) / sizeof(float));

      float *correct = Div_ExpectedOutput::outputs;

      // Checking every output value, one by one
      for (size_t i = 0; i < output.size(); ++i) {
         EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
      }
   }

TEST(ONNX, Neg)
   {
      constexpr float TOLERANCE = DEFAULT_TOLERANCE;

      // Preparing the standard input
      std::vector<float> input({
        -1.9100,  1.8811, -1.7269, -0.1094, -0.0145,  0.2509,  0.5893, -2.2733,
        -0.7077,  1.0645, -0.8607,  0.2085
      });

      TMVA_SOFIE_Neg::Session s("Neg_FromONNX.dat");
      std::vector<float> output = s.infer(input.data());

      // Checking output size
      EXPECT_EQ(output.size(), sizeof(Neg_ExpectedOutput::outputs) / sizeof(float));

      float *correct = Neg_ExpectedOutput::outputs;

      // Checking every output value, one by one
      for (size_t i = 0; i < output.size(); ++i) {
         EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
      }
   }

TEST(ONNX, Cast)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard  input
   std::vector<int64_t> input({
      1,2,3,4,5,6
   });

   TMVA_SOFIE_Cast::Session s("Cast_FromONNX.dat");

   auto output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Cast_ExpectedOutput::outputs) / sizeof(float));

   float *correct = Cast_ExpectedOutput::outputs;

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

TEST(ONNX, Tanh)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the random input
   std::vector<float> input({
     -0.3896, -0.3521,  0.0363,  1.0962,  0.5085, -0.8523, -0.6766,  0.2421,
      1.5971,  1.3873, -0.2112, -0.6895, -0.5069, -2.1395, -0.7087,  1.1658,
      1.3493,  0.8132,  1.7156, -0.8637, -0.1971,  0.0411, -0.5662, -0.2516
   });

   TMVA_SOFIE_Tanh::Session s("Tanh_FromONNX.dat");

   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Tanh_ExpectedOutput::outputs) / sizeof(float));

   float *correct = Tanh_ExpectedOutput::outputs;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, LinearWithLeakyRelu)
{
   constexpr float TOLERANCE = 1;

   // Preparing the standard all-ones input
   std::vector<float> input({
      0.4369, -0.6882,  1.0309, -1.0263, -0.1519,  1.2237, -0.7054, -0.1762,
      -0.6811, -2.2597,  1.0388, -0.7993,  0.1468,  1.3257, -0.4714, -0.0958,
      0.7057, -0.3749, -0.3310,  0.0986, -0.1370,  0.0832, -1.6465, -0.2793
   });

   TMVA_SOFIE_LinearWithLeakyRelu::Session s("LinearWithLeakyRelu_FromONNX.dat");

   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(LinearWithLeakyRelu_ExpectedOutput::outputs) / sizeof(float));

   float *correct = LinearWithLeakyRelu_ExpectedOutput::outputs;

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

TEST(ONNX, Pow){
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard  input
   std::vector<float> input1({
      1, 2, 3
   });
   std::vector<float> input2({
      4, 5, 6
   });
   
   TMVA_SOFIE_Pow::Session s("Pow_FromONNX.dat");
   std::vector<float> output = s.infer(input2.data(),input1.data());
   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Pow_ExpectedOutput::outputs) / sizeof(float));
   
   float *correct = Pow_ExpectedOutput::outputs;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }

}

TEST(ONNX, Pow_broadcast){
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard  input
   std::vector<float> input1({
      1, 2, 3, 3, 4, 5
   });
   std::vector<float> input2({
      2, 3, 4, 2, 3, 4
   });
   
   TMVA_SOFIE_Pow_broadcast::Session s("Pow_broadcast_FromONNX.dat");
   std::vector<float> output = s.infer(input2.data(),input1.data());
   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Pow_broadcast_ExpectedOutput::outputs) / sizeof(float));
   
   float *correct = Pow_broadcast_ExpectedOutput::outputs;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }

}

   TEST(ONNX, ReduceMean){
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard  input
   std::vector<float> input({
      5, 2, 3,
      5, 5, 4
   });
   
   TMVA_SOFIE_ReduceMean::Session s("ReduceMean_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());
   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ReduceMean_ExpectedOutput::output) / sizeof(float));
   
   float *correct = ReduceMean_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }

}

   TEST(ONNX, ReduceProd){
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard  input
   std::vector<float> input({
      5, 2, 3,
      5, 5, 4
   });
   
   TMVA_SOFIE_ReduceProd::Session s("ReduceProd_FromONNX.dat");
   std::vector<float> output = s.infer(input.data());
   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ReduceProd_ExpectedOutput::output) / sizeof(float));
   
   float *correct = ReduceProd_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }

}

TEST(ONNX, Max)
   {
      constexpr float TOLERANCE = DEFAULT_TOLERANCE;

      // Preparing the standard input
      std::vector<float> input1({
         1.0,  2.0, -1.0
      });
      std::vector<float> input2({
         3.0, 0.0, 4.0
      });
      TMVA_SOFIE_Max::Session s("Max_FromONNX.dat");

      std::vector<float> output = s.infer(input1.data(),input2.data());

      // Checking output size
      EXPECT_EQ(output.size(), sizeof(Max_ExpectedOutput::outputs) / sizeof(float));

      float *correct = Max_ExpectedOutput::outputs;

      // Checking every output value, one by one
      for (size_t i = 0; i < output.size(); ++i) {
         EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
      }
   }

TEST(ONNX, Shape){
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard  input
   std::vector<float> input({
      1, 2
   });
   
   TMVA_SOFIE_Shape::Session s("Shape_FromONNX.dat");
   auto output = s.infer(input.data());
   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Shape_ExpectedOutput::outputs) / sizeof(float));
   
   int *correct = Shape_ExpectedOutput::outputs;

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

TEST(ONNX, Softmax1d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({-1., 0., 1.});
   TMVA_SOFIE_Softmax1d::Session s("Softmax1d_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   EXPECT_EQ(output.size(), sizeof(Softmax1d_ExpectedOutput::output) / sizeof(float));

   float *correct = Softmax1d_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, Softmax2d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({-1., 0., 1.});
   TMVA_SOFIE_Softmax2d::Session s("Softmax2d_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   EXPECT_EQ(output.size(), sizeof(Softmax2d_ExpectedOutput::output) / sizeof(float));

   float *correct = Softmax2d_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, Softmax3d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({
        -0.8939, -0.3674,  0.1763,  1.5804, -0.4687,  1.2253, -1.3488, -0.1000,
        -0.1262,  0.4962,  1.0870,  0.6905, -0.3451, -1.6981, -0.4688,  0.4468,
        -0.5479,  0.0650,  1.0446, -1.6249, -0.7190, -1.7520,  3.7753, -1.4939});
   TMVA_SOFIE_Softmax3d::Session s("Softmax3d_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   EXPECT_EQ(output.size(), sizeof(Softmax3d_ExpectedOutput::output) / sizeof(float));

   float *correct = Softmax3d_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, Softmax4d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({
        -0.5869, -1.4272, -0.1546,  0.0096,  0.1706,  0.0388, -0.3484, -0.7829,
         1.1138, -0.5644, -0.6264, -1.1890,  1.6741, -0.7130,  0.9592,  1.7477,
        -0.4775,  1.3407, -0.3882, -0.4560,  1.0385, -0.1669,  0.5540, -1.0790,
        -0.6153, -0.6274, -1.2304, -0.6757,  1.0178, -0.2379, -0.7912, -0.0165,
        -0.5423,  0.1459,  1.3585, -0.5005, -0.2187, -1.8181, -0.6642,  0.0287,
        -1.9103,  0.7984, -0.7860,  1.5134,  1.3873, -0.6462, -0.6354, -0.1335});
   TMVA_SOFIE_Softmax4d::Session s("Softmax4d_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   EXPECT_EQ(output.size(), sizeof(Softmax4d_ExpectedOutput::output) / sizeof(float));

   float *correct = Softmax4d_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, ConvTranspose1d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(3);
   std::iota(input.begin(), input.end(), 0.0f);
   TMVA_SOFIE_ConvTranspose1d::Session s("ConvTranspose1d_FromONNX.dat");
   auto output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvTranspose1d_ExpectedOutput::output) / sizeof(float));

   float *correct = ConvTranspose1d_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, ConvTranspose2d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(9);
   std::iota(input.begin(), input.end(), 0.0f);
   TMVA_SOFIE_ConvTranspose2d::Session s("ConvTranspose2d_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvTranspose2d_ExpectedOutput::output) / sizeof(float));

   float *correct = ConvTranspose2d_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, ConvTranspose3d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(8);
   std::iota(input.begin(), input.end(), 0.0f);
   TMVA_SOFIE_ConvTranspose3d::Session s("ConvTranspose3d_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvTranspose3d_ExpectedOutput::output) / sizeof(float));

   float *correct = ConvTranspose3d_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, ConvTransposeBias2d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(9);
   std::iota(input.begin(), input.end(), 0.0f);
   TMVA_SOFIE_ConvTransposeBias2d::Session s("ConvTransposeBias2d_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvTransposeBias2d_ExpectedOutput::output) / sizeof(float));

   float *correct = ConvTransposeBias2d_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, ConvTransposeBias2dBatched)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(18);
   std::iota(input.begin(), input.end(), 0.0f);
   TMVA_SOFIE_ConvTransposeBias2dBatched::Session s("ConvTransposeBias2dBatched_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(ConvTransposeBias2dBatched_ExpectedOutput::output) / sizeof(float));

   float *correct = ConvTransposeBias2dBatched_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

