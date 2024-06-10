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

#include "Mul_FromONNX.hxx"
#include "input_models/references/Mul.ref.hxx"

#include "Div_FromONNX.hxx"
#include "input_models/references/Div.ref.hxx"

#include "Cast_FromONNX.hxx"
#include "input_models/references/Cast.ref.hxx"

#include "ReduceMean_FromONNX.hxx"
#include "input_models/references/ReduceMean.ref.hxx"

#include "ReduceProd_FromONNX.hxx"
#include "input_models/references/ReduceProd.ref.hxx"

#include "Shape_FromONNX.hxx"
#include "input_models/references/Shape.ref.hxx"

#include "Constant_FromONNX.hxx"
#include "input_models/references/Constant.ref.hxx"

#include "LinearWithLeakyRelu_FromONNX.hxx"
#include "input_models/references/LinearWithLeakyRelu.ref.hxx"

#include "Tanh_FromONNX.hxx"
#include "input_models/references/Tanh.ref.hxx"

#include "Erf_FromONNX.hxx"
#include "input_models/references/Erf.ref.hxx"

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

#include "MaxMultidirectionalBroadcast_FromONNX.hxx"
#include "input_models/references/MaxMultidirectionalBroadcast.ref.hxx"

#include "MinMultidirectionalBroadcast_FromONNX.hxx"
#include "input_models/references/MinMultidirectionalBroadcast.ref.hxx"

#include "MeanMultidirectionalBroadcast_FromONNX.hxx"
#include "input_models/references/MeanMultidirectionalBroadcast.ref.hxx"

#include "SumMultidirectionalBroadcast_FromONNX.hxx"
#include "input_models/references/SumMultidirectionalBroadcast.ref.hxx"

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

//#include "ConvTranspose3d_FromONNX.hxx"
//#include "input_models/references/ConvTranspose3d.ref.hxx"

#include "ConvTransposeBias2d_FromONNX.hxx"
#include "input_models/references/ConvTransposeBias2d.ref.hxx"

#include "ConvTransposeBias2dBatched_FromONNX.hxx"
#include "input_models/references/ConvTransposeBias2dBatched.ref.hxx"

#include "Sqrt_FromONNX.hxx"
#include "input_models/references/Sqrt.ref.hxx"

#include "Reciprocal_FromONNX.hxx"
#include "input_models/references/Reciprocal.ref.hxx"

#include "Neg_FromONNX.hxx"
#include "input_models/references/Neg.ref.hxx"

#include "Exp_FromONNX.hxx"
#include "input_models/references/Exp.ref.hxx"

#include "AddBroadcast1_FromONNX.hxx"
#include "input_models/references/AddBroadcast1.ref.hxx"

#include "AddBroadcast2_FromONNX.hxx"
#include "input_models/references/AddBroadcast2.ref.hxx"

#include "AddBroadcast3_FromONNX.hxx"
#include "input_models/references/AddBroadcast3.ref.hxx"

#include "AddBroadcast4_FromONNX.hxx"
#include "input_models/references/AddBroadcast4.ref.hxx"

#include "AddBroadcast5_FromONNX.hxx"
#include "input_models/references/AddBroadcast5.ref.hxx"

#include "AddBroadcast6_FromONNX.hxx"
#include "input_models/references/AddBroadcast6.ref.hxx"

#include "AddBroadcast7_FromONNX.hxx"
#include "input_models/references/AddBroadcast7.ref.hxx"

#include "Concat_0D_FromONNX.hxx"

#include "LayerNormalization2d_FromONNX.hxx"
#include "input_models/references/LayerNormalization2d.hxx"

#include "LayerNormalization4d_FromONNX.hxx"
#include "input_models/references/LayerNormalization4d.hxx"

#include "ExpandSameSize_FromONNX.hxx"
#include "input_models/references/ExpandSameSize.ref.hxx"

#include "ExpandDiffSize_FromONNX.hxx"
#include "input_models/references/ExpandDiffSize.ref.hxx"

#include "GatherAxis0_FromONNX.hxx"
#include "input_models/references/GatherAxis0.ref.hxx"

#include "GatherAxis1_FromONNX.hxx"
#include "input_models/references/GatherAxis1.ref.hxx"

#include "GatherAxis2_FromONNX.hxx"
#include "input_models/references/GatherAxis2.ref.hxx"

#include "GatherAxis3_FromONNX.hxx"
#include "input_models/references/GatherAxis3.ref.hxx"

#include "Gather2d_FromONNX.hxx"
#include "input_models/references/Gather2d.ref.hxx"

#include "GatherNegativeIndices_FromONNX.hxx"
#include "input_models/references/GatherNegativeIndices.ref.hxx"

#include "Slice_FromONNX.hxx"
#include "input_models/references/Slice.ref.hxx"

#include "Slice_Default_Axis_FromONNX.hxx"
#include "input_models/references/Slice_Default_Axis.ref.hxx"

#include "Slice_Default_Steps_FromONNX.hxx"
#include "input_models/references/Slice_Default_Steps.ref.hxx"

#include "Slice_Neg_FromONNX.hxx"
#include "input_models/references/Slice_Neg.ref.hxx"

#include "Log_FromONNX.hxx"
#include "input_models/references/Log.ref.hxx"

#include "Elu_FromONNX.hxx"
#include "input_models/references/Elu.ref.hxx"

#include "Equal_FromONNX.hxx"
#include "input_models/references/Equal.ref.hxx"

#include "LessOrEqual_FromONNX.hxx"
#include "input_models/references/LessOrEqual.ref.hxx"

#include "GreaterOrEqual_FromONNX.hxx"
#include "input_models/references/GreaterOrEqual.ref.hxx"

#include "Less_FromONNX.hxx"
#include "input_models/references/Less.ref.hxx"

#include "Greater_FromONNX.hxx"
#include "input_models/references/Greater.ref.hxx"

#include "EyeLike_FromONNX.hxx"
#include "input_models/references/EyeLike.ref.hxx"
#include "RangeFloat_FromONNX.hxx"
#include "input_models/references/RangeFloat.ref.hxx"

#include "RangeInt_FromONNX.hxx"
#include "input_models/references/RangeInt.ref.hxx"

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

/*TEST(ONNX, Linear32RootFeature)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(3200);
   std::fill_n(input.data(), input.size(), 1.0f);
   TMVA_SOFIE_Linear32RootFeacture::Session s("Linear_32_FromONNX.root");
   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Linear_32_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = Linear_32_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}*/

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

      std::vector<float> output = s.infer(input1.data(),input2.data());

      // Checking output size
      EXPECT_EQ(output.size(), sizeof(Sub_ExpectedOutput::outputs) / sizeof(float));

      float *correct = Sub_ExpectedOutput::outputs;

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

      std::vector<float> output = s.infer(input1.data(),input2.data());

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

TEST(ONNX, Elu)
   {
      constexpr float TOLERANCE = DEFAULT_TOLERANCE;

      // Preparing the standard input
      std::vector<float> input({
        1.0, -2.0, 3.0, 0.5, -1.0, 2.0
      });

      TMVA_SOFIE_Elu::Session s("Elu_FromONNX.dat");
      std::vector<float> output = s.infer(input.data());

      // Checking output size
      EXPECT_EQ(output.size(), sizeof(Elu_ExpectedOutput::outputs) / sizeof(float));

      float *correct = Elu_ExpectedOutput::outputs;

      // Checking every output value, one by one
      for (size_t i = 0; i < output.size(); ++i) {
         EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
      }
   }
   
TEST(ONNX, Constant)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard  input (none for Constant Op)
   // std::vector<float> input({
   //    1,2,3,4
   // });

   TMVA_SOFIE_Constant::Session s("Constant_FromONNX.dat");

   auto output = s.infer();

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Constant_ExpectedOutput::outputs) / sizeof(float));

   float *correct = Constant_ExpectedOutput::outputs;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

   TEST(ONNX, EyeLike)
   {
      constexpr float TOLERANCE = DEFAULT_TOLERANCE;

      // Preparing the standard input
      std::vector<float> input({
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
      });

      TMVA_SOFIE_EyeLike::Session s("EyeLike_FromONNX.dat");
      std::vector<float> output = s.infer(input.data());

      // Checking output size
      EXPECT_EQ(output.size(), sizeof(EyeLike_ExpectedOutput::output) / sizeof(float));

      float *correct = EyeLike_ExpectedOutput::output;

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

TEST(ONNX, Erf)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the random input
   std::vector<float> input({
     -1.0412,  0.1918,  0.9985, -0.5959,  0.6842, -2.4718,  0.1804,  0.6851,
      1.5646, -1.4981,  0.4248, -0.8504
   });

   TMVA_SOFIE_Erf::Session s("Erf_FromONNX.dat");

   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Erf_ExpectedOutput::outputs) / sizeof(float));

   float *correct = Erf_ExpectedOutput::outputs;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, Log)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the random input
   std::vector<float> input({
     1, 2, 3, 4
   });

   TMVA_SOFIE_Log::Session s("Log_FromONNX.dat");

   std::vector<float> output = s.infer(input.data());

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Log_ExpectedOutput::outputs) / sizeof(float));

   float *correct = Log_ExpectedOutput::outputs;

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
   std::vector<float> output = s.infer(input1.data(),input2.data());
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
   std::vector<float> output = s.infer(input1.data(),input2.data());
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

TEST(ONNX, MaxMultidirectionalBroadcast) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> a({0.35974154, -2.20873388,  0.95746274});
   std::vector<float> b({0.75901985, -0.46544461, -0.34920575, -0.1460754 ,  0.08269051, -0.70045695});
   std::vector<float> c({-0.41468981, -0.46591926,  0.56172534,  0.05616931});

   TMVA_SOFIE_MaxMultidirectionalBroadcast::Session s("MaxMultidirectionalBroadcast_FromONNX.dat");

   std::vector<float> output = s.infer(a.data(), b.data(), c.data());

   EXPECT_EQ(output.size(), sizeof(MaxMultidirectionalBroadcast_ExpectedOutput::output) / sizeof(float));

   float* correct = MaxMultidirectionalBroadcast_ExpectedOutput::output;

   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, MinMultidirectionalBroadcast) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> a({0.35974154, -2.20873388,  0.95746274});
   std::vector<float> b({0.75901985, -0.46544461, -0.34920575, -0.1460754 ,  0.08269051, -0.70045695});
   std::vector<float> c({-0.41468981, -0.46591926,  0.56172534,  0.05616931});

   TMVA_SOFIE_MinMultidirectionalBroadcast::Session s("MinMultidirectionalBroadcast_FromONNX.dat");

   std::vector<float> output = s.infer(a.data(), b.data(), c.data());

   EXPECT_EQ(output.size(), sizeof(MinMultidirectionalBroadcast_ExpectedOutput::output) / sizeof(float));

   float* correct = MinMultidirectionalBroadcast_ExpectedOutput::output;

   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, MeanMultidirectionalBroadcast) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> a({0.35974154, -2.20873388,  0.95746274});
   std::vector<float> b({0.75901985, -0.46544461, -0.34920575, -0.1460754 ,  0.08269051, -0.70045695});
   std::vector<float> c({-0.41468981, -0.46591926,  0.56172534,  0.05616931});

   TMVA_SOFIE_MeanMultidirectionalBroadcast::Session s("MeanMultidirectionalBroadcast_FromONNX.dat");

   std::vector<float> output = s.infer(a.data(), b.data(), c.data());

   EXPECT_EQ(output.size(), sizeof(MeanMultidirectionalBroadcast_ExpectedOutput::output) / sizeof(float));

   float* correct = MeanMultidirectionalBroadcast_ExpectedOutput::output;

   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, SumMultidirectionalBroadcast) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> a({0.35974154, -2.20873388,  0.95746274});
   std::vector<float> b({0.75901985, -0.46544461, -0.34920575, -0.1460754 ,  0.08269051, -0.70045695});
   std::vector<float> c({-0.41468981, -0.46591926,  0.56172534,  0.05616931});

   TMVA_SOFIE_SumMultidirectionalBroadcast::Session s("SumMultidirectionalBroadcast_FromONNX.dat");

   std::vector<float> output = s.infer(a.data(), b.data(), c.data());

   EXPECT_EQ(output.size(), sizeof(SumMultidirectionalBroadcast_ExpectedOutput::output) / sizeof(float));

   float* correct = SumMultidirectionalBroadcast_ExpectedOutput::output;

   for (size_t i = 0; i < output.size(); i++) {
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

/*
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
*/

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

TEST(ONNX, Sqrt)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({0.8344, 0.4716, 0.6226, 0.8448, 0.2483, 0.9467});
   TMVA_SOFIE_Sqrt::Session s("Sqrt_FromONNX.data");
   std::vector<float> output = s.infer(input.data());

   EXPECT_EQ(output.size(), sizeof(Sqrt_ExpectedOutput::output) / sizeof(float));

   float* correct = Sqrt_ExpectedOutput::output;

   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, Reciprocal)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({1.2691, -1.2160,  0.6393, -0.4438,  0.8065,  0.2011});
   TMVA_SOFIE_Reciprocal::Session s("Reciprocal_FromONNX.data");
   std::vector<float> output = s.infer(input.data());

   EXPECT_EQ(output.size(), sizeof(Reciprocal_ExpectedOutput::output) / sizeof(float));

   float* correct = Reciprocal_ExpectedOutput::output;

   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, Exp)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({1.46566453,  0.63334515,  2.4048165 ,  0.54468453,
      -1.41271672, -0.18609187,  0.2754482 ,  1.10615209,  0.88474389,  0.47531232});
   TMVA_SOFIE_Exp::Session s("Exp_FromONNX.data");
   std::vector<float> output = s.infer(input.data());

   EXPECT_EQ(output.size(), sizeof(Exp_ExpectedOutput::output) / sizeof(float));

   float* correct = Exp_ExpectedOutput::output;

   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, AddBroadcast1) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   // The shape of A is {5}
   std::vector<float> A({-0.78023305, -1.34029483, -3.01482951, 0.53641361,
                 -1.22594789});
   // The shape of B is {4, 5}
   std::vector<float> B({1.0626695,  0.43842875,  1.22476468,  0.79763274,  0.98688211,
                 0.25267614, 0.44874883,  0.31516773,  -0.78771195, 0.64565664,
                 0.50450593, -0.41265227, -0.22474539, -0.22362374, 0.00509674,
                 0.16927211, 1.06756969,  -0.81634773, 0.88467744,  0.78902059});

   TMVA_SOFIE_AddBroadcast1::Session s("AddBroadcast1_FromONNX.dat");
   std::vector<float> output(s.infer(A.data(), B.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(AddBroadcast1_ExpectedOutput::output) / sizeof(float));

   float* correct = AddBroadcast1_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, AddBroadcast2) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   // The shape of A is {5}
   std::vector<float> A({0.60081805, 0.56575772, -0.58408511, -1.50827751, 1.2396254});
   // The shape of B is {2, 3, 4, 5}
   std::vector<float> B({
        -1.22516739e+00, -2.50373737e+00, -6.14517347e-01, 4.43165956e-01,
        4.09232228e-03,  1.43520073e+00,  -8.37526920e-01, 1.18762642e+00,
        -1.42122220e+00, 3.77123343e-01,  -6.16450821e-01, 1.96641319e+00,
        -2.03568224e+00, -5.36703377e-01, -2.22149348e+00, -1.58297075e+00,
        -1.25149214e+00, 6.50629098e-01,  2.06339687e+00,  6.02281648e-01,
        -5.39034004e-01, -1.26280821e+00, 7.87767451e-01,  1.08251530e-01,
        2.32829794e+00,  -1.50890004e+00, -5.95592927e-01, -9.20059053e-02,
        1.63228625e+00,  1.94686070e+00,  7.45655684e-01,  3.86955114e-01,
        -1.83205116e+00, -1.15734817e+00, 3.80085814e-02,  -2.16949162e-01,
        -2.35165487e-01, 2.18171406e-01,  6.13588954e-02,  -8.57086260e-01,
        -2.01864267e+00, -1.61373575e+00, -2.02050258e+00, -3.25052069e-01,
        -1.07114643e-01, 4.68470099e-01,  1.99557999e-01,  -1.94637668e+00,
        2.47900553e-01,  7.76198825e-01,  -1.98736855e-01, -2.00884998e+00,
        1.46847865e+00,  9.61028795e-01,  -8.14965358e-03, 4.63333332e-01,
        -1.11316244e-01, 1.82046921e+00,  -1.00519072e-01, 2.40577520e+00,
        2.57814258e+00,  -1.51412865e+00, -6.48090386e-02, 9.22939224e-01,
        -1.31486041e+00, 3.67387151e-01,  -2.17020478e-03, -4.74744054e-01,
        -6.28942699e-01, -1.31704730e+00, -6.20633846e-01, -4.90250204e-01,
        -2.12485120e-01, -2.36786681e-02, 2.88809968e-02,  -7.44777791e-01,
        1.30091804e-02,  -1.68105549e+00, 8.22247057e-02,  -1.14939503e+00,
        -1.57565418e+00, -7.99386689e-01, -4.06411097e-01, 1.09358391e+00,
        1.58323366e+00,  -8.15174970e-02, -9.09925044e-02, 2.35596716e+00,
        -6.85364818e-02, 4.12883924e-01,  5.00495425e-01,  -1.48442647e+00,
        -5.19349052e-01, 3.81025828e-01,  -1.06188597e-01, 2.83921542e-01,
        1.13215001e+00,  1.21558052e+00,  -1.04667496e+00, -9.41151099e-01,
        -4.04363040e-02, 1.45554304e+00,  1.64025681e-01,  -3.34693361e-01,
        1.27701314e+00,  8.64744621e-01,  1.09621430e+00,  -1.06563435e+00,
        -1.55637568e+00, 2.14343040e+00,  4.69610352e-01,  9.09135609e-01,
        -6.20603382e-01, -1.04235434e+00, -1.32974691e+00, -1.35968049e-01,
        9.62438348e-01,  1.13413513e+00,  -9.24612219e-01, -2.26132356e+00});

   TMVA_SOFIE_AddBroadcast2::Session s("AddBroadcast2_FromONNX.dat");
   std::vector<float> output(s.infer(A.data(), B.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(AddBroadcast2_ExpectedOutput::output) / sizeof(float));

   float* correct = AddBroadcast2_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, AddBroadcast3) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   // The shape of A is {2, 1, 1, 5}
   std::vector<float> A({0.13225244, -0.47801406, -1.47034622, 0.87786363, -0.51388502,
                 0.77012016, 0.99407484,  -0.41014198, 1.76506248, 1.24142803});
   // The shape of B is {2, 3, 4, 5}
   std::vector<float> B({
         -0.79900037, 1.26774471,  0.10287351,  -0.00704713, 0.19927171,
        1.77125926,  0.23393901,  -0.75160577, -0.40987021, 0.02957325,
        2.48770369,  2.72426688,  0.16116267,  0.13580884,  -1.34550983,
        1.08341747,  -0.57232679, -0.27434247, 2.29759196,  0.72506479,
        -0.35984264, -1.47553974, 0.46544721,  0.45304508,  0.39350919,
        0.25335039,  -2.15455262, 0.58592831,  0.0907586,   1.32830358,
        2.16876532,  -1.31509165, -0.77901816, 1.72970744,  0.89410519,
        1.18891089,  0.58372505,  -0.6117035,  -0.83829228, 0.63917945,
        0.66626077,  -1.07667629, 0.01411519,  -0.67082652, -0.04556866,
        -0.04949148, -1.87075929, 0.25587637,  0.14715114,  -0.74584515,
        -1.19373527, -1.52142058, -0.92522942, -0.98126531, -0.07535746,
        -1.4692508,  -0.08861242, 0.64951867,  -0.16918995, 0.87015361,
        0.57688991,  1.36293834,  1.28256834,  0.39245538,  0.43308474,
        0.84529828,  -0.56686547, -0.84791844, -0.11286944, 0.60857973,
        -0.79519511, -0.20491925, -1.52951743, -0.39030064, -2.76160767,
        0.09055906,  -0.99142034, 0.33480785,  -1.09999883, 1.36149355,
        0.18557576,  0.55407001,  1.23164067,  -0.23469015, -1.37274723,
        1.80717934,  1.42966758,  0.72077395,  -0.09774939, 1.12065382,
        -0.51515613, -0.9527945,  0.87646967,  -0.59440101, -0.12440208,
        -0.71096692, -0.6301275,  0.51726169,  1.23726643,  1.56255466,
        -0.94469759, -0.38114756, -0.42021761, -0.58921487, -0.71439637,
        0.04793575,  -2.04214516, -0.45765407, -1.12307202, 0.90727137,
        0.96272832,  0.54303206,  -0.84973033, 0.28780329,  0.17027854,
        -0.11893711, -1.22414638, -1.62747593, 0.53264501,  0.53483601});

   TMVA_SOFIE_AddBroadcast3::Session s("AddBroadcast3_FromONNX.dat");
   std::vector<float> output(s.infer(A.data(), B.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(AddBroadcast3_ExpectedOutput::output) / sizeof(float));

   float* correct = AddBroadcast3_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, AddBroadcast4) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   // The shape of A is {2, 1}
   std::vector<float> A({1.94301397, 0.40606817});
   // The shape of B is {2, 4}
   std::vector<float> B({0.50898894, -0.27829921, -0.68761628,  0.33186382,  0.57915535,
        0.406858  ,  1.4203833 ,  0.19857093});
   TMVA_SOFIE_AddBroadcast4::Session s("AddBroadcast4_FromONNX.dat");
   std::vector<float> output(s.infer(A.data(), B.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(AddBroadcast4_ExpectedOutput::output) / sizeof(float));

   float* correct = AddBroadcast4_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, AddBroadcast5) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   // The shape of A is {2, 1, 4}
   std::vector<float> A({-0.45616139, -0.05853134,  1.09564217,  0.95880315,  0.94995322,
       -0.35864105,  1.08570897,  0.6028053});
   // The shape of B is {2, 3, 4}
   std::vector<float> B({1.69787452,  1.10641673,  2.19755165,  0.06709206,  0.04572308,
       -2.14504366, -0.47730702,  0.15205423, -0.25159224, -0.07529807,
        0.5174367 ,  0.08267595,  0.34015625,  0.09460231, -1.16608969,
       -0.23466058, -0.5520268 , -0.13844847,  0.53055759,  0.17068648,
       -0.49491276, -1.4246271 , -0.99973914, -0.2571329});

   TMVA_SOFIE_AddBroadcast5::Session s("AddBroadcast5_FromONNX.dat");
   std::vector<float> output(s.infer(A.data(), B.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(AddBroadcast5_ExpectedOutput::output) / sizeof(float));

   float* correct = AddBroadcast5_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, AddBroadcast6) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   // The shape of A is {2, 1, 3, 1, 2}
   std::vector<float> A({1.05498675, -1.64311041,  0.11925147, -1.59755778, -0.01445313,
       -0.69440541, -0.12011281,  0.00539323, -0.16923531,  2.34533598,
        1.30268048,  0.45699443});
   // The shape of B is {2, 2, 3, 2, 2}
   std::vector<float> B({
       0.03162163,  1.36340443, -0.34736459, -0.71856324,  0.40669968,
       -0.37595741,  0.22234952,  1.69563792,  0.91459166, -0.02081215,
       -1.64894217, -0.01189261,  0.58031339, -0.11880191,  0.70099317,
       -0.37424243, -0.23980527, -0.03178407, -0.27969109,  0.01895688,
        1.32111755,  0.02113906,  0.51450298, -1.41760768, -0.19220553,
        0.23529522,  0.95199908, -1.38971445, -0.75836965, -0.90956958,
       -0.13006828, -0.64390454, -0.0808229 ,  0.79134757,  1.00684867,
       -1.43818087, -0.14550621, -0.33635512, -0.6185612 , -0.49281407,
       -1.12947258,  1.61818821, -0.05826431, -1.47802183,  0.25637381,
       -0.1547858 ,  2.50788792,  0.30898059});

   TMVA_SOFIE_AddBroadcast6::Session s("AddBroadcast6_FromONNX.dat");
   std::vector<float> output(s.infer(A.data(), B.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(AddBroadcast6_ExpectedOutput::output) / sizeof(float));

   float* correct = AddBroadcast6_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, AddBroadcast7) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   // The shape of A is {2, 1, 3, 1}
   std::vector<float> A({-0.42164834, -0.61767078, -0.68778897, -1.14175916,  0.63204375,
       -0.60630317});
   // The shae of B is {1, 1, 3, 4}
   std::vector<float> B({1.40519865e+00, -2.87660856e-01,  7.49375999e-02,  1.22074840e+00,
       -4.86212681e-01, -6.88210109e-01, -6.77434705e-01,  3.67088873e-01,
        8.05744026e-04, -2.08031088e-01,  9.69779132e-01,  7.58373863e-01});

   TMVA_SOFIE_AddBroadcast7::Session s("AddBroadcast7_FromONNX.dat");
   std::vector<float> output(s.infer(A.data(), B.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(AddBroadcast7_ExpectedOutput::output) / sizeof(float));

   float* correct = AddBroadcast7_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, Concat0D) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   std::vector<float> input({1.40519865e+00, -2.87660856e-01});
   std::vector<float> expected_output({1.40519865e+00, -2.87660856e-01, 1.40519865e+00, -2.87660856e-01});
   TMVA_SOFIE_Concat_0D::Session s("Concat_0D_FromONNX.dat");
   std::vector<float> actual_output(s.infer(input.data()));

   // Checking the output size
   EXPECT_EQ(expected_output.size(), expected_output.size());

   float* correct = expected_output.data();

   // Checking every output value, one by one
   for (size_t i = 0; i < actual_output.size(); i++) {
      EXPECT_LE(std::abs(actual_output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, LayerNormalization2d) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   std::vector<float> x(12);
   std::iota(x.begin(), x.end(), 0.);
   TMVA_SOFIE_LayerNormalization2d::Session s("LayerNormalization2d_FromONNX.dat");
   std::vector<float> output(s.infer(x.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(LayerNormalization2d_ExpectedOutput::output) / sizeof(float));

   float* correct = LayerNormalization2d_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, LayerNormalization4d) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   std::vector<float> x(120);
   std::iota(x.begin(), x.end(), 0.);
   TMVA_SOFIE_LayerNormalization4d::Session s("LayerNormalization4d_FromONNX.dat");
   std::vector<float> output(s.infer(x.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(LayerNormalization4d_ExpectedOutput::output) / sizeof(float));

   float* correct = LayerNormalization4d_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, Equal){
   constexpr float TOLERANCE = 0;

   // Preparing the standard  input
   std::vector<float> input1({
      1.0, 2.0, 3.0
   });
   std::vector<float> input2({
      4.0, 2.0, 6.0
   });

   TMVA_SOFIE_Equal::Session s("Equal_FromONNX.dat");
   std::vector<bool> output = s.infer(input1.data(),input2.data());
   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Equal_ExpectedOutput::outputs) / sizeof(bool));

   bool *correct = Equal_ExpectedOutput::outputs;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(((correct[i]==output[i])?0:1), TOLERANCE);
   }

}

TEST(ONNX, LessOrEqual){
   constexpr float TOLERANCE = 0;

   // Preparing the standard  input
   std::vector<float> input1({
      1.0, 2.0, 3.0
   });
   std::vector<float> input2({
      4.0, 2.0, 6.0
   });

   TMVA_SOFIE_LessOrEqual::Session s("LessOrEqual_FromONNX.dat");
   std::vector<bool> output = s.infer(input1.data(),input2.data());
   // Checking output size
   EXPECT_EQ(output.size(), sizeof(LessOrEqual_ExpectedOutput::outputs) / sizeof(bool));

   bool *correct = LessOrEqual_ExpectedOutput::outputs;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(((correct[i]==output[i])?0:1), TOLERANCE);
   }

}

TEST(ONNX, GreaterOrEqual){
   constexpr float TOLERANCE = 0;

   // Preparing the standard  input
   std::vector<float> input1({
      1.0, 2.0, 3.0
   });
   std::vector<float> input2({
      4.0, 2.0, 6.0
   });

   TMVA_SOFIE_GreaterOrEqual::Session s("GreaterOrEqual_FromONNX.dat");
   std::vector<bool> output = s.infer(input1.data(),input2.data());
   // Checking output size
   EXPECT_EQ(output.size(), sizeof(GreaterOrEqual_ExpectedOutput::outputs) / sizeof(bool));

   bool *correct = GreaterOrEqual_ExpectedOutput::outputs;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(((correct[i]==output[i])?0:1), TOLERANCE);
   }

}

TEST(ONNX, Greater){
   constexpr float TOLERANCE = 0;

   // Preparing the standard  input
   std::vector<float> input1({
      1.0, 2.0, 3.0
   });
   std::vector<float> input2({
      4.0, 2.0, 6.0
   });

   TMVA_SOFIE_Greater::Session s("Greater_FromONNX.dat");
   std::vector<bool> output = s.infer(input1.data(),input2.data());
   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Greater_ExpectedOutput::outputs) / sizeof(bool));

   bool *correct = Greater_ExpectedOutput::outputs;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(((correct[i]==output[i])?0:1), TOLERANCE);
   }

}

TEST(ONNX, Less){
   constexpr float TOLERANCE = 0;

   // Preparing the standard  input
   std::vector<float> input1({
      1.0, 2.0, 3.0
   });
   std::vector<float> input2({
      4.0, 2.0, 6.0
   });

   TMVA_SOFIE_Less::Session s("Less_FromONNX.dat");
   std::vector<bool> output = s.infer(input1.data(),input2.data());
   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Less_ExpectedOutput::outputs) / sizeof(bool));

   bool *correct = Less_ExpectedOutput::outputs;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(((correct[i]==output[i])?0:1), TOLERANCE);
   }

}

TEST(ONNX, ExpandSameSize) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   std::vector<float> input({0., 1., 2.});
   TMVA_SOFIE_ExpandSameSize::Session s("ExpandSameSize_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(ExpandSameSize_ExpectedOutput::output) / sizeof(float));

   float* correct = ExpandSameSize_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, ExpandDiffSize) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   std::vector<float> input({0., 1., 2.});
   TMVA_SOFIE_ExpandDiffSize::Session s("ExpandDiffSize_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(ExpandDiffSize_ExpectedOutput::output) / sizeof(float));

   float* correct = ExpandDiffSize_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, GatherAxis0) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   std::vector<float> input(120);
   std::iota(input.begin(), input.end(), 0.);
   TMVA_SOFIE_GatherAxis0::Session s("GatherAxis0_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(GatherAxis0_ExpectedOutput::output) / sizeof(float));

   float* correct = GatherAxis0_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, GatherAxis1) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   std::vector<float> input(120);
   std::iota(input.begin(), input.end(), 0.);
   TMVA_SOFIE_GatherAxis1::Session s("GatherAxis1_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(GatherAxis1_ExpectedOutput::output) / sizeof(float));

   float* correct = GatherAxis1_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, GatherAxis2) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   std::vector<float> input(120);
   std::iota(input.begin(), input.end(), 0.);
   TMVA_SOFIE_GatherAxis2::Session s("GatherAxis2_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(GatherAxis2_ExpectedOutput::output) / sizeof(float));

   float* correct = GatherAxis2_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, GatherAxis3) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   std::vector<float> input(120);
   std::iota(input.begin(), input.end(), 0.);
   TMVA_SOFIE_GatherAxis3::Session s("GatherAxis3_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(GatherAxis3_ExpectedOutput::output) / sizeof(float));

   float* correct = GatherAxis3_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, Gather2d) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   std::vector<float> input(9);
   std::iota(input.begin(), input.end(), 0.);
   TMVA_SOFIE_Gather2d::Session s("Gather2d_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(Gather2d_ExpectedOutput::output) / sizeof(float));

   float* correct = Gather2d_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, GatherNegativeIndices) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // input
   std::vector<float> input(10);
   std::iota(input.begin(), input.end(), 0.);
   TMVA_SOFIE_GatherNegativeIndices::Session s("GatherNegativeIndices_FromONNX.dat");
   std::vector<float> output(s.infer(input.data()));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(GatherNegativeIndices_ExpectedOutput::output) / sizeof(float));

   float* correct = GatherNegativeIndices_ExpectedOutput::output;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, Slice) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input = Slice::input;
   TMVA_SOFIE_Slice::Session s("Slice.dat");
   std::vector<float> output(s.infer(input.data()));

   EXPECT_EQ(output.size(), sizeof(Slice::output) / sizeof(float));
   float *correct = Slice::output;

   for (size_t i=0; i<output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }

}

TEST(ONNX, Slice_Default_Axis) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input = Slice_Default_Axis::input;
   TMVA_SOFIE_Slice_Default_Axis::Session s("Slice_Default_Axis.dat");
   std::vector<float> output(s.infer(input.data()));

   EXPECT_EQ(output.size(), sizeof(Slice_Default_Axis::output) / sizeof(float));
   float *correct = Slice_Default_Axis::output;

   for (size_t i=0; i<output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }

}

TEST(ONNX, Slice_Default_Steps) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input = Slice_Default_Steps::input;
   TMVA_SOFIE_Slice_Default_Steps::Session s("Slice_Default_Steps.dat");
   std::vector<float> output(s.infer(input.data()));

   EXPECT_EQ(output.size(), sizeof(Slice_Default_Steps::output) / sizeof(float));
   float *correct = Slice_Default_Steps::output;

   for (size_t i=0; i<output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }

}

TEST(ONNX, Slice_Neg) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input = Slice_Neg::input;
   TMVA_SOFIE_Slice_Neg::Session s("Slice_Neg.dat");
   std::vector<float> output(s.infer(input.data()));

   EXPECT_EQ(output.size(), sizeof(Slice_Neg::output) / sizeof(float));
   float *correct = Slice_Neg::output;

   for (size_t i=0; i<output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }

}
TEST(ONNX, RangeFloat) {
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // inputs
   float start = 1.;
   float limit = 10.;
   float delta = 2.;
   TMVA_SOFIE_RangeFloat::Session s("RangeFloat_FromONNX.dat");
   std::vector<float> output(s.infer(&start, &limit, &delta));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(RangeFloat_ExpectedOutput::outputs) / sizeof(float));

   float* correct = RangeFloat_ExpectedOutput::outputs;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}

TEST(ONNX, RangeInt) {
   // inputs
   int64_t start = 1;
   int64_t limit = 10;
   int64_t delta = 2;
   TMVA_SOFIE_RangeInt::Session s("RangeInt_FromONNX.dat");
   std::vector<int64_t> output(s.infer(&start, &limit, &delta));

   // Checking the output size
   EXPECT_EQ(output.size(), sizeof(RangeInt_ExpectedOutput::outputs) / sizeof(int64_t));

   int64_t* correct = RangeInt_ExpectedOutput::outputs;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); i++) {
      EXPECT_EQ(output[i], correct[i]);
   }
}
