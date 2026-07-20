constexpr auto modelHeaderSuffix = "_FromONNX.hxx";
constexpr auto modelDataSuffix = "_FromONNX.dat";
#include "test_helpers.h"

#include "gtest/gtest.h"

TEST(ONNX, Linear16)
{
   SofieReference ref = readReference("Linear_16");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Linear_16", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

/*TEST(ONNX, Linear32RootFeature)
{
   SofieReference ref = readReference("Linear_32");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Linear_32", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}*/

TEST(ONNX, Linear32)
{
   SofieReference ref = readReference("Linear_32");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Linear_32", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Sub)
{
   SofieReference ref = readReference("Sub");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Sub", ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Add)
{
   SofieReference ref = readReference("Add");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Add", ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Mul)
{
   SofieReference ref = readReference("Mul");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Mul", ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Div)
{
   SofieReference ref = readReference("Div");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Div", ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Neg)
{
   SofieReference ref = readReference("Neg");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Neg", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Elu)
{
   SofieReference ref = readReference("Elu");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Elu", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}
TEST(ONNX, EluAlpha)
{
   SofieReference ref = readReference("EluAlpha");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "EluAlpha", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Constant)
{
   SofieReference ref = readReference("Constant");

   ASSERT_INCLUDE_AND_RUN_0(std::vector<float>, "Constant");

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ComplexTopK)
{
   SofieReference ref = readReference("ComplexTopK");

   ASSERT_INCLUDE_AND_RUN(TupleFloatInt64_t, "ComplexTopK", ref.f32("input0"));

   expectNear(std::get<0>(output), ref.f32("output0"), DEFAULT_TOLERANCE);
   expectEqual(std::get<1>(output), ref.i64("output1"));
}
TEST(ONNX, TopK)
{
   SofieReference ref = readReference("TopK");

   ASSERT_INCLUDE_AND_RUN(TupleFloatInt64_t, "TopK", ref.f32("input0"));

   expectNear(std::get<0>(output), ref.f32("output0"), DEFAULT_TOLERANCE);
   expectEqual(std::get<1>(output), ref.i64("output1"));
}
   TEST(ONNX, EyeLike)
{
   SofieReference ref = readReference("EyeLike");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "EyeLike", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Cast)
{
   SofieReference ref = readReference("Cast");

   ASSERT_INCLUDE_AND_RUN(std::vector<double>, "Cast", ref.i64("input0"));

   expectNear(output, ref.f64("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Linear64)
{
   SofieReference ref = readReference("Linear_64");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Linear_64", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}


TEST(ONNX, LinearWithSelu)
{
   SofieReference ref = readReference("LinearWithSelu");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "LinearWithSelu", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Tanh)
{
   SofieReference ref = readReference("Tanh");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Tanh", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Erf)
{
   SofieReference ref = readReference("Erf");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Erf", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Log)
{
   SofieReference ref = readReference("Log");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Log", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LinearWithLeakyRelu)
{
   SofieReference ref = readReference("LinearWithLeakyRelu");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "LinearWithLeakyRelu", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}


TEST(ONNX, LinearWithSigmoid)
{
   SofieReference ref = readReference("LinearWithSigmoid");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "LinearWithSigmoid", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}


TEST(ONNX, ConvWithPadding)
{
   SofieReference ref = readReference("ConvWithPadding");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ConvWithPadding", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}


TEST(ONNX, ConvWithoutPadding)
{
   SofieReference ref = readReference("ConvWithoutPadding");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ConvWithoutPadding", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}


TEST(ONNX, ConvWithAutopadSameLower)
{
   SofieReference ref = readReference("ConvWithAutopadSameLower");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ConvWithAutopadSameLower", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithAutopadSameUpper)
{
   SofieReference ref = readReference("ConvWithAutopadSameUpper");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ConvWithAutopadSameUpper", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithStridesPadding)
{
   SofieReference ref = readReference("ConvWithStridesPadding");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ConvWithStridesPadding", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}


TEST(ONNX, ConvWithDilation)
{
   SofieReference ref = readReference("ConvWithDilation");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ConvWithDilation", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}


TEST(ONNX, ConvWithStridesNoPadding)
{
   SofieReference ref = readReference("ConvWithStridesNoPadding");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ConvWithStridesNoPadding", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvAddRelu)
{
   SofieReference ref = readReference("ConvAddRelu");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ConvAddRelu", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithDynShapeStride)
{
   // Conv1d with dynamic spatial dimension W and stride=2.
   // Verifies fix for output dimension formula: ((W+pad-kernel)/stride+1)
   // was incorrectly generated as ((W+pad-kernel)/stride1) before the fix.
   //
   // Model: kernel=3, stride=2, pad=0, weight=all-ones, input shape (1,1,W)
   // With W=7: output shape (1,1,3), output = [0+1+2, 2+3+4, 4+5+6] = [3, 9, 15]

   std::vector<float> input = {0, 1, 2, 3, 4, 5, 6}; // shape (1,1,7)
   std::vector<float> correct_output = {3, 9, 15};

   // model is dynamic in spatial dim W, use W = 7
   ASSERT_INCLUDE_AND_RUN_SESSION_ARGS(std::vector<float>, "ConvWithDynShapeStride",
                                       "\"ConvWithDynShapeStride_FromONNX.dat\", 7", 7, input);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

// Disables test (asymmetric padding not supported)
TEST(DISABLED_ONNX, ConvWithAsymmetricPadding)
{
   SofieReference ref = readReference("ConvWithAsymmetricPadding");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ConvWithAsymmetricPadding", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MaxPool1d)
{
   SofieReference ref = readReference("MaxPool1d");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "MaxPool1d", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MaxPool2d)
{
   SofieReference ref = readReference("MaxPool2d");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "MaxPool2d", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MaxPool2d_AsymPad)
{
   SofieReference ref = readReference("MaxPool2d_AsymPad");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "MaxPool2d_AsymPad", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MaxPool2d_CeilMode)
{
   SofieReference ref = readReference("MaxPool2d_CeilMode");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "MaxPool2d_CeilMode", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MaxPool1d_CeilMode_Overhang)
{
   SofieReference ref = readReference("MaxPool1d_CeilMode_Overhang");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "MaxPool1d_CeilMode_Overhang", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MaxPool2d_CeilMode_Pads)
{
   SofieReference ref = readReference("MaxPool2d_CeilMode_Pads");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "MaxPool2d_CeilMode_Pads", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MaxPool3d)
{
   SofieReference ref = readReference("MaxPool3d");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "MaxPool3d", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AveragePool1d_CeilMode)
{
   SofieReference ref = readReference("AveragePool1d_CeilMode");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "AveragePool1d_CeilMode", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AveragePool1d_CeilMode_Overhang)
{
   SofieReference ref = readReference("AveragePool1d_CeilMode_Overhang");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "AveragePool1d_CeilMode_Overhang", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AveragePool2d_CeilMode)
{
   SofieReference ref = readReference("AveragePool2d_CeilMode");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "AveragePool2d_CeilMode", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AveragePool2d_CeilMode_Pads)
{
   SofieReference ref = readReference("AveragePool2d_CeilMode_Pads");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "AveragePool2d_CeilMode_Pads", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AveragePool2d_CeilMode_CountIncludePad)
{
   SofieReference ref = readReference("AveragePool2d_CeilMode_CountIncludePad");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "AveragePool2d_CeilMode_CountIncludePad", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AveragePool2d_Pads_CountIncludePad)
{
   SofieReference ref = readReference("AveragePool2d_Pads_CountIncludePad");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "AveragePool2d_Pads_CountIncludePad", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AveragePool3d_CeilMode)
{
   SofieReference ref = readReference("AveragePool3d_CeilMode");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "AveragePool3d_CeilMode", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AvgPool)
{
   SofieReference ref = readReference("AvgPool");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "AvgPool", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Pow)
{
   SofieReference ref = readReference("Pow");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Pow", ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Pow_broadcast)
{
   SofieReference ref = readReference("Pow_broadcast");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Pow_broadcast", ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, FMod_ConstantFolding)
{
   // Both inputs are Constant nodes, so SOFIE constant-folds via Func().
   // fmod([10, 7, 5], [3, 3, 3]) = [1, 1, 2]
   std::vector<float> correct_output = {1, 1, 2};
   ASSERT_INCLUDE_AND_RUN_0(std::vector<float>, "FMod_ConstantFolding");
   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Mod_ConstantFolding)
{
   // Both inputs are Constant nodes, so SOFIE constant-folds via Func().
   // [10, 7, 5] % [3, 3, 3] = [1, 1, 2]
   std::vector<int64_t> correct_output = {1, 1, 2};
   ASSERT_INCLUDE_AND_RUN_0(std::vector<int64_t>, "Mod_ConstantFolding");
   expectEqual(output, correct_output);
}

   TEST(ONNX, ReduceMean)
{
   SofieReference ref = readReference("ReduceMean");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ReduceMean", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ReduceMean_kFirst)
{
   // ReduceMean over axis=0 (kFirst path) on a [3,4] tensor.
   std::vector<float> input(12);
   std::iota(input.begin(), input.end(), 0.0f);
   std::vector<float> correct_output = {4, 5, 6, 7};

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ReduceMean_kFirst", input);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

   TEST(ONNX, ReduceProd)
{
   SofieReference ref = readReference("ReduceProd");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ReduceProd", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ReduceSum){
   // Preparing the standard  input
   std::vector<float> input({
      5, 2, 3,
      5, 5, 4
   });

   // test Reduce sum in all axis and  keeping the dimension
   // input tensor is shape [1,2,3]
   // output tensod is shape [1,1,1] and value = 24 (sum of all elements)

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ReduceSum", input);
   std::vector<float> correct{24};

   expectNear(output, correct, DEFAULT_TOLERANCE);
}

TEST(ONNX, ReduceSumSquare){
   // Preparing the standard  input
   std::vector<float> input({
      5, 2, 3,
      5, 5, 4
   });

   // reduce on last axis and do not keep dimension
   // output should be [1,2] and [25+4+9, 25+25+16]


   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ReduceSumSquare", input);
   std::vector<float> correct{38, 66};

   expectNear(output, correct, DEFAULT_TOLERANCE);
}

TEST(ONNX, Max)
{
   SofieReference ref = readReference("Max");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Max", ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MaxMultidirectionalBroadcast)
{
   SofieReference ref = readReference("MaxMultidirectionalBroadcast");

   ASSERT_INCLUDE_AND_RUN(
      std::vector<float>,
      "MaxMultidirectionalBroadcast",
      ref.f32("input0"),
      ref.f32("input1"),
      ref.f32("input2"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MinMultidirectionalBroadcast)
{
   SofieReference ref = readReference("MinMultidirectionalBroadcast");

   ASSERT_INCLUDE_AND_RUN(
      std::vector<float>,
      "MinMultidirectionalBroadcast",
      ref.f32("input0"),
      ref.f32("input1"),
      ref.f32("input2"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MeanMultidirectionalBroadcast)
{
   SofieReference ref = readReference("MeanMultidirectionalBroadcast");

   ASSERT_INCLUDE_AND_RUN(
      std::vector<float>,
      "MeanMultidirectionalBroadcast",
      ref.f32("input0"),
      ref.f32("input1"),
      ref.f32("input2"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, SumMultidirectionalBroadcast)
{
   SofieReference ref = readReference("SumMultidirectionalBroadcast");

   ASSERT_INCLUDE_AND_RUN(
      std::vector<float>,
      "SumMultidirectionalBroadcast",
      ref.f32("input0"),
      ref.f32("input1"),
      ref.f32("input2"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Shape)
{
   SofieReference ref = readReference("Shape");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Shape", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNBatchwise)
{
   SofieReference ref = readReference("RNNBatchwise");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "RNNBatchwise", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNBidirectional)
{
   SofieReference ref = readReference("RNNBidirectional");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "RNNBidirectional", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNBidirectionalBatchwise)
{
   SofieReference ref = readReference("RNNBidirectionalBatchwise");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "RNNBidirectionalBatchwise", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNDefaults)
{
   SofieReference ref = readReference("RNNDefaults");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "RNNDefaults", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNSeqLength)
{
   SofieReference ref = readReference("RNNSeqLength");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "RNNSeqLength", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNSequence)
{
   SofieReference ref = readReference("RNNSequence");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "RNNSequence", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNSequenceBatchwise)
{
   SofieReference ref = readReference("RNNSequenceBatchwise");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "RNNSequenceBatchwise", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LSTMBatchwise)
{
   SofieReference ref = readReference("LSTMBatchwise");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "LSTMBatchwise", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LSTMBidirectional)
{
   SofieReference ref = readReference("LSTMBidirectional");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "LSTMBidirectional", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
   expectNear(output[2], ref.f32("output2"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LSTMDefaults)
{
   SofieReference ref = readReference("LSTMDefaults");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "LSTMDefaults", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LSTMInitialBias)
{
   SofieReference ref = readReference("LSTMInitialBias");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "LSTMInitialBias", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LSTMPeepholes)
{
   SofieReference ref = readReference("LSTMPeepholes");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "LSTMPeepholes", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

// GRU tests
TEST(ONNX, GRUBatchwise)
{
   SofieReference ref = readReference("GRUBatchwise");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "GRUBatchwise", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GRUBidirectional)
{
   SofieReference ref = readReference("GRUBidirectional");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "GRUBidirectional", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GRUDefaults)
{
   SofieReference ref = readReference("GRUDefaults");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "GRUDefaults", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GRUInitialBias)
{
   SofieReference ref = readReference("GRUInitialBias");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "GRUInitialBias", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GRUSeqLength)
{
   SofieReference ref = readReference("GRUSeqLength");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "GRUSeqLength", ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Softmax1d)
{
   SofieReference ref = readReference("Softmax1d");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Softmax1d", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Softmax2d)
{
   SofieReference ref = readReference("Softmax2d");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Softmax2d", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Softmax3d)
{
   SofieReference ref = readReference("Softmax3d");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Softmax3d", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Softmax4d)
{
   SofieReference ref = readReference("Softmax4d");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Softmax4d", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvTranspose1d)
{
   SofieReference ref = readReference("ConvTranspose1d");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ConvTranspose1d", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvTranspose2d)
{
   SofieReference ref = readReference("ConvTranspose2d");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ConvTranspose2d", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

/* ConvTranspose3d is not supported yet; a ConvTranspose3d model would have
   to be added to generate_input_models.py to enable this test.
TEST(ONNX, ConvTranspose3d)
{
   SofieReference ref = readReference("ConvTranspose3d");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ConvTranspose3d", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}
*/

TEST(ONNX, ConvTransposeBias2d)
{
   SofieReference ref = readReference("ConvTransposeBias2d");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ConvTransposeBias2d", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvTransposeBias2dBatched)
{
   SofieReference ref = readReference("ConvTransposeBias2dBatched");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ConvTransposeBias2dBatched", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Sqrt)
{
   SofieReference ref = readReference("Sqrt");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Sqrt", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Reciprocal)
{
   SofieReference ref = readReference("Reciprocal");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Reciprocal", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Exp)
{
   SofieReference ref = readReference("Exp");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Exp", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AddBroadcast1)
{
   SofieReference ref = readReference("AddBroadcast1");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "AddBroadcast1", ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AddBroadcast2)
{
   SofieReference ref = readReference("AddBroadcast2");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "AddBroadcast2", ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AddBroadcast3)
{
   SofieReference ref = readReference("AddBroadcast3");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "AddBroadcast3", ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AddBroadcast4)
{
   SofieReference ref = readReference("AddBroadcast4");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "AddBroadcast4", ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AddBroadcast5)
{
   SofieReference ref = readReference("AddBroadcast5");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "AddBroadcast5", ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AddBroadcast6)
{
   SofieReference ref = readReference("AddBroadcast6");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "AddBroadcast6", ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AddBroadcast7)
{
   SofieReference ref = readReference("AddBroadcast7");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "AddBroadcast7", ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Concat0D) {
   // input
   std::vector<float> input({1.40519865e+00, -2.87660856e-01});
   std::vector<float> expected_output({1.40519865e+00, -2.87660856e-01, 1.40519865e+00, -2.87660856e-01});
   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Concat_0D", input);

   expectNear(output, expected_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, LayerNormalization2d)
{
   SofieReference ref = readReference("LayerNormalization2d");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "LayerNormalization2d", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LayerNormalization4d)
{
   SofieReference ref = readReference("LayerNormalization4d");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "LayerNormalization4d", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Equal)
{
   SofieReference ref = readReference("Equal");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::uint8_t>, "Equal", ref.f32("input0"), ref.f32("input1"));

   expectEqual(output, ref.u8("output0"));
}

TEST(ONNX, LessOrEqual)
{
   SofieReference ref = readReference("LessOrEqual");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::uint8_t>, "LessOrEqual", ref.f32("input0"), ref.f32("input1"));

   expectEqual(output, ref.u8("output0"));
}

TEST(ONNX, GreaterOrEqual)
{
   SofieReference ref = readReference("GreaterOrEqual");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::uint8_t>, "GreaterOrEqual", ref.f32("input0"), ref.f32("input1"));

   expectEqual(output, ref.u8("output0"));
}

TEST(ONNX, Greater)
{
   SofieReference ref = readReference("Greater");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::uint8_t>, "Greater", ref.f32("input0"), ref.f32("input1"));

   expectEqual(output, ref.u8("output0"));
}

TEST(ONNX, Less)
{
   SofieReference ref = readReference("Less");

   ASSERT_INCLUDE_AND_RUN(std::vector<std::uint8_t>, "Less", ref.f32("input0"), ref.f32("input1"));

   expectEqual(output, ref.u8("output0"));
}

TEST(ONNX, ExpandSameSize)
{
   SofieReference ref = readReference("ExpandSameSize");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ExpandSameSize", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ExpandDiffSize)
{
   SofieReference ref = readReference("ExpandDiffSize");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ExpandDiffSize", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GatherAxis0)
{
   SofieReference ref = readReference("GatherAxis0");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "GatherAxis0", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GatherAxis1)
{
   SofieReference ref = readReference("GatherAxis1");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "GatherAxis1", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GatherAxis2)
{
   SofieReference ref = readReference("GatherAxis2");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "GatherAxis2", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GatherAxis3)
{
   SofieReference ref = readReference("GatherAxis3");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "GatherAxis3", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Gather2d)
{
   SofieReference ref = readReference("Gather2d");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Gather2d", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GatherNegativeIndices)
{
   SofieReference ref = readReference("GatherNegativeIndices");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "GatherNegativeIndices", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Slice)
{
   SofieReference ref = readReference("Slice");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Slice", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Slice_Default_Axis)
{
   SofieReference ref = readReference("Slice_Default_Axis");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Slice_Default_Axis", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Slice_Default_Steps)
{
   SofieReference ref = readReference("Slice_Default_Steps");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Slice_Default_Steps", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Slice_Neg)
{
   SofieReference ref = readReference("Slice_Neg");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Slice_Neg", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}
TEST(ONNX, RangeFloat)
{
   SofieReference ref = readReference("RangeFloat");

   ASSERT_INCLUDE_AND_RUN_SESSION_ARGS(
      std::vector<float>,
      "RangeFloat",
      "\"RangeFloat_FromONNX.dat\", 5",
      ref.f32("input0"),
      ref.f32("input1"),
      ref.f32("input2"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RangeInt)
{
   SofieReference ref = readReference("RangeInt");

   ASSERT_INCLUDE_AND_RUN_SESSION_ARGS(
      std::vector<int64_t>,
      "RangeInt",
      "\"RangeInt_FromONNX.dat\", 5",
      ref.i64("input0"),
      ref.i64("input1"),
      ref.i64("input2"));

   expectEqual(output, ref.i64("output0"));
}
TEST(ONNX, Tile5D)
{
   SofieReference ref = readReference("Tile5D");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Tile5D", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}
TEST(ONNX, Pad) {
   // add constant pad values of zeros
   // input tensor [1,2,2] and pad in (1,0),(0,1),(2,1) -> with shape (2,3,5)
   std::vector<float> input = {1,2,3,4};
   std::vector<float> correct = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 3,
       4, 0, 0, 0, 0, 0, 0, 0};
   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Pad", input);

   expectEqual(output, correct);
}
TEST(ONNX, Where) {
   // test of Where using [[1,2]] and [[3,4],[5,6],[7,8]] with condition [[true],[false],[true]] -> [[1,2],[5,6],[1,2]]
   // test also the broadcast of boolean tensors
   std::vector<float> input1 = {1,2};
   std::vector<float> input2 = {3,4,5,6};
   std::vector<uint8_t> cond = {true, false, true};
   std::vector<float> correct = {1,2,5,6,1,2};
   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Where", input1, input2, cond);

   expectEqual(output, correct);
}

TEST(ONNX, Sin)
{
   // Preparing some random input
   std::vector<float> input({
     -0.786738,-0.197796,-0.187787,0.142758,0.876096,-0.653239,0.145444,-1.107658,2.259171,-0.947054,-0.506689,1.801250
   });

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Sin", input);

   std::vector<float> correct_output;
   for (float x : input)
      correct_output.push_back(std::sin(x));

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Cos)
{
   // Preparing the random input
   std::vector<float> input({
     1.152504,-1.459324,0.691594,0.347690,-1.307323,1.832516,-1.261772,0.014224,1.311477,1.147405,-0.567206,-0.530606
   });

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Cos", input);

   std::vector<float> correct_output;
   for (float x : input)
      correct_output.push_back(std::cos(x));

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Abs)
{
   // Preparing the random input
   std::vector<float> input({1.,-2.,-3,4,-5.,6});

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Abs", input);

   std::vector<float> correct_output;
   for (float x : input)
      correct_output.push_back(std::abs(x));

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Softplus)
{
   // Inputs spanning stable region, threshold boundary, and overflow-prone range
   std::vector<float> input({0.1f, -0.2f, 100.0f, 89.0f, 0.0f, 50.0f});

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Softplus", input);

   ASSERT_EQ(output.size(), input.size());

   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_FALSE(std::isinf(output[i])) << "Inf at input=" << input[i];
      EXPECT_FALSE(std::isnan(output[i])) << "NaN at input=" << input[i];
      // For large positive x (>= 20.0), softplus(x) ≈ x
      if (input[i] >= 20.0f) {
         EXPECT_NEAR(output[i], input[i], DEFAULT_TOLERANCE);
      } else {
         float exp_value = std::log1p(std::exp(input[i]));
         EXPECT_LE(std::abs(output[i] - exp_value), DEFAULT_TOLERANCE);
      }
   }
}
// tests of Einsum operator
TEST(ONNX, Einsum_matmul)
{
   std::vector<float> input1{1, 2, 3, 4};
   std::vector<float> input2{5, 6, 7, 8};
   std::vector<float> correct_output = {19, 22, 43, 50};

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Einsum_matmul", input1, input2);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}
// test dot prod using Einsum
TEST(ONNX, Einsum_dotprod)
{
   std::vector<float> input1{1, 2, 3};
   std::vector<float> input2{5, 6, 7};
   std::vector<float> correct_output {5 +  12 + 21};

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Einsum_dotprod", input1, input2);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}
// test tensor contraction of rank 3 tensors
TEST(ONNX, Einsum_3)
{
   // test abc,abd->ad   [2,2,3] , [2,2,3] -> [2,3]
   std::vector<float> input1 {1.,2.,3,4,5,6,7,8,9,10,11,12};
   std::vector<float> input2 {1.,2.,3,4,5,6,7,8,9,10,11,12};
   std::vector<float> correct_output {66. , 87. , 108., 498.,  555., 612. };

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Einsum_3", input1, input2);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}
// test tensor contraction of rank 4 tensors
TEST(ONNX, Einsum_4)
{
   // test abcd,abed->abce  [2,1,2,3] , [2,1,3,3] -> [2,1,2,3]
   std::vector<float> input1 {1.,2.,3,4,5,6,7,8,9,10,11,12};
   std::vector<float> input2 {1.,2.,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
   std::vector<float> correct_output { 14., 32.,  50., 32.,  77.,  122.,
                                      266., 338., 410., 365., 464., 563. };

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Einsum_4", input1, input2);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}
TEST(ONNX, RandomUniform)
{
   // output of gRandom->Uniform(10,20) with seed 111 - > shape(2,3)
   std::vector<float> correct_output = {16.1217, 11.2076, 11.6907, 13.0179, 14.3606, 18.5391};

   ASSERT_INCLUDE_AND_RUN_0(std::vector<float>, "RandomUniform");

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, RandomNormal)
{
    // output of gRandom->Gaus(1,3) with seed 111 - > shape(2,3)
   std::vector<float> correct_output = {-0.808389, -0.985581, 0.616354, 2.1887, 1.13927, -0.228048};

   ASSERT_INCLUDE_AND_RUN_0(std::vector<float>, "RandomNormal");

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Split_0)
{
   // split in axis 0  in 2 tensor {2,2,3}
   std::vector<float> input {1.,2.,3,4,5,6,7,8,9,10,11,12};
   std::vector<std::vector<float>> correct_output ={ {1,2,3,4,5,6}, {7,8,9,10,11,12} };

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "Split_0", input);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Split_1)
{
   // split in axis 1  in 2 tensor {2,2,3}
   std::vector<float> input {1.,2.,3,4,5,6,7,8,9,10,11,12};
   std::vector<std::vector<float>> correct_output ={ {1,2,3,7,8,9}, {4,5,6,10,11,12} };

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "Split_1", input);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Split_2)
{
   // split in axis 2  in 2 tensor {2,2,3} -> { 2,2,2} and {2,2,1}
   std::vector<float> input {1.,2.,3,4,5,6,7,8,9,10,11,12};
   std::vector<std::vector<float>> correct_output ={ {1,2,4,5,7,8,10,11}, {3,6,9,12} };

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<float>>, "Split_2", input);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, ScatterElements)
{
   // test scatter elements (similar test as in ONNX doc)
   std::vector<float> input(9, 0.);    // input tensor shape is (3.3)
   std::vector<int64_t> indices = { 1, 0, 2, 0, 2, 1};
   std::vector<float> updates = { 1, 1.1, 1.2, 2, 2.1, 2.2};
   std::vector<float> correct_output = {2, 1.1, 0., 1., 0., 2.2, 0., 2.1, 1.2 };

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ScatterElements", input, indices, updates);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, MatMul_Stacked)
{
   // test stacked matrix multiplication with same second matrix
   std::vector<float> input1 = {1,2,3,4,5,6,7,8};    // input tensor shape is (2,2,2)
   std::vector<float> input2 = {2,3};                // shape is (2,1)

   std::vector<float> correct_output = {8,18, 28,38};

   // model is dynamic , use N = 2
   ASSERT_INCLUDE_AND_RUN_SESSION_ARGS(std::vector<float>, "MatMul_Stacked", "\"MatMul_Stacked_FromONNX.dat\", 2", 2, input1, input2);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, MatMul_Stacked2)
{
   // test stacked matrix multiplication with different second matrix
   std::vector<float> input1 = {1,2,3,4,5,6,7,8};    // input tensor shape is (2,2,2)
   std::vector<float> input2 = {2,3,3,2};                // shape is (2,2,1)

   std::vector<float> correct_output = {8,18, 27,37};

   // model is dynamic , use N = 2
   ASSERT_INCLUDE_AND_RUN_SESSION_ARGS(std::vector<float>, "MatMul_Stacked2", "\"MatMul_Stacked2_FromONNX.dat\", 2", 2, input1, input2);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, GatherND_1)
{
   // test  gatherND elements
   std::vector<float> input(18, 0.);    // input tensor shape is (2, 3, 3)
   std::iota(input.begin(), input.end(), 1.);
   // input : {1,2,3},{4,5,6},{7,8,9}  {10,11,12}{13,14,15}{16,17,18}
   std::vector<int64_t> indices = { 1, 0, 2,   0, 2, 1};  // get x(1,0,2) and x(0,2,1)
   std::vector<float> correct_output = {12, 8};

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "GatherND_1", input, indices);

   expectEqual(output, correct_output);
}

TEST(ONNX, GatherND_2)
{
   // test GatherND using slices
   std::vector<float> input(18, 0.);    // input tensor shape is (2, 3, 3)
   // input : {1,2,3},{4,5,6},{7,8,9}......
   std::iota(input.begin(), input.end(), 1.);  // { 1,...,18}
   std::vector<int64_t> indices = { 1, 1, 0, 2}; // get x(1,1,:) and x(0,2:)
   std::vector<float> correct_output = {13,14,15, 7,8,9};

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "GatherND_2", input, indices);

   expectEqual(output, correct_output);
}

TEST(ONNX, GatherND_3)
{
   // test GatherND elements using batch size as first dim (bs=2)
   std::vector<float> input(24, 0.);    // input tensor shape is (2, 3, 2, 2)
   std::iota(input.begin(), input.end(), 1.);  // { 1,...,24}
   std::vector<int64_t> indices = { 2, 0, 0, 1}; // shape is (2,2,1)
   // indices are { [[2],[0]] , [[0],[1]]}  :
   // data[0,2,:] data[0,0:] ,  data[1,0:] data[1,1,:]
   std::vector<float> correct_output = {9,10,11,12, 1,2,3,4, 13,14,15,16, 17,18,19,20};

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "GatherND_3", input, indices);

   expectEqual(output, correct_output);
}

TEST(ONNX, NonZero)
{
   // test with input uint8_t   (note int8_t is not supported in the test_helper code)
   std::vector<uint8_t> input = {0,1,0, 1,1,0, 0,0,1, 0,1,1 }; // shape is (2x2x3)
   // output is tensor shape { 3, number of non zeros}
   std::vector<int64_t> correct_output = { 0,0,0,1,1,1 ,   0,1,1,0,1,1 ,    1,0,1,2,1,2 };

   ASSERT_INCLUDE_AND_RUN(std::vector<int64_t>, "NonZero", input);

   expectEqual(output, correct_output);
}

TEST(ONNX, NonZero_Constant)
{
   // input is a constant tensor in the model
   // output is tensor shape { 3, number of non zeros}
   std::vector<int64_t> correct_output = { 0,0,0,1,1,1 ,   0,1,1,0,1,1 ,    1,0,1,2,1,2 };

   ASSERT_INCLUDE_AND_RUN_0(std::vector<int64_t>, "NonZero_Constant");

   expectEqual(output, correct_output);
}
TEST(ONNX, IsInf)
{
   // expected input
   std::vector<float> input = { 1, static_cast<float>(1./0.), 2.};
   std::vector<uint8_t> correct_output = { 0,1,0 };

   // not cannot use input.size() in string because input symbol  will not be visible when running inference
   ASSERT_INCLUDE_AND_RUN_SESSION_ARGS(std::vector<uint8_t>, "IsInf",std::string("\"\", ") + std::to_string(input.size()), input.size(),input);

   expectEqual(output, correct_output);
}

TEST(ONNX, NotIsNaN)
{
   // expected input
   std::vector<float> input = { 1, static_cast<float>(0./0.), 2.};
   std::vector<uint8_t> correct_output = { 1,0,1 };

   ASSERT_INCLUDE_AND_RUN_SESSION_ARGS(std::vector<uint8_t>, "NotIsNaN",std::string("\"\", ") + std::to_string(input.size()), input.size(),input);

   expectEqual(output, correct_output);
}

TEST(ONNX, ScatterND_1)
{
   // test 1-D scatter (k=1, scalar slice)
   std::vector<float> input = {1.,2.,3.,4.,5.};  // shape {5}
   std::vector<int64_t> indices = { 0, 2, 4};    // shape {3,1}
   std::vector<float> updates = { 10.,30.,50.};  // shape {3}
   std::vector<float> correct_output = {10., 2., 30., 4., 50.};

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ScatterND_1", input, indices, updates);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, ScatterND_2)
{
   // test 2-d Scatter - scatter rows - reduction = 'add
   std::vector<float> input = {1.,1.,2.,2.,3.,3.};  // shape {3,2}
   std::vector<int64_t> indices = { 0, 1};          // shape {2,1}
   std::vector<float> updates = { 10.,10.,20.,20.};  // shape { 2,2}
   std::vector<float> correct_output = {11., 11., 22., 22., 3., 3.};

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ScatterND_2", input, indices, updates);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, ScatterND_3)
{
   // test element wise scatter (k==rank input)  reduction = 'mul'
   std::vector<float> input = {1.,2.,3.,4.};  // shape {2,2}
   std::vector<int64_t> indices = { 0,0, 1,1};          // shape {2,2}
   std::vector<float> updates = { 11.,22.};  // shape { 2}
   std::vector<float> correct_output = {11., 2., 3., 88.};

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "ScatterND_3", input, indices, updates);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Clip)
{
   // test Clip operator : input is [N,2,2] use N= 2 using min/max of -1,1
   std::vector<float> input = {-2.0,  0.5, 1.5, -0.3, 0.0,  3.0, -1.5,  0.8};
   std::vector<float> correct_output1 = {-1, 0.5, 1., -0.3, 0., 1.0, -1, 0.8};
   std::vector<float> correct_output2 = {-1, 0.5, 1.5, -0.3, 0., 3.0, -1, 0.8};

   ASSERT_INCLUDE_AND_RUN_SESSION_ARGS(std::vector<std::vector<float>>, "Clip", "\"Clip_FromONNX.dat\", 2", 2, input);

   ASSERT_EQ(output.size(), 2u);
   expectNear(output[0], correct_output1, DEFAULT_TOLERANCE);
   expectNear(output[1], correct_output2, DEFAULT_TOLERANCE);
}

TEST(ONNX, Gelu)
{
   SofieReference ref = readReference("Gelu");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Gelu", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Swish)
{
   SofieReference ref = readReference("Swish");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Swish", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, HardSigmoid)
{
   SofieReference ref = readReference("HardSigmoid");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "HardSigmoid", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, HardSwish)
{
   SofieReference ref = readReference("HardSwish");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "HardSwish", ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ComparisonBroadcast)
{
   // A shape [1, 4]
   std::vector<float> input_A = {0.0f, 1.0f, 2.0f, 3.0f};

   // B shape [4]
   std::vector<float> input_B = {4.0f, 4.0f, 2.0f, 2.0f};

   // (A < B)
   std::vector<uint8_t> expected_output_less = {1, 1, 0, 0};

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<uint8_t>>, "Comparison_broadcast", input_A, input_B);

   ASSERT_EQ(output.size(), 3u);
   const std::vector<uint8_t> &output_less = output[2];

   expectEqual(output_less, expected_output_less);
}

TEST(ONNX, ComparisonBroadcast3d)
{
   std::vector<float> input_A = {1.0f, 6.0f, 2.0f, 9.0f, 0.0f, 5.0f, 3.0f, 1.0f,
                                 2.0f, 4.0f, 4.0f, 2.0f, 1.0f, 7.0f, 0.0f, 3.0f};

   std::vector<float> input_B = {1.0f, 5.0f, 3.0f, 2.0f};

   // (A > B)
   // [
   //   [[F, T, F, T], [F, F, F, F]], -> {0,1,0,1, 0,0,0,0}
   //   [[T, F, T, F], [F, T, F, T]]  -> {1,0,1,0, 0,1,0,1}
   // ]
   std::vector<uint8_t> expected_greater = {0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1};

   // (A == B)
   // [
   //   [[T, F, F, F], [F, T, T, F]], -> {1,0,0,0, 0,1,1,0}
   //   [[F, F, F, T], [T, F, F, F]]  -> {0,0,0,1, 1,0,0,0}
   // ]
   std::vector<uint8_t> expected_equal = {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0};

   // (A < B)
   // [
   //   [[F, F, T, F], [T, F, F, T]], -> {0,0,1,0, 1,0,0,1}
   //   [[F, T, F, F], [F, F, T, F]]  -> {0,1,0,0, 0,0,1,0}
   // ]
   std::vector<uint8_t> expected_less = {0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0};

   ASSERT_INCLUDE_AND_RUN(std::vector<std::vector<uint8_t>>, "Comparison_broadcast_3d", input_A, input_B);

   ASSERT_EQ(output.size(), 3);

   const std::vector<uint8_t> &output_greater = output[0];
   const std::vector<uint8_t> &output_equal = output[1];
   const std::vector<uint8_t> &output_less = output[2];

   ASSERT_EQ(output_greater, expected_greater);
   ASSERT_EQ(output_equal, expected_equal);
   ASSERT_EQ(output_less, expected_less);
}
