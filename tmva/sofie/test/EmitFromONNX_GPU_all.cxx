// Author: Federico Sossai
// Last modified: 2021/07/30
// Description:
//    SOFIE command line compiler.
//    This program is automatically run when the target 'TestCustomModelsFromONNX' is built.
//    Usage example: $./sofiec indir/mymodel.onnx outdir/myname.hxx

#include <iostream>

#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"

using namespace TMVA::Experimental::SOFIE;

int EmitModel(std::string filename, std::string outname){

   RModelParser_ONNX parser;
   RModel model = parser.Parse(filename);
   model.GenerateGPU();
   model.OutputGeneratedGPU(outname+"GPU_FromONNX.hxx");

   return 0;
}

int main(int argc, char *argv[]){

EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Add.onnx", "Add");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/AddBroadcast1.onnx", "AddBroadcast1");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/AddBroadcast2.onnx", "AddBroadcast2");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/AddBroadcast3.onnx", "AddBroadcast3");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/AddBroadcast4.onnx", "AddBroadcast4");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/AddBroadcast5.onnx", "AddBroadcast5");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/AddBroadcast6.onnx", "AddBroadcast6");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/AddBroadcast7.onnx", "AddBroadcast7");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/AvgPool.onnx", "AvgPool");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Cast.onnx", "Cast");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Concat_0D.onnx", "Concat_0D");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/ConvTranspose1d.onnx", "ConvTranspose1d");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/ConvTranspose2d.onnx", "ConvTranspose2d");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/ConvTransposeBias2d.onnx", "ConvTransposeBias2d");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/ConvTransposeBias2dBatched.onnx", "ConvTransposeBias2dBatched");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/ConvWithAsymmetricPadding.onnx", "ConvWithAsymmetricPadding");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/ConvWithAutopadSameLower.onnx", "ConvWithAutopadSameLower");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/ConvWithPadding.onnx", "ConvWithPadding");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/ConvWithStridesNoPadding.onnx", "ConvWithStridesNoPadding");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/ConvWithStridesPadding.onnx", "ConvWithStridesPadding");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/ConvWithoutPadding.onnx", "ConvWithoutPadding");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Div.onnx", "Div");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Exp.onnx", "Exp");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/ExpandDiffSize.onnx", "ExpandDiffSize");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/ExpandSameSize.onnx", "ExpandSameSize");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/GRUBatchwise.onnx", "GRUBatchwise");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/GRUBidirectional.onnx", "GRUBidirectional");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/GRUDefaults.onnx", "GRUDefaults");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/GRUInitialBias.onnx", "GRUInitialBias");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/GRUSeqLength.onnx", "GRUSeqLength");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Gather2d.onnx", "Gather2d");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/GatherAxis0.onnx", "GatherAxis0");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/GatherAxis1.onnx", "GatherAxis1");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/GatherAxis2.onnx", "GatherAxis2");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/GatherAxis3.onnx", "GatherAxis3");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/GatherNegativeIndices.onnx", "GatherNegativeIndices");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/LSTMBatchwise.onnx", "LSTMBatchwise");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/LSTMBidirectional.onnx", "LSTMBidirectional");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/LSTMDefaults.onnx", "LSTMDefaults");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/LSTMInitialBias.onnx", "LSTMInitialBias");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/LSTMPeepholes.onnx", "LSTMPeepholes");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/LayerNormalization2d.onnx", "LayerNormalization2d");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/LayerNormalization4d.onnx", "LayerNormalization4d");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/LinearWithLeakyRelu.onnx", "LinearWithLeakyRelu");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/LinearWithSelu.onnx", "LinearWithSelu");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/LinearWithSigmoid.onnx", "LinearWithSigmoid");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Linear_16.onnx", "Linear_16");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Linear_32.onnx", "Linear_32");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Linear_64.onnx", "Linear_64");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Max.onnx", "Max");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/MaxMultidirectionalBroadcast.onnx", "MaxMultidirectionalBroadcast");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/MaxPool1d.onnx", "MaxPool1d");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/MaxPool2d.onnx", "MaxPool2d");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/MaxPool3d.onnx", "MaxPool3d");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/MeanMultidirectionalBroadcast.onnx", "MeanMultidirectionalBroadcast");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/MinMultidirectionalBroadcast.onnx", "MinMultidirectionalBroadcast");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Mul.onnx", "Mul");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Neg.onnx", "Neg");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Pow.onnx", "Pow");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Pow_broadcast.onnx", "Pow_broadcast");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/RNNBatchwise.onnx", "RNNBatchwise");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/RNNBidirectional.onnx", "RNNBidirectional");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/RNNBidirectionalBatchwise.onnx", "RNNBidirectionalBatchwise");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/RNNDefaults.onnx", "RNNDefaults");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/RNNSeqLength.onnx", "RNNSeqLength");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/RNNSequence.onnx", "RNNSequence");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/RNNSequenceBatchwise.onnx", "RNNSequenceBatchwise");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Reciprocal.onnx", "Reciprocal");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/ReduceMean.onnx", "ReduceMean");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/ReduceProd.onnx", "ReduceProd");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Shape.onnx", "Shape");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Softmax1d.onnx", "Softmax1d");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Softmax2d.onnx", "Softmax2d");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Softmax3d.onnx", "Softmax3d");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Softmax4d.onnx", "Softmax4d");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Sqrt.onnx", "Sqrt");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Sub.onnx", "Sub");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/SumMultidirectionalBroadcast.onnx", "SumMultidirectionalBroadcast");
EmitModel( "/home/ionna/root/root_src/tmva/sofie/test/input_models/Tanh.onnx", "Tanh") ;

}

