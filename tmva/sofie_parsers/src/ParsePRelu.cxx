#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_PRelu.hxx"
#include "onnx.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParsePRelu = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;
   ETensorType slope_type;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser PRelu op has input tensor " + input_name +
                               " but its type is not yet registered");
   }
   auto slope_name = nodeproto.input(1);
   if (parser.IsRegisteredTensorType(slope_name)) {
      slope_type = parser.GetTensorType(slope_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser PRelu op has slope tensor " + slope_name +
                               " but its type is not yet registered");
   }
   if (slope_type != input_type) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser PRelu op has slope tensor " + slope_name + " of type " +
                               ConvertTypeToString(slope_type) + " but input tensor " + input_name + " is of type " +
                               ConvertTypeToString(input_type) + "; they must be the same");
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   // PRelu has no attributes (the legacy opset-1 'consumed_inputs' is ignored)
   switch (input_type) {
   case ETensorType::FLOAT: op.reset(new ROperator_PRelu<float>(input_name, slope_name, output_name)); break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator PRelu does not yet support input type " +
                               ConvertTypeToString(input_type));
   }
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
