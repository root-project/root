#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Where.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseWhere = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {

   if (nodeproto.input_size() != 3) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Where op has invalid input size");
   }
   // condition boolean vector is input 0
   if (!parser.IsRegisteredTensorType(nodeproto.input(1))){
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Where op has input tensor " +  nodeproto.input(1)
                                + " but its type is not yet registered");
   }
   if (!parser.IsRegisteredTensorType(nodeproto.input(2))){
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Where op has input tensor " +  nodeproto.input(2)
                                + " but its type is not yet registered");
   }
   ETensorType input_type = parser.GetTensorType(nodeproto.input(1));
   if (parser.GetTensorType(nodeproto.input(2)) != input_type) {
      throw std::runtime_error("TMVA::SOFIE ONNX parser Where op has input tensors of different types: " +
                  nodeproto.input(2) + " : " + ConvertTypeToString(parser.GetTensorType(nodeproto.input(2))) +
                     " and " +  nodeproto.input(1) + " : " + ConvertTypeToString(input_type));
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_Where<float>(nodeproto.input(1), nodeproto.input(2), nodeproto.input(0), output_name));
      break;
   case ETensorType::INT64:
      op.reset(new ROperator_Where<int64_t>(nodeproto.input(1), nodeproto.input(2), nodeproto.input(0), output_name));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Where Operator does not yet support input type " +
                               std::to_string(static_cast<int>(input_type)));
   }

   // Infer the output type
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};


} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
