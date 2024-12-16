#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Einsum.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseEinsum = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {

   ETensorType input_type = ETensorType::UNDEFINED;
   int input_size = nodeproto.input_size();
   std::vector<std::string> input_names(input_size);
   for (int i = 0; i < input_size; i++) {
      if (!parser.IsRegisteredTensorType(nodeproto.input(i))){
        throw std::runtime_error("TMVA::SOFIE ONNX Parser Einsum op has input tensor " +  nodeproto.input(i)
                                + " but its type is not yet registered");
      }
      if (i == 0)
         input_type = parser.GetTensorType(nodeproto.input(0));
      if (parser.GetTensorType(nodeproto.input(i)) != input_type) {
         throw std::runtime_error("TMVA::SOFIE ONNX parser Einsum op has input tensors of different types: " +
                  nodeproto.input(i) + " : " + ConvertTypeToString(parser.GetTensorType(nodeproto.input(2))) +
                     " and " +  nodeproto.input(0) + " : " + ConvertTypeToString(input_type));
      }
      input_names[i] = nodeproto.input(i);
   }

   // equation attribute should be existing
   if (nodeproto.attribute_size() == 0)
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Einsum op has  no attribute defining the equation");
   if (nodeproto.attribute(0).name() != "equation")
       throw std::runtime_error("TMVA::SOFIE ONNX Parser Einsum op has wrong attribute name: " + nodeproto.attribute(0).name());
   std::string equation = nodeproto.attribute(0).s();

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);



   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_Einsum<float>(equation, input_names, output_name));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Einsum Operator does not yet support input type " +
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
