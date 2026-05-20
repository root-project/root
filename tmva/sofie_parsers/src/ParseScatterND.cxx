#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_ScatterND.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseScatterND = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {

   if (nodeproto.input_size() != 3) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser ScatterND op has invalid input size");
   }
   // data is input 0
   for (int i = 0; i < 3; i++) {
      if (!parser.IsRegisteredTensorType(nodeproto.input(i)))
         throw std::runtime_error("TMVA::SOFIE ONNX Parser ScatterND op has input tensor " +  nodeproto.input(i)
                                + " but its type is not yet registered");
   }

   ETensorType input_type = parser.GetTensorType(nodeproto.input(0));
   // type of update tensor (input(2)) must be the same of data tensor (input(0)
   if (parser.GetTensorType(nodeproto.input(2)) != input_type) {
      throw std::runtime_error("TMVA::SOFIE ONNX parser ScatterND op has input tensors of different types: " +
                  nodeproto.input(2) + " : " + ConvertTypeToString(parser.GetTensorType(nodeproto.input(2))) +
                     " and " +  nodeproto.input(0) + " : " + ConvertTypeToString(input_type));
   }

   std::string reduction;
   for (int i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "reduction")
         reduction = nodeproto.attribute(i).s();
   }

   std::string output_name = nodeproto.output(0);

   auto op = std::make_unique<ROperator_ScatterND>(nodeproto.input(0), nodeproto.input(1), nodeproto.input(2),
                                          output_name, reduction);

   // Infer the output type
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};


} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
