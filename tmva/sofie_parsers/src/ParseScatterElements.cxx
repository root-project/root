#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_ScatterElements.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseScatterElements = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {

   if (nodeproto.input_size() != 3) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser ScatterElements op has invalid input size");
   }
   // data is input 0
   if (!parser.IsRegisteredTensorType(nodeproto.input(0))){
      throw std::runtime_error("TMVA::SOFIE ONNX Parser ScatterElements op has input tensor " +  nodeproto.input(0)
                                + " but its type is not yet registered");
   }
   if (!parser.IsRegisteredTensorType(nodeproto.input(1))){
      throw std::runtime_error("TMVA::SOFIE ONNX Parser ScatterElements op has input tensor " +  nodeproto.input(1)
                                + " but its type is not yet registered");
   }
   if (!parser.IsRegisteredTensorType(nodeproto.input(2))){
      throw std::runtime_error("TMVA::SOFIE ONNX Parser ScatterElements op has input tensor " +  nodeproto.input(2)
                                + " but its type is not yet registered");
   }
   ETensorType input_type = parser.GetTensorType(nodeproto.input(0));
   if (parser.GetTensorType(nodeproto.input(2)) != input_type) {
      throw std::runtime_error("TMVA::SOFIE ONNX parser ScatterElements op has input tensors of different types: " +
                  nodeproto.input(2) + " : " + ConvertTypeToString(parser.GetTensorType(nodeproto.input(2))) +
                     " and " +  nodeproto.input(0) + " : " + ConvertTypeToString(input_type));
   }

   int axis = 0;
   std::string reduction;
   for (int i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "axis")
         axis = nodeproto.attribute(i).i();
      else if (attribute_name == "reduction")
         reduction = nodeproto.attribute(i).s();
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   op.reset(new ROperator_ScatterElements(nodeproto.input(0), nodeproto.input(1), nodeproto.input(2),
                                          output_name, axis, reduction));

   // Infer the output type
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};


} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
