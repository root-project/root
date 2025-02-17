#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Pad.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParsePad = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   std::string input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Pad op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   if (nodeproto.input_size() < 2) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Pad op has invalid input size < 2");
   }

   // pads is second inputs
   std::string pads_name = nodeproto.input(1);
   if (!parser.IsRegisteredTensorType(pads_name)) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Pad op has input tensor" + pads_name +
                                  " but its type is not yet registered");
   }
   // in case of optional inputs
   std::string cvalue_name;
   if (nodeproto.input_size() > 2) {
      cvalue_name = nodeproto.input(2);
   }
   std::string axes_name;
   if (nodeproto.input_size() > 3) {
      axes_name = nodeproto.input(3);
   }

   // get  attributes
   std::string mode = "constant";
   if (nodeproto.attribute_size() > 0 ) {
      std::string attribute_name = nodeproto.attribute(0).name();
      if (attribute_name == "mode") {
         mode = nodeproto.attribute(0).s();
      }
   }
   std::string output_name = nodeproto.output(0);

   std::unique_ptr<ROperator> op;
   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_Pad<float>(input_name, pads_name, cvalue_name, axes_name, output_name, mode));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Pad does not yet support input type " +
                               std::to_string(static_cast<int>(input_type)));
   }

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
