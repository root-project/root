#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Gelu.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseGelu = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type = ETensorType::UNDEFINED;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser gelu op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   // Read optional 'approximate' attribute (default: "none")
   std::string approximate = "none";
   for (int i = 0; i < nodeproto.attribute_size(); i++) {
      const auto &attr = nodeproto.attribute(i);
      if (attr.name() == "approximate") {
         approximate = attr.s();
      }
   }

   std::string output_name = nodeproto.output(0);

   auto  op = std::make_unique<ROperator_Gelu>(input_name, output_name, approximate);

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
