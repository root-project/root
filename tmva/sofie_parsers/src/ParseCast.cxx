#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Cast.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseCast = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   auto input_name = nodeproto.input(0);
   if (!parser.IsRegisteredTensorType(input_name)) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Cast op has input tensor" + input_name +
                               "  but its type is not yet registered");
   }

   ETensorType output_type = ETensorType::UNDEFINED;

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "to") {
         output_type = static_cast<ETensorType>(nodeproto.attribute(i).i());
      }
   }
   if (output_type == ETensorType::UNDEFINED)
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Cast op has invalid output type");

   std::string output_name = nodeproto.output(0);
   auto op = std::make_unique<ROperator_Cast>(output_type, nodeproto.input(0), output_name);

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, output_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
