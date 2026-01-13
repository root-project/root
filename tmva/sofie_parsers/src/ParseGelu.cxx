#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Gelu.hxx"
#include "onnx_proto3.pb.h"

#include <stdexcept>
#include <string>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseGelu = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   // --- Handle ONNX Gelu attribute: approximate ---
   // ONNX Gelu has attribute "approximate": "none" (default) or "tanh"
   std::string approximate = "none";
   for (const auto &attr : nodeproto.attribute()) {
      if (attr.name() == "approximate") {
         if (attr.type() != onnx::AttributeProto::STRING)
            throw std::runtime_error(
               "TMVA::SOFIE ONNX Parser: Gelu attribute 'approximate' must be a string");

         approximate = attr.s();
      }
   }

   if (approximate != "none") {
      if (approximate == "tanh") {
         throw std::runtime_error(
            "TMVA::SOFIE ONNX Parser: Gelu attribute approximate='tanh' not supported yet");
      }

      throw std::runtime_error(
         "TMVA::SOFIE ONNX Parser: Gelu attribute approximate='" + approximate +
         "' not supported (expected 'none' or 'tanh')");
   }

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Gelu op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT: op.reset(new ROperator_Gelu<float>(input_name, output_name)); break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Gelu does not yet support input type " +
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
