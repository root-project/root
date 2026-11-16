#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_RMSNormalization.hxx"
#include "onnx.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseRMSNormalization = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;
   ETensorType scale_type;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser RMSNormalization op has input tensor " + input_name +
                               " but its type is not yet registered");
   }
   if (nodeproto.input_size() < 2) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser RMSNormalization op requires a scale input");
   }
   auto scale_name = nodeproto.input(1);
   if (parser.IsRegisteredTensorType(scale_name)) {
      scale_type = parser.GetTensorType(scale_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser RMSNormalization op has scale tensor " + scale_name +
                               " but its type is not yet registered");
   }
   if (scale_type != input_type) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser RMSNormalization op has scale tensor " + scale_name +
                               " of type " + ConvertTypeToString(scale_type) + " but input tensor " + input_name +
                               " is of type " + ConvertTypeToString(input_type) + "; they must be the same");
   }

   int64_t axis = -1;
   float epsilon = 1e-5;
   int64_t stash_type = 1;
   for (int64_t i = 0; i < nodeproto.attribute_size(); i++) {
      const std::string &attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "axis") {
         axis = nodeproto.attribute(i).i();
      } else if (attribute_name == "epsilon") {
         epsilon = nodeproto.attribute(i).f();
      } else if (attribute_name == "stash_type") {
         stash_type = nodeproto.attribute(i).i();
      }
   }
   // The operator accumulates in float, which is what stash_type = 1 (the default) asks for.
   if (stash_type != 1) {
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator RMSNormalization only supports stash_type = 1, "
                               "got " +
                               std::to_string(stash_type));
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_RMSNormalization<float>(axis, epsilon, input_name, scale_name, output_name));
      break;
   default:
      throw std::runtime_error(
         "TMVA::SOFIE - Unsupported - Operator RMSNormalization does not yet support input type " +
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
