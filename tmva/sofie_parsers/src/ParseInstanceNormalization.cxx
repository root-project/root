#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_InstanceNormalization.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseInstanceNormalization = [](RModelParser_ONNX &parser,
                                                    const onnx::NodeProto &nodeproto) -> std::unique_ptr<ROperator> {
   ETensorType input_type = ETensorType::UNDEFINED;
   const std::string input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser InstanceNormalization op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   float epsilon = 1e-5;
   for (int64_t i = 0; i < nodeproto.attribute_size(); i++) {
      if (nodeproto.attribute(i).name() == "epsilon") {
         epsilon = nodeproto.attribute(i).f();
      }
   }

   // Inputs: X (0), scale (1), B (2)
   const std::string name_scale = nodeproto.input(1);
   const std::string name_bias = nodeproto.input(2);
   const std::string output_name = nodeproto.output(0);

   std::unique_ptr<ROperator> op;
   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_InstanceNormalization<float>(epsilon, input_name, name_scale, name_bias, output_name));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE ONNX parser Operator with input type " + ConvertTypeToString(input_type) +
                               " not supported.");
      break;
   }

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
