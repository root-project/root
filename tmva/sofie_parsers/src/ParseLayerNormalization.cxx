#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_LayerNormalization.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseLayerNormalization = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
-> std::unique_ptr<ROperator> {
   ETensorType input_type = ETensorType::UNDEFINED;
   const std::string input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser LayerNormalizaion op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   int64_t axis = -1;
   float epsilon = 1e-5;
   int64_t stash_type = 1;
   for (int64_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "axis") {
         axis = nodeproto.attribute(i).i();
      } else if (attribute_name == "epsilon") {
         epsilon = nodeproto.attribute(i).f();
      } else if (attribute_name == "stash_type") {
         stash_type = nodeproto.attribute(i).i();
      }
   }
   size_t input_size = nodeproto.input_size();
   std::string name_scale = "";
   if (input_size > 1) {
      name_scale = nodeproto.input(1);
   }
   std::string name_bias = "";
   if (input_size > 2) {
      name_bias = nodeproto.input(2);
   }

   const std::string output_name = nodeproto.output(0);
   size_t output_size = nodeproto.output_size();
   std::string name_mean = "";
   if (output_size > 1) {
      name_mean = nodeproto.output(1);
   }
   std::string name_std = "";
   if (output_size > 2) {
      name_std = nodeproto.output(2);
   }

   std::unique_ptr<ROperator> op;
   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_LayerNormalization<float>(axis, epsilon, stash_type, input_name, name_scale, name_bias,
                                                       output_name, name_mean, name_std));
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

}
} // namespace Experimental
} // namespace TMVA
