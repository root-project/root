#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_EyeLike.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseEyeLike = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Eyelike op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;

   int attr_dtype = static_cast<int>(ETensorType::FLOAT);
   int attr_k = 0;

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "dtype")
         attr_dtype = nodeproto.attribute(i).i();
      if(attribute_name == "k"){
         attr_k = nodeproto.attribute(i).i();
      }

   }

   std::string output_name = nodeproto.output(0);
   switch (input_type) {
   case ETensorType::FLOAT: op.reset(new ROperator_EyeLike<float>(attr_dtype, attr_k, input_name, output_name)); break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator EyeLike does not yet support input type " +
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