#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_NonZero.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseNonZero = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser NonZero op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   std::string output_name = nodeproto.output(0);

   std::unique_ptr<ROperator> op;
   switch (input_type) {
      case ETensorType::FLOAT:
         op.reset(new ROperator_NonZero<float>(input_name, output_name));
         break;
      case ETensorType::INT64:
         op.reset(new ROperator_NonZero<int64_t>(input_name, output_name));
         break;
      case ETensorType::INT32:
         op.reset(new ROperator_NonZero<int32_t>(input_name, output_name));
         break;
      case ETensorType::INT8:
         op.reset(new ROperator_NonZero<int8_t>(input_name, output_name));
         break;
      case ETensorType::BOOL:
         op.reset(new ROperator_NonZero<bool>(input_name, output_name));
         break;
      default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Relu does not yet support input type " +
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
