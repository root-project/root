#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Expand.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseExpand = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   std::unique_ptr<ROperator> op;

   ETensorType input_type = ETensorType::UNDEFINED;
   const std::string input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error(
        "TMVA::SOFIE ONNX Parser Expand op has input tensor " + input_name +
        " but its type is not yet registered");
   }

   const std::string shape_name = nodeproto.input(1);
   if (parser.IsRegisteredTensorType(shape_name)) {
      if (parser.GetTensorType(shape_name) != ETensorType::INT64) {
         throw
            std::runtime_error("TMVA::SOFIE - ONNX Parser Expand Op shape type not supported");
      }
   } else {
      throw std::runtime_error(
        "TMVA::SOFIE ONNX Parser Sign op has input tensor " + input_name +
        " but its type is not yet registered");
   }

   const std::string output_name = nodeproto.output(0);
   switch (input_type) {
      case ETensorType::FLOAT:
         op.reset(new ROperator_Expand<float>(input_name, shape_name, output_name));
         break;
      case ETensorType::INT64:
         op.reset(new ROperator_Expand<int64_t>(input_name, shape_name, output_name));
         break;
      default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Expand Operator does "
                             "not support input type " +
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
