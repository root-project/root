#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Neg.hxx"
#include "onnx_proto3.pb.h"
#include <stdexcept>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseNeg = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   std::unique_ptr<ROperator> op;

   ETensorType input_type = ETensorType::UNDEFINED;
   const std::string input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Neg op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   const std::string output_name = nodeproto.output(0);
   switch (input_type) {
   case ETensorType::FLOAT: op.reset(new ROperator_Neg<float>(input_name, output_name)); break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Unary Operator does not support imput type " +
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
