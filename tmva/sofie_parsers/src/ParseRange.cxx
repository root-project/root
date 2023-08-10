#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Range.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseRange = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   auto start = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(start)) {
      input_type = parser.GetTensorType(start);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Tanh op has input tensor" + start +
                               " but its type is not yet registered");
   }

   auto limit = nodeproto.input(1);
   if (!parser.IsRegisteredTensorType(limit)) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Tanh op has input tensor" + limit +
                               " but its type is not yet registered");
   }

   auto delta = nodeproto.input(2);
   if (!parser.IsRegisteredTensorType(delta)) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Tanh op has input tensor" + delta +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT: op.reset(new ROperator_Range<float>(start, limit, delta, output_name)); break;
   case ETensorType::INT64: op.reset(new ROperator_Range<int64_t>(start, limit, delta, output_name)); break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Range does not yet support input type " +
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
