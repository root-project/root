#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Equal.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseEqual = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   std::unique_ptr<ROperator> op;

   ETensorType input_type = ETensorType::UNDEFINED;

   for (int i = 0; i < 2; ++i) {
      auto input_name = nodeproto.input(i);
      if (parser.IsRegisteredTensorType(input_name)) {
         // according to ONNX both inputs have same type
         if (i == 0)
            input_type = parser.GetTensorType(input_name);
         else
            if (input_type != parser.GetTensorType(input_name)) {
               throw
                  std::runtime_error("TMVA::SOFIE ONNX parser Equal op has input tensors of different types");
            }
      } else {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Equal op has input tensor " + input_name +
                                  " but its type is not yet registered");
      }
   }

   const std::string output_name = nodeproto.output(0);

   switch (input_type) {
      case ETensorType::FLOAT:
         op.reset(new ROperator_Equal<float>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Equal Operator does not yet support input type " +
                               std::to_string(static_cast<int>(input_type)));
   }

   // Infer the output type
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, ETensorType::BOOL);
   }

   return op;
};


} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
