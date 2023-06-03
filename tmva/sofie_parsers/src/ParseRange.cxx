#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Range.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseRange = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {

ETensorType input_type;

   for (int i = 0; i < 3; ++i) {
      auto input_name = nodeproto.input(i);
      if (parser.IsRegisteredTensorType(input_name)) {
         // according to ONNX all inputs have same type
         if (i == 0)
            input_type = parser.GetTensorType(input_name);
         else
            if (input_type != parser.GetTensorType(input_name)) {
               throw
                  std::runtime_error("TMVA::SOFIE ONNX parser Range op has input tensors of different types");
            }
      } else {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Range op has input tensor " + input_name +
                                  " but its type is not yet registered");
      }
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

  switch (input_type) {
   case ETensorType::UNDEFINED: //Inputs are not a tensor
      op.reset(new ROperator_Range<float>(nodeproto.input(0), nodeproto.input(1), nodeproto.input(2), output_name));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Range Operator does not yet support input type " +
                               std::to_string(static_cast<int>(input_type)));
   }

   // Infer the output type
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
