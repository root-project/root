#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_BatchNormalization.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseBatchNormalization = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser BatchNorm op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);
   float fepsilon = 1e-05;
   float fmomentum = 0.9;
   std::size_t ftraining_mode = 0;

   switch (input_type) {
   case ETensorType::FLOAT:
      if (nodeproto.input_size() == 5) {
         op.reset(new ROperator_BatchNormalization<float>(fepsilon, fmomentum, ftraining_mode, nodeproto.input(0),
                                                          nodeproto.input(1), nodeproto.input(2), nodeproto.input(3),
                                                          nodeproto.input(4), output_name));
      }
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator BatchNorm does not yet support input type " +
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
