#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_HardSigmoid.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseHardSigmoid = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser HardSigmoid op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;

   float attr_alpha = 0.2;
   float attr_beta = 0.5;

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "alpha")
         attr_alpha = nodeproto.attribute(i).f();
      if (attribute_name == "beta")
         attr_beta = nodeproto.attribute(i).f();
   }

   std::string output_name = nodeproto.output(0);
   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_HardSigmoid<float>(attr_alpha, attr_beta, input_name, output_name));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator HardSigmoid does not yet support input type " +
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
