#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Selu.hxx"
#include "onnx.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseSelu = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type = ETensorType::UNDEFINED;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser selu op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;

   // default values from the ONNX specification
   float attr_alpha = 1.67326319217681884765625;
   float attr_gamma = 1.05070102214813232421875;

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "alpha")
         attr_alpha = nodeproto.attribute(i).f();
      else if (attribute_name == "gamma")
         attr_gamma = nodeproto.attribute(i).f();
   }

   std::string output_name = nodeproto.output(0);
   switch (input_type) {
   case ETensorType::FLOAT: op.reset(new ROperator_Selu<float>(attr_alpha, attr_gamma, input_name, output_name)); break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Selu does not yet support input type " +
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
