#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Cast.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseCast = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;
   auto input_name = nodeproto.input(0);
   if (!parser.IsRegisteredTensorType(input_name)) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Cast op has input tensor" + input_name +
                               "  but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;
   std::string attr_type;

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "to")
         attr_type = ConvertTypeToString(static_cast<ETensorType>(nodeproto.attribute(i).i()));
   }

   std::string output_name = nodeproto.output(0);
   op.reset(new ROperator_Cast(attr_type, nodeproto.input(0), output_name));

   if (!parser.IsRegisteredTensorType(output_name)) {
      ETensorType output_type = ConvertStringToType(attr_type);
      parser.RegisterTensorType(output_name, output_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
