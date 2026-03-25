#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Transpose.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseTranspose = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser tranpose op has input tensor" + input_name +
                               " but its type is not yet registered");
   }


   std::string output_name = nodeproto.output(0);
   std::vector<int_t> attr_perm;

   if (nodeproto.attribute_size() == 1) {
      attr_perm.assign(nodeproto.attribute(0).ints().begin(), nodeproto.attribute(0).ints().end());
   }

   auto op = std::make_unique<ROperator_Transpose>(attr_perm, nodeproto.input(0), nodeproto.output(0));

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
