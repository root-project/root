#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Softmax.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseSoftmax = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Softmax op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   int64_t attr_axis = -1;
   if (nodeproto.attribute_size() == 1 && nodeproto.attribute(0).name() == "axis")
      attr_axis = nodeproto.attribute(0).i();

   switch (input_type) {
   case ETensorType::FLOAT: op.reset(new ROperator_Softmax<float>(attr_axis, input_name, output_name)); break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Softmax does not yet support input type " +
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
