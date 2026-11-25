#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_ArgMax.hxx"
#include "onnx.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseArgMax = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   auto input_name = nodeproto.input(0);
   if (!parser.IsRegisteredTensorType(input_name)) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser ArgMax op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   // default values as in the ONNX specification
   int64_t attr_axis = 0;
   int attr_keepdims = 1;
   int attr_select_last_index = 0;
   for (int i = 0; i < nodeproto.attribute_size(); i++) {
      const std::string &attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "axis")
         attr_axis = nodeproto.attribute(i).i();
      else if (attribute_name == "keepdims")
         attr_keepdims = nodeproto.attribute(i).i();
      else if (attribute_name == "select_last_index")
         attr_select_last_index = nodeproto.attribute(i).i();
   }

   std::string output_name = nodeproto.output(0);
   std::unique_ptr<ROperator> op(
      new ROperator_ArgMax(attr_axis, attr_keepdims, attr_select_last_index, input_name, output_name));

   // the output contains indices, so it is always INT64
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, ETensorType::INT64);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
