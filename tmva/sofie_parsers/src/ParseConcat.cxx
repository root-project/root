#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Concat.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseConcat = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type = ETensorType::UNDEFINED;
   std::vector<std::string> inputs;
   size_t size = nodeproto.input_size();
   inputs.reserve(size);
   for (int i = 0; i < nodeproto.input_size(); ++i) {
      auto input_name = nodeproto.input(i);
      if (parser.IsRegisteredTensorType(input_name)) {
         if (i == 0)
            input_type = parser.GetTensorType(input_name);
         else
            assert(parser.GetTensorType(input_name) == input_type);
      } else {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Concat op has input tensor" + input_name +
                                  " but its type is not yet registered");
      }
      inputs.emplace_back(input_name);
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   int attr_axis = 0;
   int attr_new_axis = 0;
   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "axis")
         attr_axis = nodeproto.attribute(i).i();
      else if (attribute_name == "new_axis") // this is for ConcatFromSequence (that is equivalent to np.stack)
         attr_new_axis = nodeproto.attribute(i).i();
   }
   //switch (input_type) {
   //case ETensorType::FLOAT:
   op.reset(new ROperator_Concat(inputs, attr_axis, attr_new_axis, output_name));
   //break;
   //default:
   //   throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Concat does not yet support input type " +
   //                            std::to_string(static_cast<int>(input_type)));
  // }

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }
   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
