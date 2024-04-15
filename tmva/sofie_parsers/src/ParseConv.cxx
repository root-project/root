#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Conv.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseConv = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Conv op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;

   std::string attr_auto_pad = "NOTSET";
   std::vector<size_t> attr_dilations;
   size_t attr_group = 0;
   std::vector<size_t> attr_kernel_shape;
   std::vector<size_t> attr_pads;
   std::vector<size_t> attr_strides;

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "auto_pad") {
         attr_auto_pad = nodeproto.attribute(i).s();
      } else if (attribute_name == "dilations") {
         attr_dilations =
            std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attribute_name == "group") {
         attr_group = nodeproto.attribute(i).i();
      } else if (attribute_name == "kernel_shape") {
         attr_kernel_shape =
            std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attribute_name == "pads") {
         attr_pads = std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attribute_name == "strides") {
         attr_strides =
            std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else {
         std::cout << "TMVA::SOFIE Warning - Model Loading - Attribute " << attribute_name << " in OperatorNode "
                   << nodeproto.name() << " is not defined in ONNX IR and not applied!\n";
      }
   }

   std::string name_b = "";
   if (nodeproto.input_size() > 2) {
      name_b = nodeproto.input(2);
   }
   std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_Conv<float>(attr_auto_pad, attr_dilations, attr_group, attr_kernel_shape, attr_pads,
                                         attr_strides, nodeproto.input(0), nodeproto.input(1), name_b, output_name));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Conv does not yet support input type " +
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
