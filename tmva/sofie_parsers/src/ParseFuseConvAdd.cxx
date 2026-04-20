#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Conv.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuseFuncSignature ParseFuseConvAdd = [](RModelParser_ONNX &parser, const onnx::NodeProto &convnode,
                                              const onnx::NodeProto &addnode) {
   if (addnode.input_size() != 2) {
      throw std::runtime_error("TMVA::SOFIE - Cannot fuse Conv and Add - Add input size is not 2");
   }
   std::string name_b;
   if (convnode.output(0) == addnode.input(0))
      name_b = addnode.input(1);
   else if (convnode.output(0) == addnode.input(1))
      name_b = addnode.input(0);
   else
      throw std::runtime_error("TMVA::SOFIE - Cannot fuse Conv and Add - output of Conv is not input to Add");

   ETensorType input_type;
   auto input_name = convnode.input(0);
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

   for (int_t i = 0; i < convnode.attribute_size(); i++) {
      std::string attribute_name = convnode.attribute(i).name();
      if (attribute_name == "auto_pad") {
         attr_auto_pad = convnode.attribute(i).s();
      } else if (attribute_name == "dilations") {
         attr_dilations = std::vector<size_t>({convnode.attribute(i).ints().begin(), convnode.attribute(i).ints().end()});
      } else if (attribute_name == "group") {
         attr_group = convnode.attribute(i).i();
      } else if (attribute_name == "kernel_shape") {
         attr_kernel_shape = std::vector<size_t>({convnode.attribute(i).ints().begin(), convnode.attribute(i).ints().end()});
      } else if (attribute_name == "pads") {
         attr_pads = std::vector<size_t>({convnode.attribute(i).ints().begin(), convnode.attribute(i).ints().end()});
      } else if (attribute_name == "strides") {
         attr_strides = std::vector<size_t>({convnode.attribute(i).ints().begin(), convnode.attribute(i).ints().end()});
      } else {
         std::cout << "TMVA::SOFIE Warning - Model Loading - Attribute " << attribute_name << " in OperatorNode "
                   << convnode.name() << " is not defined in ONNX IR and not applied!\n";
      }
   }

   std::string output_name = addnode.output(0);
   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_Conv<float>(attr_auto_pad, attr_dilations, attr_group, attr_kernel_shape, attr_pads,
                                         attr_strides, convnode.input(0), convnode.input(1), name_b, output_name));
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
