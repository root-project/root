#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Pool.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParsePool = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   PoolOpMode op_mode = InvalidPool;
   if (nodeproto.op_type() == "MaxPool")
      op_mode = MaxPool;
   else if (nodeproto.op_type() == "AveragePool")
      op_mode = AveragePool;
   else if (nodeproto.op_type() == "GlobalAveragePool")
      op_mode = GlobalAveragePool;

   assert(op_mode != InvalidPool);

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Pool op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;

   RAttributes_Pool attr;

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "auto_pad") {
         attr.auto_pad = nodeproto.attribute(i).s();
      } else if (attribute_name == "ceil_mode") {
         attr.ceil_mode = nodeproto.attribute(i).i();
      } else if (attribute_name == "count_include_pad" && op_mode == AveragePool) {
         attr.count_include_pad = nodeproto.attribute(i).i();
      } else if (attribute_name == "storage_order" && op_mode == MaxPool) {
         attr.storage_order = nodeproto.attribute(i).i();
      } else if (attribute_name == "dilations" && op_mode == MaxPool) {
         attr.dilations =
            std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attribute_name == "kernel_shape") {
         attr.kernel_shape =
            std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attribute_name == "pads") {
         attr.pads = std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attribute_name == "strides") {
         attr.strides =
            std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else {
         std::cout << "TMVA::SOFIE Warning - Model Loading - Attribute " << attribute_name << " in OperatorNode "
                   << nodeproto.name() << " is not defined in ONNX IR and not applied!\n";
      }
   }

   std::string output_name = nodeproto.output(0);
   switch (input_type) {
   case ETensorType::FLOAT: op.reset(new ROperator_Pool<float>(op_mode, attr, input_name, output_name)); break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Pool does not yet support input type " +
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
