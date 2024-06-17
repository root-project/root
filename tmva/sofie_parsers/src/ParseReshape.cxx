#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Reshape.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseReshape = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   // make Reshape operator
   ETensorType input_type = ETensorType::UNDEFINED;

   ReshapeOpMode opMode = Reshape;
   if (nodeproto.op_type() == "Flatten")
      opMode = Flatten;
   else if (nodeproto.op_type() == "Squeeze")
      opMode = Squeeze;
   else if (nodeproto.op_type() == "Unsqueeze")
      opMode = Unsqueeze;

   // reshape has as extra input shape tensor (int64) but
   // it is not present for Flatten, Squeeze and Unsquueze
   auto input_name = nodeproto.input(0);
   // for squeeze is optional ?
   auto shape_name = ((nodeproto.input_size() > 1) && ( opMode == Reshape || opMode == Unsqueeze || opMode == Squeeze) )
      ? nodeproto.input(1) : "";
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Reshape op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   // Reshape is having one attribute: allowzero (int) (default = 0)
   // Flatten is having one attribute: axis (int) (default=1)
   // old version of reshape and squeeze have axes as attributes
   std::unique_ptr<ROperator> op;
   int attr_value = (opMode == Reshape) ? 0 : 1;
   if (opMode == Reshape && nodeproto.attribute_size() > 0)
      attr_value = nodeproto.attribute(0).i();

   std::vector<int64_t> attr_axes = {};
   if (nodeproto.input_size() == 1 && (opMode == Squeeze || opMode == Unsqueeze)) {
      std::string attribute_name = nodeproto.attribute(0).name();
      if (attribute_name == "axes")
         attr_axes = {nodeproto.attribute(0).ints().begin(), nodeproto.attribute(0).ints().end()};
   }

   std::string output_name = nodeproto.output(0);

   if (attr_axes.empty())
      op.reset(new ROperator_Reshape(opMode, attr_value, input_name, shape_name, output_name));
   else // for old Squeeze and Unsqueeze
      op.reset(new ROperator_Reshape(opMode, attr_axes, input_name, output_name));

   //   throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Reshape does not yet support input type " +
   //                            ConvertTypeToString(input_type));

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
