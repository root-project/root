#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Slice.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseSlice = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   // make Slice operator
   ETensorType input_type = ETensorType::UNDEFINED;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Slice op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   std::vector<std::string> axisTensorNames;
   if (nodeproto.input_size() > 1)
      axisTensorNames.push_back(nodeproto.input(1));
   if (nodeproto.input_size() > 2)
      axisTensorNames.push_back(nodeproto.input(2));
   if (nodeproto.input_size() > 3)
      axisTensorNames.push_back(nodeproto.input(3));
   if (nodeproto.input_size() > 4)
      axisTensorNames.push_back(nodeproto.input(4));

   // not sure how to find here type of the integer inputs
   ETensorType axis_type = ETensorType::INT64;
   // for version < 10
   std::vector<int64_t> attr_starts = {};
   std::vector<int64_t> attr_ends = {};
   std::vector<int64_t> attr_axes = {};
   if (nodeproto.input_size() == 1) {
      for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
         std::string attribute_name = nodeproto.attribute(i).name();
         if (attribute_name == "starts")
            attr_starts = {nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()};
         if (attribute_name == "ends")
            attr_ends = {nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()};
         if (attribute_name == "axes")
            attr_axes = {nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()};
      }
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);
   //switch (input_type) {
   //case ETensorType::FLOAT:
      if (axisTensorNames.size() > 0) {
         // for version >= 10
         if (axis_type == ETensorType::INT32)
            op.reset(new ROperator_Slice<int32_t>(input_name, axisTensorNames, output_name));
         else if (axis_type == ETensorType::INT64)
            op.reset(new ROperator_Slice<int64_t>(input_name, axisTensorNames, output_name));
         else
            throw std::runtime_error(
               "TMVA::SOFIE - Unsupported - Operator Slice has invalid input type for input axis descriptors " +
               std::to_string(static_cast<int>(axis_type)));
      } else if (attr_starts.size() > 0 && attr_ends.size() > 0) {
         op.reset(new ROperator_Slice<int64_t>(input_name, attr_starts, attr_ends, attr_axes, output_name));
      } else {
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Slice has invalid attribues");
      }
      //break;
   //default:
   //   throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Slice does not yet support input type " +
   //                            std::to_string(static_cast<int>(input_type)));
   //}

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
