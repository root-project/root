#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_TopK.hxx"
#include "onnx_proto3.pb.h"
#include <memory>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

std::unique_ptr<ROperator> ParseTopK(RModelParser_ONNX& parser, const onnx::NodeProto& nodeproto) {
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
         throw std::runtime_error("TMVA::SOFIE ONNX Parser TopK op has input tensor " + input_name +
                                  " but its type is not yet registered");
      }
      inputs.emplace_back(input_name);
   }

   int attr_axis = 1;
   int attr_largest = 1;
   int attr_sorted = 1;
   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "axis")
         attr_axis = nodeproto.attribute(i).i();
      else if (attribute_name == "largest")
         attr_largest = nodeproto.attribute(i).i();
      else if(attribute_name == "sorted"){
         attr_sorted = nodeproto.attribute(i).i();
      }
   }



   if (nodeproto.output_size() != 2) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser TopK op must have exactly 2 output tensors, but " +
                               std::to_string(nodeproto.output_size()) + " are specified");
   }

   std::unique_ptr<ROperator> op;
   std::string output_name_values = nodeproto.output(0);
   std::string output_name_indices = nodeproto.output(1);

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_TopK<float>(inputs, attr_axis, attr_largest, attr_sorted, output_name_values, output_name_indices));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator TopK does not yet support input type " +
                               ConvertTypeToString(input_type));
   }

   if (!parser.IsRegisteredTensorType(output_name_values)) {
      parser.RegisterTensorType(output_name_values, input_type);
   }

   if (!parser.IsRegisteredTensorType(output_name_indices)) {
      parser.RegisterTensorType(output_name_indices, ETensorType::INT64);
   }

   return op;
}

ParserFuncSignature ParseTopK = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseTopK(parser, nodeproto);
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA