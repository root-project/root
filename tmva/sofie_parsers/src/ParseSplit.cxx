#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Split.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseSplit = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   std::string input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Split op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   std::string split_name;
   if (nodeproto.input_size() > 1) {
      split_name = nodeproto.input(1);
      if (!parser.IsRegisteredTensorType(split_name)) {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Split op has input tensor" + split_name +
                                  " but its type is not yet registered");
      }
   }

   int axis = 0;
   int num_outputs = 0;
   for (int i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "axis") {
         axis = nodeproto.attribute(i).i();
      }
      else if (attribute_name == "num_outputs") {
         num_outputs = nodeproto.attribute(i).i();
      }
      else
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Split operator: attribute" + attribute_name +  "is not yet supported");
   }

   // number of splits are given by the number of output tensors
   int output_size = nodeproto.output_size();
   std::vector<std::string> output_names(output_size);
   for (int i = 0; i < output_size; i++)
      output_names[i] = nodeproto.output(i);

   if (num_outputs > 0 && num_outputs != output_size)
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Split - invalid output size: " + std::to_string(output_size) + " instead of " +
         std::to_string(num_outputs));

   std::unique_ptr<ROperator> op(new ROperator_Split(input_name, split_name, axis, output_names));


   for (int i = 0; i < output_size; i++) {
      if (!parser.IsRegisteredTensorType(output_names[i])) {
        parser.RegisterTensorType(output_names[i], input_type);
      }
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
