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

   // ignore for time being attributes
   if (nodeproto.attribute_size() > 0 )
      std::cout << "WARNING: TMVA::SOFIE ONNX Parser Split operator: attributes are not yet supported- they are ignored" << std::endl;

   // number of splits are given by the number of output tensors
   size_t output_size = nodeproto.output_size();
   std::vector<std::string> output_names(output_size);
   for (size_t i = 0; i < output_size; i++)
      output_names[i] = nodeproto.output(i);

   std::unique_ptr<ROperator> op(new ROperator_Split<float>(input_name, split_name, output_names));

   for (size_t i = 0; i < output_size; i++) {
      if (!parser.IsRegisteredTensorType(output_names[i])) {
        parser.RegisterTensorType(output_names[i], input_type);
      }
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
