#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_TopK.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseTopK = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type = ETensorType::UNDEFINED;

   std::string input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser TopK op has input tensor " + input_name +
                               " but its type is not yet registered");
   }
   std::string k_name = nodeproto.input(1);
   if (!parser.IsRegisteredTensorType(k_name)) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser TopK op has input tensor " + k_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;

   std::string outputVal_name = nodeproto.output(0);
   std::string outputInd_name = nodeproto.output(1);
   int attr_axis = -1;
   int attr_largest = 1;
   int attr_sorted = 1;

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "axis")
         attr_axis = nodeproto.attribute(i).i();
      if (attribute_name == "largest")
         attr_largest = nodeproto.attribute(i).i();
      if (attribute_name == "sorted")
         attr_sorted = nodeproto.attribute(i).i();
   }
   op.reset(new ROperator_TopK<float>(attr_axis, attr_largest, attr_sorted, k_name, input_name, outputVal_name, outputInd_name));

   if (!parser.IsRegisteredTensorType(outputVal_name)) {
      parser.RegisterTensorType(outputVal_name, input_type);
   }
   if (!parser.IsRegisteredTensorType(outputInd_name)) {
      parser.RegisterTensorType(outputInd_name, ETensorType::INT64);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
