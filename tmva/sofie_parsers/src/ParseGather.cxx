#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Gather.hxx"
#include "onnx_proto3.pb.h"
#include <stdexcept>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseGather = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type = ETensorType::UNDEFINED;
   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Gather op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   ETensorType indices_type = ETensorType::UNDEFINED;
   auto indices_name = nodeproto.input(1);
   // indices_type can be an initialized tensor, no need to emit an error if it is not registered
   if (parser.IsRegisteredTensorType(indices_name)) {
      indices_type = parser.GetTensorType(indices_name);
      if (indices_type != ETensorType::INT64 && indices_type != ETensorType::INT32) {
         throw
            std::runtime_error("TMVA::SOFIE ONNX Parser Gather op Indices tensor type not supported.");
      }
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);
   int64_t attr_axis = 0;
   if (nodeproto.attribute_size() == 1) {
      attr_axis = nodeproto.attribute(0).i();
   }

   op.reset(new ROperator_Gather(attr_axis, input_name, indices_name, nodeproto.output(0)));

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
