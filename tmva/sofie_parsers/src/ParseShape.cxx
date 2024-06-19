#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Shape.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseShape = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   auto input_name = nodeproto.input(0);
   if (!parser.IsRegisteredTensorType(input_name)) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Shape op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);
   int attr_start = 0;
   int attr_end = INT_MAX;  // cannot use 0 or -1 as default

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "start")
         attr_start = nodeproto.attribute(i).i();
      if (attribute_name == "end")
         attr_end = nodeproto.attribute(i).i();
   }

   op.reset(new ROperator_Shape(attr_start, attr_end, input_name, output_name));

   // output of Shpe is always an int64 tensor
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, ETensorType::INT64);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
