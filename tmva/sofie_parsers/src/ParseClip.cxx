#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Clip.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseClip = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Clip op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);
   std::string opt_input_min;
   std::string opt_input_max;

   for (int_t i = 1; i < nodeproto.input_size(); i++) {
      std::string opt_input_name = nodeproto.input(i);
      if (opt_input_name == "min") {
         opt_input_min = opt_input_name;
         if(!parser.IsRegisteredTensorType(opt_input_min)) {
            throw std::runtime_error("TMVA::SOFIE ONNX Parser Clip op has input tensor" + opt_input_min +
                               " but its type is not yet registered");
         }
      }
      if (opt_input_name == "max") {
         opt_input_max = opt_input_name;
         if(!parser.IsRegisteredTensorType(opt_input_name)) {
            throw std::runtime_error("TMVA::SOFIE ONNX Parser Clip op has input tensor" + opt_input_max +
                               " but its type is not yet registered");
         }
      }
   }
   op.reset(new ROperator_Clip<float>(opt_input_min, opt_input_max, input_name, output_name));
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
