#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_BasicNary.hxx"
#include "onnx_proto3.pb.h"
#include <memory>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template<EBasicNaryOperator Op>
std::unique_ptr<ROperator> ParseBasicNary(RModelParser_ONNX& parser, const onnx::NodeProto& nodeproto) {
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
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Max op has input tensor" + input_name +
                                  " but its type is not yet registered");
      }
      inputs.emplace_back(input_name);
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT: op.reset(new ROperator_BasicNary<float, Op>(inputs, output_name)); break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Max does not yet support input type " + ConvertTypeToString(input_type));
   }

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
}


ParserFuncSignature ParseMax = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicNary<EBasicNaryOperator::Max>(parser, nodeproto);
};

ParserFuncSignature ParseMin= [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicNary<EBasicNaryOperator::Min>(parser, nodeproto);
};

ParserFuncSignature ParseMean = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicNary<EBasicNaryOperator::Mean>(parser, nodeproto);
};

ParserFuncSignature ParseSum = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicNary<EBasicNaryOperator::Sum>(parser, nodeproto);
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
