#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Comparision.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <EComparisionOperator Op>
std::unique_ptr<ROperator> ParseComparision(RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
{
   ETensorType input_type = ETensorType::UNDEFINED;

   for (int i = 0; i < 2; ++i) {
      auto input_name = nodeproto.input(i);
      if (parser.IsRegisteredTensorType(input_name)) {
         // according to ONNX both inputs have same type
         if (i == 0)
            input_type = parser.GetTensorType(input_name);
         else
            if (input_type != parser.GetTensorType(input_name)) {
               throw
                  std::runtime_error("TMVA::SOFIE ONNX parser Comparision op has input tensors of different types");
            }
      } else {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Comparision op has input tensor " + input_name +
                                  " but its type is not yet registered");
      }
   }


   std::string output_name = nodeproto.output(0);

   std::unique_ptr<ROperator> op;
   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_Comparision<float, Op>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   case ETensorType::INT64:
      op.reset(new ROperator_Comparision<int64_t, Op>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   case ETensorType::INT32:
      op.reset(new ROperator_Comparision<int32_t, Op>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Comparision Operator does not yet support input type " +
                               ConvertTypeToString(input_type));
   }

   // Infer the output type
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, ETensorType::BOOL);
   }

   return op;
};

// Parse Equal
ParserFuncSignature ParseEq = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseComparision<EComparisionOperator::Eq>(parser, nodeproto);
};

// Parse Less
ParserFuncSignature ParseLess = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseComparision<EComparisionOperator::Less>(parser, nodeproto);
};

// Parse Mul
ParserFuncSignature ParseLessEq = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseComparision<EComparisionOperator::LessEq>(parser, nodeproto);
};

// Parse Div
ParserFuncSignature ParseGreater = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseComparision<EComparisionOperator::Greater>(parser, nodeproto);
};

// Parse Pow
ParserFuncSignature ParseGreaterEq = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseComparision<EComparisionOperator::GreaterEq>(parser, nodeproto);
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
