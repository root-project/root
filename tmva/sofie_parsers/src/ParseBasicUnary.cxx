#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_BasicUnary.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <EBasicUnaryOperator Op>
std::unique_ptr<ROperator> ParseBasicUnary(RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
{
   ETensorType input_type = ETensorType::UNDEFINED;

   std::string input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw
         std::runtime_error("TMVA::SOFIE ONNX Parser Unary op has input tensor " + input_name +
                                  " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_BasicUnary<float, Op>(input_name, output_name));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Binary Operator does not yet support input type " +
                               std::to_string(static_cast<int>(input_type)));
   }

   // Infer the output type
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

// Parse Sqrt
ParserFuncSignature ParseSqrt = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicUnary<EBasicUnaryOperator::kSqrt>(parser, nodeproto);
};

// Parse Reciprocal
ParserFuncSignature ParseReciprocal = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicUnary<EBasicUnaryOperator::kReciprocal>(parser, nodeproto);
};

// Parse Neg
ParserFuncSignature ParseNeg = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicUnary<EBasicUnaryOperator::kNeg>(parser, nodeproto);
};

// Parse Exp
ParserFuncSignature ParseExp = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicUnary<EBasicUnaryOperator::kExp>(parser, nodeproto);
};

// Parse Log
ParserFuncSignature ParseLog = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicUnary<EBasicUnaryOperator::kLog>(parser, nodeproto);
};

// Parse Sin
ParserFuncSignature ParseSin = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicUnary<EBasicUnaryOperator::kSin>(parser, nodeproto);
};

// Parse Cos
ParserFuncSignature ParseCos = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicUnary<EBasicUnaryOperator::kCos>(parser, nodeproto);
};

// Parse Abs
ParserFuncSignature ParseAbs = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicUnary<EBasicUnaryOperator::kAbs>(parser, nodeproto);
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
