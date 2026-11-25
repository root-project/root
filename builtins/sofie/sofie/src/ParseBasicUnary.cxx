#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_BasicUnary.hxx"
#include "onnx.hxx"

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

void RegisterBasicUnaryParsers(RModelParser_ONNX &parser)
{
   parser.RegisterOperator("Sqrt", ParseBasicUnary<EBasicUnaryOperator::kSqrt>);
   parser.RegisterOperator("Reciprocal", ParseBasicUnary<EBasicUnaryOperator::kReciprocal>);
   parser.RegisterOperator("Neg", ParseBasicUnary<EBasicUnaryOperator::kNeg>);
   parser.RegisterOperator("Exp", ParseBasicUnary<EBasicUnaryOperator::kExp>);
   parser.RegisterOperator("Log", ParseBasicUnary<EBasicUnaryOperator::kLog>);
   parser.RegisterOperator("Sin", ParseBasicUnary<EBasicUnaryOperator::kSin>);
   parser.RegisterOperator("Cos", ParseBasicUnary<EBasicUnaryOperator::kCos>);
   parser.RegisterOperator("Abs", ParseBasicUnary<EBasicUnaryOperator::kAbs>);
   parser.RegisterOperator("Softplus", ParseBasicUnary<EBasicUnaryOperator::kSoftplus>);
   parser.RegisterOperator("Atan", ParseBasicUnary<EBasicUnaryOperator::kAtan>);
   parser.RegisterOperator("Asinh", ParseBasicUnary<EBasicUnaryOperator::kAsinh>);
   parser.RegisterOperator("Acosh", ParseBasicUnary<EBasicUnaryOperator::kAcosh>);
   parser.RegisterOperator("Atanh", ParseBasicUnary<EBasicUnaryOperator::kAtanh>);
   parser.RegisterOperator("Floor", ParseBasicUnary<EBasicUnaryOperator::kFloor>);
}

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
