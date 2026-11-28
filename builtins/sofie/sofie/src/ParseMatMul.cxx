#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "onnx.hxx"

namespace SOFIE {

ParserFuncSignature ParseMatMul = [](RModelParser_ONNX &parser, const onnx::NodeProto &matmulnode) {
   ETensorType input_type = ETensorType::UNDEFINED;

   // check input type - only first input from MatMul
   auto input_name = matmulnode.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("SOFIE ONNX Parser MatMul op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;

   // for MatMul there is no alpha and beta : use alpha=1 and beta=0
   float attr_alpha = 1.0;
   float attr_beta = 0.0;
   int_t attr_transA = 0;
   int_t attr_transB = 0;

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_Gemm<float>(attr_alpha, attr_beta, attr_transA, attr_transB, matmulnode.input(0),
                                          matmulnode.input(1), matmulnode.output(0)));
      break;
   default:
      throw std::runtime_error(
         "SOFIE - Unsupported - Operator for fusing MatMul and Add to Gemm does not yet support input type " +
         std::to_string(static_cast<int>(input_type)));
   }

   std::string output_name = matmulnode.output(0);
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
