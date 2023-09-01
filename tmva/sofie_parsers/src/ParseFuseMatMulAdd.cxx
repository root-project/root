#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Gemm.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuseFuncSignature ParseFuseMatMulAdd = [](RModelParser_ONNX &parser, const onnx::NodeProto &matmulnode,
                                                const onnx::NodeProto &addnode) {
   ETensorType input_type = ETensorType::UNDEFINED;

   // check input tye - only first input from MatMul
   auto input_name = matmulnode.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser MatMul op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   if (addnode.input_size() != 2)
      throw std::runtime_error("TMVA::SOFIE ONNX Parser : cannot fuse MatMul if Add does not have 2 inputs");
   // output of matmul should be one of the input of Add
   std::string nameBias;
   if (matmulnode.output(0) == addnode.input(0))
      nameBias = addnode.input(1);
   else if (matmulnode.output(0) == addnode.input(1))
      nameBias = addnode.input(0);
   else
      throw std::runtime_error("TMVA::SOFIE ONNX Parser : cannot fuse MatMul and Add since have different inputs");

   // we don't check input type of ADD since it is not be registered
   std::unique_ptr<ROperator> op;

   float attr_alpha = 1.0;
   float attr_beta = 1.0;
   int_t attr_transA = 0;
   int_t attr_transB = 0;

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_Gemm<float>(attr_alpha, attr_beta, attr_transA, attr_transB, matmulnode.input(0),
                                          matmulnode.input(1), nameBias, addnode.output(0)));
      break;
   default:
      throw std::runtime_error(
         "TMVA::SOFIE - Unsupported - Operator for fusing MatMul and Add to Gemm does not yet support input type " +
         std::to_string(static_cast<int>(input_type)));
   }

   std::string output_name = addnode.output(0);
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
