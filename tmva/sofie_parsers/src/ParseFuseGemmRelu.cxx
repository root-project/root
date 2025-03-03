#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Gemm.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuseFuncSignature ParseFuseGemmRelu = [](RModelParser_ONNX &parser, const onnx::NodeProto &gemmnode,
                                                const onnx::NodeProto &relunode) -> std::unique_ptr<ROperator> {
   ETensorType input_type = ETensorType::UNDEFINED;
   // check input type - only first input from Gemm
   auto input_name = gemmnode.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser MatMul op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   // we don't check input type of ADD since it is not be registered
   std::unique_ptr<ROperator> op;

   float attr_alpha = 1.0;
   float attr_beta = 1.0;
   int_t attr_transA = 0;
   int_t attr_transB = 0;

   for (int i = 0; i < gemmnode.attribute_size(); i++) {
      std::string attribute_name = gemmnode.attribute(i).name();
      if (attribute_name == "alpha") {
         attr_alpha = gemmnode.attribute(i).f();
      } else if (attribute_name == "beta") {
         attr_beta = gemmnode.attribute(i).f();
      } else if (attribute_name == "transA") {
         attr_transA = gemmnode.attribute(i).i();
         if (attr_transA != 0 && attr_transA != 1)
            throw std::runtime_error("TMVA::SOFIE Error - Model Loading - attribute transA in Operator Gemm not 0/1");
      } else if (attribute_name == "transB") {
         attr_transB = gemmnode.attribute(i).i();
         if (attr_transB != 0 && attr_transB != 1)
            throw std::runtime_error("TMVA::SOFIE Error - Model Loading - attribute transB in Operator Gemm not 0/1");
      } else {
         std::cout << "TMVA::SOFIE Warning - Model Loading - Attribute " << attribute_name << " in OperatorNode "
                   << gemmnode.name() << " is not defined in ONNX IR and not applied!\n";
      }
   }
   switch (input_type) {
   case ETensorType::FLOAT:
      if (gemmnode.input_size() == 2) {
         op.reset(new ROperator_Gemm<float>(attr_alpha, attr_beta, attr_transA, attr_transB, gemmnode.input(0),
                                            gemmnode.input(1), relunode.output(0), EActivationType::RELU));
      } else {
         op.reset(new ROperator_Gemm<float>(attr_alpha, attr_beta, attr_transA, attr_transB, gemmnode.input(0),
                                            gemmnode.input(1), gemmnode.input(2), relunode.output(0), EActivationType::RELU));
      }
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Gemm does not yet support input type " +
                               std::to_string(static_cast<int>(input_type)));
   }

   std::string output_name = relunode.output(0);
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
