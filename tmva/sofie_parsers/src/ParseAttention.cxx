#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Attention.hxx"
#include "onnx.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseAttention = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   if (nodeproto.input_size() < 3) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Attention op requires the inputs Q, K and V");
   }
   auto optionalInput = [&](int i) { return (nodeproto.input_size() > i) ? nodeproto.input(i) : std::string{}; };
   auto optionalOutput = [&](int i) { return (nodeproto.output_size() > i) ? nodeproto.output(i) : std::string{}; };

   const std::string q_name = nodeproto.input(0);
   const std::string k_name = nodeproto.input(1);
   const std::string v_name = nodeproto.input(2);
   const std::string mask_name = optionalInput(3);
   if (!optionalInput(4).empty() || !optionalInput(5).empty() || !optionalOutput(1).empty() ||
       !optionalOutput(2).empty()) {
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Attention does not yet support the "
                               "past_key/past_value inputs and present_key/present_value outputs");
   }
   if (!optionalInput(6).empty()) {
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Attention does not yet support the "
                               "nonpad_kv_seqlen input");
   }
   if (!optionalOutput(3).empty()) {
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Attention does not yet support the "
                               "qk_matmul_output output");
   }

   for (const std::string &name : {q_name, k_name, v_name, mask_name}) {
      if (!name.empty() && !parser.IsRegisteredTensorType(name)) {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Attention op has input tensor " + name +
                                  " but its type is not yet registered");
      }
   }
   const ETensorType input_type = parser.GetTensorType(q_name);
   for (const std::string &name : {k_name, v_name}) {
      if (parser.GetTensorType(name) != input_type) {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Attention op has input tensor " + name + " of type " +
                                  ConvertTypeToString(parser.GetTensorType(name)) + " but Q " + q_name +
                                  " is of type " + ConvertTypeToString(input_type) + "; they must be the same");
      }
   }

   int64_t is_causal = 0;
   float scale = 0;
   float softcap = 0;
   int64_t q_num_heads = 0;
   int64_t kv_num_heads = 0;
   int64_t softmax_precision = 0;
   for (int64_t i = 0; i < nodeproto.attribute_size(); i++) {
      const std::string &attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "is_causal") {
         is_causal = nodeproto.attribute(i).i();
      } else if (attribute_name == "scale") {
         scale = nodeproto.attribute(i).f();
      } else if (attribute_name == "softcap") {
         softcap = nodeproto.attribute(i).f();
      } else if (attribute_name == "q_num_heads") {
         q_num_heads = nodeproto.attribute(i).i();
      } else if (attribute_name == "kv_num_heads") {
         kv_num_heads = nodeproto.attribute(i).i();
      } else if (attribute_name == "softmax_precision") {
         softmax_precision = nodeproto.attribute(i).i();
      }
      // qk_matmul_output_mode only matters for the qk_matmul_output output, which is rejected above
   }
   if (q_num_heads < 0 || kv_num_heads < 0) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Attention op has a negative q_num_heads or kv_num_heads");
   }
   // the softmax is computed in float
   if (softmax_precision != 0 && softmax_precision != static_cast<int64_t>(ETensorType::FLOAT)) {
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Attention only supports a float "
                               "softmax_precision, got " +
                               std::to_string(softmax_precision));
   }

   std::unique_ptr<ROperator> op;
   const std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_Attention<float>(is_causal != 0, scale, softcap, q_num_heads, kv_num_heads, q_name, k_name,
                                              v_name, mask_name, output_name));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Attention does not yet support input type " +
                               ConvertTypeToString(input_type));
   }
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
