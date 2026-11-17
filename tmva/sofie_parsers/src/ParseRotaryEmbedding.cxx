#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_RotaryEmbedding.hxx"
#include "onnx.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseRotaryEmbedding = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   if (nodeproto.input_size() < 3) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser RotaryEmbedding op requires the inputs X, cos_cache and "
                               "sin_cache");
   }
   const std::string input_name = nodeproto.input(0);
   const std::string cos_name = nodeproto.input(1);
   const std::string sin_name = nodeproto.input(2);
   const std::string position_ids_name = (nodeproto.input_size() > 3) ? nodeproto.input(3) : "";

   for (const std::string &name : {input_name, cos_name, sin_name}) {
      if (!parser.IsRegisteredTensorType(name)) {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser RotaryEmbedding op has input tensor " + name +
                                  " but its type is not yet registered");
      }
   }
   const ETensorType input_type = parser.GetTensorType(input_name);
   for (const std::string &name : {cos_name, sin_name}) {
      if (parser.GetTensorType(name) != input_type) {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser RotaryEmbedding op has cache tensor " + name + " of type " +
                                  ConvertTypeToString(parser.GetTensorType(name)) + " but input tensor " + input_name +
                                  " is of type " + ConvertTypeToString(input_type) + "; they must be the same");
      }
   }

   int64_t interleaved = 0;
   int64_t rotary_embedding_dim = 0;
   int64_t num_heads = 0;
   for (int64_t i = 0; i < nodeproto.attribute_size(); i++) {
      const std::string &attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "interleaved") {
         interleaved = nodeproto.attribute(i).i();
      } else if (attribute_name == "rotary_embedding_dim") {
         rotary_embedding_dim = nodeproto.attribute(i).i();
      } else if (attribute_name == "num_heads") {
         num_heads = nodeproto.attribute(i).i();
      }
   }
   if (rotary_embedding_dim < 0 || num_heads < 0) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser RotaryEmbedding op has negative rotary_embedding_dim or "
                               "num_heads");
   }

   std::unique_ptr<ROperator> op;
   const std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_RotaryEmbedding<float>(interleaved != 0, rotary_embedding_dim, num_heads, input_name,
                                                    cos_name, sin_name, position_ids_name, output_name));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator RotaryEmbedding does not yet support input type " +
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
