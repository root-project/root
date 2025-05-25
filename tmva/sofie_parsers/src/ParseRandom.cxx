#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Random.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseRandom = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {

   RandomOpMode opMode = kUniform;
   auto op_type = nodeproto.op_type();
   if (op_type == "RandomNormal" || op_type == "RandomNormalLike")
      opMode = kNormal;


   ETensorType input_type = ETensorType::FLOAT; // default value
   std::string input_name;
   // case of NormalLike and UniformLike , type is given by the input
   if (nodeproto.input_size() > 0) {
      input_name = nodeproto.input(0);
      if (parser.IsRegisteredTensorType(input_name)) {
         input_type = parser.GetTensorType(input_name);
      } else {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Randomr op has input tensor" + input_name +
                               " but its type is not yet registered");
      }
   }
   // get  attributes
   float seed = 0;
   std::map<std::string, float> paramMap;
   std::vector<size_t> shape;
   for (int i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      auto attr_type = nodeproto.attribute(i).type();
      if (attribute_name == "dtype")
         input_type = static_cast<ETensorType>(nodeproto.attribute(i).i());
      else if (attribute_name == "seed") {
         if (attr_type == onnx::AttributeProto::FLOAT )
            seed = nodeproto.attribute(i).f();
         else if (attr_type == onnx::AttributeProto::INT)
            seed = nodeproto.attribute(i).i();
         else
            throw std::runtime_error("TMVA::SOFIE ONNX Parser Random  op has invalid type for attribute seed");
      }
      else if (attribute_name == "shape") {
         if (attr_type != onnx::AttributeProto::INTS)
            throw std::runtime_error("TMVA::SOFIE ONNX Parser Random  op has invalid type for attribute shape");
         shape = std::vector<size_t>(nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end());
      }
      else {
         float value = 0;
         if (attr_type == onnx::AttributeProto::FLOAT)
            value = nodeproto.attribute(i).f();
         else if (attr_type == onnx::AttributeProto::INT)
            value = nodeproto.attribute(i).i();
         else
            throw std::runtime_error("TMVA::SOFIE ONNX Parser Random  op has invalid type for attribute " + attribute_name);
         paramMap[attribute_name] = value;
      }
   }

   std::string output_name = nodeproto.output(0);

   std::unique_ptr<ROperator> op(new ROperator_Random(opMode, input_type, input_name, output_name, shape, paramMap, seed));

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
