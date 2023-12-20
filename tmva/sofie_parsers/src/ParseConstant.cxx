#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Constant.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

// same function used to parse Constant and ConstantOfShape

ParserFuncSignature ParseConstant = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   std::string input_name;
   auto ninputs = nodeproto.input_size();
   if (ninputs > 0) {  // case of ConstantOfShape
      input_name = nodeproto.input(0);
      if (parser.IsRegisteredTensorType(input_name)) {
         ETensorType input_type = parser.GetTensorType(input_name);
         // input type should be int64
         //if (input_type != ETensorType::INT64)
         //   throw std::runtime_error("TMVA::SOFIE ONNX Parser ConstantOfShape op has invalid input type " + ConvertTypeToString(input_type));
      } else {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser ConstantOfShape op has input tensor" + input_name +
                                  "  but its type is not yet registered");
      }
   }

   std::unique_ptr<ROperator> op;
   std::string attr_type;

   std::string output_name = nodeproto.output(0);
   ETensorType output_type = ETensorType::FLOAT;
   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "value") {
         const onnx::TensorProto & t = nodeproto.attribute(i).t();
         output_type = static_cast<ETensorType>(t.data_type());
         std::vector<std::size_t> shape;
         //std::size_t length = 1;
         for (int j = 0; j < t.dims_size(); j++) {
            shape.push_back(t.dims(j));
            //length *= t.dims(j);
         }
         switch(output_type) {
         case ETensorType::INT64: {
            std::vector<int64_t> values(t.int64_data_size());
            for (size_t j = 0; j < values.size(); j++) values[j] = t.int64_data(j);
            op.reset(new ROperator_Constant<int64_t>("int64_t", values, shape, input_name, output_name));
            break;
         }
         case ETensorType::FLOAT: {
            std::vector<float> values(t.float_data_size());
            for (size_t j = 0; j < values.size(); j++) values[j] = t.float_data(j);
            op.reset(new ROperator_Constant<float>("float",values, shape, input_name, output_name));
            break;
         }
         default:
           throw std::runtime_error("Data type in Constant op attribute " + ConvertTypeToString(output_type) +
                                       " is not supported!\n");
         }
         break;
      }
      else {
         throw std::runtime_error("Attribute " + attribute_name +  " in Constant op  is not supported!\n");
      }
      // else if (attribute_name == "value_int")
      //    attr_type = std::to_string( static_cast<int64_t>(nodeproto.attribute(i).i()) );
   }


   //op.reset(new ROperator_Constant(attr_type, nodeproto.input(0), output_name));


   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, output_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
