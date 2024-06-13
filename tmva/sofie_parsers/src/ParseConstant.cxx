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
   bool isConstantOfShape = false;
   // case of ConstantOfShape (Constant has zero inputs)
   if (ninputs > 0) {
      input_name = nodeproto.input(0);
      isConstantOfShape = true;
      if (!parser.IsRegisteredTensorType(input_name)) {
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

      // tensor input
      if (attribute_name == "value") {
         const onnx::TensorProto & t = nodeproto.attribute(i).t();
         output_type = static_cast<ETensorType>(t.data_type());
         //std::cout << "found attribute value with type " << ConvertTypeToString(output_type) << "\n";
         std::vector<std::size_t> shape;
         std::size_t length = 1;
         for (int j = 0; j < t.dims_size(); j++) {
            shape.push_back(t.dims(j));
            length *= t.dims(j);
         }
         switch(output_type) {
            // need to use raw_data() to get the tensor values
         case ETensorType::INT64: {
            std::vector<int64_t> values(length);
            auto raw_data_ptr = reinterpret_cast<int64_t *>(const_cast<char *>(t.raw_data().c_str()));
            std::memcpy(values.data(), raw_data_ptr, length * sizeof(int64_t));
            op.reset(new ROperator_Constant<int64_t>("int64_t", values, shape, input_name, output_name));
            break;
         }
         case ETensorType::FLOAT: {
            std::vector<float> values(length);
            auto raw_data_ptr = reinterpret_cast<float *>(const_cast<char *>(t.raw_data().c_str()));
            std::memcpy(values.data(), raw_data_ptr, length * sizeof(float));
            //for (size_t j = 0; j < values.size(); j++) std::cout << values[j] << "\n";
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
         // neither constant nor ConstantOfShape
         if (!isConstantOfShape)
            throw std::runtime_error("Attribute " + attribute_name +  " in Constant op  is not yet supported!\n");
         // case of ConstantOfShape
         else {
            // if attribute is not there use by default float type with zero values
            std::vector<float> values(1);
            std::vector<size_t> shape(1,1);
            op.reset(new ROperator_Constant<float>("float",values,shape, input_name, output_name));
         }
      }

      // other cases of Constant operator not required currently
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