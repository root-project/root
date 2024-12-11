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

   if (parser.Verbose()) {
      std::cout << "\t.... ";
      if (isConstantOfShape)
         std::cout << "ConstantOfShape " << nodeproto.input(0) << "  -> ";
      else
         std::cout << "Constant  --> ";
      std::cout << nodeproto.output(0) << std::endl;
   }

   std::unique_ptr<ROperator> op;
   std::string attr_type;

   std::string output_name = nodeproto.output(0);
   ETensorType output_type = ETensorType::FLOAT;
   std::vector<std::size_t> shape;   // output shape (use in case of constant operator)
   // it should be only one attribute (Constant or 1 or 0 COnstant of Shape)
   if (nodeproto.attribute_size() > 1)
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Constant or ConstantOfShape and attribute size is larger than 1");
   if (nodeproto.attribute_size() > 0) {
      std::string attribute_name = nodeproto.attribute(0).name();
      // tensor input
      if (attribute_name == "value") {
         const onnx::TensorProto & t = nodeproto.attribute(0).t();
         output_type = static_cast<ETensorType>(t.data_type());

         std::size_t length = 1;
         for (int j = 0; j < t.dims_size(); j++) {
            shape.push_back(t.dims(j));
            length *= t.dims(j);
         }
         if (isConstantOfShape) {
            // value tensor should be one-element tensor
            if (length != 1)
               throw std::runtime_error("TMVA::SOFIE ONNX Parser ConstantOfShape has invalid tensor size " + std::to_string(length));
         }
         switch(output_type) {
         // to get the tensor values one needs to use the given data types or the raw_data.
         // it depends how the operator was created. We cannot get size of the raw_data
         case ETensorType::INT32: {
            std::vector<int32_t> values(length);
            if (t.int32_data_size() == int(length)) {
               for (size_t i = 0; i < length; i++)
                  values[i] = t.int32_data(i);
            } else {
               auto raw_data_ptr = reinterpret_cast<int32_t *>(const_cast<char *>(t.raw_data().c_str()));
               std::memcpy(values.data(), raw_data_ptr, length * sizeof(int32_t));
            }
            op.reset(new ROperator_Constant<int32_t>("int32_t", values, shape, input_name, output_name));
            break;
         }
         case ETensorType::INT64: {
            std::vector<int64_t> values(length);
            if (t.int64_data_size() == int(length)) {
               for (size_t i = 0; i < length; i++)
                  values[i] = t.int64_data(i);
            } else  { // cannot get size of raw data : assume is ok
               auto raw_data_ptr = reinterpret_cast<int64_t *>(const_cast<char *>(t.raw_data().c_str()));
               std::memcpy(values.data(), raw_data_ptr, length * sizeof(int64_t));
            }
            op.reset(new ROperator_Constant<int64_t>("int64_t", values, shape, input_name, output_name));
            break;
         }
         case ETensorType::FLOAT: {
            std::vector<float> values(length);
             if (t.float_data_size() == int(length)) {
               for (size_t i = 0; i < length; i++)
                  values[i] = t.float_data(i);
            } else {
               auto raw_data_ptr = reinterpret_cast<float *>(const_cast<char *>(t.raw_data().c_str()));
               std::memcpy(values.data(), raw_data_ptr, length * sizeof(float));
            }
            op.reset(new ROperator_Constant<float>("float",values, shape, input_name, output_name));
            break;
         }
         case ETensorType::DOUBLE: {
            std::vector<double> values(length);
             if (t.double_data_size() == int(length)) {
               for (size_t i = 0; i < length; i++)
                  values[i] = t.double_data(i);
            } else {
               auto raw_data_ptr = reinterpret_cast<double *>(const_cast<char *>(t.raw_data().c_str()));
               std::memcpy(values.data(), raw_data_ptr, length * sizeof(double));
            }
            op.reset(new ROperator_Constant<double>("double",values, shape, input_name, output_name));
            break;
         }
         case ETensorType::BOOL: {
            std::vector<bool> values(length);
            auto raw_data_ptr = reinterpret_cast<bool *>(const_cast<char *>(t.raw_data().c_str()));
            // cannot use values.data() for vector of bools
            std::copy(raw_data_ptr, raw_data_ptr + length, values.begin());
            //std::memcpy(values.data(), raw_data_ptr, length * sizeof(float));
            op.reset(new ROperator_Constant<bool>("bool",values, shape, input_name, output_name));
            break;
         }
         default:
           throw std::runtime_error("Data type in Constant op attribute " + ConvertTypeToString(output_type) +
                                       " is not supported!\n");
         }
      }
      else {
         // neither constant nor ConstantOfShape
         if (!isConstantOfShape) {
            // case of ConstantOfShape
            if (attribute_name == "value_float") {
               std::vector<float> values(1);
               values[0] = nodeproto.attribute(0).f();
               shape.push_back(1);
               op.reset(new ROperator_Constant<float>("float",values, shape, input_name, output_name));
            }
            else if (attribute_name == "value_floats") {
               auto values = std::vector<float>({nodeproto.attribute(0).floats().begin(), nodeproto.attribute(0).floats().end()});
               shape.push_back(values.size());
               op.reset(new ROperator_Constant<float>("float",values, shape, input_name, output_name));
            }
            else if (attribute_name == "value_int") {
               std::vector<int64_t> values(1);
               values[0] = nodeproto.attribute(0).i();
               shape.push_back(1);
               op.reset(new ROperator_Constant<int64_t>("int64_t",values, shape, input_name, output_name));
            }
            else if (attribute_name == "value_ints") {
               auto values = std::vector<int64_t>({nodeproto.attribute(0).ints().begin(), nodeproto.attribute(0).ints().end()});
               shape.push_back(values.size());
               op.reset(new ROperator_Constant<int64_t>("int64_t",values, shape, input_name, output_name));
            } else {
               throw std::runtime_error("TMVA::SOFIE ONNX Parser Constant op: not yet supporting attribute " + attribute_name);
            }
         } else {
            throw std::runtime_error("TMVA::SOFIE ONNX Parser ConstantOfShape op: parsed invalid attribute " + attribute_name);
         }
      }

   // case when there is no attribute
   }  else {
      // case of Constant of Shape : if attribute is not there use by default float type with zero values
      if (isConstantOfShape) {
         std::vector<float> values(1);
         std::vector<size_t> constantShape(1,1);
         op.reset(new ROperator_Constant<float>("float",values,constantShape, input_name, output_name));
      } else {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Constant has no attribute");
      }
   }

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, output_type);
   }

   if (parser.Verbose())
      std::cout << "\t ParseConstant: operator created\n";

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA