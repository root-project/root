#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_BasicBinary.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <EBasicBinaryOperator Op>
std::unique_ptr<ROperator> ParseBasicBinary(RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
{
   ETensorType input_type = ETensorType::UNDEFINED;

   for (int i = 0; i < 2; ++i) {
      auto input_name = nodeproto.input(i);
      if (parser.IsRegisteredTensorType(input_name)) {
         // according to ONNX both inputs have same type
         if (i == 0)
            input_type = parser.GetTensorType(input_name);
         else {
            ETensorType input_type2 = parser.GetTensorType(input_name);
            if (input_type2 != input_type) {
               throw
                  std::runtime_error("TMVA::SOFIE ONNX parser Binary op has input tensors of different types: " +
                     input_name + " : " + ConvertTypeToString(input_type2) +
                     " and " +  nodeproto.input(0) + " : " + ConvertTypeToString(input_type));
            }
         }
      } else {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Binary op has input tensor " + input_name +
                                  " but its type is not yet registered");
      }
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_BasicBinary<float, Op>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   case ETensorType::DOUBLE:
      op.reset(new ROperator_BasicBinary<double, Op>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   case ETensorType::INT32:
      op.reset(new ROperator_BasicBinary<int32_t, Op>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   case ETensorType::INT64:
      op.reset(new ROperator_BasicBinary<int64_t, Op>(nodeproto.input(0), nodeproto.input(1), output_name));
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


// Parse Add
ParserFuncSignature ParseAdd = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicBinary<EBasicBinaryOperator::Add>(parser, nodeproto);
};

// Parse Sub
ParserFuncSignature ParseSub = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicBinary<EBasicBinaryOperator::Sub>(parser, nodeproto);
};

// Parse Mul
ParserFuncSignature ParseMul = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicBinary<EBasicBinaryOperator::Mul>(parser, nodeproto);
};

// Parse Div
ParserFuncSignature ParseDiv = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicBinary<EBasicBinaryOperator::Div>(parser, nodeproto);
};

// Parse Pow
ParserFuncSignature ParsePow = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicBinary<EBasicBinaryOperator::Pow>(parser, nodeproto);
};

// Mod (and fmod) is a special case di BasicBinary

ParserFuncSignature ParseMod = [] (RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {

   ETensorType input_type = ETensorType::UNDEFINED;
   for (int i = 0; i < 2; ++i) {
      auto input_name = nodeproto.input(i);
      if (parser.IsRegisteredTensorType(input_name)) {
         // according to ONNX both inputs have same type
         if (i == 0)
            input_type = parser.GetTensorType(input_name);
         else {
            ETensorType input_type2 = parser.GetTensorType(input_name);
            if (input_type2 != input_type) {
               throw
                  std::runtime_error("TMVA::SOFIE ONNX parser Binary op has input tensors of different types: " +
                     input_name + " : " + ConvertTypeToString(input_type2) +
                     " and " +  nodeproto.input(0) + " : " + ConvertTypeToString(input_type));
            }
         }
      } else {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Binary op has input tensor " + input_name +
                                  " but its type is not yet registered");
      }
   }
   // in case of Mod there can be an attribute
   int fmod = 0;
   if (nodeproto.attribute_size() > 0) {
      fmod = nodeproto.attribute(0).i();
   }
   // case of float or double fmod must be 1
   if (input_type ==ETensorType::FLOAT || input_type ==ETensorType::FLOAT ) {
      if (fmod != 1)
         std::runtime_error("TMVA::SOFIE ONNX parser Mod operator has fmod = 0 for floating inputs");
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_BasicBinary<float,EBasicBinaryOperator::FMod >(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   case ETensorType::DOUBLE:
      op.reset(new ROperator_BasicBinary<double, EBasicBinaryOperator::FMod>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   case ETensorType::INT32:
      if (fmod == 1)
         op.reset(new ROperator_BasicBinary<int32_t, EBasicBinaryOperator::FMod>(nodeproto.input(0), nodeproto.input(1), output_name));
      else
         op.reset(new ROperator_BasicBinary<int32_t, EBasicBinaryOperator::Mod>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   case ETensorType::INT64:
      if (fmod == 1)
         op.reset(new ROperator_BasicBinary<int64_t, EBasicBinaryOperator::FMod>(nodeproto.input(0), nodeproto.input(1), output_name));
      else
         op.reset(new ROperator_BasicBinary<int64_t, EBasicBinaryOperator::Mod>(nodeproto.input(0), nodeproto.input(1), output_name));
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


} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
