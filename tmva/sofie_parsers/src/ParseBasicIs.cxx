#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Basic_Is.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <EBasicIsOperator Op>
std::unique_ptr<ROperator> ParseBasicIs(RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
{
   ETensorType input_type = ETensorType::UNDEFINED;

   std::string input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw
         std::runtime_error("TMVA::SOFIE ONNX Parser " + IsOpTraits<Op>::Name() + " op has input tensor " + input_name +
                                  " but its type is not yet registered");
   }

   // get attributes for the IsInf operator
   int detect_negative = 1;
   int detect_positive = 1;
   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "detect_negative")
         detect_negative = nodeproto.attribute(i).i();
       if (attribute_name == "detect_positive")
         detect_positive = nodeproto.attribute(i).i();
   }

   if (detect_positive == 0 && detect_negative == 0)
      throw std::runtime_error("TMVA::SOFIE ONNX Parser IsInf op has invalide attributes");


   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   if (nodeproto.attribute_size() == 0 || (detect_negative == 1 && detect_positive == 1))
      op.reset(new ROperator_Basic_Is<Op>(input_name, output_name));
   else if (nodeproto.attribute_size() > 0) {
      // case detect_negative or detective_positive are set
      if (detect_negative == 0)
         op.reset(new ROperator_Basic_Is<EBasicIsOperator::kIsInfPos>(input_name, output_name));
      else if (detect_positive == 0)
         op.reset(new ROperator_Basic_Is<EBasicIsOperator::kIsInfNeg>(input_name, output_name));
   } else
      throw std::runtime_error("TMVA::SOFIE ONNX Parser " + IsOpTraits<Op>::Name() + " operator - invalid attributes");

   // Infer the output type
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

// Parse IsNaN
ParserFuncSignature ParseIsNaN = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicIs<EBasicIsOperator::kIsNaN>(parser, nodeproto);
};

// Parse IsInf
ParserFuncSignature ParseIsInf = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicIs<EBasicIsOperator::kIsInf>(parser, nodeproto);
};


} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
