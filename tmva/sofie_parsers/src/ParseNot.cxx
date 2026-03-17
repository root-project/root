#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Not.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseNot = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
{
   ETensorType input_type = ETensorType::UNDEFINED;

   if (nodeproto.input_size() != 1 || nodeproto.output_size() != 1)
      std::runtime_error("TMVA::SOFIE ONNX Parser Not op has invalid input or output size ");

   std::string input_name = nodeproto.input(0);

   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
      std::cout << "NOT op input type is " << static_cast<int>(input_type) << "  " << ConvertTypeToString(input_type) << std::endl;
      if (input_type !=ETensorType::BOOL  && input_type !=ETensorType::UINT8 )
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Not op has invalid input type " + ConvertTypeToString(input_type));
   } else {
      throw
         std::runtime_error("TMVA::SOFIE ONNX Parser Not op has input tensor " + input_name +
                                  " but its type is not yet registered");
   }

   std::string output_name = nodeproto.output(0);
   std::unique_ptr<ROperator> op(new ROperator_Not(input_name, output_name));

   // Infer the output type
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};


} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
