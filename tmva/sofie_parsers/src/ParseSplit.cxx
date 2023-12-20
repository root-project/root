#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Split.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

// same function used to parse Constant and ConstantOfShape

ParserFuncSignature ParseSplit = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   std::string input_name;
   auto ninputs = nodeproto.input_size();
   // first input is input tensor
   input_name = nodeproto.input(0);
   if (!parser.IsRegisteredTensorType(input_name)) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Split op has input tensor" + input_name +
                                  "  but its type is not yet registered");
   }

   std::string split_name;
   if (ninputs > 1) {  // case of providing split tensor
      split_name = nodeproto.input(1);
      if (!parser.IsRegisteredTensorType(split_name)) {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Split op has input tensor" + split_name +
                                  "  but its type is not yet registered");
      }
   }

   int axis = 0;
   int noutputs = 0;
   // do here attribute parsing if needed
   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "axis") {
         axis = nodeproto.attribute(i).i();
      }
      else if (attribute_name == "num_outputs") {
         noutputs = nodeproto.attribute(i).i();
      }
   }

   ETensorType output_type = parser.GetTensorType(input_name);
   if (noutputs == 0)
      noutputs = nodeproto.output_size();
   else {
      if (noutputs != nodeproto.output_size())
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Split op has invalid num outputs " +
            std::to_string(noutputs) + " and " + std::to_string(nodeproto.output_size()) );
   }
   std::vector<std::string>  output_names(noutputs);
   for (int j = 0; j < noutputs; j++)
      output_names[j] = nodeproto.output(j);



   std::unique_ptr<ROperator> op(new ROperator_Split(axis, input_name, split_name, output_names));


   for (int j = 0; j < noutputs; j++) {
      if (!parser.IsRegisteredTensorType(output_names[j])) {
         parser.RegisterTensorType(output_names[j], output_type);
      }
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
