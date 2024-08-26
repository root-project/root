#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_SubGraph.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseIf = [] (RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   //ETensorType input_type;

   auto input_name = nodeproto.input(0);
   if (!parser.IsRegisteredTensorType(input_name)) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser If op has input tensor" + input_name +
                               " but its type is not yet registered");
   }
   // attributes containing the graphs
   if (nodeproto.attribute_size() != 2) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser If op has not 2 attributes");
   }
   int then_index = -1;
   int else_index = -1;
   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "then_branch") {
         then_index = i;
      }
      else if (attribute_name == "else_branch") {
         else_index = i;
      }
   }
   if (else_index < 0 || then_index < 0)
     throw std::runtime_error("TMVA::SOFIE ONNX Parser If has wrong attributes");

   auto then_graph = nodeproto.attribute(then_index).g();
   auto else_graph = nodeproto.attribute(else_index).g();
   // need here to parse the graphs

   auto model_then = std::make_unique<RModel>(then_graph.name(),"");
   auto model_else = std::make_unique<RModel>(else_graph.name(),"");

   std::vector<std::string> outputNames;
   for (int i = 0; i < nodeproto.output_size(); i++) {
      outputNames.push_back(nodeproto.output(i));
   }
   parser.ParseONNXGraph(*model_then, then_graph);
   parser.ParseONNXGraph(*model_else, else_graph);


   std::unique_ptr<ROperator> op(new ROperator_If(input_name, outputNames, std::move(model_then), std::move(model_else)));
   // output of if are the output of the branches
   for (size_t i = 0; i < outputNames.size(); i++) {
      // get type from the output of subgraph models
      std::string out_g_name = then_graph.output(i).name();
      auto type = parser.GetTensorType(out_g_name);
      std::string output_name = outputNames[i];
      if (!parser.IsRegisteredTensorType(output_name))
         parser.RegisterTensorType(output_name, type);
   }
   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
