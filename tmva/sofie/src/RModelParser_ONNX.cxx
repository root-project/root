#include "RModelParser_ONNX.hxx"

#include <string>
#include <memory>

namespace TMVA{
namespace Experimental{
namespace SOFIE{



namespace INTERNAL{

std::unique_ptr<ROperator> make_ROperator(size_t idx, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type){
   const auto& nodeproto = graphproto.node(idx);
   auto find = mapOptypeOperator.find(nodeproto.op_type());
   if (find == mapOptypeOperator.end()){
      throw std::runtime_error("TMVA::SOFIE - Operator type " + nodeproto.op_type() + " is not yet supported");
   }else{
      return (find->second)(nodeproto, graphproto, tensor_type);
   }
}

std::unique_ptr<ROperator> make_ROperator_Transpose(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type){

   ETensorType input_type;

   auto input_name =  nodeproto.input(0);
   auto it = tensor_type.find(input_name);
   if (it != tensor_type.end()){
      input_type = it->second;
   }else{
      throw std::runtime_error("TMVA::SOFIE ONNX Parser tranpose op has input tensor" + input_name + " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;
   std::vector<int_t> attr_perm;

   if (nodeproto.attribute_size() == 1){
      attr_perm.assign(nodeproto.attribute(0).ints().begin(), nodeproto.attribute(0).ints().end());
   }

   switch(input_type){
   case ETensorType::FLOAT:
      if (attr_perm.size() != 0){
         op.reset(new ROperator_Transpose<float>(attr_perm, nodeproto.input(0), nodeproto.output(0)));
      }else{
         op.reset(new ROperator_Transpose<float> (nodeproto.input(0), nodeproto.output(0)));
      }
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Transpose does not yet support input type " + std::to_string(static_cast<int>(input_type)));
   }

   ETensorType output_type = (op->TypeInference({input_type}))[0];
   auto it2 = tensor_type.find(nodeproto.output(0));
   if (it2 == tensor_type.end()){
      tensor_type[nodeproto.output(0)] = output_type;
   }


   return std::move(op);
}

std::unique_ptr<ROperator> make_ROperator_Relu(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type){

   ETensorType input_type;

   auto input_name =  nodeproto.input(0);
   auto it = tensor_type.find(input_name);
   if (it != tensor_type.end()){
      input_type = it->second;
   }else{
      throw std::runtime_error("TMVA::SOFIE ONNX Parser relu op has input tensor" + input_name + " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;


   switch(input_type){
   case ETensorType::FLOAT:
      op.reset(new ROperator_Relu<float>(nodeproto.input(0), nodeproto.output(0)));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Relu does not yet support input type " + std::to_string(static_cast<int>(input_type)));
   }

   ETensorType output_type = (op->TypeInference({input_type}))[0];
   auto it2 = tensor_type.find(nodeproto.output(0));
   if (it2 == tensor_type.end()){
      tensor_type[nodeproto.output(0)] = output_type;
   }

   return std::move(op);
}

std::unique_ptr<ROperator> make_ROperator_Gemm(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type){

   ETensorType input_type;

   auto input_name =  nodeproto.input(0);
   auto it = tensor_type.find(input_name);
   if (it != tensor_type.end()){
      input_type = it->second;
   }else{
      throw std::runtime_error("TMVA::SOFIE ONNX Parser gemm op has input tensor" + input_name + " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;

   float attr_alpha =1.0;
   float attr_beta =1.0;
   int_t attr_transA =0;
   int_t attr_transB =0;

   for (int i = 0; i < nodeproto.attribute_size(); i++){
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "alpha"){
         attr_alpha = nodeproto.attribute(i).f();
      }else if(attribute_name == "beta"){
         attr_beta = nodeproto.attribute(i).f();
      }else if(attribute_name == "transA"){
         attr_transA = nodeproto.attribute(i).i();
         if (attr_transA != 0 && attr_transA != 1) throw std::runtime_error("TMVA::SOFIE Error - Model Loading - attribute transA in Operator Gemm not 0/1");
      }else if(attribute_name == "transB"){
         attr_transB = nodeproto.attribute(i).i();
         if (attr_transB != 0 && attr_transB != 1) throw std::runtime_error("TMVA::SOFIE Error - Model Loading - attribute transB in Operator Gemm not 0/1");
      }else{
         std::cout << "TMVA::SOFIE Warning - Model Loading - Attribute " << attribute_name << " in OperatorNode " << nodeproto.name() << " is not defined in ONNX IR and not applied!\n";
      }
   }


   switch(input_type){
   case ETensorType::FLOAT:
      if (nodeproto.input_size() == 2){
         op.reset(new ROperator_Gemm<float>(attr_alpha, attr_beta, attr_transA, attr_transB, nodeproto.input(0), nodeproto.input(1), nodeproto.output(0)));
      }else{
         op.reset(new ROperator_Gemm<float>(attr_alpha, attr_beta, attr_transA, attr_transB, nodeproto.input(0), nodeproto.input(1), nodeproto.input(2), nodeproto.output(0)));
      }
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Relu does not yet support input type " + std::to_string(static_cast<int>(input_type)));
   }

   ETensorType output_type = (op->TypeInference({input_type, input_type}))[0];
   auto it2 = tensor_type.find(nodeproto.output(0));
   if (it2 == tensor_type.end()){
      tensor_type[nodeproto.output(0)] = output_type;
   }


   return std::move(op);
}

} //INTERNAL







}//SOFIE
}//Experimental
}//TMVA
