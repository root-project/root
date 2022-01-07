#include "TMVA/RModelParser_ONNX.hxx"
#include "onnx_proto3.pb.h"

#include <string>
#include <memory>
#include <cassert>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

namespace INTERNAL{

std::unique_ptr<ROperator> make_ROperator(size_t idx, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type){
   const auto& nodeproto = graphproto.node(idx);
   auto find = mapOptypeOperator.find(nodeproto.op_type());
   if (find == mapOptypeOperator.end()){
      throw std::runtime_error("TMVA::SOFIE - Operator type " + nodeproto.op_type() + " is not yet supported");
      // std::unique_ptr<ROperator> op;
      // return op;
   } else {
      //std::cout << "create operator " << nodeproto.op_type() << std::endl;
      return (find->second)(nodeproto, graphproto, tensor_type);
   }
}

std::unique_ptr<ROperator> make_ROperator_Add(const onnx::NodeProto& nodeproto, const onnx::GraphProto& /*graphproto */, std::unordered_map<std::string, ETensorType>& tensor_type){

   ETensorType input_type = ETensorType::UNDEFINED;

   for (int i = 0; i < 2; ++i) {
      auto input_name = nodeproto.input(i);
      auto it = tensor_type.find(input_name);
      if (it != tensor_type.end()){
         // according to ONNX both inputs have same time
         if (i == 0) input_type = it->second;
         else
            assert(it->second == input_type);
      } else {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Add op has input tensor" + input_name + " but its type is not yet registered");
      }
   }

   std::unique_ptr<ROperator> op;

   switch(input_type){
   case ETensorType::FLOAT:
      op.reset(new ROperator_Add<float>(nodeproto.input(0), nodeproto.input(1), nodeproto.output(0)));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Add does not yet support input type " + std::to_string(static_cast<int>(input_type)));
   }

   ETensorType output_type = (op->TypeInference({input_type}))[0];
   auto it2 = tensor_type.find(nodeproto.output(0));
   if (it2 == tensor_type.end()){
      tensor_type[nodeproto.output(0)] = output_type;
   }

   return op;
}
std::unique_ptr<ROperator> make_ROperator_Transpose(const onnx::NodeProto& nodeproto, const onnx::GraphProto& /*graphproto*/, std::unordered_map<std::string, ETensorType>& tensor_type){

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
      if (!attr_perm.empty()){
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

   return op;
}

std::unique_ptr<ROperator> make_ROperator_Relu(const onnx::NodeProto& nodeproto, const onnx::GraphProto& /*graphproto */, std::unordered_map<std::string, ETensorType>& tensor_type){

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

   return op;
}

std::unique_ptr<ROperator> make_ROperator_Selu(const onnx::NodeProto& nodeproto, const onnx::GraphProto& /*graphproto */, std::unordered_map<std::string, ETensorType>& tensor_type){

   ETensorType input_type;

   auto input_name =  nodeproto.input(0);
   auto it = tensor_type.find(input_name);
   if (it != tensor_type.end()){
      input_type = it->second;
   }else{
      throw std::runtime_error("TMVA::SOFIE ONNX Parser selu op has input tensor" + input_name + " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;


   switch(input_type){
   case ETensorType::FLOAT:
      op.reset(new ROperator_Selu<float>(nodeproto.input(0), nodeproto.output(0)));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Selu does not yet support input type " + std::to_string(static_cast<int>(input_type)));
   }

   ETensorType output_type = (op->TypeInference({input_type}))[0];
   auto it2 = tensor_type.find(nodeproto.output(0));
   if (it2 == tensor_type.end()){
      tensor_type[nodeproto.output(0)] = output_type;
   }

   return op;
}

std::unique_ptr<ROperator> make_ROperator_Sigmoid(const onnx::NodeProto& nodeproto, const onnx::GraphProto& /*graphproto */, std::unordered_map<std::string, ETensorType>& tensor_type){

   ETensorType input_type;

   auto input_name =  nodeproto.input(0);
   auto it = tensor_type.find(input_name);
   if (it != tensor_type.end()){
      input_type = it->second;
   }else{
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Sigmoid op has input tensor" + input_name + " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;


   switch(input_type){
   case ETensorType::FLOAT:
      op.reset(new ROperator_Sigmoid<float>(nodeproto.input(0), nodeproto.output(0)));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Sigmoid does not yet support input type " + std::to_string(static_cast<int>(input_type)));
   }

   ETensorType output_type = (op->TypeInference({input_type}))[0];
   auto it2 = tensor_type.find(nodeproto.output(0));
   if (it2 == tensor_type.end()){
      tensor_type[nodeproto.output(0)] = output_type;
   }

   return op;
}

std::unique_ptr<ROperator> make_ROperator_Gemm(const onnx::NodeProto& nodeproto, const onnx::GraphProto& /* graphproto */, std::unordered_map<std::string, ETensorType>& tensor_type){

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

   return op;
}

std::unique_ptr<ROperator> make_ROperator_Conv(const onnx::NodeProto& nodeproto, const onnx::GraphProto& /* graphproto */, std::unordered_map<std::string, ETensorType>& tensor_type) {

   ETensorType input_type;

   auto input_name = nodeproto.input(0);
   auto it = tensor_type.find(input_name);
   if (it != tensor_type.end()) {
      input_type = it->second;
   } else {
      throw
         std::runtime_error("TMVA::SOFIE ONNX Parser Conv op has input tensor " + input_name + " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;

   std::string attr_auto_pad = "NOTSET";
   std::vector<size_t> attr_dilations;
   size_t attr_group = 0;
   std::vector<size_t> attr_kernel_shape;
   std::vector<size_t> attr_pads;
   std::vector<size_t> attr_strides;

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "auto_pad") {
         attr_auto_pad = nodeproto.attribute(i).s();
      } else if (attribute_name == "dilations") {
         attr_dilations = std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attribute_name == "group") {
         attr_group= nodeproto.attribute(i).i();
      } else if (attribute_name == "kernel_shape") {
         attr_kernel_shape = std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attribute_name == "pads") {
         attr_pads = std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attribute_name == "strides") {
         attr_strides = std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else {
         std::cout << "TMVA::SOFIE Warning - Model Loading - Attribute " << attribute_name << " in OperatorNode " << nodeproto.name() << " is not defined in ONNX IR and not applied!\n";
      }
   }

   std::string name_b = "";
   if (nodeproto.input_size() > 2) {
      name_b = nodeproto.input(2);
   }

   switch(input_type) {
      case ETensorType::FLOAT:
         op.reset(new ROperator_Conv<float>(attr_auto_pad, attr_dilations, attr_group, attr_kernel_shape, attr_pads, attr_strides, nodeproto.input(0), nodeproto.input(1), name_b, nodeproto.output(0)));
         break;
      default:
         throw
            std::runtime_error("TMVA::SOFIE - Unsupported - Operator Conv does not yet support input type " + std::to_string(static_cast<int>(input_type)));
   }

   ETensorType output_type = (op->TypeInference({input_type, input_type}))[0];
   auto it2 = tensor_type.find(nodeproto.output(0));
   if (it2 == tensor_type.end()) {
      tensor_type[nodeproto.output(0)] = output_type;
   }

   return op;
}

std::unique_ptr<ROperator> make_ROperator_Pool(const onnx::NodeProto& nodeproto, const onnx::GraphProto& /* graphproto */, std::unordered_map<std::string, ETensorType>& tensor_type) {

   ETensorType input_type;

   PoolOpMode op_mode = InvalidPool;
   if (nodeproto.op_type() == "MaxPool")
      op_mode = MaxPool;
   else if (nodeproto.op_type() == "AveragePool")
      op_mode = AveragePool;
   else if (nodeproto.op_type() == "GlobalAveragePool")
      op_mode = GlobalAveragePool;

   assert(op_mode != InvalidPool);

   auto input_name = nodeproto.input(0);
   auto it = tensor_type.find(input_name);
   if (it != tensor_type.end()) {
      input_type = it->second;
   } else {
      throw
         std::runtime_error("TMVA::SOFIE ONNX Parser Pool op has input tensor " + input_name + " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;

   RAttributes_Pool attr;
   // std::string attr_auto_pad = "NOTSET";
   // int attr_ceil_mode = 0;
   // int attr_count_include_pad = 0;      
   // int attr_storage_order = 0;          // not for AveragePool
   // std::vector<size_t> attr_dilations;  // not for AveragePool
   // std::vector<size_t> attr_kernel_shape;
   // std::vector<size_t> attr_pads;
   // std::vector<size_t> attr_strides;

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "auto_pad") {
         attr.auto_pad = nodeproto.attribute(i).s();
      } else if (attribute_name == "ceil_mode") {
         attr.ceil_mode = nodeproto.attribute(i).i();
      } else if (attribute_name == "count_include_pad" && op_mode == AveragePool) {
         attr.count_include_pad = nodeproto.attribute(i).i();
      } else if (attribute_name == "storage_order" && op_mode == MaxPool) {
         attr.storage_order = nodeproto.attribute(i).i();
      } else if (attribute_name == "dilations" && op_mode == MaxPool) {
         attr.dilations = std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attribute_name == "kernel_shape") {
         attr.kernel_shape = std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attribute_name == "pads") {
         attr.pads = std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attribute_name == "strides") {
         attr.strides = std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else {
         std::cout << "TMVA::SOFIE Warning - Model Loading - Attribute " << attribute_name << " in OperatorNode "
                   << nodeproto.name() << " is not defined in ONNX IR and not applied!\n";
      }
   }

   switch(input_type) {
      case ETensorType::FLOAT:
         op.reset(new ROperator_Pool<float>(op_mode, attr, nodeproto.input(0), nodeproto.output(0)));
         break;
      default:
         throw
            std::runtime_error("TMVA::SOFIE - Unsupported - Operator Pool does not yet support input type " + std::to_string(static_cast<int>(input_type)));
   }

   ETensorType output_type = (op->TypeInference({input_type}))[0];
   auto it2 = tensor_type.find(nodeproto.output(0));
   if (it2 == tensor_type.end()) {
      tensor_type[nodeproto.output(0)] = output_type;
   }

   return op;
}

std::unique_ptr<ROperator> make_ROperator_Reshape(const onnx::NodeProto &nodeproto,
                                                  const onnx::GraphProto & /*graphproto */,
                                                  std::unordered_map<std::string, ETensorType> &tensor_type)
{
   // make Reshape operator
   ETensorType input_type = ETensorType::UNDEFINED;

   
   ReshapeOpMode opMode = Reshape;
   if (nodeproto.op_type() == "Flatten") 
      opMode = Flatten;
   else if (nodeproto.op_type() == "Squeeze") 
      opMode = Squeeze;
   else if (nodeproto.op_type() == "Unsqueeze")
      opMode = Unsqueeze;

  
   //bool hasShapeInput = (opMode == Reshape) ? true : false;

   // reshape has as extra input shape tensor (int64) but 
   // it is not present for Flatten, Squeeze and Unsquueze
   auto input_name = nodeproto.input(0);
   // for squeeze is optional ? 
   auto shape_name = (opMode == Reshape || opMode == Unsqueeze) ? nodeproto.input(1) : "";
   auto it = tensor_type.find(input_name);
   if (it != tensor_type.end()) {
      input_type = it->second;
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Reshape op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   // Reshape is having one attribute: allowzero (int) (default = 0)
   // Flatten is having one attribute: axis (int) (default=1)
   // old version of reshape and squeeze have axes as attributes
   std::unique_ptr<ROperator> op;
   int attr_value = (opMode == Reshape) ? 0 : 1;     
   if (opMode == Reshape && nodeproto.attribute_size() > 0 )
      attr_value = nodeproto.attribute(0).i();

   std::vector<int64_t> attr_axes = {};
   if (nodeproto.input_size() == 1 && (opMode == Squeeze || opMode == Unsqueeze)) {
      std::string attribute_name = nodeproto.attribute(0).name();
      if (attribute_name == "axes")
         attr_axes = {nodeproto.attribute(0).ints().begin(), nodeproto.attribute(0).ints().end()};
   }

   switch (input_type) {
   case ETensorType::FLOAT:
      if (attr_axes.empty())
         op.reset(new ROperator_Reshape<float>(opMode, attr_value, input_name, shape_name, nodeproto.output(0)));
      else // for old Squeeze and Unsqueeze
         op.reset(new ROperator_Reshape<float>(opMode, attr_axes, input_name, nodeproto.output(0)));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Reshape does not yet support input type " +
                               std::to_string(static_cast<int>(input_type)));
   }

   ETensorType output_type = (op->TypeInference({input_type}))[0];
   auto it2 = tensor_type.find(nodeproto.output(0));
   if (it2 == tensor_type.end()) {
      tensor_type[nodeproto.output(0)] = output_type;
   }

   return op;
}

std::unique_ptr<ROperator> make_ROperator_Slice(const onnx::NodeProto &nodeproto,
                                                  const onnx::GraphProto & /*graphproto */,
                                                  std::unordered_map<std::string, ETensorType> &tensor_type)
{
   // make Slice operator
   ETensorType input_type = ETensorType::UNDEFINED;

   auto input_name = nodeproto.input(0);
   auto it = tensor_type.find(input_name);
   if (it != tensor_type.end()) {
      input_type = it->second;
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Slice op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   std::vector<std::string> axisTensorNames;
   if (nodeproto.input_size() > 1)
      axisTensorNames.push_back(nodeproto.input(1));
   if (nodeproto.input_size() > 2)
      axisTensorNames.push_back(nodeproto.input(1));
   if (nodeproto.input_size() > 3)
      axisTensorNames.push_back(nodeproto.input(3));
   if (nodeproto.input_size() > 4)
      axisTensorNames.push_back(nodeproto.input(4));

   // not sure how to find here type of the integer inputs
   //std::cout << "Slice input(1) " << nodeproto.input(1) << "  " << nodeproto.input(2) << std::endl;
   ETensorType axis_type = ETensorType::INT64;
   //(tensor_type.find(starts_name) != tensor_type.end()) ? tensor_type.find(starts_name)->second
   //                                                                             : ETensorType::UNDEFINED;
   // for version < 10
    std::vector<int64_t> attr_starts = {};
    std::vector<int64_t> attr_ends = {};
    std::vector<int64_t> attr_axes = {};
    if (nodeproto.input_size() == 1) {
       for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
          std::string attribute_name = nodeproto.attribute(i).name();
          if (attribute_name == "starts")
             attr_starts = {nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()};
          if (attribute_name == "ends")
             attr_ends = {nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()};
          if (attribute_name == "axes")
             attr_axes = {nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()};
       }
    }

   std::unique_ptr<ROperator> op;
   switch (input_type) {
   case ETensorType::FLOAT:
      if (axisTensorNames.size() > 0) {
         // for version >= 10
         if (axis_type == ETensorType::INT32)
            op.reset(new ROperator_Slice<float, int32_t>(input_name, axisTensorNames, nodeproto.output(0)));
         else if (axis_type == ETensorType::INT64)
            op.reset(new ROperator_Slice<float, int64_t>(input_name, axisTensorNames, nodeproto.output(0)));
         else
            throw std::runtime_error(
               "TMVA::SOFIE - Unsupported - Operator Slice has invalid input type for input axis descriptors " +
               std::to_string(static_cast<int>(axis_type)));
      } else if (attr_starts.size() > 0 && attr_ends.size() > 0) {
         op.reset(
            new ROperator_Slice<float, int64_t>(input_name, attr_starts, attr_ends, attr_axes, nodeproto.output(0)));
      } else {
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Slice has invalid attribues");
      }
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Slice does not yet support input type " +
                               std::to_string(static_cast<int>(input_type)));
   }

   ETensorType output_type = (op->TypeInference({input_type}))[0];
   auto it2 = tensor_type.find(nodeproto.output(0));
   if (it2 == tensor_type.end()) {
      tensor_type[nodeproto.output(0)] = output_type;
   }

   return op;
}

std::unique_ptr<ROperator> make_ROperator_RNN(const onnx::NodeProto& nodeproto, const onnx::GraphProto& /* graphproto */, std::unordered_map<std::string, ETensorType>& tensor_type) {

   ETensorType input_type;

   auto input_name = nodeproto.input(0);
   auto it = tensor_type.find(input_name);
   if (it != tensor_type.end()) {
      input_type = it->second;
   } else {
      throw
         std::runtime_error("TMVA::SOFIE ONNX Parser RNN op has input tensor " + input_name + " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;

   std::vector<float> attr_activation_alpha = {};
   std::vector<float> attr_activation_beta = {};
   std::vector<std::string> attr_activations = {};
   float attr_clip = 0.;
   std::string attr_direction = "forward";
   size_t attr_hidden_size = 0;
   size_t attr_layout = 0;

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "activation_alpha") {
         attr_activation_alpha = {nodeproto.attribute(i).floats().begin(), nodeproto.attribute(i).floats().end()};
      } else if (attribute_name == "activation_beta") {
         attr_activation_beta = {nodeproto.attribute(i).floats().begin(), nodeproto.attribute(i).floats().end()};
      } else if (attribute_name == "activations") {
         attr_activations = {nodeproto.attribute(i).strings().begin(), nodeproto.attribute(i).strings().end()};
      } else if (attribute_name == "clip") {
         attr_clip = nodeproto.attribute(i).i();
      } else if (attribute_name == "direction") {
         attr_direction = nodeproto.attribute(i).s();
      } else if (attribute_name == "hidden_size") {
         attr_hidden_size = nodeproto.attribute(i).i();
      } else if (attribute_name == "layout") {
         attr_layout = nodeproto.attribute(i).i();
      } else {
         std::cout << "TMVA SOFIE Warning - Model Loading - Attribute " << attribute_name << " in OperatorNode " << nodeproto.name() << " is not defined in ONNX IR and not applied!\n";
      }
   }

   // Optional inputs and outputs
   std::string name_b = "";
   std::string name_sequence_lens = "";
   std::string name_initial_h = "";
   std::string name_y = "";
   std::string name_y_h = "";
   if (nodeproto.input_size() > 3) {
      name_b = nodeproto.input(3);
   }
   if (nodeproto.input_size() > 4) {
      name_sequence_lens = nodeproto.input(4);
   }
   if (nodeproto.input_size() > 5) {
      name_initial_h = nodeproto.input(5);
   }
   if (nodeproto.output_size() > 0) {
      name_y = nodeproto.output(0);
   }
   if (nodeproto.output_size() > 1) {
      name_y_h = nodeproto.output(1);
   }

   switch(input_type) {
      case ETensorType::FLOAT:
            op.reset(new ROperator_RNN<float>(attr_activation_alpha, attr_activation_beta, attr_activations,
               attr_clip, attr_direction, attr_hidden_size, attr_layout,
               nodeproto.input(0), nodeproto.input(1), nodeproto.input(2),
               name_b, name_sequence_lens, name_initial_h, name_y, name_y_h));
         break;
      default:
         throw
            std::runtime_error("TMVA::SOFIE - Unsupported - Operator RNN does not yet support input type " + std::to_string(static_cast<int>(input_type)));
   }

   auto output_type = op->TypeInference({input_type, input_type});
   for (size_t i = 0; i < 2; i++) {
      if (tensor_type.find(nodeproto.output(i)) == tensor_type.end()) {
         tensor_type[nodeproto.output(i)] = output_type[i];
      }
   }

   return op;
}

std::unique_ptr<ROperator> make_ROperator_LSTM(const onnx::NodeProto& nodeproto, const onnx::GraphProto& /* graphproto */, std::unordered_map<std::string, ETensorType>& tensor_type) {

   ETensorType input_type;

   auto input_name = nodeproto.input(0);
   auto it = tensor_type.find(input_name);
   if (it != tensor_type.end()) {
      input_type = it->second;
   } else {
      throw
         std::runtime_error("TMVA::SOFIE ONNX Parser LSTM op has input tensor " + input_name + " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;

   std::vector<float> attr_activation_alpha;
   std::vector<float> attr_activation_beta;
   std::vector<std::string> attr_activations;
   float attr_clip = 0.;
   std::string attr_direction = "forward";
   size_t attr_hidden_size = 0;
   size_t attr_input_forget = 0;
   size_t attr_layout = 0;

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "activation_alpha") {
         attr_activation_alpha = {nodeproto.attribute(i).floats().begin(), nodeproto.attribute(i).floats().end()};
      } else if (attribute_name == "activation_beta") {
         attr_activation_beta = {nodeproto.attribute(i).floats().begin(), nodeproto.attribute(i).floats().end()};
      } else if (attribute_name == "activations") {
         attr_activations = {nodeproto.attribute(i).strings().begin(), nodeproto.attribute(i).strings().end()};
      } else if (attribute_name == "clip") {
         attr_clip = nodeproto.attribute(i).f();
      } else if (attribute_name == "direction") {
         attr_direction = nodeproto.attribute(i).s();
      } else if (attribute_name == "hidden_size") {
         attr_hidden_size = nodeproto.attribute(i).i();
      } else if (attribute_name == "input_forget") {
         attr_input_forget = nodeproto.attribute(i).i();
      } else if (attribute_name == "layout") {
         attr_layout = nodeproto.attribute(i).i();
      } else {
         std::cout << "TMVA SOFIE Warning - Model Loading - Attribute " << attribute_name << " in OperatorNode " << nodeproto.name() << " is not defined in ONNX IR and not applied!\n";
      }
   }

   // Optional inputs and outputs
   std::string name_b;
   std::string name_sequence_lens;
   std::string name_initial_h;
   std::string name_initial_c;
   std::string name_p;
   std::string name_y;
   std::string name_y_h;
   std::string name_y_c;
   if (nodeproto.input_size() > 3) {
      name_b = nodeproto.input(3);
   }
   if (nodeproto.input_size() > 4) {
      name_sequence_lens = nodeproto.input(4);
   }
   if (nodeproto.input_size() > 5) {
      name_initial_h = nodeproto.input(5);
   }
   if (nodeproto.input_size() > 6) {
      name_initial_c = nodeproto.input(6);
   }
   if (nodeproto.input_size() > 7) {
      name_p = nodeproto.input(7);
   }
   if (nodeproto.output_size() > 0) {
      name_y = nodeproto.output(0);
   }
   if (nodeproto.output_size() > 1) {
      name_y_h = nodeproto.output(1);
   }
   if (nodeproto.output_size() > 2) {
      name_y_c = nodeproto.output(2);
   }

   switch(input_type) {
      case ETensorType::FLOAT:
         op.reset(new ROperator_LSTM<float>(attr_activation_alpha, attr_activation_beta, attr_activations,
            attr_clip, attr_direction, attr_hidden_size, attr_input_forget, attr_layout,
            nodeproto.input(0), nodeproto.input(1), nodeproto.input(2),
            name_b, name_sequence_lens, name_initial_h, name_initial_c, name_p,
            name_y, name_y_h, name_y_c));
         break;
      default:
         throw
            std::runtime_error("TMVA::SOFIE - Unsupported - Operator LSTM does not yet support input type " + std::to_string(static_cast<int>(input_type)));
   }

   auto output_type = op->TypeInference({input_type, input_type});
   for (size_t i = 0; i < 2; i++) {
      if (tensor_type.find(nodeproto.output(i)) == tensor_type.end()) {
         tensor_type[nodeproto.output(i)] = output_type[i];
      }
   }

   return op;
}
std::unique_ptr<ROperator> make_ROperator_BatchNormalization(const onnx::NodeProto &nodeproto,
                                                             const onnx::GraphProto &/*graphproto*/,
                                                             std::unordered_map<std::string, ETensorType> &tensor_type)
{

   ETensorType input_type;

   auto input_name = nodeproto.input(0);
   auto it = tensor_type.find(input_name);
   if (it != tensor_type.end()) {
      input_type = it->second;
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser BatchNorm op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;
   float fepsilon = 1e-05;
	float fmomentum = 0.9;
	std::size_t ftraining_mode = 0;

   switch(input_type) {
      case ETensorType::FLOAT:
         if (nodeproto.input_size() == 5) {
            op.reset(new ROperator_BatchNormalization<float>(fepsilon, fmomentum, ftraining_mode, nodeproto.input(0), nodeproto.input(1), nodeproto.input(2), nodeproto.input(3), nodeproto.input(4), nodeproto.output(0)));
         }
         break;
      default:
         throw
            std::runtime_error("TMVA::SOFIE - Unsupported - Operator BatchNorm does not yet support input type " + std::to_string(static_cast<int>(input_type)));
   }

   ETensorType output_type = (op->TypeInference({input_type, input_type, input_type, input_type, input_type}))[0];
   auto it2 = tensor_type.find(nodeproto.output(0));
   if (it2 == tensor_type.end()) {
      tensor_type[nodeproto.output(0)] = output_type;
   }

   return op;
}

} //INTERNAL



RModel RModelParser_ONNX::Parse(std::string filename){
   char sep = '/';
   #ifdef _WIN32
      sep = '\\';
   #endif
   size_t isep = filename.rfind(sep, filename.length());
   std::string filename_nodir = filename;
   if (isep != std::string::npos){
      filename_nodir = (filename.substr(isep+1, filename.length() - isep));
   }



   std::time_t ttime = std::time(0);
   std::tm* gmt_time = std::gmtime(&ttime);
   std::string parsetime (std::asctime(gmt_time));




   GOOGLE_PROTOBUF_VERIFY_VERSION;
   //model I/O
   onnx::ModelProto model;
   RModel rmodel(filename_nodir, parsetime);

   std::unordered_map<std::string, ETensorType> tensor_type;

   std::fstream input(filename, std::ios::in | std::ios::binary);
   if (!model.ParseFromIstream(&input)){
      throw std::runtime_error("TMVA::SOFIE - Failed to parse onnx file");
   }

   const onnx::GraphProto& graph = model.graph(); //not a memory leak. model freed automatically at the end.
   google::protobuf::ShutdownProtobufLibrary();

   // ONNX version is ir_version()  - model_version() returns 0
   // std::cout << "ONNX Version " << model.ir_version() << std::endl;

   std::unordered_set<std::string> initializer_names;
   for (int i=0; i < graph.initializer_size(); i++){
      initializer_names.insert(graph.initializer(i).name());
   }


   for (int i=0; i < graph.input_size(); i++){

      tensor_type[graph.input(i).name()] = static_cast<ETensorType>(graph.input(i).type().tensor_type().elem_type());

      if (initializer_names.find(graph.input(i).name()) != initializer_names.end())  continue;

      //input datanode is not a weight node (has no initializer)
      const onnx::ValueInfoProto& valueinfoproto = graph.input(i);
      std::string input_name = valueinfoproto.name();

      ETensorType type = static_cast<ETensorType>(valueinfoproto.type().tensor_type().elem_type());
      if (type != ETensorType::FLOAT && type != ETensorType::INT32 && type != ETensorType::INT64) {
         throw std::runtime_error("TMVA::SOFIE Data type in input tensor " + input_name + " not supported!\n");
      }

      std::vector<Dim> fShape;
      bool existParam = false;
      if (!valueinfoproto.type().tensor_type().has_shape()) throw std::runtime_error("TMVA::SOFIE datanode with no shape restrictions is not supported yet");
      for (int j = 0; j < valueinfoproto.type().tensor_type().shape().dim_size(); j++){
         Dim dim;
         if (valueinfoproto.type().tensor_type().shape().dim(j).value_case() == onnx::TensorShapeProto_Dimension::ValueCase::kDimValue){
            dim.dim = valueinfoproto.type().tensor_type().shape().dim(j).dim_value();
         }else if (valueinfoproto.type().tensor_type().shape().dim(j).value_case() == onnx::TensorShapeProto_Dimension::ValueCase::kDimParam){
            dim.isParam = true;
            existParam = true;
            dim.param = valueinfoproto.type().tensor_type().shape().dim(j).dim_param();
         }else{
            throw std::runtime_error("TMVA::SOFIE ONNX file error: Valueinfoproto " + input_name + " has neither dim_value nor dim_param! \n");
         }
         fShape.push_back(dim);
      }
      if (valueinfoproto.type().tensor_type().shape().dim_size() == 0){
         Dim dim;
         dim.dim = 1;
         fShape.push_back(dim);
      } //in case this TensorShapeProto has no dimension message: ONNX IR defines this to be a scalar

      if (!existParam){
         std::vector<size_t> fShape_sizet;
         for (auto& j: fShape){
            fShape_sizet.push_back(j.dim);
         }

         rmodel.AddInputTensorInfo(input_name, type, fShape_sizet);
      }else{
         rmodel.AddInputTensorInfo(input_name, type, fShape);
      }

   }

   for (int i=0; i < graph.initializer_size(); i++){
      onnx::TensorProto* tensorproto = const_cast<onnx::TensorProto*>(&graph.initializer(i));
      std::vector<std::size_t> fShape;
      std::size_t fLength = 1;
      for (int j = 0; j < tensorproto->dims_size(); j++){
         fShape.push_back(tensorproto->dims(j));
         fLength *= tensorproto->dims(j);
      }

      std::string input_name = graph.initializer(i).name();

      switch(static_cast<ETensorType>(graph.initializer(i).data_type())){
         case ETensorType::FLOAT : {
            //void* data = malloc (fLength * sizeof(float));
            std::shared_ptr<void> data(malloc(fLength * sizeof(float)), free);

            if (tensorproto->raw_data().empty() == false){
               auto raw_data_ptr = reinterpret_cast<float*>(const_cast<char*>(tensorproto->raw_data().c_str()));
               std::memcpy(data.get(), raw_data_ptr, fLength * sizeof(float));
            }else{
               tensorproto->mutable_float_data()->ExtractSubrange(0, tensorproto->float_data_size(), static_cast<float*>(data.get()));
            }

            rmodel.AddInitializedTensor(input_name, ETensorType::FLOAT, fShape, data);
            break;
         }
         default: throw std::runtime_error("Data type in weight tensor " + graph.initializer(i).name() + " not supported!\n");
      }
   }



   for (int i=0; i < graph.node_size(); i++){
      auto op = INTERNAL::make_ROperator(i, graph, tensor_type);
      if (!op) {
         break;
      }
      rmodel.AddOperator(std::move(op));
      std::string op_type = graph.node(i).op_type();
      if (op_type == "Gemm") {
         rmodel.AddBlasRoutines({"Gemm", "Gemv"});
      } else if (op_type == "Conv") {
         rmodel.AddBlasRoutines({"Gemm", "Axpy"});
      } else if (op_type == "RNN") {
         rmodel.AddBlasRoutines({"Gemm", "Axpy"});
      } else if (op_type == "Selu" || op_type == "Sigmoid") {
         rmodel.AddNeededStdLib("cmath");
      } else if (op_type == "LSTM") {
         rmodel.AddBlasRoutines({"Gemm", "Axpy"});
      } else if (op_type == "BatchNormalization") {
         rmodel.AddBlasRoutines({"Copy", "Axpy"});
      }
   }

   std::vector<std::string> outputnames;
   for (int i=0; i < graph.output_size(); i++){
      outputnames.push_back(graph.output(i).name());
   }
   rmodel.AddOutputTensorNameList(outputnames);

   return rmodel;

}



}//SOFIE
}//Experimental
}//TMVA
