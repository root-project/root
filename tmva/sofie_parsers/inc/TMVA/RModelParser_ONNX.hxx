#ifndef TMVA_SOFIE_RMODELPARSER_ONNX
#define TMVA_SOFIE_RMODELPARSER_ONNX



#include "TMVA/SOFIE_common.hxx"
#include "TMVA/RModel.hxx"
#include "TMVA/OperatorList.hxx"

#include <string>
#include <fstream>
#include <memory>
#include <ctime>

//forward delcaration
namespace onnx{
   class NodeProto;
   class GraphProto;
}

namespace TMVA{
namespace Experimental{
namespace SOFIE{

namespace INTERNAL{

std::unique_ptr<ROperator> make_ROperator_Transpose(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_Relu(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_Selu(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_Sigmoid(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_Gemm(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_Conv(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_RNN(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_LSTM(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_BatchNormalization(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);

std::unique_ptr<ROperator> make_ROperator_Pool(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);

using factoryMethodMap = std::unordered_map<std::string, std::unique_ptr<ROperator> (*)(const onnx::NodeProto&, const onnx::GraphProto&, std::unordered_map<std::string, ETensorType>&)>;
const factoryMethodMap mapOptypeOperator = {
      {"Gemm", &make_ROperator_Gemm},
      {"Transpose", &make_ROperator_Transpose},
      {"Relu", &make_ROperator_Relu},
      {"Conv", &make_ROperator_Conv},
      {"RNN", &make_ROperator_RNN},
      {"Selu", &make_ROperator_Selu},
      {"Sigmoid", &make_ROperator_Sigmoid},
      {"LSTM", &make_ROperator_LSTM},
      {"BatchNormalization", &make_ROperator_BatchNormalization},
      {"AveragePool", &make_ROperator_Pool},
      {"MaxPool", &make_ROperator_Pool}
   };


std::unique_ptr<ROperator> make_ROperator(size_t idx, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
}//INTERNAL



class RModelParser_ONNX{
public:
   RModel Parse(std::string filename);
};



}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RMODELPARSER_ONNX
