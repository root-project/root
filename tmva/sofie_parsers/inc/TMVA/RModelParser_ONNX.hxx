#ifndef TMVA_SOFIE_RMODELPARSER_ONNX
#define TMVA_SOFIE_RMODELPARSER_ONNX



#include "TMVA/SOFIE_common.hxx"
#include "TMVA/RModel.hxx"
#include "TMVA/OperatorList.hxx"

#include <string>
#include <fstream>
#include <memory>
#include <ctime>

//forward declaration
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
std::unique_ptr<ROperator> make_ROperator_Tanh(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_LeakyRelu(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_Selu(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_Sigmoid(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_Gemm(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_Conv(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_RNN(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_LSTM(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_BatchNormalization(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_Pool(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_GemmFromMatMulandAdd(const onnx::NodeProto& nodeproto1,const onnx::NodeProto& nodeproto2, const onnx::GraphProto& graphproto , std::unordered_map<std::string, ETensorType>& tensor_type);
std::unique_ptr<ROperator> make_ROperator_Reshape(const onnx::NodeProto &nodeproto, const onnx::GraphProto &graphproto, std::unordered_map<std::string, ETensorType> &tensor_type);
std::unique_ptr<ROperator> make_ROperator_Slice(const onnx::NodeProto &nodeproto, const onnx::GraphProto &graphproto, std::unordered_map<std::string, ETensorType> &tensor_type);
std::unique_ptr<ROperator> make_ROperator_GRU(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
template <EBasicBinaryOperator Op1>
std::unique_ptr<ROperator> make_ROperator_BasicBinary(const onnx::NodeProto &nodeproto, const onnx::GraphProto &graphproto, std::unordered_map<std::string, ETensorType> &tensor_type);
std::unique_ptr<ROperator> make_ROperator_Neg(const onnx::NodeProto &nodeproto, const onnx::GraphProto &graphproto, std::unordered_map<std::string, ETensorType> &tensor_type);
std::unique_ptr<ROperator> make_ROperator_Identity(const onnx::NodeProto &nodeproto, const onnx::GraphProto &graphproto, std::unordered_map<std::string, ETensorType> &tensor_type);
std::unique_ptr<ROperator> make_ROperator_Softmax(const onnx::NodeProto &nodeproto, const onnx::GraphProto &graphproto, std::unordered_map<std::string, ETensorType> &tensor_type);
std::unique_ptr<ROperator> make_ROperator_Max(const onnx::NodeProto &nodeproto, const onnx::GraphProto &graphproto, std::unordered_map<std::string, ETensorType> &tensor_type);
std::unique_ptr<ROperator> make_ROperator_Concat(const onnx::NodeProto &nodeproto, const onnx::GraphProto &graphproto, std::unordered_map<std::string, ETensorType> &tensor_type);
std::unique_ptr<ROperator> make_ROperator_Cast(const onnx::NodeProto &nodeproto, const onnx::GraphProto &graphproto, std::unordered_map<std::string, ETensorType> &tensor_type);
template <EReduceOpMode Op1>
std::unique_ptr<ROperator> make_ROperator_Reduce(const onnx::NodeProto &nodeproto, const onnx::GraphProto &graphproto, std::unordered_map<std::string, ETensorType> &tensor_type);
std::unique_ptr<ROperator> make_ROperator_Shape(const onnx::NodeProto &nodeproto, const onnx::GraphProto &graphproto, std::unordered_map<std::string, ETensorType> &tensor_type);

using factoryMethodMap = std::unordered_map<std::string, std::unique_ptr<ROperator> (*)(const onnx::NodeProto&, const onnx::GraphProto&, std::unordered_map<std::string, ETensorType>&)>;
const factoryMethodMap mapOptypeOperator = {
   {"Gemm", &make_ROperator_Gemm},
   {"Transpose", &make_ROperator_Transpose},
   {"Relu", &make_ROperator_Relu},
   {"Tanh", &make_ROperator_Tanh},
   {"LeakyRelu", &make_ROperator_LeakyRelu},
   {"Conv", &make_ROperator_Conv},
   {"RNN", &make_ROperator_RNN},
   {"Selu", &make_ROperator_Selu},
   {"Sigmoid", &make_ROperator_Sigmoid},
   {"LSTM", &make_ROperator_LSTM},
   {"GRU", &make_ROperator_GRU},
   {"BatchNormalization", &make_ROperator_BatchNormalization},
   {"AveragePool", &make_ROperator_Pool},
   {"GlobalAveragePool", &make_ROperator_Pool},
   {"MaxPool", &make_ROperator_Pool},
   {"Add", &make_ROperator_BasicBinary<Add>},
   {"Sub", &make_ROperator_BasicBinary<Sub>},
   {"Mul", &make_ROperator_BasicBinary<Mul>},
   {"Div", &make_ROperator_BasicBinary<Div>},
   {"Pow", &make_ROperator_BasicBinary<Pow>},
   {"Neg", &make_ROperator_Neg},
   {"ReduceMean", &make_ROperator_Reduce<ReduceMean>},
   {"ReduceSumsquare", &make_ROperator_Reduce<ReduceSumsquare>},
   {"ReduceProd", &make_ROperator_Reduce<ReduceProd>},
   {"Reshape", &make_ROperator_Reshape},
   {"Flatten", &make_ROperator_Reshape},
   {"Slice", &make_ROperator_Slice},
   {"Squeeze", &make_ROperator_Reshape},
   {"Unsqueeze", &make_ROperator_Reshape},
   {"Flatten", &make_ROperator_Reshape},
   {"Identity", &make_ROperator_Identity},
   {"Softmax", &make_ROperator_Softmax},
   {"Concat", &make_ROperator_Concat},
   {"Cast", &make_ROperator_Cast},
   {"Max", &make_ROperator_Max},
   {"Shape", &make_ROperator_Shape}
};

using factoryMethodMap1 = std::unordered_map<std::string, std::unique_ptr<ROperator> (*)(const onnx::NodeProto&,const onnx::NodeProto&, const onnx::GraphProto&, std::unordered_map<std::string, ETensorType>&)>;
const factoryMethodMap1 mapOptypeOperator1 = {
    {"MatMul", &make_ROperator_GemmFromMatMulandAdd}
};
std::unique_ptr<ROperator> make_ROperator(size_t idx, const onnx::GraphProto& graphproto, std::unordered_map<std::string, ETensorType>& tensor_type);
}//INTERNAL



class RModelParser_ONNX{
public:
   RModel Parse(std::string filename, bool verbose = false);
};



}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RMODELPARSER_ONNX
