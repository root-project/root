#include "Byteswap.h"
#include "TMVA/RModelParser_ONNX.hxx"
#include "onnx_proto3.pb.h"

#include <stdexcept>
#include <string>
#include <memory>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <functional>
#include "TMVA/SOFIE_common.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

// Declaration of operators
// Unary operators
extern ParserFuncSignature ParseSqrt;
extern ParserFuncSignature ParseReciprocal;
extern ParserFuncSignature ParseNeg;
extern ParserFuncSignature ParseExp;
extern ParserFuncSignature ParseLog;
extern ParserFuncSignature ParseSin;
extern ParserFuncSignature ParseCos;
extern ParserFuncSignature ParseAbs;
// Binary operators
extern ParserFuncSignature ParseAdd;
extern ParserFuncSignature ParseSub;
extern ParserFuncSignature ParseMul;
extern ParserFuncSignature ParseDiv;
extern ParserFuncSignature ParsePow;
// Nary operators
extern ParserFuncSignature ParseMax;
extern ParserFuncSignature ParseMin;
extern ParserFuncSignature ParseMean;
extern ParserFuncSignature ParseSum;
//Comparision Operators
extern ParserFuncSignature ParseEq;
extern ParserFuncSignature ParseLess;
extern ParserFuncSignature ParseLessEq;
extern ParserFuncSignature ParseGreater;
extern ParserFuncSignature ParseGreaterEq;
// Reduce operators
extern ParserFuncSignature ParseReduceMean;
extern ParserFuncSignature ParseReduceSum;
extern ParserFuncSignature ParseReduceSumSquare;
extern ParserFuncSignature ParseReduceProd;
// Others
extern ParserFuncSignature ParseBatchNormalization;
extern ParserFuncSignature ParseConstant;
extern ParserFuncSignature ParseTranspose;
extern ParserFuncSignature ParseRelu;
extern ParserFuncSignature ParseTanh;
extern ParserFuncSignature ParseConv;
extern ParserFuncSignature ParseConvTranspose;
extern ParserFuncSignature ParseLeakyRelu;
extern ParserFuncSignature ParseSelu;
extern ParserFuncSignature ParseSigmoid;
extern ParserFuncSignature ParseGemm;
extern ParserFuncSignature ParseRNN;
extern ParserFuncSignature ParseLSTM;
extern ParserFuncSignature ParsePool;
extern ParserFuncSignature ParseReshape;
extern ParserFuncSignature ParseSlice;
extern ParserFuncSignature ParseGRU;
extern ParserFuncSignature ParseIdentity;
extern ParserFuncSignature ParseSoftmax;
extern ParserFuncSignature ParseConcat;
extern ParserFuncSignature ParseCast;
extern ParserFuncSignature ParseExpand;
extern ParserFuncSignature ParseShape;
extern ParserFuncSignature ParseMatMul;
extern ParserFuncSignature ParseLayerNormalization;
extern ParserFuncSignature ParseGather;
extern ParserFuncSignature ParseErf;
extern ParserFuncSignature ParseElu;
extern ParserFuncSignature ParseEyeLike;
extern ParserFuncSignature ParseRange;
extern ParserFuncSignature ParseTopK;
extern ParserFuncSignature ParseTile;
extern ParserFuncSignature ParseSplit;
extern ParserFuncSignature ParseIf;
extern ParserFuncSignature ParsePad;
extern ParserFuncSignature ParseWhere;
extern ParserFuncSignature ParseEinsum;
extern ParserFuncSignature ParseRandom;
extern ParserFuncSignature ParseScatterElements;
// Declaration of fused operators
extern ParserFuseFuncSignature ParseFuseConvAdd;
extern ParserFuseFuncSignature ParseFuseGemmRelu;
// extern ParserFuseFuncSignature ParseFuseBatchnormRelu;
extern ParserFuseFuncSignature ParseFuseConvTransposeAdd;
extern ParserFuseFuncSignature ParseFuseMatMulAdd;

// Definition of  RModelParser_ONNX::OperatorsMap
struct RModelParser_ONNX::OperatorsMapImpl {
   // Registered operators
   std::unordered_map<std::string, ParserFuncSignature> fOperatorsMap;
};

// helper function to get initialized tensor data
template<typename T>
struct ExtractDataFromTP {
};
// trait function to extract data from TensorProto
template<>
struct ExtractDataFromTP<float> {
   static void Copy(onnx::TensorProto * tensor, void * data) {
      tensor->mutable_float_data()->ExtractSubrange(0, tensor->float_data_size(),
                                                            static_cast<float *>(data));
   }
};
template<>
struct ExtractDataFromTP<double> {
   static void Copy(onnx::TensorProto * tensor, void * data) {
      tensor->mutable_double_data()->ExtractSubrange(0, tensor->double_data_size(),
                                                            static_cast<double *>(data));
   }
};
template<>
struct ExtractDataFromTP<int32_t> {
   static void Copy(onnx::TensorProto * tensor, void * data) {
      tensor->mutable_int32_data()->ExtractSubrange(0, tensor->int32_data_size(),
                                                            static_cast<int32_t *>(data));
   }
};
template<>
struct ExtractDataFromTP<int64_t> {
   static void Copy(onnx::TensorProto * tensor, void * data) {
      tensor->mutable_int64_data()->ExtractSubrange(0, tensor->int64_data_size(),
                                                            static_cast<int64_t *>(data));
   }
};
template<typename T>
std::shared_ptr<void> GetInitializedTensorData(onnx::TensorProto * tensorproto, size_t length) {
   std::shared_ptr<void> data(malloc(length * sizeof(T)), free);

   if (!tensorproto->raw_data().empty()) {
#ifdef R__BYTESWAP
      std::memcpy(data.get(), tensorproto->raw_data().c_str(), length * sizeof(T));
#else
      for (std::size_t k = 0; k < length; ++k)
         (reinterpret_cast<typename RByteSwap<sizeof(T)>::value_type *>(data.get()))[k] =
            RByteSwap<sizeof(T)>::bswap((reinterpret_cast<const typename RByteSwap<sizeof(T)>::value_type *>(tensorproto->raw_data().c_str()))[k]);
#endif
   } else {
      ExtractDataFromTP<T>::Copy(tensorproto, data.get());
   }
   return data;
}

// Constructor of the parser
RModelParser_ONNX::RModelParser_ONNX() noexcept : fOperatorsMapImpl(std::make_unique<OperatorsMapImpl>()) {
   // Register operators
   // Unary operators
   RegisterOperator("Sqrt", ParseSqrt);
   RegisterOperator("Reciprocal", ParseReciprocal);
   RegisterOperator("Neg", ParseNeg);
   RegisterOperator("Exp", ParseExp);
   RegisterOperator("Log", ParseLog);
   RegisterOperator("Sin", ParseSin);
   RegisterOperator("Cos", ParseCos);
   RegisterOperator("Abs", ParseAbs);
   // Binary operators
   RegisterOperator("Add", ParseAdd);
   RegisterOperator("Sub", ParseSub);
   RegisterOperator("Mul", ParseMul);
   RegisterOperator("Div", ParseDiv);
   RegisterOperator("Pow", ParsePow);
   // Nary operators
   RegisterOperator("Max", ParseMax);
   RegisterOperator("Min", ParseMin);
   RegisterOperator("Mean", ParseMean);
   RegisterOperator("Sum", ParseSum);
   //Comparision Operators
   RegisterOperator("Equal", ParseEq);
   RegisterOperator("Less", ParseLess);
   RegisterOperator("LessOrEqual", ParseLessEq);
   RegisterOperator("Greater", ParseGreater);
   RegisterOperator("GreaterOrEqual", ParseGreaterEq);
   // Reduce operators
   RegisterOperator("ReduceMean", ParseReduceMean);
   RegisterOperator("ReduceSum", ParseReduceSum);
   RegisterOperator("ReduceSumSquare", ParseReduceSumSquare);
   RegisterOperator("ReduceProd", ParseReduceProd);
   // Others
   RegisterOperator("BatchNormalization", ParseBatchNormalization);
   RegisterOperator("Constant", ParseConstant);
   RegisterOperator("ConstantOfShape", ParseConstant);
   RegisterOperator("Cast", ParseCast);
   RegisterOperator("Concat", ParseConcat);
   RegisterOperator("Conv", ParseConv);
   RegisterOperator("ConvTranspose", ParseConvTranspose);
   RegisterOperator("Gemm", ParseGemm);
   RegisterOperator("GRU", ParseGRU);
   RegisterOperator("Identity", ParseIdentity);
   RegisterOperator("LeakyRelu", ParseLeakyRelu);
   RegisterOperator("LSTM", ParseLSTM);
   RegisterOperator("AveragePool", ParsePool);
   RegisterOperator("GlobalAveragePool", ParsePool);
   RegisterOperator("MaxPool", ParsePool);
   RegisterOperator("Relu", ParseRelu);
   RegisterOperator("Reshape", ParseReshape);
   RegisterOperator("Flatten", ParseReshape);
   RegisterOperator("Squeeze", ParseReshape);
   RegisterOperator("Unsqueeze", ParseReshape);
   RegisterOperator("RNN", ParseRNN);
   RegisterOperator("Selu", ParseSelu);
   RegisterOperator("Shape", ParseShape);
   RegisterOperator("Sigmoid", ParseSigmoid);
   RegisterOperator("Slice", ParseSlice);
   RegisterOperator("Softmax", ParseSoftmax);
   RegisterOperator("Tanh", ParseTanh);
   RegisterOperator("Transpose", ParseTranspose);
   RegisterOperator("MatMul", ParseMatMul);
   RegisterOperator("LayerNormalization", ParseLayerNormalization);
   RegisterOperator("Expand", ParseExpand);
   RegisterOperator("Gather", ParseGather);
   RegisterOperator("Erf", ParseErf);
   RegisterOperator("Elu", ParseElu);
   RegisterOperator("EyeLike", ParseEyeLike);
   RegisterOperator("Range", ParseRange);
   RegisterOperator("TopK", ParseTopK);
   RegisterOperator("Tile", ParseTile);
   RegisterOperator("Split", ParseSplit);
   RegisterOperator("If", ParseIf);
   RegisterOperator("Pad", ParsePad);
   RegisterOperator("Where", ParseWhere);
   RegisterOperator("Einsum", ParseEinsum);
   RegisterOperator("RandomNormal", ParseRandom);
   RegisterOperator("RandomNormalLike", ParseRandom);
   RegisterOperator("RandomUniform", ParseRandom);
   RegisterOperator("RandomUniformLike", ParseRandom);
   RegisterOperator("ScatterElements", ParseScatterElements);
}

// Destructor of the parser
RModelParser_ONNX::~RModelParser_ONNX() = default;

void RModelParser_ONNX::RegisterOperator(const std::string &name, ParserFuncSignature func)
{
   fOperatorsMapImpl->fOperatorsMap[name] = func;
}

bool RModelParser_ONNX::IsRegisteredOperator(const std::string &name)
{
   return fOperatorsMapImpl->fOperatorsMap.find(name) != fOperatorsMapImpl->fOperatorsMap.end();
}

std::vector<std::string> RModelParser_ONNX::GetRegisteredOperators()
{
   std::vector<std::string> ops;
   ops.reserve(fOperatorsMapImpl->fOperatorsMap.size());
   for (auto &it : fOperatorsMapImpl->fOperatorsMap) {
      ops.emplace_back(it.first);
   }
   // return sorted list in alphabetical order
   std::sort(ops.begin(), ops.end());
   return ops;
}

void RModelParser_ONNX::RegisterTensorType(const std::string &name, ETensorType type)
{
   fTensorTypeMap[UTILITY::Clean_name(name)] = type;
}

bool RModelParser_ONNX::IsRegisteredTensorType(const std::string &name)
{
   return fTensorTypeMap.find(UTILITY::Clean_name(name)) != fTensorTypeMap.end();
}

ETensorType RModelParser_ONNX::GetTensorType(const std::string &name)
{
   return fTensorTypeMap[UTILITY::Clean_name(name)];
}

// Parse an operator
std::unique_ptr<ROperator>
RModelParser_ONNX::ParseOperator(const size_t i, const onnx::GraphProto &graphproto, const std::vector<size_t> &nodes, const std::vector<int> & children)
{
   if (i >= nodes.size())
      throw std::runtime_error("TMVA::SOFIE - Error in parsing ordered operators " + std::to_string(i) + " is >=  " + std::to_string(nodes.size()));
   int idx = nodes[i];
   const auto &nodeproto = graphproto.node(idx);
   const std::string op_type = nodeproto.op_type();
   if (fVerbose)
      std::cout << "Parsing operator " << op_type << std::endl;

   // skip already fused operators
   if (fFusedOperators[idx]) return nullptr;

   // try to fuse with following operator in case it is not last one
   if (children.size() == 1) {
      int idx2 = children.front();
      if (op_type == "MatMul") {
        // Fuse MatMul and Add
         if (idx2 < graphproto.node_size() && graphproto.node(idx2).op_type() == "Add") {
            fFusedOperators[idx2] = true;
            return ParseFuseMatMulAdd(*this, graphproto.node(idx), graphproto.node(idx2));
         }
         else {
            return ParseMatMul(*this, graphproto.node(idx));
         }
      } else if (nodeproto.op_type() == "Conv" || nodeproto.op_type() == "ConvTranspose") {
      // Fuse Conv or ConvTranspose without bias and Add
         if (idx2 < graphproto.node_size() && graphproto.node(idx2).op_type() == "Add") {
            if (nodeproto.op_type() == "Conv") {
               fFusedOperators[idx2] = true;
               return ParseFuseConvAdd(*this, graphproto.node(idx), graphproto.node(idx2));
            } else {
               fFusedOperators[idx2] = true;
               return ParseFuseConvTransposeAdd(*this, graphproto.node(idx), graphproto.node(idx2));
            }
         }
      } else if (nodeproto.op_type() == "Gemm") {
         // Fuse Gemm with activation operators
         if (idx2 < graphproto.node_size() && graphproto.node(idx2).op_type() == "Relu") {
            fFusedOperators[idx2] = true;
            return ParseFuseGemmRelu(*this, graphproto.node(idx), graphproto.node(idx2));
         }
      } 
      // else if (nodeproto.op_type() == "BatchNormalization") {
      //    if (idx2 < graphproto.node_size() && graphproto.node(idx2).op_type() == "Relu") {
      //       fFusedOperators[idx2] = true;
      //       return ParseFuseBatchnormRelu(*this, graphproto.node(idx), graphproto.node(idx2));
      //    }
      // }
   }



   auto it = fOperatorsMapImpl->fOperatorsMap.find(op_type);
   if (it == fOperatorsMapImpl->fOperatorsMap.end()) {
      std::cout << "operator " << op_type << " is not supported" << std::endl;
      throw std::runtime_error("TMVA::SOFIE Operator type " + op_type + " is not yet supported");
   }
   if (fVerbose) {
      std::cout << "\tCreating operator " << op_type << std::endl;
   }
   return it->second(*this, nodeproto);
}

// Parse a model
RModel RModelParser_ONNX::Parse(std::string filename, bool verbose)
{
   fVerbose = verbose;

   fTensorTypeMap.clear();

   auto model = LoadModel(filename);
   if (!model)
      throw std::runtime_error("TMVA::SOFIE - Failed to load onnx file " + filename);

   const onnx::GraphProto &graph = model->graph(); // not a memory leak. model freed automatically at the end.


   std::time_t ttime = std::time(0);
   std::tm *gmt_time = std::gmtime(&ttime);
   std::string parsetime(std::asctime(gmt_time));

   // get name of model (filename without directory name)
   char sep = '/';
#ifdef _WIN32
   sep = '\\';
#endif
   size_t isep = filename.rfind(sep, filename.length());
   std::string filename_nodir = filename;
   if (isep != std::string::npos) {
      filename_nodir = (filename.substr(isep + 1, filename.length() - isep));
   }

   RModel rmodel(filename_nodir, parsetime);
   ParseONNXGraph(rmodel, graph, filename_nodir);
   return rmodel;
}

std::unique_ptr<onnx::ModelProto> RModelParser_ONNX::LoadModel(std::string filename) {

   GOOGLE_PROTOBUF_VERIFY_VERSION;
   auto model = std::make_unique<onnx::ModelProto>();

   std::fstream input(filename, std::ios::in | std::ios::binary);
   if (!model->ParseFromIstream(&input)) {
      std::cerr << "TMVA::SOFIE - Failed to open onnx file " <<  filename << std::endl;
      return std::unique_ptr<onnx::ModelProto>();
   }

   // ONNX version is ir_version()  - model_version() returns 0
   if (fVerbose) {
      std::cout << "ONNX Version " << model->ir_version() << std::endl;
   }
   google::protobuf::ShutdownProtobufLibrary();
   return model;

}

void RModelParser_ONNX::CheckGraph(const onnx::GraphProto & graph, int & level, std::map<std::string, int> & missingOperators) {
   if (fVerbose)
      std::cout << "\n" << graph.name() << " Graph operator list\n";
   for (int i = 0; i < graph.node_size(); i++) {
      const auto & node = graph.node(i);
      const std::string opType =  node.op_type();
      if (fVerbose) {
         std::cout << "\tOperator " << i << " : " << opType << " (" << node.name() << "), " << graph.node(i).input_size()
                      << " inputs : {";
            for (int j = 0; j < graph.node(i).input_size(); j++) {
               std::cout << graph.node(i).input(j);
               if (j < graph.node(i).input_size() - 1)
                  std::cout << ", ";
            }
         std::cout << " }" << std::endl;
      }
      // check if operator exists
      if (!IsRegisteredOperator(opType))
         missingOperators[opType] = level;
      // see if sub-graph exists as node attributes
      for (int j = 0; j < node.attribute_size(); j++) {
         const auto & attribute = node.attribute(j);
         if (attribute.has_g()) {
            const auto & subGraph = attribute.g();
            level += 1;
            CheckGraph(subGraph, level, missingOperators);
         }
      }
   }
}

bool RModelParser_ONNX::CheckModel(std::string filename, bool verbose) {

   fVerbose = verbose;
   auto model = LoadModel(filename);
   if (!model) return false;

   const onnx::GraphProto &graph = model->graph();
    // Initial operator order
   if (fVerbose)
      std::cout << "\nModel operator list " << model->producer_name() << "\n";

   std::map<std::string, int> missingOperators;
   int level = 1;
   CheckGraph(graph, level, missingOperators);

   if (!missingOperators.empty()) {
      std::cout << "List of missing operators for model loaded from file " << filename << std::endl;
      for (auto & op : missingOperators) {
         std::cout << op.first << "  " << op.second << std::endl;
      }
      return false;
   }
   std::cout << "All operators in the loaded model are supported!\n";
   return true;
}

void RModelParser_ONNX::ParseONNXGraph(RModel & rmodel, const onnx::GraphProto & graph, std::string  graphName)
{
   bool verbose = fVerbose;

   if (graphName.empty())
      graphName = graph.name();

   if (verbose)
      std::cout << "\nParsing Graph - " << graphName << std::endl;

   std::unordered_set<std::string> initializer_names;
   for (int i = 0; i < graph.initializer_size(); i++) {
      initializer_names.insert(graph.initializer(i).name());
   }

   if (verbose)
      std::cout << "Parsing model inputs...." << std::endl;
   /// Loop on model inputs
   for (int i = 0; i < graph.input_size(); i++) {
      RegisterTensorType(graph.input(i).name(),
                         static_cast<ETensorType>(graph.input(i).type().tensor_type().elem_type()));

      if (verbose)
         std::cout << "\tgraph input " << i << " name " << graph.input(i).name() << " type "
                   << graph.input(i).type().tensor_type().elem_type() << std::endl;

      if (initializer_names.find(graph.input(i).name()) != initializer_names.end())
         continue;

      // input data node is not a weight node (has no initializer)
      const onnx::ValueInfoProto &valueinfoproto = graph.input(i);
      std::string input_name = valueinfoproto.name();

      ETensorType type = static_cast<ETensorType>(valueinfoproto.type().tensor_type().elem_type());

      std::vector<Dim> fShape;
      bool existParam = false;
      if (!valueinfoproto.type().tensor_type().has_shape())
         throw std::runtime_error("TMVA::SOFIE data node with no shape restrictions is not supported yet");
      for (int j = 0; j < valueinfoproto.type().tensor_type().shape().dim_size(); j++) {
         Dim dim;
         if (valueinfoproto.type().tensor_type().shape().dim(j).value_case() ==
             onnx::TensorShapeProto_Dimension::ValueCase::kDimValue) {
             int dim_value = valueinfoproto.type().tensor_type().shape().dim(j).dim_value();
             dim.dim = dim_value;
             // case input dim is -1 - set a parametric shape
             if (dim_value < 0) {
               dim.isParam = true;
               existParam = true;
               dim.param = UTILITY::Clean_name(input_name) + "_size";
             }
         } else if (valueinfoproto.type().tensor_type().shape().dim(j).value_case() ==
                    onnx::TensorShapeProto_Dimension::ValueCase::kDimParam) {
            dim.isParam = true;
            existParam = true;
            dim.param = valueinfoproto.type().tensor_type().shape().dim(j).dim_param();
         } else {
            throw std::runtime_error("TMVA::SOFIE ONNX file error: Valueinfoproto " + input_name +
                                     " has neither dim_value nor dim_param! \n");
         }
         fShape.push_back(dim);
      }
      if (valueinfoproto.type().tensor_type().shape().dim_size() == 0) {
         Dim dim;
         dim.dim = 1;
         fShape.push_back(dim);
      } // in case this TensorShapeProto has no dimension message: ONNX IR defines this to be a scalar

      if (!existParam) {
         std::vector<size_t> fShape_sizet;
         for (auto &j : fShape) {
            fShape_sizet.push_back(j.dim);
         }

         rmodel.AddInputTensorInfo(input_name, type, fShape_sizet);
      } else {
         rmodel.AddInputTensorInfo(input_name, type, fShape);
      }
      rmodel.AddInputTensorName(input_name); // store also names in given order
   }

   std::map<std::string, int> allInitializedTensors;

   if (verbose)
      std::cout << "\nParsing graph initializer list and fill model initialized tensors" << std::endl;

   for (int i = 0; i < graph.initializer_size(); i++) {
      onnx::TensorProto *tensorproto = const_cast<onnx::TensorProto *>(&graph.initializer(i));
      std::vector<std::size_t> shape;
      std::size_t fLength = 1;
      for (int j = 0; j < tensorproto->dims_size(); j++) {
         shape.push_back(tensorproto->dims(j));
         fLength *= tensorproto->dims(j);
      }
      // in case of scalars keep an empty shape but with length =1

      std::string input_name = graph.initializer(i).name();

      if (verbose)
         std::cout << "\t initializer " << i << " name " << input_name << " type " << graph.initializer(i).data_type()
                   << std::endl;

      // register also the initialized tensors
      auto tensor_type = static_cast<ETensorType>(graph.initializer(i).data_type());
      RegisterTensorType(input_name, tensor_type);

      switch (tensor_type) {
      case ETensorType::FLOAT: {
         std::shared_ptr<void> data = GetInitializedTensorData<float>(tensorproto, fLength);
         if (verbose) std::cout << "add FLOAT initialized tensor " << input_name << " shape " << ConvertShapeToString(shape) << std::endl;
         rmodel.AddInitializedTensor(input_name, ETensorType::FLOAT, shape, data);
         allInitializedTensors[input_name] = i;
         break;
      }
      case ETensorType::DOUBLE: {
         std::shared_ptr<void> data = GetInitializedTensorData<double>(tensorproto, fLength);
         if (verbose) std::cout << "add DOUBLE initialized tensor " << input_name << " shape " << ConvertShapeToString(shape) << std::endl;
         rmodel.AddInitializedTensor(input_name, ETensorType::DOUBLE, shape, data);
         allInitializedTensors[input_name] = i;
         break;
      }
      case ETensorType::INT32: {
         std::shared_ptr<void> data = GetInitializedTensorData<int32_t>(tensorproto, fLength);
         if (verbose) std::cout << "add INT32 initialized tensor " << input_name << " shape " << ConvertShapeToString(shape) << std::endl;
         rmodel.AddInitializedTensor(input_name, ETensorType::INT32, shape, data);
         allInitializedTensors[input_name] = i;
         break;
      }
      case ETensorType::INT64: {
         std::shared_ptr<void> data = GetInitializedTensorData<int64_t>(tensorproto, fLength);
         if (verbose) std::cout << "add INT64 initialized tensor " << input_name << " shape " << ConvertShapeToString(shape) << std::endl;
         rmodel.AddInitializedTensor(input_name, ETensorType::INT64, shape, data);
         allInitializedTensors[input_name] = i;
         break;
      }
      default:
         throw std::runtime_error("Data type in weight tensor " + graph.initializer(i).name() + " not supported!\n");
      }
   }

   // Initial operator order
   if (verbose) {
      std::cout << "\nGraph operator list (ONNX order)\n";
      for (int i = 0; i < graph.node_size(); i++) {
         std::cout << "\tOperator " << i << " : " << graph.node(i).op_type() << " , " << graph.node(i).input_size()
                   << " inputs : {";
         for (int j = 0; j < graph.node(i).input_size(); j++) {
            std::cout << graph.node(i).input(j);
            if (j < graph.node(i).input_size() - 1)
               std::cout << ", ";
         }
         std::cout << " }" << std::endl;
      }
   }

   // make order of nodes:
   if (verbose)
      std::cout << "\n***********************\nRe-Order graph operator list\n*************************\n";
   std::vector<size_t> nodesOrder;
   nodesOrder.reserve(graph.node_size());
   std::vector<bool> foundNodes(graph.node_size());

   // loop at graph inputs
   std::map<std::string, int> allInputs;
   for (int i = 0; i < graph.input_size(); i++) {
      allInputs[graph.input(i).name()] = -1;
   }
   do {
      auto psize = nodesOrder.size();
      for (int i = 0; i < graph.node_size(); i++) {
         if (foundNodes[i])
            continue;
         // check if all input exists add to list
         bool existInputs = true;
         int input_size = graph.node(i).input_size();
         // special case for Reshape where shape is input and not a weight tensor
         if (fVerbose)
            std::cout << "Checking input of  Node " << i << " : " << graph.node(i).name() << std::endl;
         for (int j = 0; j < input_size; j++) {
            std::string name = graph.node(i).input(j);
            // skip empty names
            if (!name.empty()) {
               existInputs &= (allInputs.find(name) != allInputs.end() ||
                               allInitializedTensors.find(name) != allInitializedTensors.end());
               if (fVerbose) {
                  std::cout << "\t\t input " << name << " "
                     << bool(allInputs.find(name) != allInputs.end()) << "  " <<
                     bool(allInitializedTensors.find(name) != allInitializedTensors.end()) << "  " <<
                     existInputs << std::endl;
               }
            }
         }
         if (!existInputs) {
            if (fVerbose) {
               std::cout << "skip node " << graph.node(i).op_type() << "  " << graph.node(i).name() << " inputs are not existing ";
               for (int j = 0; j < input_size; j++) {
                  std::cout << graph.node(i).input(j) << " ";
               }
               std::cout << std::endl;
            }
            continue;
         }

         // adding node to the currectly ordered list
         if (verbose)
            std::cout << "===> New node " << graph.node(i).op_type() << "  " << graph.node(i).name() << " order " << i << std::endl;

         nodesOrder.push_back(i);
         foundNodes[i] = true;
         // register the outputs
         for (int j = 0; j < graph.node(i).output_size(); j++) {
            if (fVerbose) std::cout << "\toutput : " << graph.node(i).output(j) << std::endl;
            allInputs[graph.node(i).output(j)] = i;
         }
      }
      // no increment in nodes - something wrong
      if (nodesOrder.size() == psize) {
         int ilast = nodesOrder.back();
         std::cout << "cannot find a new node after " << graph.node(ilast).op_type() << " " << graph.node(ilast).name() << std::endl;
         throw std::runtime_error("TMVA::SOFIE - cannot find a new node ");
      }
   } while ((int)nodesOrder.size() < graph.node_size());


   // find list of children for each operator (used for fusing operators)
   std::vector<std::vector<int>> nodesChildren(graph.node_size());

   for (int k = 0; k < graph.node_size(); k++) {
      int i = nodesOrder[k];
      // compute the number of output for the operators
      if (graph.node(i).output_size() > 0) nodesChildren[i].reserve(graph.node(i).output_size());
      for (const auto& output_name : graph.node(i).output()) {
         // loop on all nodes
         for (int l = k; l < graph.node_size(); l++) {
            int j = nodesOrder[l];
            for (const auto& input_name : graph.node(j).input()) {
               if (input_name == output_name)
                  nodesChildren[i].push_back(j);
            }
         }
      }
   }

   // print lit of order operators with list of inputs and list of children nodes
   if (verbose) {
      std::cout << "\nGraph operator list (re-ordered)\n";
      for (int k = 0; k < graph.node_size(); k++) {
         int i = nodesOrder[k];
         std::cout << "\tOperator " << i << " : " << graph.node(i).op_type() << " , " << graph.node(i).name() << " input tensors : {";
            for (int j = 0; j < graph.node(i).input_size(); j++) {
            std::cout << graph.node(i).input(j);
            if (j < graph.node(i).input_size() - 1)
               std::cout << ", ";
         }
         std::cout << " } ";
         std::cout << " children : {";
         for ( const auto & ichild : nodesChildren[i]) {
            std::cout << " [ " << ichild << " " << graph.node(ichild).op_type() << " , " << graph.node(ichild).name() << "]";
         }
         std::cout << "}" << std::endl;
      }
   }

   // fill model with operators
   if (verbose) {
      std::cout << "Fill RModel with operators...\n";
   }

   // we have to record order of node execution separately to
   // account for fused operators
   size_t node_order_exec = 0;
   fFusedOperators = std::vector<bool>(graph.node_size(), false);
   for (int i = 0; i < graph.node_size(); i++) {
      std::string op_type = graph.node(nodesOrder[i]).op_type();

      if (verbose) {
         std::cout << "\t" << i << "  " << nodesOrder[i] << " parsing operator " << op_type << std::endl;
      }

      std::unique_ptr<ROperator> op = ParseOperator(i, graph, nodesOrder, nodesChildren[i]);
      if (!op) {
         if (verbose) {
            std::cout << "\t\tskipping operator since it is fused with previous one" << std::endl;
         }
         // for skipping the fused nodes like Add after MatMul
         continue;
      }
      rmodel.AddOperator(std::move(op), node_order_exec++);
   }

   std::vector<std::string> outputnames;
   if (verbose)
      std::cout << "\nParsing Graph output list\n";
   for (int i = 0; i < graph.output_size(); i++) {
      if (verbose)
         std::cout << "\toutput " << i << " name " << graph.output(i).name() << std::endl;
      outputnames.push_back(graph.output(i).name());
   }
   rmodel.AddOutputTensorNameList(outputnames);

   return;
}

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
