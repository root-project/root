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
extern ParserFuncSignature ParseReduceSumsquare;
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
// Decalaration of fused operators
extern ParserFuseFuncSignature ParseFuseConvAdd;
extern ParserFuseFuncSignature ParseFuseConvTransposeAdd;
extern ParserFuseFuncSignature ParseFuseMatMulAdd;

// Definition of  RModelParser_ONNX::OperatorsMap
struct RModelParser_ONNX::OperatorsMapImpl {
   // Registered operators
   std::unordered_map<std::string, ParserFuncSignature> fOperatorsMap;
};

// Constructor of the parser
RModelParser_ONNX::RModelParser_ONNX() noexcept : fOperatorsMapImpl(std::make_unique<OperatorsMapImpl>()) {
   // Register operators
   // Unary operators
   RegisterOperator("Sqrt", ParseSqrt);
   RegisterOperator("Reciprocal", ParseReciprocal);
   RegisterOperator("Neg", ParseNeg);
   RegisterOperator("Exp", ParseExp);
   RegisterOperator("Log", ParseLog);
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
   RegisterOperator("ReduceSumsquare", ParseReduceSumsquare);
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
RModelParser_ONNX::ParseOperator(const size_t i, const onnx::GraphProto &graphproto, const std::vector<size_t> &nodes)
{
   if (i >= nodes.size())
      throw std::runtime_error("TMVA::SOFIE - Error in parsing ordered operators " + std::to_string(i) + " is >=  " + std::to_string(nodes.size()));
   int idx = nodes[i];
   const auto &nodeproto = graphproto.node(idx);
   const std::string op_type = nodeproto.op_type();
   if (fVerbose)
      std::cout << "Parsing an operator " << op_type << std::endl;

   // try to fuse with following operator in case it is not last one
   if (i < nodes.size() - 1) {
      int idx2 = nodes[i+1];
      if (op_type == "MatMul") {
        // Fuse MatMul and Add
         if (idx2 < graphproto.node_size() && graphproto.node(idx2).op_type() == "Add") {
            return ParseFuseMatMulAdd(*this, graphproto.node(idx), graphproto.node(idx2));
         }
         else {
            return ParseMatMul(*this, graphproto.node(idx));
         }
      } else if (nodeproto.op_type() == "Conv" || nodeproto.op_type() == "ConvTranspose") {
      // Fuse Conv or ConvTranspose without bias and Add
         if (idx2 < graphproto.node_size() && graphproto.node(idx2).op_type() == "Add") {
            if (nodeproto.op_type() == "Conv") {
               return ParseFuseConvAdd(*this, graphproto.node(idx), graphproto.node(idx2));
            } else {
               return ParseFuseConvTransposeAdd(*this, graphproto.node(idx), graphproto.node(idx2));
            }
         }
      }
   }

   // skip then the following Add if it was fused before
   if (idx > 0 && op_type == "Add") {
      int idx0 = nodes[i - 1];
      if (graphproto.node(idx0).op_type() == "MatMul")
         return nullptr;
      else if (graphproto.node(idx0).op_type() == "ConvTranspose")
         return nullptr;
   }

   auto it = fOperatorsMapImpl->fOperatorsMap.find(op_type);
   if (it == fOperatorsMapImpl->fOperatorsMap.end()) {
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
   char sep = '/';
#ifdef _WIN32
   sep = '\\';
#endif
   size_t isep = filename.rfind(sep, filename.length());
   std::string filename_nodir = filename;
   if (isep != std::string::npos) {
      filename_nodir = (filename.substr(isep + 1, filename.length() - isep));
   }

   std::time_t ttime = std::time(0);
   std::tm *gmt_time = std::gmtime(&ttime);
   std::string parsetime(std::asctime(gmt_time));

   GOOGLE_PROTOBUF_VERIFY_VERSION;
   // model I/O
   onnx::ModelProto model;
   RModel rmodel(filename_nodir, parsetime);

   fTensorTypeMap.clear();

   std::fstream input(filename, std::ios::in | std::ios::binary);
   if (!model.ParseFromIstream(&input)) {
      throw std::runtime_error("TMVA::SOFIE - Failed to parse onnx file " + filename);
   }

   const onnx::GraphProto &graph = model.graph(); // not a memory leak. model freed automatically at the end.
   google::protobuf::ShutdownProtobufLibrary();

   // ONNX version is ir_version()  - model_version() returns 0
   if (fVerbose) {
      std::cout << "ONNX Version " << model.ir_version() << std::endl;
   }

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
      if (type != ETensorType::FLOAT && type != ETensorType::INT32 && type != ETensorType::INT64) {
         throw std::runtime_error("TMVA::SOFIE Data type in input tensor " + input_name + " not supported!\n");
      }

      std::vector<Dim> fShape;
      bool existParam = false;
      if (!valueinfoproto.type().tensor_type().has_shape())
         throw std::runtime_error("TMVA::SOFIE datanode with no shape restrictions is not supported yet");
      for (int j = 0; j < valueinfoproto.type().tensor_type().shape().dim_size(); j++) {
         Dim dim;
         if (valueinfoproto.type().tensor_type().shape().dim(j).value_case() ==
             onnx::TensorShapeProto_Dimension::ValueCase::kDimValue) {
            dim.dim = valueinfoproto.type().tensor_type().shape().dim(j).dim_value();
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
         std::shared_ptr<void> data(malloc(fLength * sizeof(float)), free);

         if (!tensorproto->raw_data().empty()) {
#ifdef R__BYTESWAP
            std::memcpy(data.get(), tensorproto->raw_data().c_str(), fLength * sizeof(float));
#else
            for (std::size_t k = 0; k < fLength; ++k)
               (reinterpret_cast<uint32_t *>(data.get()))[k] =
                  Rbswap_32((reinterpret_cast<const uint32_t *>(tensorproto->raw_data().c_str()))[k]);
#endif
         } else {
            tensorproto->mutable_float_data()->ExtractSubrange(0, tensorproto->float_data_size(),
                                                               static_cast<float *>(data.get()));
         }

         if (verbose) std::cout << "add FLOAT initialized tensor " << input_name << " shape " << ConvertShapeToString(shape) << std::endl;
         rmodel.AddInitializedTensor(input_name, ETensorType::FLOAT, shape, data);
         allInitializedTensors[input_name] = i;
         break;
      }
      case ETensorType::INT64: {
         std::shared_ptr<void> data(malloc(fLength * sizeof(int64_t)), free);

         if (!tensorproto->raw_data().empty()) {
#ifdef R__BYTESWAP
            std::memcpy(data.get(), tensorproto->raw_data().c_str(), fLength * sizeof(int64_t));
#else
            for (std::size_t k = 0; k < fLength; ++k)
               (reinterpret_cast<uint64_t *>(data.get()))[k] =
                  Rbswap_64((reinterpret_cast<const uint64_t *>(tensorproto->raw_data().c_str()))[k]);
#endif
         } else {
            tensorproto->mutable_int64_data()->ExtractSubrange(0, tensorproto->int64_data_size(),
                                                               static_cast<int64_t *>(data.get()));
         }

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
      std::cout << "\nRe-Order graph operator list\n";
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
         for (int j = 0; j < input_size; j++) {
            std::string name = graph.node(i).input(j);
            // skip empty names
            if (!name.empty()) {
               existInputs &= (allInputs.find(name) != allInputs.end() ||
                               allInitializedTensors.find(name) != allInitializedTensors.end());
               if (fVerbose) {
                  std::cout << graph.node(i).op_type() << " input " << name << " "
                     << bool(allInputs.find(name) != allInputs.end()) << "  " <<
                     bool(allInitializedTensors.find(name) != allInitializedTensors.end()) <<
                     existInputs << std::endl;
               }
            }
         }
         if (!existInputs) {
            if (fVerbose) {
               std::cout << "skip op " << graph.node(i).op_type() << " inputs are ";
               for (int j = 0; j < input_size; j++) {
                  std::cout << graph.node(i).input(j) << " ";
               }
               std::cout << std::endl;
            }
            continue;
         }
         if (verbose)
            std::cout << "\tadd node " << graph.node(i).op_type() << " order " << i << std::endl;

         nodesOrder.push_back(i);
         foundNodes[i] = true;
         // register the outputs
         for (int j = 0; j < graph.node(i).output_size(); j++) {
            allInputs[graph.node(i).output(j)] = i;
         }
      }
      // no increment in nodes - something wrong
      if (nodesOrder.size() == psize) {
         throw std::runtime_error("TMVA::SOFIE - cannot find a new node ");
      }
   } while ((int)nodesOrder.size() < graph.node_size());

   // scan operators for orders
   if (verbose) {
      std::cout << "\nGraph operator list (re-ordered)\n";
      for (int k = 0; k < graph.node_size(); k++) {
         int i = nodesOrder[k];
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

   // fill model with operators
   if (verbose) {
      std::cout << "Fill RModel with operators...\n";
   }
   for (int i = 0; i < graph.node_size(); i++) {
      std::string op_type = graph.node(nodesOrder[i]).op_type();

      if (verbose) {
         std::cout << "\t" << i << "  " << nodesOrder[i] << " parsing operator " << op_type << std::endl;
      }

      std::unique_ptr<ROperator> op = ParseOperator(i, graph, nodesOrder);
      if (!op) {
         if (verbose) {
            std::cout << "\t\tskipping operator since it is fused with previous one" << std::endl;
         }
         // for skipping the fused nodes like Add after MatMul
         continue;
      }
      rmodel.AddOperator(std::move(op));
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

   return rmodel;
}

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA