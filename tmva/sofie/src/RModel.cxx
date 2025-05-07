#include <limits>
#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#ifdef SOFIE_SUPPORT_ROOT_BINARY
#include "TFile.h"
#endif

#include "TMVA/RModel.hxx"
#include "TMVA/SOFIE_common.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

namespace {
const std::string SP = "   ";
}

std::underlying_type_t<Options> operator|(Options opA, Options opB) {
    return static_cast<std::underlying_type_t<Options>>(opA) | static_cast<std::underlying_type_t<Options>>(opB);
}
std::underlying_type_t<Options> operator|(std::underlying_type_t<Options> opA, Options opB) {
    return opA | static_cast<std::underlying_type_t<Options>>(opB);
}

const std::vector<size_t>& RModel::GetTensorShape(std::string name) const {
    auto f = fReadyInputTensorInfos.find(name);
    if (f != fReadyInputTensorInfos.end()) {
        return f->second.shape;
    }
    auto f2 = fInitializedTensors.find(name);
    if (f2 != fInitializedTensors.end()) {
        return f2->second.shape();
    }
    auto f3 = fInputTensorInfos.find(name);
    if (f3 != fInputTensorInfos.end()) {
        throw std::runtime_error("TMVA SOFIE tensor [" + name + "] is an input tensor with unspecified dimension parameter");
    }
    auto f4 = fIntermediateTensorInfos.find(name);
    if (f4 != fIntermediateTensorInfos.end()) {
        return f4->second.shape;
    }
    if (fDynamicTensorInfos.find(name) != fDynamicTensorInfos.end())
      throw std::runtime_error("TMVA SOFIE tensor [" + name + "] is a dynamic tensor. Use GetDynamicTensorShape instead of GetTensorShape");

   if (fIsSubGraph && fParentGraph)
      return fParentGraph->GetTensorShape(name);

    throw std::runtime_error("TMVA SOFIE tensor [" + name + "] for which the shape is requested is not found");
}

std::vector<Dim> RModel::GetDynamicTensorShape(std::string name) const {
   if (auto f = fDynamicTensorInfos.find(name); f != fDynamicTensorInfos.end()) {
      return f->second.shape;
   }
   if (auto f = fInputTensorInfos.find(name); f != fInputTensorInfos.end()) {
      return f->second.shape;
   }
   // in case is not a dynamic tensor convert normal shape to Dim one
   // for this we need to return the vector by value
   return ConvertShapeToDim(GetTensorShape(name));
}

const ETensorType& RModel::GetTensorType(std::string name) const {
    auto f = fReadyInputTensorInfos.find(name);
    if (f != fReadyInputTensorInfos.end()) {
        return f->second.type;
    }
    auto f2 = fInitializedTensors.find(name);
    if (f2 != fInitializedTensors.end()) {
        return f2->second.type();
    }
    auto f3 = fInputTensorInfos.find(name);
    if (f3 != fInputTensorInfos.end()) {
        return f3->second.type;
    }
    auto f4 = fIntermediateTensorInfos.find(name);
    if (f4 != fIntermediateTensorInfos.end()) {
        return f4->second.type;
    }
    auto f5 = fDynamicTensorInfos.find(name);
    if (f5 != fDynamicTensorInfos.end()){
      return f5->second.type;
    }

    if (fIsSubGraph && fParentGraph)
      return fParentGraph->GetTensorType(name);

    throw std::runtime_error("TMVA SOFIE tensor [" + name + "] for which the type is requested is not found, model name: " + fName);
}

bool RModel::CheckIfTensorAlreadyExist(std::string tensor_name) {
    if (fReadyInputTensorInfos.find(tensor_name) != fReadyInputTensorInfos.end())  return true;
    if (fInputTensorInfos.find(tensor_name) != fInputTensorInfos.end()) return true;
    if (fInitializedTensors.find(tensor_name) != fInitializedTensors.end()) return true;
    if (fIntermediateTensorInfos.find(tensor_name) != fIntermediateTensorInfos.end()) return true;
    if (fDynamicTensorInfos.find(tensor_name) != fDynamicTensorInfos.end()) return true;
    if (fIsSubGraph && fParentGraph) return fParentGraph->CheckIfTensorAlreadyExist(tensor_name);
    return false;
}

void RModel::AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<Dim> shape) {
    input_name = UTILITY::Clean_name(input_name);
    if (CheckIfTensorAlreadyExist(input_name)) {
        throw std::runtime_error("TMVA-SOFIE: input tensor with name " + input_name + " already exists \n");
    }

    InputTensorInfo inputInfo { type, shape };
    fInputTensorInfos[input_name] = inputInfo;
}

void RModel::AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<size_t> shape) {
    input_name = UTILITY::Clean_name(input_name);
    if (CheckIfTensorAlreadyExist(input_name)) {
        throw std::runtime_error("TMVA-SOFIE: input tensor with name " + input_name + " already exists \n");
    }
    TensorInfo inputInfo { type, shape };
    fReadyInputTensorInfos[input_name] = inputInfo;
}

void RModel::AddInputTensorName(std::string input_name) {
    fInputTensorNames.emplace_back(UTILITY::Clean_name(input_name));
}

void RModel::AddOperator(std::unique_ptr<ROperator> op, int order_execution) {
    AddBlasRoutines(op->GetBlasRoutines());
    auto libs = op->GetStdLibs();
    auto op_input_tensors = op->GetOpInputTensors();
    for (auto& stdlib : libs) {
        AddNeededStdLib(stdlib);
    }
    if (order_execution >= 0) {
        fOperators.insert(fOperators.begin() + order_execution, std::move(op));
    } else {
        fOperators.push_back(std::move(op));
    }

    // storing the last usage of tensors which are input to
    // operators (but are not inputs to the model, i.e. they are intermediate
    // tensors). This information is needed to keep a check on when a
    // particular intermediate tensor can be flushed to free up memory for reuse.
   for(size_t index = 0; index<op_input_tensors.size() &&
         fInitializedTensors.find(UTILITY::Clean_name(std::string(op_input_tensors[index]))) == fInitializedTensors.end() &&
         std::find(fInputTensorNames.begin(), fInputTensorNames.end(),
                   UTILITY::Clean_name(std::string(op_input_tensors[index]))) == fInputTensorNames.end() &&
         fDynamicTensorInfos.find(UTILITY::Clean_name(std::string(op_input_tensors[index]))) == fDynamicTensorInfos.end();
         ++index){
            fIntermediateTensorFrequencyLookup[op_input_tensors[index]] = order_execution;
   }
}

void RModel::AddInitializedTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape, std::shared_ptr<void> data) {
    tensor_name = UTILITY::Clean_name(tensor_name);
    //NB: own data
    if (CheckIfTensorAlreadyExist(tensor_name)) {
        throw std::runtime_error("TMVA-SOFIE: initialized tensor with name " + tensor_name + " already exists \n");
    }
    InitializedTensor new_tensor {type, shape, data};
    fInitializedTensors[tensor_name] = new_tensor;
}

void RModel::AddConstantTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape, std::shared_ptr<void> data) {
    tensor_name = UTILITY::Clean_name(tensor_name);
    //NB: own data
    if (CheckIfTensorAlreadyExist(tensor_name)) {
        throw std::runtime_error("TMVA-SOFIE: initialized tensor with name " + tensor_name + " already exists \n");
    }
    InitializedTensor new_tensor {type, shape, data, true};   // add here flag to specify is a constant tensor
    fInitializedTensors[tensor_name] = new_tensor;
}

bool RModel::IsInitializedTensor(const std::string& tensorName) const {
    std::string name = UTILITY::Clean_name(tensorName);
    return fInitializedTensors.find(name) != fInitializedTensors.end();
}
bool RModel::IsConstantTensor(const std::string& tensorName) const {
    std::string name = UTILITY::Clean_name(tensorName);
    auto itr = fInitializedTensors.find(name);
    if (itr == fInitializedTensors.end()) return false;
    return itr->second.IsConstantTensor();
}

bool RModel::IsDynamicTensor(const std::string& tensorName) const {
   std::string name = UTILITY::Clean_name(tensorName);
   return fDynamicTensorInfos.find(name) != fDynamicTensorInfos.end();
}
bool RModel::IsDimInputTensor(const std::string& tensorName) const {
   std::string name = UTILITY::Clean_name(tensorName);
   return fInputTensorInfos.find(name) != fInputTensorInfos.end();
}
bool RModel::IsReadyInputTensor(const std::string& tensorName) const {
   std::string name = UTILITY::Clean_name(tensorName);
   return fReadyInputTensorInfos.find(name) != fReadyInputTensorInfos.end();
}

// generic addition of a tensor
void RModel::AddIntermediateTensor(std::string tensor_name, ETensorType type, std::vector<Dim> dim_shape) {
   auto int_shape = ConvertShapeToInt(dim_shape);
   if (!int_shape.empty())
      AddIntermediateTensor(tensor_name, type, int_shape);
   else
      AddDynamicTensor(tensor_name, type, dim_shape);
}

void RModel::AddIntermediateTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape) {
    tensor_name = UTILITY::Clean_name(tensor_name);
    if (CheckIfTensorAlreadyExist(tensor_name)) {
        throw std::runtime_error("TMVA-SOFIE: intermediate tensor with name " + tensor_name + " already exists \n");
    }
    TensorInfo new_tensor {type, shape};
    fIntermediateTensorInfos[tensor_name] = new_tensor;
}

void RModel::AddDynamicTensor(std::string tensor_name, ETensorType type, std::vector<Dim> shape){
   tensor_name = UTILITY::Clean_name(tensor_name);
   if (CheckIfTensorAlreadyExist(tensor_name)){
      throw std::runtime_error("TMVA-SOFIE: intermediate tensor with name " + tensor_name + " already exists \n");
   }
   DynamicTensorInfo new_tensor {type, shape};
   fDynamicTensorInfos[tensor_name] = new_tensor;
   // store shape parameter if not existing
   for (auto &d : shape) {
      if (d.isParam) {
         if (fShapeParams.count(d.param) == 0) {
            // case parameter is an expression of some other existing parameter, no need to
            // register it
            if (d.dim != size_t(-1)) {
              fShapeParams[d.param] = std::to_string(d.dim);
            }
         }
      }
   }
}

void RModel::AddOutputTensorNameList(std::vector<std::string> outputtensornames) {
    fOutputTensorNames.clear();
    for(auto& it : outputtensornames) {
        fOutputTensorNames.emplace_back(UTILITY::Clean_name(it));
    }
}

void RModel::UpdateOutputTensorList(std::vector<std::string> curr_output_tensors, std::vector<std::string> new_output_tensors) {
    for(auto& it:curr_output_tensors) {
        fOutputTensorNames.erase(std::remove(fOutputTensorNames.begin(), fOutputTensorNames.end(), it), fOutputTensorNames.end());
    }
    fOutputTensorNames.insert(fOutputTensorNames.end(), new_output_tensors.begin(), new_output_tensors.end());
}

void RModel::UpdateInitializedTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape, std::shared_ptr<void> data) {
    tensor_name = UTILITY::Clean_name(tensor_name);
    if (!CheckIfTensorAlreadyExist(tensor_name)) {
        throw std::runtime_error("TMVA-SOFIE: tensor " + tensor_name + " not found when trying to update it");
    }
    InitializedTensor new_tensor {type, shape, data};
    fInitializedTensors[tensor_name] = new_tensor;
}

std::shared_ptr<void> RModel::GetInitializedTensorData(std::string tensor_name) {
    auto f = fInitializedTensors.find(tensor_name);
    if (f == fInitializedTensors.end()) {
        throw std::runtime_error("TMVA-SOFIE: tensor " + tensor_name + " not found when trying to get its data");
    } else {
        return f->second.sharedptr();
    }
}

void RModel::SetNotWritableInitializedTensor(const std::string & tensor_name) {
      auto t = fInitializedTensors.find(tensor_name);
      if (t == fInitializedTensors.end()) {
         throw std::runtime_error("TMVA-SOFIE: initialized tensor " + tensor_name + " not found when trying to get its info");
      }
      t->second.SetNotWritable();
   }

std::string RModel::AllocateIntermediateMemory(std::span<const std::string_view> op_output_tensors)
{
   std::stringstream code;

   auto declareIntermediateTensor = [this, &code](std::string const &name, size_t size, size_t location) {
      std::string typeName = ConvertTypeToString(GetTensorType(name));
      code << "\n // Allocating memory for intermediate tensor " << name << " with size " << size << " bytes";
      code << "\n"
           << typeName << "* tensor_" << name << " = reinterpret_cast<" << typeName
           << "*>(fIntermediateMemoryPool.data() + " << location << ");\n";
   };

   for (auto &it : op_output_tensors) {
         std::string name = std::string{it};
         bool allocated = false;
         if (GetTensorType(name) == ETensorType::BOOL ||
            fInitializedTensors.find(name) != fInitializedTensors.end() ||
            fDynamicTensorInfos.find(name) != fDynamicTensorInfos.end()) continue;

         auto tensor_size = GetTypeSize(GetTensorType(name)) * ConvertShapeToLength(GetTensorShape(name));

            for (auto chunk = fIntermediateMemoryInfo.available_stack.begin(); chunk != fIntermediateMemoryInfo.available_stack.end(); ) {

                  // check if available memory chunks can accommodate the tensor
                  if (chunk->second >= tensor_size) {
                     auto new_chunk = fIntermediateMemoryInfo.total_stack[chunk->first].split(it, tensor_size);
                     auto new_chunk_location = chunk->first+chunk->second-tensor_size;
                     fIntermediateMemoryInfo.total_stack[new_chunk_location] = new_chunk;

                     declareIntermediateTensor(name, tensor_size, new_chunk_location);
                     chunk->second -= tensor_size;

                     allocated = true;

                     if (chunk->second == 0) {
                        chunk = fIntermediateMemoryInfo.available_stack.erase(chunk);
                     }

                     break;
                  }
                  ++chunk;
            }

         if (!allocated) {
               size_t chunk_idx = fIntermediateMemoryInfo.total_stack.empty()
                                 ? 0
                                 : fIntermediateMemoryInfo.total_stack.rbegin()->first + fIntermediateMemoryInfo.total_stack.rbegin()->second.tensor_size;

               fIntermediateMemoryInfo.total_stack[chunk_idx] = {it, tensor_size};

               declareIntermediateTensor(name, tensor_size, chunk_idx);
         }
   }
   return code.str();
}

void RModel::CheckAndFlushIntermediateMemory(std::span<const std::string_view> op_input_tensors, const size_t& op_idx){
   for (auto &it : op_input_tensors){
      // last occurence of the tensor is reached => flush it from memory
      if (fIntermediateTensorFrequencyLookup[it] == op_idx) {
         for (auto chunk = fIntermediateMemoryInfo.total_stack.begin();
               chunk != fIntermediateMemoryInfo.total_stack.end(); ++chunk ) {
               if (chunk->second.tensor_name == it) {

                     // check if nearby chunks in available memory can coalesce
                     auto first_greater = fIntermediateMemoryInfo.available_stack.upper_bound(chunk->first); // smallest element greater than the flushed chunk idx
                     auto last_smaller = (first_greater == fIntermediateMemoryInfo.available_stack.begin()) ? fIntermediateMemoryInfo.available_stack.end() : std::prev(first_greater); // largest element smaller than the flushed chunk idx

                     // check if the next stack entry is actually adjacent in memory
                     if (last_smaller->first+last_smaller->second + 1 == chunk->first){
                        last_smaller->second += chunk->second.tensor_size;
                        fIntermediateMemoryInfo.total_stack[last_smaller->first].merge(chunk->second);

                        if (last_smaller->first + last_smaller->second + 1 == first_greater->first){
                              fIntermediateMemoryInfo.total_stack[last_smaller->first].merge(fIntermediateMemoryInfo.total_stack[first_greater->first]);
                              first_greater = fIntermediateMemoryInfo.available_stack.erase(first_greater);
                        }
                     } else{
                        if (chunk->first + chunk->second.tensor_size + 1 == first_greater->first){
                           fIntermediateMemoryInfo.total_stack[chunk->first].merge(fIntermediateMemoryInfo.total_stack[first_greater->first]);
                           first_greater = fIntermediateMemoryInfo.available_stack.erase(first_greater);
                        }
                        fIntermediateMemoryInfo.available_stack.insert({
                           chunk->first,
                           chunk->second.tensor_size
        });
                     }
               }
         }
      }
   }
}



void RModel::Initialize(int batchSize, bool verbose) {
   std::map<std::string, size_t> inputParams;
   if (batchSize > 0) {
      inputParams["input_size"] = batchSize;
      inputParams["batch_size"] = batchSize;
      inputParams["bs"] = batchSize;
   }
   Initialize(inputParams, verbose);
   fIntermediateMemoryInfo = MemoryPoolInfo();
}
void RModel::Initialize(const std::map<std::string, size_t> & inputParams, bool verbose) {

   fVerbose = int(verbose);

   if (fIsInitialized) {
      if (verbose)
         std::cout << "Model is already initialized  - skip initialization " << std::endl;
      return;
   }
   fIntermediateTensorInfos.clear();
   fDynamicTensorInfos.clear();

   // loop on inputs and see if shape can be  full specified
   // if the batch size is provided it can be used to specify the full shape
   // Add the full specified tensors in fReadyInputTensors collection
   auto originalInputTensorInfos = fInputTensorInfos; // need to copy because we may delete elements
   for (auto &input : originalInputTensorInfos) {
      if (verbose) std::cout << "looking at the tensor " << input.first << std::endl;
      // if a parameter (e.g. batch_size) is specified use for converting parametric shape in defined one
      if (!inputParams.empty()) {
         for (auto &d : input.second.shape) {
            if (d.isParam) {
               std::string pname = d.param;
               if (pname == input.first + "_size") pname = "input_size";
               auto itr = inputParams.find(pname);
               if (itr != inputParams.end() ) {
                  d = Dim{ itr->second };
                  if (verbose)
                     std::cout << "Tensor: " << input.first << " - fix parametric shape " << itr->first << " to " << itr->second << std::endl;
               }
            }
         }
      }
      // see if shape now is fully defined
      auto shape = ConvertShapeToInt(input.second.shape);
      if (verbose)
         std::cout << "converting input shape for " << input.first << " " << ConvertShapeToString(shape) << " from "
            << ConvertDynamicShapeToString(input.second.shape) << std::endl;
      if (!shape.empty()) {
         // case shape is defined (not parametric) we add the tensor in the fReadyInputTensorInfos map and
         // we remove the tensor from the fInputTensorInfo where th eold parametric shape was stored
         fInputTensorInfos.erase(input.first);
         // add to the ready input tensor information the new fixed shape
         AddInputTensorInfo(input.first, input.second.type, shape);
         // check consistency
         assert( fReadyInputTensorInfos.size() + fInputTensorInfos.size() == fInputTensorNames.size());
      }
      // store the parameters of the input tensors
      else {
         // store the found parametric shape parameters
         for (auto &d : input.second.shape) {
            if (d.isParam)
               fShapeParams[d.param] = std::to_string(d.dim);
         }
      }
   }

   if (verbose) {
      PrintRequiredInputTensors();
      PrintDynamicTensors();
   }

   // check if there are initialized tensors to write in a weight file
   // support for the time being only weight of FLOAT type
   if (fUseWeightFile) {
      bool modelHasWeights = false;
      for (auto &i : fInitializedTensors) {
         if (i.second.type() == ETensorType::FLOAT) {
            modelHasWeights = true;
            break;
         }
      }
      if (!modelHasWeights)
         fUseWeightFile = false;
   }
   // Go through model and initialize each operator
   int i = 0;

   std::vector<size_t> temp_available_stack; // vector stores individual chunks of available memory that maybe reused

   for(size_t op_idx = 0; op_idx < fOperators.size(); ++op_idx){
      if (verbose) {
         auto& r = *fOperators[op_idx].get();
         std::cout << "Initializing operator " << i << "  " << typeid(r).name() << std::endl;
      }
      fOperators[op_idx]->Initialize(*this);
      for(auto &it:fOperators[op_idx]->GetOpOutputTensors()){
         std::string name = std::string{it};
         if (fIntermediateTensorFrequencyLookup.find(it) == fIntermediateTensorFrequencyLookup.end() &&
             std::find(fOutputTensorNames.begin(), fOutputTensorNames.end(), name) == fOutputTensorNames.end() &&
             fInitializedTensors.find(name) == fInitializedTensors.end() &&
             fDynamicTensorInfos.find(name) == fDynamicTensorInfos.end()){
            fIntermediateTensorFrequencyLookup[it] = op_idx;
         }
      }
      i++;
   }

   fIsInitialized = true;
}

void RModel::InitializeSubGraph(std::shared_ptr<RModel>  graph) {
   // add the subgraph to the list
   fSubGraphs.push_back(graph);
   //this needs to be done before initializing
   graph->fParentGraph = this;
   graph->fIsSubGraph = true;

   graph->Initialize(fBatchSize, fVerbose);
   // set the same options as parent model
   graph->fWeightFile = fWeightFile;
   graph->fUseWeightFile = fUseWeightFile;
   graph->fUseSession = fUseSession;
   // add needed blas routines and libs
   std::vector<std::string> blasRoutines;
   for (auto & e : graph->fNeededBlasRoutines)
      blasRoutines.push_back(e);
   AddBlasRoutines(blasRoutines);
   for (auto e : graph->fNeededStdLib)
      AddNeededStdLib(e);

   // add parent input tensors to current graph
   for (auto & name : fInputTensorNames)
      graph->fInputTensorNames.emplace_back(name);

   // clean graph name
   graph->fName = UTILITY::Clean_name(graph->fName);

}

// Function to generate the code for declaring and initializing constant tensors
// This is for tensors which are not part of weight files and can be created from the Constant operator
template <typename T>
std::string GenerateConstantTensorCode(const std::pair<std::string, InitializedTensor> &t)
{
   std::stringstream strs;
   std::string type = ConvertTypeToString(t.second.type());
   size_t length = ConvertShapeToLength(t.second.shape());
   // avoid using stack sizes for constant tensors to reduce compilation time
   bool allocateOnStack = (length > 100) ? false : true;

   const T *data = t.second.data<T>();

   // and check if all values are the same
   bool sameData = false;
   // for non stack allocation check if data are the same
   if (!allocateOnStack && length > 1) {
      size_t idx = 1;
      do {
         sameData = (data[idx] == data[idx - 1]);
         idx++;
      } while (sameData && idx < length);
   }
   if (allocateOnStack) {
      strs << type << " tensor_" << t.first << "[" << length << "] = " << ConvertValuesToString(length, data) << ";\n";
   } else {
      strs << "std::vector<" << type << "> fTensor_" << t.first << " = ";
      if (sameData)
         strs << "std::vector<" << type << ">(" << length << ", " << ConvertValToString(data[0]) << ");\n";
      else {
         strs << ConvertValuesToString(length, data) << ";\n";
      }
      strs << "const " << type << " * tensor_" + t.first + " = fTensor_" + t.first + ".data();\n";
   }
   return strs.str();
}

void RModel::GenerateInitializedTensorInfo()
{
   if (!fInitializedTensors.empty())
      fGC += "// initialized tensors\n";

   for (auto &i : fInitializedTensors) {
      if (!fUseWeightFile || i.second.IsConstantTensor()) {
         if (i.second.type() == ETensorType::FLOAT)
            fGC += GenerateConstantTensorCode<float>(i);
         else if (i.second.type() == ETensorType::INT64)
            fGC += GenerateConstantTensorCode<int64_t>(i);

      } else {
         // case of tensors which are read from a file
         size_t length = ConvertShapeToLength(i.second.shape());
         if (i.second.type() == ETensorType::FLOAT) {
            fGC += "std::vector<float> fTensor_" + i.first + " = std::vector<float>(" + std::to_string(length) + ");\n";
            fGC += "float * tensor_" + i.first + " = fTensor_" + i.first + ".data();\n";
         }
      }
   }
}

void RModel::GenerateIntermediateMemoryPool() {
   if (fIntermediateMemoryInfo.total_stack.empty()) return;
   fGC += "\n//--- Allocating session memory pool to be used for allocating intermediate tensors\n";

   // char memory block is allocated since char takes 1 byte, thus easier to allocate tensors
   // of other data types
   auto const &totalStack = fIntermediateMemoryInfo.total_stack;
   const size_t memPoolSize = totalStack.rbegin()->first + totalStack.rbegin()->second.tensor_size;
   fGC += "std::vector<char> fIntermediateMemoryPool = std::vector<char>(" + std::to_string(memPoolSize) + ");\n\n";
}

void RModel::GenerateIntermediateTensorInfo() {
   if (!fIntermediateTensorInfos.empty()) {
      std::string tensor_declaration_block = "";
      for (auto &i : fIntermediateTensorInfos) {
         if (i.second.type == ETensorType::BOOL) {
               tensor_declaration_block += "std::vector<std::uint8_t> fTensor_" + i.first + " = std::vector<std::uint8_t>(" + std::to_string(ConvertShapeToLength(i.second.shape)) + ");\n";
               tensor_declaration_block += "std::uint8_t * tensor_" + i.first + " = fTensor_" + i.first + ".data();\n";
               continue;
         }
         bool is_extended = (fOptimizationLevel == OptimizationLevel::kExtended);
         bool not_in_freq_map =
            (fIntermediateTensorFrequencyLookup.find(i.first) == fIntermediateTensorFrequencyLookup.end());
         bool not_in_output_names =
            (std::find(fOutputTensorNames.begin(), fOutputTensorNames.end(), i.first) == fOutputTensorNames.end());

         if ((not_in_freq_map && not_in_output_names) || (!not_in_freq_map && !is_extended && not_in_output_names)) {
            size_t length = ConvertShapeToLength(i.second.shape);

            if (i.second.type == ETensorType::FLOAT) {
               tensor_declaration_block += "std::vector<float> fTensor_" + i.first + " = std::vector<float>(" + std::to_string(length) + ");\n";
               tensor_declaration_block += "float * tensor_" + i.first + " = fTensor_" + i.first + ".data();\n";
            }
            else if (i.second.type == ETensorType::DOUBLE) {
               tensor_declaration_block += "std::vector<double> fTensor_" + i.first + " = std::vector<double>(" + std::to_string(length) + ");\n";
               tensor_declaration_block += "double * tensor_" + i.first + " = fTensor_" + i.first + ".data();\n";
            }
            else if (i.second.type == ETensorType::INT64) {
               tensor_declaration_block += "std::vector<int64_t> fTensor_" + i.first + " = std::vector<int64_t>(" + std::to_string(length) + ");\n";
               tensor_declaration_block += "int64_t * tensor_" + i.first + " = fTensor_" + i.first + ".data();\n";
            }
         }
      }

      if (tensor_declaration_block.length()) {
         fGC += "\n//--- declare and allocate the intermediate tensors\n" + tensor_declaration_block;
      }
   }
   // add also the dynamic tensors (only declarations, allocation will be done later)
   if (!fDynamicTensorInfos.empty()) {
      fGC += "//--- declare the dynamic tensors\n";
      for (auto &i : fDynamicTensorInfos) {
         if (i.second.type == ETensorType::FLOAT) {
            fGC += "std::vector<float> fTensor_" + i.first + ";\n";
            fGC += "float * tensor_" + i.first + " = nullptr;\n";
         } else if (i.second.type == ETensorType::DOUBLE) {
            fGC += "std::vector<double> fTensor_" + i.first + ";\n";
            fGC += "double * tensor_" + i.first + " = nullptr;\n";
         } else if (i.second.type == ETensorType::INT64) {
            fGC += "std::vector<int64_t> fTensor_" + i.first + ";\n";
            fGC += "int64_t * tensor_" + i.first + " = nullptr;\n";
         }
      }
   }
}

// generate code for specific operator declarations  to be defined in the Session class
void RModel::GenerateOperatorDeclarations() {
   std::string strcode;
   for (auto & op : fOperators) {
      strcode += op->GenerateDeclCode();
   }
   if (strcode.empty()) return;
   fGC += "\n//---- operator declarations \n";
   fGC += strcode;
   fGC += "\n";
}

void RModel::GenerateDynamicTensorInfo()
{
   std::stringstream out;
   for (auto &i : fDynamicTensorInfos) {
      auto length = ConvertDynamicShapeToLength(i.second.shape);
      out << SP << "if (" << length << " > 0) {\n";
      out << SP << SP << "fTensor_" << i.first << ".resize(" << length << ");\n";
      out << SP << SP << "tensor_" << i.first << " = fTensor_" << i.first << ".data();\n";
      out << SP << "}\n";
   }
   fGC += out.str();
}

std::string RModel::GenerateInferSignature(bool isdecl) {
   // generate the infer signature given the inputs: eg. "float * tensor1, float * tensor2"
   // if (decl = false) generate only calling signature (tensor1,tensor2,....)
   std::string rGC;
   std::unordered_map<std::string, int> inputParams;
   int i_input = 0;
   for (auto &name : fInputTensorNames) {
      // if is a dynamic tensor pass initial parameters
      if (IsDimInputTensor(name)) {
         auto shape = GetDynamicTensorShape(name);
         for (auto &d : shape) {
            std::string pName = d.param;
            // need to check if the input parameters is already existing in another input tensor
            if (d.isParam && inputParams.count(pName) == 0) {
               if (isdecl) rGC += "size_t ";
               rGC += d.param + ",";
               inputParams[pName] = i_input;
            }
         }
      }
      if (isdecl) {
         std::string type = ConvertTypeToString(GetTensorType(name));
         if (type == "other")
            throw std::runtime_error("TMVA-SOFIE: input tensor " + name +
                                     " is of a data type which is not yet supported.");
         rGC += type + " const* ";
      }
      rGC += "tensor_" + name + ",";
      i_input++;
   }

   if (fInputTensorNames.size() > 0) rGC.pop_back();// remove last ","
   return rGC;
}

namespace {

std::string typeForOutput(ETensorType t) {
   // The std::vector<bool> is a special type that is not wrapping continuous memory.
   // We don't want to use it as a return type.
   if (t == ETensorType::BOOL) t = ETensorType::UINT8;
   return ConvertTypeToString(t);
}

}

void RModel::GenerateOutput()
{
   size_t outputSize = fOutputTensorNames.size();
   // assume output types are all the same

   bool sameOutputTypes = true;
   std::string inferReturnType; // type return by infer function
   ETensorType eFirstOutputType = GetTensorType(*fOutputTensorNames.begin());
   fGC += "\n\n";
   if (outputSize == 1) {
      fGC += "std::vector<" + typeForOutput(eFirstOutputType) + ">";
   } else {
      // if all output types are the same we return an std::vector - otherwise a tuple
      for (std::string const &name : fOutputTensorNames) {
         if (GetTensorType(name) != eFirstOutputType)
            sameOutputTypes = false;
      }
      if (sameOutputTypes)
         fGC += "std::vector<std::vector<" + typeForOutput(eFirstOutputType) + ">>";
      else {
         inferReturnType = "std::tuple<";
         for (size_t i = 0; i < outputSize; i++) {
            inferReturnType += "std::vector<" + typeForOutput(GetTensorType(fOutputTensorNames[i])) + ">";
            if (i < outputSize - 1)
               inferReturnType += ",";
         }
         inferReturnType += ">";
         fGC += inferReturnType;
      }
   }

   fGC += " infer(" + GenerateInferSignature() + "){\n";

   std::string doInferArgs = GenerateInferSignature(false);
   if (!doInferArgs.empty())
      doInferArgs += ",";
   for (std::string const &name : fOutputTensorNames) {
      fGC += SP + "std::vector<" + typeForOutput(GetTensorType(name)) + " > output_tensor_" + name + ";\n";
      doInferArgs += " output_tensor_" + name + ",";
   }
   if (!doInferArgs.empty())
      doInferArgs.back() = ' ';

   fGC += SP + "doInfer(" + doInferArgs + ");\n";

   fGC += SP + "return {";
   for (size_t i = 0; i < fOutputTensorNames.size(); i++) {
      fGC += "output_tensor_" + fOutputTensorNames[i];
      if (i < fOutputTensorNames.size() - 1)
         fGC += ",";
   }
   fGC += "};\n";
   fGC += "}\n"; // end of infer function scope
}

void RModel::GenerateSessionCode()
{
   // Determine the signature of the actual inference function
   std::string doInferSignature = GenerateInferSignature();
   if (!doInferSignature.empty())
      doInferSignature += ", ";
   for (auto const &name : fOutputTensorNames) {
      doInferSignature += " std::vector<" + typeForOutput(GetTensorType(name)) + "> &output_tensor_" + name + ",";
   }
   doInferSignature.back() = ' ';

   doInferSignature = "void doInfer(" + doInferSignature + ")";

   // define the Session struct (for GNN this is generated in RModel_GNN)
   if (fUseSession && !fIsGNNComponent) {
      if (!fIsSubGraph)
         fGC += "struct Session {\n";
      else
         fGC += "struct Session_" + fName + " {\n";
   }

   // generate code for declaring the initialized tensors
   GenerateInitializedTensorInfo();

   if (fOptimizationLevel == OptimizationLevel::kExtended) {
      // evaluate total intermediate memory and position intermediate tensor addresses
      std::string intermediate_memory_alloc_string = "";
      intermediate_memory_alloc_string += "\n// --- Positioning intermediate tensor memory --";
      for (size_t op_idx = 0; op_idx < fOperators.size(); ++op_idx) {
         intermediate_memory_alloc_string += AllocateIntermediateMemory(fOperators[op_idx]->GetOpOutputTensors());
         CheckAndFlushIntermediateMemory(fOperators[op_idx]->GetOpInputTensors(), op_idx);
      }

      // to check remaining unused fragments after memory allocation (lesser the better)
      // for (const auto &it: fIntermediateMemoryInfo.available_stack){
      //    std::cout<<"chunk_idx: "<<it.first<<", chunk_size: "<<it.second<<"\n";
      // }

      // generate the memory pool to be used by intermediate tensors
      GenerateIntermediateMemoryPool();

      // position intermediate tensors
      fGC += intermediate_memory_alloc_string;
   }

   // generate the declaring the intermediate tensors
   GenerateIntermediateTensorInfo();
   // generate code for declarations of some specific operators
   GenerateOperatorDeclarations();



   // add subgraph session
   if (!fSubGraphs.empty()) fGC += "//   subgraph sessions\n";
   for (auto & graph : fSubGraphs) {
      fGC += "Session_" + graph->fName + "  fSession_" + graph->fName + ";\n";
   }

   // Generate code for Session constructor
   if (fUseSession) {
      std::string sessionName = "Session";
      if (fIsSubGraph)
         sessionName += "_" + fName;
      // add here specific operator code that needs to define session data members
      fGC += "\n";
      for (size_t id = 0; id < fOperators.size(); id++) {
         std::string opName = std::to_string(id);
         fGC += fOperators[id]->GenerateSessionMembersCode(opName);
      }
      fGC += "\n";
      // here add initialization and reading of weight tensors
      if (fUseWeightFile) {
         std::string fileName = fName;
         if (fWeightFile == WeightFileType::Text) {
            fileName += ".dat";
         }
         if (fWeightFile == WeightFileType::RootBinary) {
            fileName += ".root";
         }
         fGC += sessionName + "(std::string filename =\"" + fileName + "\"";
      } else {
         // no need to pass weight file since it is not used
         // keep passing a string for compatibility
         fGC += sessionName + "(std::string = \"\"";
      }
      // add initialization of shape parameters
      // assume all parameters are of type size_t
      if (!fShapeParams.empty()) {
         for (auto &p : fShapeParams) {
            fGC += ",\n";
            fGC += "        size_t " + p.first + " = " + p.second;
         }
      }
      fGC += ") {\n";

      if (fUseWeightFile) {
         fGC += "\n//--- reading weights from file\n";
         ReadInitializedTensorsFromFile(fReadPos);
         fGC += "\n";
         // fUseWeightFile = fUseWeightFile;
      }

      // now we have passed the parameters we can allocate the dynamic tensors
      GenerateDynamicTensorInfo();

      // add here initialization code  for operator
      for (size_t id = 0; id < fOperators.size(); id++) {
         fGC += fOperators[id]->GenerateInitCode();
      }

      fGC += "}\n\n";
   }

   fGC += doInferSignature + "{\n";
   fGC += "\n";

   // generate the inference code
   if (fVerbose)
      std::cout << "Generating main inference code for " << fName << std::endl;

   if (fOutputTensorNames.size() == 0)
      throw std::runtime_error("TMVA-SOFIE: output size=0 are not supported");

   for (size_t op_idx = 0; op_idx < fOperators.size(); ++op_idx) {
      if (fVerbose)
         std::cout << "Generating code for operator .... " << op_idx << std::endl;
      fGC += (fOperators[op_idx]->Generate(std::to_string(op_idx)));
   }

   fGC += SP + "using TMVA::Experimental::SOFIE::UTILITY::FillOutput;\n\n";

   for (std::string const &name : fOutputTensorNames) {
      // need to check is size is the same (don't want to return a vector with
      // larger size) in that case better to copy
      bool isIntermediate = fIntermediateTensorInfos.count(name) > 0;
      std::string n = isIntermediate ? std::to_string(ConvertShapeToLength(GetTensorShape(name)))
                                     : ConvertDynamicShapeToLength(GetDynamicTensorShape(name));
      fGC += SP + "FillOutput(tensor_" + name + ", output_tensor_" + name + ", " + n + ");\n";
   }

   fGC += "}\n\n";

   // generate the inference overload that returns an output struct
   GenerateOutput();

   // end of session
   if (fUseSession && !fIsGNNComponent) {
      fGC += "};   // end of Session\n\n";
   }
}

void RModel::Generate(std::underlying_type_t<Options> options, int batchSize, long pos, bool verbose)
{
   fVerbose = verbose;
   fBatchSize = batchSize;
   fReadPos = pos;

   // session flag is used in operator initialize
   if (static_cast<std::underlying_type_t<Options>>(Options::kNoSession) & options) {
      fUseSession = false;
      fWeightFile = WeightFileType::None;
   }
   if (static_cast<std::underlying_type_t<Options>>(Options::kNoWeightFile) & options) {
      fUseWeightFile = false;
      fWeightFile = WeightFileType::None;
   }
   if (static_cast<std::underlying_type_t<Options>>(Options::kRootBinaryWeightFile) & options) {
      fUseWeightFile = true;
      fWeightFile = WeightFileType::RootBinary;
   }
   if (fUseWeightFile && !fUseSession) {
      throw std::runtime_error(
         "TMVA-SOFIE: RModel::Generate: cannot use a separate weight file without generating a Session class");
   }

   if (static_cast<std::underlying_type_t<Options>>(Options::kGNN) & options)
      fIsGNN = true;
   if (static_cast<std::underlying_type_t<Options>>(Options::kGNNComponent) & options)
      fIsGNNComponent = true;

   // initialize the model including all operators and sub-graphs
   Initialize(batchSize, verbose);

   std::string hgname;
   if (!fIsGNNComponent && !fIsSubGraph) {
      fGC.clear();
      GenerateHeaderInfo(hgname);
   }

   // generate first code for the subgraphs
   for (auto &graph : fSubGraphs) {
      if (fVerbose)
         std::cout << "generate session code for subgraph " << graph->fName << std::endl;
      graph->GenerateSessionCode();
      fGC += graph->fGC;
   }

   if (fVerbose)
      std::cout << "generate Main session code - model  " << fName << std::endl;

   // generate main session code
   GenerateSessionCode();

   if (!fIsGNNComponent && !fIsSubGraph) {
      fGC += ("} //TMVA_SOFIE_" + fName + "\n");
      fGC += "\n#endif  // " + hgname + "\n";
   }
}

void RModel::ReadInitializedTensorsFromFile(long pos) {
    // generate the code to read initialized tensors from a text data file
    if (fWeightFile == WeightFileType::Text) {
        if (fInitializedTensors.empty()) return;

        fGC += "   std::ifstream f;\n";
        fGC += "   f.open(filename);\n";
        fGC += "   if (!f.is_open()) {\n";
        fGC += "      throw std::runtime_error(\"tmva-sofie failed to open file \" + filename + \" for input weights\");\n";
        fGC += "   }\n";

        if(fIsGNNComponent) {
            fGC += "   f.seekg(" + std::to_string(pos) + ");\n";
        }

        fGC += "   using TMVA::Experimental::SOFIE::ReadTensorFromStream;\n";

        // loop on tensors and parse the file
        for (auto& i: fInitializedTensors) {
            // skip Constant and shape tensors (not written in a file)
            if (!i.second.IsWeightTensor()) continue;
            std::string tensor_name = "tensor_" + i.first;
            if (i.second.type() == ETensorType::FLOAT) {
               std::string length = std::to_string(ConvertShapeToLength(i.second.shape()));
               fGC += "   ReadTensorFromStream(f, " + tensor_name + ", \"" + tensor_name + "\", " + length + ");\n";
            } else {
               std::runtime_error("tmva-sofie tensor " + tensor_name + " with type " + ConvertTypeToString(i.second.type()) + " cannot be read from a file");
            }
        }
        fGC += "   f.close();\n";
    }

    // generate the code to read initialized tensors from a ROOT data file
    if(fWeightFile == WeightFileType::RootBinary) {
#ifdef SOFIE_SUPPORT_ROOT_BINARY
        fGC += "  {\n";
        fGC += "   std::unique_ptr<TFile> rootFile(TFile::Open(filename.c_str(), \"READ\"));\n";
        fGC += "   if (!rootFile->IsOpen()) {\n";
        fGC += "      throw std::runtime_error(\"tmva-sofie failed to open ROOT file for input weights\");\n";
        fGC += "   }\n";

        std::string dirName = fName + "_weights";
        fGC += "   if (!rootFile->GetKey(\"" + dirName + "\")) {\n";
        fGC += "      throw std::runtime_error(\"tmva-sofie failed to open ROOT directory for input weights\");\n";
        fGC += "   }\n";

        for (auto &i : fInitializedTensors) {
            // skip Constant and shape tensors
            if (!i.second.IsWeightTensor()) continue;
            fGC += "  {\n";
            std::string tensor_name = "tensor_" + i.first;
            if (i.second.type() == ETensorType::FLOAT) {
               fGC += "      fTensor_" + i.first + " = *reinterpret_cast<std::vector<float>*>(rootFile->Get(\"";
               fGC += dirName + "/" + tensor_name + "\"));\n";
            } else if (i.second.type() == ETensorType::DOUBLE) {
               fGC += "      fTensor_" + i.first + " = *reinterpret_cast<std::vector<double>*>(rootFile->Get(\"";
               fGC += dirName + + "/" + tensor_name + "\"));\n";
            } else if (i.second.type() == ETensorType::INT64) {
               fGC += "      fTensor_" + i.first + " = *reinterpret_cast<std::vector<int64_t>*>(rootFile->Get(\"";
               fGC += dirName + "/" + tensor_name + "\"));\n";
            } else {
               std::runtime_error("tmva-sofie tensor " + tensor_name + " with type " + ConvertTypeToString(i.second.type()) + " cannot be read from a ROOT file");
            }
            fGC += "  }\n";
        }
        fGC += "  }\n";
#else
        throw std::runtime_error("SOFIE was not built with ROOT file support.");
#endif // SOFIE_SUPPORT_ROOT_BINARY
    }
}

long RModel::WriteInitializedTensorsToFile(std::string filename) {
    // Determine the file extension based on the weight file type
    std::string fileExtension;
    switch (fWeightFile) {
    case WeightFileType::None:
        fileExtension = ".dat";
        break;
    case WeightFileType::RootBinary:
        fileExtension = ".root";
        break;
    case WeightFileType::Text:
        fileExtension = ".dat";
        break;
    }

    // If filename is empty, use the model name as the base filename
    if (filename.empty()) {
        filename = fFileName + fileExtension;
    }

    // Write the initialized tensors to the file
    if (fWeightFile == WeightFileType::RootBinary) {
#ifdef SOFIE_SUPPORT_ROOT_BINARY
        if(fIsGNNComponent || fIsGNN) {
            throw std::runtime_error("SOFIE-GNN yet not supports writing to a ROOT file.");
        }
        std::unique_ptr<TFile> outputFile(TFile::Open(filename.c_str(), "UPDATE"));

        std::string dirName = fName + "_weights";
        // check if directory exists, in case delete to replace with new one
        if (outputFile->GetKey(dirName.c_str()))
            outputFile->rmdir(dirName.c_str());

        auto outputDir = outputFile->mkdir(dirName.c_str());

        for (const auto& item : fInitializedTensors) {
            // skip Constant tensors and tensors which are not writable (e.g. shape tensors)
            if (!item.second.IsWeightTensor()) continue;
            std::string tensorName = "tensor_" + item.first;
            size_t length = 1;
            length = ConvertShapeToLength(item.second.shape());
            if(item.second.type() == ETensorType::FLOAT) {
               const float* data = item.second.data<float>();
                std::vector<float> tensorDataVector(data, data + length);
               outputDir->WriteObjectAny(&tensorDataVector, "std::vector<float>", tensorName.c_str());
            }
            else if(item.second.type() == ETensorType::DOUBLE) {
               const double* data = item.second.data<double>();
               std::vector<double> tensorDataVector(data, data + length);
               outputDir->WriteObjectAny(&tensorDataVector, "std::vector<double>", tensorName.c_str());
            }
            else if(item.second.type() == ETensorType::INT64) {
               const int64_t* data = item.second.data<int64_t>();
               std::vector<int64_t> tensorDataVector(data, data + length);
               outputDir->WriteObjectAny(&tensorDataVector, "std::vector<int64_t>", tensorName.c_str());
            }
            else {
               std::runtime_error("tmva-sofie tensor " + tensorName + " with type " + ConvertTypeToString(item.second.type()) +
                                  " cannot be written to a ROOT file");
            }
        }
        outputFile->Write(filename.c_str());

        // this needs to be changed, similar to the text file
        return -1;

#else
        throw std::runtime_error("SOFIE was not built with ROOT file support.");
#endif // SOFIE_SUPPORT_ROOT_BINARY
    } else if (fWeightFile == WeightFileType::Text) {
        std::ofstream f;
        if(fIsGNNComponent) {
            // appending all GNN components into the same file
            f.open(filename, std::ios::app);
        } else {
            f.open(filename);
        }
        if (!f.is_open())
            throw
            std::runtime_error("tmva-sofie failed to open file " + filename + " for tensor weight data");
        for (auto& i: fInitializedTensors) {
             // skip Constant tensors and not writable tensors (e.g. shape tensors)
            if (!i.second.IsWeightTensor()) {
               continue;
            }
            size_t length = ConvertShapeToLength(i.second.shape());
            std::string tensor_name = "tensor_" + i.first;
            f << tensor_name << " " << length << "\n";
            if (i.second.type() == ETensorType::FLOAT) {
               const float * data = i.second.data<float>();
               for (size_t idx = 0; idx < length; idx++) {
                  // round to zero sub-normal values
                  float value = data[idx];
                  if (value != 0. && std::abs(value) < std::numeric_limits<float>::min() ) value = 0;
                  f << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
                  f <<  ( (idx < length-1) ? " " : "\n" );
               }
            }
            else {
               std::runtime_error("tmva-sofie tensor " + tensor_name + " with type " + ConvertTypeToString(i.second.type()) + " cannot be written to a file");
            }
            if (f.fail())
               std::runtime_error("tmva-sofie failed to write tensor data to file for  " + tensor_name);
        }
        long curr_pos = f.tellp();
        f.close();
        return curr_pos;
    } else {
        return -1;
    }
}

void RModel::PrintRequiredInputTensors() {
    std::cout << "Model requires following inputs:\n";
    for (auto& inputInfo: fInputTensorInfos) {
        std::cout << "Parametrised Tensor name: " << inputInfo.first << "\t";
        std::cout << "type: " << ConvertTypeToString(inputInfo.second.type) << "\t";
        std::cout << "shape: [";
        for (size_t i = 0; i < inputInfo.second.shape.size(); i++) {
            if (inputInfo.second.shape[i].isParam) {
                std::cout << inputInfo.second.shape[i].param;
            } else {
                std::cout << inputInfo.second.shape[i].dim ;
            }
            if (i < inputInfo.second.shape.size() - 1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
    }

    for (auto& inputInfo: fReadyInputTensorInfos) {
        std::cout << "Fully Specified Tensor name: " << inputInfo.first << "\t";
        std::cout << "type: " << ConvertTypeToString(inputInfo.second.type) << "\t";
        std::cout << "shape: [";
        for (size_t i = 0; i < inputInfo.second.shape.size(); i++) {
            std::cout << inputInfo.second.shape[i];
            if (i < inputInfo.second.shape.size() - 1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "\n";
}

void RModel::PrintInitializedTensors() {
    std::cout << "Model initialized the following tensors:\n";
    for (auto& it: fInitializedTensors) {
        std::cout << "Tensor name: \"" << it.first << "\"\t";
        std::cout << "type: " << ConvertTypeToString(it.second.type()) << "\t";
        std::cout << "shape: [";
        for (size_t i = 0; i < it.second.shape().size(); i++) {
            std::cout << it.second.shape()[i];
            if (i < it.second.shape().size() - 1) std::cout << ",";
        }
        std::cout << "]";
        if (it.second.IsConstantTensor()) std::cout << " (Constant)";
        else if (!it.second.IsWeightTensor()) std::cout << " (Not Writable)";
        std::cout << std::endl;
    }
    std::cout << "\n";
}

void RModel::PrintIntermediateTensors() {
    std::cout << "Model specify the following intermediate tensors:\n";
    for (auto& it: fIntermediateTensorInfos) {
        std::cout << "Tensor name: \"" << it.first << "\"\t";
        std::cout << "type: " << ConvertTypeToString(it.second.type) << "\t";
        std::cout << "shape: [";
        for (size_t i = 0; i < it.second.shape.size(); i++) {
            std::cout << it.second.shape[i];
            if (i < it.second.shape.size() - 1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "\n";
}

void RModel::PrintDynamicTensors() {
    std::cout << "Model specify the following dynamic tensors:\n";
    for (auto& it: fDynamicTensorInfos) {
        std::cout << "Tensor name: \"" << it.first << "\"\t";
        std::cout << "type: " << ConvertTypeToString(it.second.type) << "\t";
        std::cout << "shape: [";
        for (size_t i = 0; i < it.second.shape.size(); i++) {
            std::cout << it.second.shape[i].GetVal();
            if (i < it.second.shape.size() - 1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "\n";
}

void RModel::PrintOutputTensors() {
    std::cout << "Model specify the following output tensors:\n";
    for (auto& it: fOutputTensorNames) {
        std::cout << "Tensor name: \"" << it << "\"\t";
        if (!IsDynamicTensor(it))
          std::cout << "shape: " << ConvertShapeToString(GetTensorShape(it)) << std::endl;
       else
          std::cout << "shape: " << ConvertDynamicShapeToString(GetDynamicTensorShape(it)) << std::endl;
    }
    std::cout << "\n";
}

void RModel::HeadInitializedTensors(std::string name, int n_print) {
    auto it = fInitializedTensors.find(name);
    if (it == fInitializedTensors.end()) {
        std::cout << "Tensor " << name << " not found in model's initialized tensor list" << std::endl;
        return;
    }

    std::cout << "Tensor name: " << it->first << "\t";
    std::cout << "type: " << ConvertTypeToString(it->second.type()) << "\t";
    int length =1;
    std::cout << "shape: [";
    for (size_t i = 0; i < it->second.shape().size(); i++) {
        std::cout << it->second.shape()[i];
        length *= it->second.shape()[i];
        if (i < it->second.shape().size() - 1) std::cout << ",";
    }
    std::cout << "]" << std::endl;
    bool ellipsis = true;
    if (n_print > length) {
        n_print = length;
        ellipsis = false;
    }

    std::cout << "data: [" << std::endl;
    if (it->second.type() == ETensorType::FLOAT) {
        auto converted_data = it->second.data<float>();
        for (int i =0; i < n_print; i++) {
            std::cout << converted_data[i];
            if (i < n_print - 1) std::cout << " ,";
        }
    }
    if (ellipsis) std::cout << ", ...";
    std::cout << "]" << std::endl;

}

void RModel::OutputGenerated(std::string filename, bool append) {

    RModel_Base::OutputGenerated(filename, append);

    // write weights in a text file
    if (fUseWeightFile) {
        if (!filename.empty()) {
            size_t pos = filename.find(".hxx");
            if (fWeightFile == WeightFileType::Text)
                filename.replace(pos, 4, ".dat");
            if (fWeightFile == WeightFileType::RootBinary)  {
                filename = filename.erase(pos, 4);
                filename += ".root";
            }
        } else {
            filename = fName;
            filename += fWeightFile == WeightFileType::Text ? ".dat" : ".root";
        }
        WriteInitializedTensorsToFile(filename);
    }
}

void RModel::Streamer(TBuffer &R__b) {
    if (R__b.IsReading()) {
        RModel::Class()->ReadBuffer(R__b, this);
        for (auto & i : fInitializedTensors) {
            i.second.CastPersistentToShared();
        }
    }
    else {
        for (auto & i : fInitializedTensors) {
            i.second.CastSharedToPersistent();
        }
        RModel::Class()->WriteBuffer(R__b, this);
    }
}

}//SOFIE
}//Experimental
}//TMVA
