#include <limits>
#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include "TFile.h"

#include "TMVA/RModel.hxx"
#include "TMVA/SOFIE_common.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

std::underlying_type_t<Options> operator|(Options opA, Options opB) {
    return static_cast<std::underlying_type_t<Options>>(opA) | static_cast<std::underlying_type_t<Options>>(opB);
}
std::underlying_type_t<Options> operator|(std::underlying_type_t<Options> opA, Options opB) {
    return opA | static_cast<std::underlying_type_t<Options>>(opB);
}

RModel::RModel(RModel&& other) {
    fInputTensorInfos = std::move(other.fInputTensorInfos);
    fReadyInputTensorInfos = std::move(other.fReadyInputTensorInfos);
    fOutputTensorNames = other.fOutputTensorNames;
    fInputTensorNames = other.fInputTensorNames;
    fOperators = std::move(other.fOperators);
    fInitializedTensors = std::move(other.fInitializedTensors);
    fIntermediateTensorInfos = std::move(other.fIntermediateTensorInfos);
    fName = other.fName;
    fFileName = other.fFileName;
    fParseTime = other.fParseTime;
    fGC = other.fGC;
    fNeededBlasRoutines = other.fNeededBlasRoutines;
    fNeededStdLib = other.fNeededStdLib;
}

RModel& RModel::operator=(RModel&& other) {
    fInputTensorInfos = std::move(other.fInputTensorInfos);
    fReadyInputTensorInfos = std::move(other.fReadyInputTensorInfos);
    fOutputTensorNames = other.fOutputTensorNames;
    fInputTensorNames = other.fInputTensorNames;
    fOperators = std::move(other.fOperators);
    fInitializedTensors = std::move(other.fInitializedTensors);
    fIntermediateTensorInfos = std::move(other.fIntermediateTensorInfos);
    fName = other.fName;
    fFileName = other.fFileName;
    fParseTime = other.fParseTime;
    fGC = other.fGC;
    fNeededBlasRoutines = other.fNeededBlasRoutines;
    fNeededStdLib = other.fNeededStdLib;
    return *this;
}

const std::vector<size_t>& RModel::GetTensorShape(std::string name) {
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

    throw std::runtime_error("TMVA SOFIE tensor [" + name + "] for which the shape is requested is not found");
}

std::vector<Dim> RModel::GetDynamicTensorShape(std::string name) {
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

const ETensorType& RModel::GetTensorType(std::string name) {
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

    throw std::runtime_error("TMVA SOFIE tensor [" + name + "] for which the type is requested is not found");
}

bool RModel::CheckIfTensorAlreadyExist(std::string tensor_name) {
    if (fReadyInputTensorInfos.find(tensor_name) != fReadyInputTensorInfos.end())  return true;
    if (fInputTensorInfos.find(tensor_name) != fInputTensorInfos.end()) return true;
    if (fInitializedTensors.find(tensor_name) != fInitializedTensors.end()) return true;
    if (fIntermediateTensorInfos.find(tensor_name) != fIntermediateTensorInfos.end()) return true;
    if (fDynamicTensorInfos.find(tensor_name) != fDynamicTensorInfos.end()) return true;
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
    fInputTensorNames.push_back(UTILITY::Clean_name(input_name));
}

void RModel::AddOperator(std::unique_ptr<ROperator> op, int order_execution) {
    AddBlasRoutines(op->GetBlasRoutines());
    auto libs = op->GetStdLibs();
    for (auto& stdlib : libs) {
        AddNeededStdLib(stdlib);
    }
    if (order_execution >= 0) {
        fOperators.insert(fOperators.begin() + order_execution, std::move(op));
    } else {
        fOperators.push_back(std::move(op));
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

bool RModel::IsDynamicTensor(const std::string& tensorName) const {
   std::string name = UTILITY::Clean_name(tensorName);
   return fDynamicTensorInfos.find(name) != fDynamicTensorInfos.end();
}
bool RModel::IsInputTensor(const std::string& tensorName) const {
   std::string name = UTILITY::Clean_name(tensorName);
   return fInputTensorInfos.find(name) != fInputTensorInfos.end();
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
        fOutputTensorNames.push_back(UTILITY::Clean_name(it));
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

void RModel::Initialize(int batchSize, bool verbose) {

   fIntermediateTensorInfos.clear();
   fDynamicTensorInfos.clear();

   // loop on inputs and see if shape can be  full specified
   // if the batch size is provided it can be used to specify the full shape
   // Add the full specified tensors in fReadyInputTensors collection
   auto originalInputTensorInfos = fInputTensorInfos; // need to copy because we may delete elements
   for (auto &input : originalInputTensorInfos) {
      if (verbose) std::cout << "looking at the tensor " << input.first << std::endl;
      // if a batch size is provided convert batch size
      // assume is parameterised as "bs" or "batch_size"
      if (batchSize > 0) {
         // std::vector<Dim> shape;
         // shape.reserve(input.second.shape.size());
         // assume first parameter is teh batch size
         if (!input.second.shape.empty()) {
            auto & d0 = input.second.shape[0];
            if (d0.isParam) {
               if (verbose) std::cout << "Fix the batch size to " << batchSize << std::endl;
               d0 = Dim{static_cast<size_t>(batchSize)};
            }
            else {  // look for cases that a bs or bath_size is specified in tensor shape
               for (auto &d : input.second.shape) {
                  if (d.isParam && (d.param == "bs" || d.param == "batch_size")) {
                     d = Dim{static_cast<size_t>(batchSize)};
                     if (verbose) std::cout << "Input shape has bs or batch_size as names. Fix the batch size to " << batchSize << std::endl;
                  }
               }
            }
         }
      }
      auto shape = ConvertShapeToInt(input.second.shape);
      if (!shape.empty()) {
         // remove from the tensor info old dynamic shape
         fInputTensorInfos.erase(input.first);
         // add to the ready input tensor information the new fixed shape
         AddInputTensorInfo(input.first, input.second.type, shape);
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
   for (auto &op : fOperators) {
      if (verbose) {
         auto& r = *op.get();
         std::cout << "Initializing operator " << i << "  " << typeid(r).name() << std::endl;
      }
      op->Initialize(*this);
      i++;
   }
}

void RModel::GenerateInitializedTensorInfo() {
    if (!fInitializedTensors.empty())
      fGC += "// initialized tensors\n";
    for (auto& i: fInitializedTensors) {

         size_t length = ConvertShapeToLength(i.second.shape());
            // in case we are not using weight files or for tensor created from Constant operator
         if (!fUseWeightFile || i.second.IsConstantTensor() ) {
            //std::cout << "write tensor " << i.first << std::endl;
            std::stringstream strs;
            if (i.second.type() == ETensorType::FLOAT) {
               strs << "float tensor_" << i.first << "[" << length << "] = {";
               float const *data = i.second.data<float>();
               for (size_t idx = 0; idx < length; idx++) {
                  strs << std::setprecision(std::numeric_limits<float>::max_digits10) << data[idx];
                  if (idx < length-1) strs << ", ";
               }
               strs << "};\n";
            }
            else if (i.second.type() == ETensorType::INT64) {
               strs << "int64_t tensor_" << i.first << "[" << length << "] = {";
               int64_t const *data = i.second.data<int64_t>();
               for (size_t idx = 0; idx < length; idx++) {
                  strs << data[idx];
                  if (idx < length-1) strs << ", ";
               }
               strs << "};\n";
            }
            fGC += strs.str();
         }
         // case of tensors which are read from a file
         else {
            if (i.second.type() == ETensorType::FLOAT) {
               fGC += "std::vector<float> fTensor_" + i.first + " = std::vector<float>(" + std::to_string(length) + ");\n";
               fGC += "float * tensor_" + i.first + " = fTensor_" + i.first + ".data();\n";
            }
         }
    }
}

void RModel::GenerateIntermediateTensorInfo() {
   if (!fIntermediateTensorInfos.empty()) {
      fGC += "\n//--- declare and allocate the intermediate tensors\n";
      for (auto &i : fIntermediateTensorInfos) {
         size_t length = ConvertShapeToLength(i.second.shape);
         if (i.second.type == ETensorType::FLOAT) {
            fGC += "std::vector<float> fTensor_" + i.first + " = std::vector<float>(" + std::to_string(length) + ");\n";
            fGC += "float * tensor_" + i.first + " = fTensor_" + i.first + ".data();\n";
         }
         if (i.second.type == ETensorType::DOUBLE) {
            fGC += "std::vector<double> fTensor_" + i.first + " = std::vector<double>(" + std::to_string(length) + ");\n";
            fGC += "double * tensor_" + i.first + " = fTensor_" + i.first + ".data();\n";
         }
         if (i.second.type == ETensorType::INT64) {
            fGC += "std::vector<int64_t> fTensor_" + i.first + " = std::vector<int64_t>(" + std::to_string(length) + ");\n";
            fGC += "int64_t * tensor_" + i.first + " = fTensor_" + i.first + ".data();\n";
         }
         if (i.second.type == ETensorType::BOOL) {
            fGC += "std::vector<bool> fTensor_" + i.first + " = std::vector<bool>(" + std::to_string(length) + ");\n";
            // don't allocate pointer since boolean vector don't have the .data() member
         }
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

void RModel::GenerateDynamicTensorInfo() {
    fGC += "//---- allocate the intermediate dynamic tensors\n";
    std::stringstream out;
    for (auto & i: fDynamicTensorInfos) {
        auto length = ConvertDynamicShapeToLength(i.second.shape);
        out << SP <<  "if (" << length << " > 0) {\n";
        out << SP << SP <<  "fTensor_" <<  i.first  <<  ".resize(" <<  length << ");\n";
        out << SP << SP <<  "tensor_" << i.first << " = fTensor_" << i.first  << ".data();\n";
        out << SP << "}\n";
    }
    fGC += out.str();
}

void RModel::GenerateOutput() {

   size_t outputSize = fOutputTensorNames.size();
   // assume output types are all the same
   if (outputSize == 0)
      throw std::runtime_error("TMVA-SOFIE: output size=0 are not supported");

   std::string outputType;
   ETensorType eOutputType;
   eOutputType = GetTensorType(fOutputTensorNames[0]);
   outputType = ConvertTypeToString(eOutputType);
   if (outputSize == 1) {
      fGC += "std::vector<" + outputType + "> ";
   } else {
      // we assume all output types are the same
      for (size_t i = 1; i < outputSize; i++) {
         if (GetTensorType(fOutputTensorNames[i]) != eOutputType)
            throw std::runtime_error("TMVA-SOFIE: different output tensor types are not supported");
      }
      fGC += "std::vector<std::vector<" + outputType + ">> ";
   }

   fGC += "infer(";

   std::unordered_map<std::string, int> inputParams;
   int i_input = 0;
   for (auto &name : fInputTensorNames) {
      // if is a dynamic tensor pass initial parameters
      if (IsInputTensor(name)) {
         auto shape = GetDynamicTensorShape(name);
         for (auto &d : shape) {
            std::string pName = d.param;
            // need to check if the input parameters is already existing in another input tensor
            if (d.isParam && inputParams.count(pName) == 0) {
               fGC += "size_t " + d.param + ",";
               inputParams[pName] = i_input;
            }
         }
      }
      switch (GetTensorType(name)) {
      case ETensorType::FLOAT: {
         fGC += "float* tensor_" + name + ",";
         break;
      }
      case ETensorType::INT32: {
         fGC += "int32_t* tensor_" + name + ",";
         break;
      }
      case ETensorType::INT64: {
         fGC += "int64_t* tensor_" + name + ",";
         break;
      }
      case ETensorType::DOUBLE: {
         fGC += "double* tensor_" + name + ",";
         break;
      }
      case ETensorType::BOOL: {
         fGC += "bool* tensor_" + name + ",";
         break;
      }
      default: {
         throw std::runtime_error("TMVA-SOFIE: input tensor " + name +
                                  " is of a data type which is not yet supported.");
      }
      }
      i_input++;
   }

   if (fInputTensorNames.size() > 0) fGC.pop_back();// remove last ","
   fGC += "){\n";

   for (size_t id = 0; id < fOperators.size(); id++) {
      fGC += (fOperators[id]->Generate(std::to_string(id)));
   }

   if (outputSize == 1) {
      std::string tensorName = fOutputTensorNames[0];
      if (fIntermediateTensorInfos.count(tensorName) > 0) {
         // need to check is size is the same(don't want to return a vector with larger size)
         // in that case better to copy
         fGC += SP + "return fTensor_" + tensorName + ";\n";
      } else {
         // include also dynamic tensors since the vectors can be allocated with a size larger than their output
         // we need a special handling for bool type allocated as vector<bool>
         auto outputLength = ConvertDynamicShapeToLength(GetDynamicTensorShape(tensorName));
         if (IsDynamicTensor(tensorName) && eOutputType == ETensorType::BOOL) {
            fGC += SP + "std::vector<bool> ret (fTensor_" + tensorName + ".begin(), fTensor_" + tensorName +
                   ".begin() + " + outputLength + ");\n";
         } else {
            fGC += SP + "std::vector<" + outputType + "> ret (tensor_" + tensorName + ", tensor_" + tensorName + " + " +
                  outputLength + ");\n";
         }
         fGC += SP + "return ret;\n";
      }
   } else {
      // here we assume all outputs have same type
      fGC += SP + "std::vector<std::vector<" + outputType + ">> ret({";
      for (size_t i = 0; i < outputSize; i++) {
         std::string tensorName = fOutputTensorNames[i];
         if (!tensorName.empty()) {
            if (fIntermediateTensorInfos.count(tensorName) > 0) {
               fGC += "fTensor_" + tensorName;
            } else {
               auto outputLength = ConvertDynamicShapeToLength(GetDynamicTensorShape(tensorName));
               if (IsDynamicTensor(tensorName) && eOutputType == ETensorType::BOOL) {
                  fGC += "std::vector<bool>(fTensor_" + tensorName + ".begin(), fTensor_" + tensorName + ".begin() + " +
                        outputLength + ");\n";
               } else {
                  fGC += "std::vector<" + outputType + ">(tensor_" + tensorName + ", tensor_" + tensorName + " + " +
                        outputLength + ")";
               }
            }
            if (i < outputSize - 1)
               fGC += ",";
         } else {
            fGC += "{}";
         }
      }
      fGC += "});\n";
      fGC += SP + "return ret;\n";
   }
   fGC += "}\n";
}

void RModel::Generate(std::underlying_type_t<Options> options, int batchSize, long pos, bool verbose) {
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
        throw
        std::runtime_error("TMVA-SOFIE: RModel::Generate: cannot use a separate weight file without generating a Session class");
    }

    if (static_cast<std::underlying_type_t<Options>>(Options::kGNN) & options)
        fIsGNN = true;
    if (static_cast<std::underlying_type_t<Options>>(Options::kGNNComponent) & options)
        fIsGNNComponent = true;

    Initialize(batchSize, verbose);
    std::string hgname;
    if(!fIsGNNComponent) {
        fGC.clear();
        GenerateHeaderInfo(hgname);
        if (fUseSession) {
            fGC += "struct Session {\n";
        }
    }

    GenerateInitializedTensorInfo();
    GenerateIntermediateTensorInfo();

    if (fUseSession) {
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
            fGC += "Session(std::string filename =\"" + fileName + "\"";
        } else {
            // no need to pass weight file since it is not used
            // keep passing a string for compatibility
            fGC += "Session(std::string = \"\"";
        }
        // add initialization of shape parameters
        // assume all parameters are of type size_t
        if (!fShapeParams.empty()) {
            for (auto & p : fShapeParams) {
               fGC += ",\n";
               fGC += "        size_t " + p.first + " = " + p.second;
            }
        }
        fGC += ") {\n";

        if (fUseWeightFile) {
            fGC += "\n//--- reading weights from file\n";
            ReadInitializedTensorsFromFile(pos);
            fGC += "\n";
            //fUseWeightFile = fUseWeightFile;
        }

        // now we have passed the parameters we can allocate the dynamic tensors
        GenerateDynamicTensorInfo();

        // add here initialization code  for operator
        for (size_t id = 0; id < fOperators.size() ; id++) {
            fGC += fOperators[id]->GenerateInitCode();
        }

        fGC += "}\n\n";
    }

    GenerateOutput();

    if(!fIsGNNComponent) {
        if (fUseSession) {
            fGC += "};\n";
        }
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

        fGC += "   std::string tensor_name;\n";
        fGC += "   size_t length;\n";

        // loop on tensors and parse the file
        for (auto& i: fInitializedTensors) {
             // skip Constant tensors
            std::cout << "initisalized tensor " << i.first << std::endl;
            if (i.second.IsConstantTensor()) continue;
            if (i.second.type() == ETensorType::FLOAT) {
                size_t length = 1;
                length = ConvertShapeToLength(i.second.shape());
                std::string tensor_name = "tensor_" + i.first;
                std::string slength = std::to_string(length);
                fGC += "   f >> tensor_name >> length;\n";
                fGC += "   if (tensor_name != \"" + tensor_name + "\" ) {\n";
                fGC += "      std::string err_msg = \"TMVA-SOFIE failed to read the correct tensor name; expected name is " +
                       tensor_name + " , read \" + tensor_name;\n";
                fGC += "      throw std::runtime_error(err_msg);\n";
                fGC += "    }\n";
                fGC += "   if (length != " + slength + ") {\n";
                fGC += "      std::string err_msg = \"TMVA-SOFIE failed to read the correct tensor size; expected size is " +
                       slength + " , read \" + std::to_string(length) ;\n";
                fGC += "      throw std::runtime_error(err_msg);\n";
                fGC += "    }\n";
                fGC += "   for (size_t i = 0; i < length; ++i)\n";
                fGC += "      f >> " + tensor_name + "[i];\n";
                fGC += "   if (f.fail()) {\n";
                fGC += "      throw std::runtime_error(\"TMVA-SOFIE failed to read the values for tensor " + tensor_name + "\");\n";
                fGC += "   }\n";
            }
        }
        fGC += "   f.close();\n";
    }

    // generate the code to read initialized tensors from a ROOT data file
    if(fWeightFile == WeightFileType::RootBinary) {
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
            }
            fGC += "  }\n";
        }
        fGC += "  }\n";
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
            // skip Constant tensors
            if (item.second.IsConstantTensor()) continue;
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
        }
        outputFile->Write(filename.c_str());

        // this needs to be changed, similar to the text file
        return -1;

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
            std::cout << "looking at tensor " << i.first << std::endl;
             // skip Constant tensors
            //if (i.second.IsConstantTensor()) continue;
            if (i.second.IsConstantTensor()) {
               std::cout << "skip constant tensor " << i.first << std::endl;
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
        std::cout << "Parametraised Tensor name: " << inputInfo.first << "\t";
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
        for(auto i=RModel::fInitializedTensors.begin(); i!=RModel::fInitializedTensors.end(); ++i) {
            i->second.CastPersistentToShared();
        }
    }
    else {
        for(auto i=RModel::fInitializedTensors.begin(); i!=RModel::fInitializedTensors.end(); ++i) {
            i->second.CastSharedToPersistent();
        }
        RModel::Class()->WriteBuffer(R__b, this);
    }
}

}//SOFIE
}//Experimental
}//TMVA
