#include <limits>
#include <algorithm>
#include <cctype>

#include "TMVA/RModel.hxx"
#include "TMVA/SOFIE_common.hxx"




namespace TMVA{
namespace Experimental{
namespace SOFIE{

   std::underlying_type_t<Options> operator|(Options opA, Options opB) {
      return static_cast<std::underlying_type_t<Options>>(opA) | static_cast<std::underlying_type_t<Options>>(opB);
   }
   std::underlying_type_t<Options> operator|(std::underlying_type_t<Options> opA, Options opB) {
      return opA | static_cast<std::underlying_type_t<Options>>(opB);
   }

   RModel::RModel(RModel&& other){
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

   RModel& RModel::operator=(RModel&& other){
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

   RModel::RModel(std::string name, std::string parsedtime): fFileName (name), fParseTime(parsedtime) {
      fName = fFileName.substr(0, fFileName.rfind("."));
   }

   const std::vector<size_t>& RModel::GetTensorShape(std::string name){
      auto f = fReadyInputTensorInfos.find(name);
      if (f != fReadyInputTensorInfos.end()){
         return f->second.shape;
      }
      auto f2 = fInitializedTensors.find(name);
      if (f2 != fInitializedTensors.end()){
         return f2->second.fShape;
      }
      auto f3 = fInputTensorInfos.find(name);
      if (f3 != fInputTensorInfos.end()){
         throw std::runtime_error("TMVA SOFIE tensor [" + name + "] is an input tensor with unspecified dimension parameter");
      }
      auto f4 = fIntermediateTensorInfos.find(name);
      if (f4 != fIntermediateTensorInfos.end()){
         return f4->second.shape;
      }

      throw std::runtime_error("TMVA SOFIE tensor [" + name + "] for which the shape is requested is not found");
   }

   const ETensorType& RModel::GetTensorType(std::string name){
      auto f = fReadyInputTensorInfos.find(name);
      if (f != fReadyInputTensorInfos.end()){
         return f->second.type;
      }
      auto f2 = fInitializedTensors.find(name);
      if (f2 != fInitializedTensors.end()){
         return f2->second.fType;
      }
      auto f3 = fInputTensorInfos.find(name);
      if (f3 != fInputTensorInfos.end()){
         return f3->second.type;
      }
      auto f4 = fIntermediateTensorInfos.find(name);
      if (f4 != fIntermediateTensorInfos.end()){
         return f4->second.type;
      }

      throw std::runtime_error("TMVA SOFIE tensor [" + name + "] for which the type is requested is not found");
   }

   bool RModel::CheckIfTensorAlreadyExist(std::string tensor_name){
      if (fReadyInputTensorInfos.find(tensor_name) != fReadyInputTensorInfos.end())  return true;
      if (fInitializedTensors.find(tensor_name) != fInitializedTensors.end()) return true;
      if (fIntermediateTensorInfos.find(tensor_name) != fIntermediateTensorInfos.end()) return true;
      return false;
   }

   void RModel::AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<Dim> shape){
      input_name = UTILITY::Clean_name(input_name);
      if (CheckIfTensorAlreadyExist(input_name)){
         throw std::runtime_error("TMVA-SOFIE: input tensor with name " + input_name + " already exists \n");
      }

      InputTensorInfo inputInfo { type, shape };
      fInputTensorInfos[input_name] = inputInfo;
   }

   void RModel::AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<size_t> shape){
      input_name = UTILITY::Clean_name(input_name);
      if (CheckIfTensorAlreadyExist(input_name)){
         throw std::runtime_error("TMVA-SOFIE: input tensor with name " + input_name + " already exists \n");
      }
      TensorInfo inputInfo { type, shape };
      fReadyInputTensorInfos[input_name] = inputInfo;
   }

   void RModel::AddInputTensorName(std::string input_name) {
       fInputTensorNames.push_back(UTILITY::Clean_name(input_name));
   }

   void RModel::AddOperator(std::unique_ptr<ROperator> op, int order_execution){
      AddBlasRoutines(op->GetBlasRoutines());
      auto libs = op->GetStdLibs();
      for (auto& stdlib : libs) {
         AddNeededStdLib(stdlib);
      }
      if (order_execution >= 0) {
         fOperators.insert(fOperators.begin() + order_execution, std::move(op));
      }else{
         fOperators.push_back(std::move(op));
      }
   }

   void RModel::AddInitializedTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape, std::shared_ptr<void> data){
      tensor_name = UTILITY::Clean_name(tensor_name);
      //NB: own data
      if (CheckIfTensorAlreadyExist(tensor_name)){
         throw std::runtime_error("TMVA-SOFIE: initialized tensor with name " + tensor_name + " already exists \n");
      }
      InitializedTensor new_tensor {type, shape, data};
      fInitializedTensors[tensor_name] = new_tensor;

   }

   bool RModel::IsInitializedTensor(const std::string& tensorName) const {
      std::string name = UTILITY::Clean_name(tensorName);
      return fInitializedTensors.find(name) != fInitializedTensors.end();
   }

   void RModel::AddIntermediateTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape){
      tensor_name = UTILITY::Clean_name(tensor_name);
      if (CheckIfTensorAlreadyExist(tensor_name)){
         throw std::runtime_error("TMVA-SOFIE: intermediate tensor with name " + tensor_name + " already exists \n");
      }
      TensorInfo new_tensor {type, shape};
      fIntermediateTensorInfos[tensor_name] = new_tensor;
   }

   void RModel::AddOutputTensorNameList(std::vector<std::string> outputtensornames){
      for(auto& it : outputtensornames){
         fOutputTensorNames.push_back(UTILITY::Clean_name(it));
      }
   }

   void RModel::UpdateOutputTensorList(std::vector<std::string> curr_output_tensors, std::vector<std::string> new_output_tensors){
      for(auto& it:curr_output_tensors){
         fOutputTensorNames.erase(std::remove(fOutputTensorNames.begin(), fOutputTensorNames.end(), it), fOutputTensorNames.end());
      }
      fOutputTensorNames.insert(fOutputTensorNames.end(), new_output_tensors.begin(), new_output_tensors.end());
   }

   void RModel::UpdateInitializedTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape, std::shared_ptr<void> data){
      tensor_name = UTILITY::Clean_name(tensor_name);
      if (!CheckIfTensorAlreadyExist(tensor_name)){
         throw std::runtime_error("TMVA-SOFIE: tensor " + tensor_name + " not found when trying to update it");
      }
      InitializedTensor new_tensor {type, shape, data};
      fInitializedTensors[tensor_name] = new_tensor;
   }

   std::shared_ptr<void> RModel::GetInitializedTensorData(std::string tensor_name){
      auto f = fInitializedTensors.find(tensor_name);
      if (f == fInitializedTensors.end()){
         throw std::runtime_error("TMVA-SOFIE: tensor " + tensor_name + " not found when trying to get its data");
      }else{
         return f->second.fData;
      }
   }

   void RModel::Initialize(int batchSize){
      // check if there are only parametrized input tensor and convert in
      // ready input tensor according to batch size
      // convert parametric shape to a dimensional shape
      fIntermediateTensorInfos.clear();
      if (fReadyInputTensorInfos.size() != fInputTensorNames.size()) {
         if ( fReadyInputTensorInfos.size() + fInputTensorInfos.size() != fInputTensorNames.size())
            throw std::runtime_error("TMVA-SOFIE: RModel::Initializes: invalid inputs");
         for (auto & input : fInputTensorInfos) {
            std::vector<size_t> shape;
            shape.reserve(input.second.shape.size());
            for (auto & d : input.second.shape){
               if (d.isParam)
                  shape.push_back(batchSize);
               else
                  shape.push_back(d.dim);
            }
            AddInputTensorInfo(input.first, input.second.type, shape);
         }
      }
      // check if there are initialized tensors to write in a weight file
      // support for the time being only wheight of FLOAT type
      if (fUseWeightFile) {
         bool modelHasWeights = false;
         for (auto& i: fInitializedTensors){
            if (i.second.fType == ETensorType::FLOAT) {
               modelHasWeights = true;
               break;
            }
         }
         if (!modelHasWeights) fUseWeightFile = false;
      }


      for (auto& i : fOperators){
         //std::cout << "initialize operator  " << typeid(*i).name() << std::endl;
         i->Initialize(*this);
      }
   }

   void RModel::Generate(std::underlying_type_t<Options> options, int batchSize){
      // session flag is used in operator initialize
      if (static_cast<std::underlying_type_t<Options>>(Options::kNoSession) & options)
         fUseSession = false;
      if (static_cast<std::underlying_type_t<Options>>(Options::kNoWeightFile) & options)
         fUseWeightFile = false;
      if (fUseWeightFile && !fUseSession) {
         throw std::runtime_error("TMVA-SOFIE: RModel::Generate: cannot use a separate weight file without generating a Session class");
      }
      fGC.clear();
      Initialize(batchSize);
      fGC += ("//Code generated automatically by TMVA for Inference of Model file [" + fFileName + "] at [" + fParseTime.substr(0, fParseTime.length()-1) +"] \n");
      // add header guards
      std::string hgname = fName;
      std::transform(hgname.begin(), hgname.end(), hgname.begin(), [](unsigned char c){ return std::toupper(c);} );
      hgname = "TMVA_SOFIE_" + hgname;
      fGC += "\n#ifndef " + hgname + "\n";
      fGC += "#define " + hgname + "\n\n";
      for (auto& i: fNeededStdLib) {
         fGC += "#include<" + i + ">\n";
      }
      for (auto& i: fCustomOpHeaders) {
         fGC += "#include \"" + i + "\"\n";
      }
      // for the session we need to include SOFIE_Common functions
      //needed for convolution operator (need to add a flag)
      fGC += "#include \"TMVA/SOFIE_common.hxx\"\n";
      if (fUseWeightFile)
         fGC += "#include <fstream>\n";

      fGC += "\nnamespace TMVA_SOFIE_" + fName + "{\n";
      if (!fNeededBlasRoutines.empty()) {
         fGC += ("namespace BLAS{\n");
         for (auto &routine : fNeededBlasRoutines) {
            if (routine == "Gemm") {
               fGC += ("\textern \"C\" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,\n"
                       "\t                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,\n"
                       "\t                       const float * beta, float * C, const int * ldc);\n");
            } else if (routine == "Gemv") {
               fGC += ("\textern \"C\" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,\n"
                       "\t                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);\n");
            } else if (routine == "Axpy") {
               fGC += ("\textern \"C\" void saxpy_(const int * n, const float * alpha, const float * x,\n"
                       "\t                         const int * incx, float * y, const int * incy);\n");
            } else if (routine == "Copy") {
               fGC += ("\textern \"C\" void scopy_(const int *n, const float* x, const int *incx, float* y, const int* incy);\n");
            }
         }
         fGC += ("}//BLAS\n");
      }
      if (fUseSession) {
         fGC += "struct Session {\n";
      }
      for (auto& i: fInitializedTensors){
         if (i.second.fType == ETensorType::FLOAT){
            size_t length = 1;
            for (auto & dim: i.second.fShape){
               length *= dim;
            }
            if (!fUseWeightFile) {
               fGC += "float tensor_" + i.first + "[" + std::to_string(length) + "] = {";
               std::shared_ptr<float> data = std::static_pointer_cast<float>(i.second.fData);
               std::stringstream floats;
               for (size_t idx = 0; idx < length-1; idx++){
                  floats << std::setprecision(std::numeric_limits<float>::max_digits10) << data.get()[idx] << ", ";
               }
               floats << std::setprecision(std::numeric_limits<float>::max_digits10) << data.get()[length-1];
               fGC += floats.str();
               fGC += "};\n";
            }
            else {
               fGC += "std::vector<float> fTensor_" + i.first + " = std::vector<float>(" + std::to_string(length) + ");\n";
               fGC += "float * tensor_" + i.first + " = fTensor_" + i.first + ".data();\n";
            }

         }
      }
      for (auto&i: fIntermediateTensorInfos){
         size_t length = ConvertShapeToLength(i.second.shape);
         if (i.second.type == ETensorType::FLOAT){
            fGC += "std::vector<float> fTensor_" + i.first  + " = std::vector<float>(" + std::to_string(length) + ");\n";
            fGC += "float * tensor_" + i.first + " = fTensor_" + i.first  + ".data();\n";
         }
         if (i.second.type == ETensorType::DOUBLE){
            fGC += "std::vector<double> fTensor_" + i.first  + " = std::vector<double>(" + std::to_string(length) + ");\n";
            fGC += "double * tensor_" + i.first + " = fTensor_" + i.first  + ".data();\n";
         }
         if (i.second.type == ETensorType::INT64){
            fGC += "std::vector<int64_t> fTensor_" + i.first  + " = std::vector<int64_t>(" + std::to_string(length) + ");\n";
            fGC += "int64_t * tensor_" + i.first + " = fTensor_" + i.first  + ".data();\n";
         }
      }
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
            fGC += "Session(std::string filename =\"\") {\n";
            fGC += "   if (filename.empty()) filename = \"" + fName + ".dat\";\n";
            ReadInitializedTensorsFromFile();
            //fUseWeightFile = fUseWeightFile;
         } else {
            // no need to pass weight file since it is not used
            // keep passing a string for compatibility
            fGC += "Session(std::string = \"\") {\n";
         }
         // add here initialization code
         for (size_t id = 0; id < fOperators.size() ; id++){
            fGC += fOperators[id]->GenerateInitCode();
         }
         fGC += "}\n\n";
      }

      size_t outputSize = fOutputTensorNames.size();
      // assume output types are all the same
      std::string outputType;
      if (outputSize == 1) {
         auto f = fIntermediateTensorInfos.find(fOutputTensorNames[0]);
         if (f == fIntermediateTensorInfos.end()){
            throw std::runtime_error("TMVA-SOFIE: output tensor " + fOutputTensorNames[0] + " not found when trying to get its info");
         }else{
            outputType = ConvertTypeToString(f->second.type);
            fGC += "std::vector<" + outputType + "> ";
         }
      } else {
         std::vector<ETensorType> outputTensorsTypes(outputSize);
         for (size_t i = 0; i < outputSize; i++) {
            auto f = fIntermediateTensorInfos.find(fOutputTensorNames[i]);
            if (f == fIntermediateTensorInfos.end()) {
               throw std::runtime_error("TMVA-SOFIE: output tensor " + fOutputTensorNames[i]
                  + " not found when trying to get its info");
            } else {
               outputTensorsTypes[i] = f->second.type;
            }
         }
         // assume all output types are the same
         outputType = ConvertTypeToString(outputTensorsTypes[0]);
         for (size_t i = 0; i < outputSize; i++) {
            if (outputTensorsTypes[i] != outputTensorsTypes[0]) {
               throw std::runtime_error("TMVA-SOFIE: output tensor " + fOutputTensorNames[i] + " is of different type.");
            }
         }
         fGC += "std::vector<std::vector<" + outputType + ">> ";
      }

      fGC += "infer(";
      for(size_t i = 0; i<fInputTensorNames.size(); ++i){
         switch((fReadyInputTensorInfos[fInputTensorNames[i]]).type){
            case  ETensorType::FLOAT :{
               fGC += "float* tensor_" + fInputTensorNames[i] + ",";
               break;
            }
            case  ETensorType::INT32 :{
               fGC += "int32_t* tensor_" + fInputTensorNames[i] + ",";
               break;
            }
            case  ETensorType::INT64 :{
               fGC += "int64_t* tensor_" + fInputTensorNames[i] + ",";
               break;
            }
            case  ETensorType::DOUBLE :{
               fGC += "double* tensor_" + fInputTensorNames[i] + ",";
               break;
            }
            default: {
               throw std::runtime_error("TMVA-SOFIE: input tensor " + fInputTensorNames[i] + " is of a data type which is not yet supported.");
            }
         }
      }
      fGC.pop_back(); //remove last ","
      fGC += "){\n";

      const std::string SP = "   ";

      for (size_t id = 0; id < fOperators.size() ; id++){
         fGC+= (fOperators[id]->Generate(std::to_string(id)));
      }
      if (outputSize == 1) {
         size_t outputLength = ConvertShapeToLength(GetTensorShape(fOutputTensorNames[0]));

         fGC += SP + "std::vector<" + outputType + "> ret (tensor_" + fOutputTensorNames[0] + ", tensor_" + fOutputTensorNames[0] + " + " +
               std::to_string(outputLength) + ");\n";
      } else {
         for (size_t i = 0; i < outputSize; i++) {
            if (!fOutputTensorNames[i].empty()) {
               size_t outputLength = ConvertShapeToLength(GetTensorShape(fOutputTensorNames[i]));
               fGC += SP + "std::vector<" + outputType + "> ret_";
               fGC += std::to_string(i);
               fGC += " (tensor_" + fOutputTensorNames[i] + ", tensor_" + fOutputTensorNames[i] + " + " +
               std::to_string(outputLength) + ");\n";
            }
         }
         fGC += SP + "std::vector<std::vector<" + outputType + ">> ret({";
         for (size_t i = 0; i < outputSize; i++) {
            if (fOutputTensorNames[i].empty()) {
               fGC += "{}";
            } else {
               fGC += "ret_";
               fGC += std::to_string(i);
            }
            if (i < outputSize - 1) {
               fGC += ",";
            }
         }
         fGC += "});\n";
      }
      fGC += SP + "return ret;\n";
      fGC += "}\n";
      if (fUseSession) {
         fGC += "};\n";
      }
      fGC += ("} //TMVA_SOFIE_" + fName + "\n");
      fGC += "\n#endif  // " + hgname + "\n";
   }

   void RModel::ReadInitializedTensorsFromFile() {
      // generate the code to read initialized tensors from a text data file
      if (fInitializedTensors.empty()) return;

      fGC += "   std::ifstream f;\n";
      fGC += "   f.open(filename);\n";
      fGC += "   if (!f.is_open()){\n";
      fGC += "      throw std::runtime_error(\"tmva-sofie failed to open file for input weights\");\n";
      fGC += "   }\n";
      fGC += "   std::string tensor_name;\n";
      fGC += "   int length;\n";

      //loop on tensors and parse the file
      for (auto& i: fInitializedTensors){
         if (i.second.fType == ETensorType::FLOAT){
            size_t length = 1;
            for (auto & dim: i.second.fShape){
               length *= dim;
            }
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
            fGC += "    for (int i =0; i < length; ++i) \n";
            fGC += "       f >> " + tensor_name + "[i];\n";
         }
      }
      fGC += "   f.close();\n";
   }

   void RModel::WriteInitializedTensorsToFile(std::string filename) {
      // write the initialized tensors in a text file
      if (filename == ""){
         filename = fName + ".data";
      }

      std::ofstream f;
      f.open(filename);
      if (!f.is_open()){
         throw std::runtime_error("tmva-sofie failed to open file for tensor weight data");
      }
      for (auto& i: fInitializedTensors){
         if (i.second.fType == ETensorType::FLOAT){
            size_t length = 1;
            for (auto &dim : i.second.fShape) {
               length *= dim;
            }
            std::string tensor_name = "tensor_" + i.first;
            f << tensor_name << " " << length << "\n";
            const float * data = (std::static_pointer_cast<float>(i.second.fData)).get();
            for (size_t idx = 0; idx < length - 1; idx++) {
               f << std::setprecision(std::numeric_limits<float>::max_digits10) << data[idx] << " ";
            }
            f << std::setprecision(std::numeric_limits<float>::max_digits10) << data[length - 1];
            f << "\n";
         }
      }
      f.close();
   }

   void RModel::PrintRequiredInputTensors(){
      std::cout << "Model requires following inputs:\n";
      for (auto& inputInfo: fInputTensorInfos){
         std::cout << "Parameterised Tensor name: " << inputInfo.first << "\t";
         std::cout << "type: " << ConvertTypeToString(inputInfo.second.type) << "\t";
         std::cout << "shape: [";
         for (size_t i = 0; i < inputInfo.second.shape.size(); i++){
            if (inputInfo.second.shape[i].isParam){
               std::cout << inputInfo.second.shape[i].param;
            }else{
               std::cout << inputInfo.second.shape[i].dim ;
            }
            if (i < inputInfo.second.shape.size() - 1) std::cout << ",";
         }
         std::cout << "]" << std::endl;
      }

      for (auto& inputInfo: fReadyInputTensorInfos){
         std::cout << "Fully Specified Tensor name: " << inputInfo.first << "\t";
         std::cout << "type: " << ConvertTypeToString(inputInfo.second.type) << "\t";
         std::cout << "shape: [";
         for (size_t i = 0; i < inputInfo.second.shape.size(); i++){
            std::cout << inputInfo.second.shape[i];
            if (i < inputInfo.second.shape.size() - 1) std::cout << ",";
         }
         std::cout << "]" << std::endl;
      }

   }

   void RModel::PrintInitializedTensors(){
      std::cout << "Model initialized the following tensors:\n";
      for (auto& it: fInitializedTensors){
         std::cout << "Tensor name: \"" << it.first << "\"\t";
         std::cout << "type: " << ConvertTypeToString(it.second.fType) << "\t";
         std::cout << "shape: [";
         for (size_t i = 0; i < it.second.fShape.size(); i++){
            std::cout << it.second.fShape[i];
            if (i < it.second.fShape.size() - 1) std::cout << ",";
         }
         std::cout << "]" << std::endl;
      }
   }

   void RModel::PrintIntermediateTensors(){
      std::cout << "Model specify the following intermediate tensors:\n";
      for (auto& it: fIntermediateTensorInfos){
         std::cout << "Tensor name: \"" << it.first << "\"\t";
         std::cout << "type: " << ConvertTypeToString(it.second.type) << "\t";
         std::cout << "shape: [";
         for (size_t i = 0; i < it.second.shape.size(); i++){
            std::cout << it.second.shape[i];
            if (i < it.second.shape.size() - 1) std::cout << ",";
         }
         std::cout << "]" << std::endl;
      }
   }

   void RModel::HeadInitializedTensors(std::string name, int n_print){
      auto it = fInitializedTensors.find(name);
      if (it == fInitializedTensors.end()){
         std::cout << "Tensor " << name << " not found in model's intialized tensor list" << std::endl;
         return;
      }

      std::cout << "Tensor name: " << it->first << "\t";
      std::cout << "type: " << ConvertTypeToString(it->second.fType) << "\t";
      int length =1;
      std::cout << "shape: [";
      for (size_t i = 0; i < it->second.fShape.size(); i++){
         std::cout << it->second.fShape[i];
         length *= it->second.fShape[i];
         if (i < it->second.fShape.size() - 1) std::cout << ",";
      }
      std::cout << "]" << std::endl;
      bool ellipsis = true;
      if (n_print > length){
         n_print = length;
         ellipsis = false;
      }

      std::cout << "data: [" << std::endl;
      //switch(it->second.type){
      //   case ETensorType::FLOAT : {
      if (it->second.fType == ETensorType::FLOAT) {
         auto converted_data = std::static_pointer_cast<float>(it->second.fData).get();
         for (int i =0; i < n_print; i++){
            std::cout << converted_data[i];
            if (i < n_print - 1) std::cout << " ,";
         }
         //   break;
         // }
      }
      if (ellipsis) std::cout << ", ...";
      std::cout << "]" << std::endl;

   }

   void RModel::OutputGenerated(std::string filename){
      if (filename == ""){
         filename = fName + ".hxx";
      }
      std::ofstream f;
      f.open(filename);
      if (!f.is_open()){
         throw std::runtime_error("tmva-sofie failed to open file for output generated inference code");
      }
      f << fGC;
      f.close();

      // write weights in a text file
      size_t pos = filename.find(".hxx");
      filename.replace(pos,4,".dat");
      if (fUseWeightFile) WriteInitializedTensorsToFile(filename);
   }

   void RModel::Streamer(TBuffer &R__b){
       if (R__b.IsReading()) {
           RModel::Class()->ReadBuffer(R__b, this);
           for(auto i=RModel::fInitializedTensors.begin(); i!=RModel::fInitializedTensors.end();++i){
               i->second.CastPersistentToShared();
           }
       }
       else {
          for(auto i=RModel::fInitializedTensors.begin(); i!=RModel::fInitializedTensors.end();++i){
               i->second.CastSharedToPersistent();
           }
          RModel::Class()->WriteBuffer(R__b, this);
       }
   }

}//SOFIE
}//Experimental
}//TMVA
