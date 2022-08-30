#ifndef TMVA_SOFIE_RMODEL
#define TMVA_SOFIE_RMODEL

#include "TMVA/RModel_Base.hxx"
#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TBuffer.h"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

<<<<<<< HEAD
enum class Options {
   kDefault = 0x0,
   kNoSession = 0x1,
   kNoWeightFile = 0x2,
   kRootBinaryWeightFile = 0x4
   kGNN = 0x8,
   kGNNComponent = 0x16,
};

std::underlying_type_t<Options> operator|(Options opA, Options opB);
std::underlying_type_t<Options> operator|(std::underlying_type_t<Options> opA, Options opB);

enum class WeightFileType {None, RootBinary, Text};

class RModel: public TObject{
=======
class RModel: public RModel_Base{
>>>>>>> 3b618fbcad (feat: modify RModel architecture)

private:
   WeightFileType fWeightFile = WeightFileType::Text;

   std::unordered_map<std::string, InputTensorInfo> fInputTensorInfos; //graph input only; not including operator input (intermediate tensors)
   std::unordered_map<std::string, TensorInfo> fReadyInputTensorInfos;
   std::unordered_map<std::string, InitializedTensor> fInitializedTensors;
   std::unordered_map<std::string, TensorInfo> fIntermediateTensorInfos;
   std::vector<std::string> fOutputTensorNames;
   std::vector<std::string> fInputTensorNames;  //input tensor names using ONNX order

   std::vector<std::unique_ptr<ROperator>> fOperators;

<<<<<<< HEAD
   std::string fFileName; //file name of original model file for identification
   std::string fParseTime; //UTC date and time string at parsing

   std::unordered_set<std::string> fNeededBlasRoutines;

   const std::unordered_set<std::string> fAllowedStdLib = {"vector", "algorithm", "cmath"};
   std::unordered_set<std::string> fNeededStdLib = {"vector"};
   std::unordered_set<std::string> fCustomOpHeaders;

protected:
   std::string fName="UnnamedModel";
   std::string fGC; //generated code
   bool fUseWeightFile = true;
   bool fUseSession = true;
   bool fIsGNN = false;
   bool fIsGNNComponent = false;

=======
>>>>>>> d8ae68bedf (feat: modify RModel architecture)
public:

   //explicit move ctor/assn
   RModel(RModel&& other);

   RModel& operator=(RModel&& other);

   //disallow copy
   RModel(const RModel& other) = delete;
   RModel& operator=(const RModel& other) = delete;

   RModel(){}
   RModel(std::string name, std::string parsedtime): RModel_Base(name, parsedtime){}
   
   // For GNN Functions usage
   RModel(std::string function_name):RModel_Base(function_name){}

   const std::vector<size_t>& GetTensorShape(std::string name);
   const ETensorType& GetTensorType(std::string name);

   bool CheckIfTensorAlreadyExist(std::string tensor_name);
   void AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<Dim> shape);
   void AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<size_t> shape);
   void AddOperator(std::unique_ptr<ROperator> op, int order_execution = -1);
   void AddInitializedTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape, std::shared_ptr<void> data);
   // Check if a tensor is initialized
   bool IsInitializedTensor(const std::string& name) const;
   void AddIntermediateTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape);
   
   void AddNeededCustomHeader(std::string filename) {
      fCustomOpHeaders.insert(filename);
   }
   
   void AddInputTensorName(std::string name);
   void AddOutputTensorNameList(std::vector<std::string> outputtensornames);
   void UpdateOutputTensorList(std::vector<std::string> curr_output_tensor, std::vector<std::string> modify_output_tensor);
   void UpdateInitializedTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape, std::shared_ptr<void> data);
   std::shared_ptr<void> GetInitializedTensorData(std::string tensor_name);


   void Initialize(int batchSize=1);
   void GenerateInitializedTensorInfo();
   void GenerateIntermediateTensorInfo();
   void GenerateOutput();
   void Generate(std::underlying_type_t<Options> options, int batchSize = 1, long pos = 0);
   void Generate(Options options = Options::kDefault, int batchSitmva/sofie/inc/TMVA/RModel.hxx
ze = 1, int pos = 0) {
      Generate(static_cast<std::underlying_type_t<Options>>(options), batchSize, pos);
   }

   void ReadInitializedTensorsFromFile(long);
   long WriteInitializedTensorsToFile(std::string filename = "");

<<<<<<< HEAD
   void PrintGenerated() {
      std::cout << fGC;
   }

   std::string ReturnGenerated(){
      return fGC;
   }
=======
>>>>>>> 3b618fbcad (feat: modify RModel architecture)
   void PrintIntermediateTensors();
   void OutputGenerated(std::string filename = "", bool append = false);
   std::vector<std::string> GetOutputTensorNames(){
      return fOutputTensorNames;
   void SetFilename(std::string filename){
      fName = filename;
   }

/*
   template <typename T>
   void AddInitializedTensor(std::string tensor_name, RTensor<T> new_tensor){
      //a view only
      T obj;
      if (fInitializedTensors.find(tensor_name) != fInitializedTensors.end()){
         throw std::runtime_error("TMVA-SOFIE: initialized tensor with name " + tensor_name + " already exists \n");
      }
      InitializedTensor new_tensor_ {GetTemplatedType(obj), new_tensor.GetShape() , static_cast<void>(new_tensor.GetData())};
      fInitializedTensors[tensor_name] = new_tensor_;
   }
*/

   void PrintRequiredInputTensors();
   void PrintInitializedTensors();
   void HeadInitializedTensors(std::string name, int n_print = 50);

   bool UseSession() const { return fUseSession;}

   ~RModel() {}

   ClassDef(RModel,1);
};

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RMODEL
