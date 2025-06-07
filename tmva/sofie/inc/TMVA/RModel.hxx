#ifndef TMVA_SOFIE_RMODEL
#define TMVA_SOFIE_RMODEL

#include "TMVA/RModel_Base.hxx"
#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModel final : public RModel_Base {

private:
   bool fIsInitialized = false;
   bool fIsSubGraph = false;
   int fVerbose = 0;
   int fBatchSize = -1;
   long fReadPos = 0;  // reading file position

   OptimizationLevel fOptimizationLevel = OptimizationLevel::kExtended;

   std::unordered_map<std::string, InputTensorInfo> fInputTensorInfos; // input tensors where shape may not fully defined or other graph inputs?
   std::unordered_map<std::string, TensorInfo> fReadyInputTensorInfos; // input tensors where shape is full defined
   std::unordered_map<std::string, InitializedTensor> fInitializedTensors;
   std::unordered_map<std::string, TensorInfo> fIntermediateTensorInfos;
   std::unordered_map<std::string, DynamicTensorInfo> fDynamicTensorInfos;
   std::unordered_map<std::string, std::string>
      fShapeParams; // parameters defining the dynamic shape (e.g. batch size), store also its default value
   std::vector<std::string> fOutputTensorNames;
   std::vector<std::string> fInputTensorNames; // input tensor names using ONNX order

   std::vector<std::unique_ptr<ROperator>> fOperators;
   std::vector<std::unique_ptr<ROperator>> fConstantOperators;

   std::vector<std::shared_ptr<RModel>> fSubGraphs;    ///<!  sub-graph models (transient)
   RModel * fParentGraph = nullptr;

   // memory pool information for intermediate tensors
   MemoryPoolInfo fIntermediateMemoryInfo;    ///<!  intermediate memory info (transient)
   std::unordered_map<std::string_view, size_t> fIntermediateTensorFrequencyLookup;    ///<!  lookup table for intermediate tensor frequency (transient)

public:
   /**
       Default constructor. Needed to allow serialization of ROOT objects. See
       https://root.cern/manual/io_custom_classes/#restrictions-on-types-root-io-can-handle
   */
   RModel() = default;
   RModel(std::string name, std::string parsedtime) : RModel_Base(name, parsedtime) {}

   // For GNN Functions usage
   RModel(std::string function_name) : RModel_Base(function_name) {}

   int Verbose() const { return fVerbose;}

   const std::vector<size_t> &GetTensorShape(std::string name) const;
   std::vector<Dim> GetDynamicTensorShape(std::string name) const;
   const ETensorType &GetTensorType(std::string name) const;

   bool CheckIfTensorAlreadyExist(std::string tensor_name);
   void AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<Dim> shape);
   void AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<size_t> shape);
   void AddOperator(std::unique_ptr<ROperator> op, size_t order_execution = -1);
   void AddOperatorReference(ROperator *op, size_t order_execution = -1)
   {
      std::unique_ptr<ROperator> tmp(op);
      AddOperator(std::move(tmp), order_execution);
   }
   void AddConstantOperator(std::unique_ptr<ROperator> op);
   void AddInitializedTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape,
                             std::shared_ptr<void> data);
   void AddConstantTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape,
                             std::shared_ptr<void> data);

   template<class T>
   void AddConstantTensor(const std::string & name, const std::vector<size_t> & shape, const T * data) {
      size_t length = ConvertShapeToLength(shape);
      std::shared_ptr<void> data_ptr(malloc(length * sizeof(T)), free);
      std::memcpy(data_ptr.get(), (void*) data, length * sizeof(T));
      AddConstantTensor(name, GetTemplatedType<T>(T()), shape, data_ptr);
   }
   // for boolean can be more convenient passing an std::vector
   template<class T>
   void AddConstantTensor(const std::string & name, const std::vector<size_t> & shape, const std::vector<T> & data) {
      size_t length = data.size();
      std::shared_ptr<void> data_ptr(malloc(length * sizeof(T)), free);
      std::copy(data.begin(), data.end(), (T*) data_ptr.get());
      //std::memcpy(data_ptr.get(), (void*) data, length * sizeof(T));
      AddConstantTensor(name, GetTemplatedType<T>(T()), shape, data_ptr);
   }

   template <typename T>
   void AddInitializedTensor(const std::string & tensor_name, const std::vector<std::size_t> & shape, T *raw_data)
   {
      size_t size = ConvertShapeToLength(shape);
      std::shared_ptr<void> data(malloc(size * sizeof(T)), free);
      std::memcpy(data.get(), raw_data, size * sizeof(T));
      AddInitializedTensor(tensor_name,  GetTemplatedType(T()), shape, data);
   }

   // add and initialize subgraph to the model
   void InitializeSubGraph(std::shared_ptr<RModel>  graph);

   // set a flag to indicate tensor does not need to be written in a weight file
   // (e.g. shape tensors used as input to define a shape (in Reshape))
   void SetNotWritableInitializedTensor(const std::string & tensor_name);

   // Check if a tensor is initialized
   bool IsInitializedTensor(const std::string &name) const;
   // Check if a tensor is Constant (note a Constant tensor is also initialized)
   bool IsConstantTensor(const std::string &name) const;
   bool IsDynamicTensor(const std::string &name) const;
   // Check if tensor is a input dynamic tensor (without a specified shape, based on Sim structure
   bool IsDimInputTensor(const std::string &name) const;
   // check if tensor is a fully specified input tensor
   bool IsReadyInputTensor(const std::string &name) const;

   // Add intermediate tensor
   void AddIntermediateTensor(std::string tensor_name, ETensorType type, std::vector<Dim> dim_shape);
   void AddIntermediateTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape);
   // Add an intermediate dynamic tensor
   void AddDynamicTensor(std::string tensor_name, ETensorType type, std::vector<Dim> shape);

   void AddInputTensorName(std::string name);
   void AddOutputTensorNameList(std::vector<std::string> output_tensor_names);
   void
   UpdateOutputTensorList(std::vector<std::string> curr_output_tensor, std::vector<std::string> modify_output_tensor);
   void UpdateInitializedTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape,
                                std::shared_ptr<void> data);
   std::shared_ptr<void> GetInitializedTensorData(std::string tensor_name);

   void Initialize(int batchSize = -1, bool verbose = false);
   void Initialize(const std::map<std::string,size_t> & inputParams, bool verbose = false);

   void Generate(std::underlying_type_t<Options> options, int batchSize = -1, long pos = 0, bool verbose = false);
   void Generate(Options options = Options::kDefault, int batchSize = -1, int pos = 0, bool verbose = false)
   {
      Generate(static_cast<std::underlying_type_t<Options>>(options), batchSize, pos, verbose);
   }
   // generate the infer function signature. If isdecl= false generate the calling infer function
   // used to infer the sub-graphs
   std::string GenerateInferSignature(bool isdecl = true);

   // calculate total intermediate memory and position intermediate tensor addresses
   std::string AllocateIntermediateMemory(std::span<const std::string> op_output_tensors, std::set<std::string>& allocated_tensors);
   void CheckAndFlushIntermediateMemory(std::span<const std::string> op_output_tensors, const size_t& op_idx);

   void SetOptimizationLevel(const OptimizationLevel &optim_level) { fOptimizationLevel = optim_level; }

protected:
   // internal functions
   // generate code for the initialized tensors
   void GenerateInitializedTensorInfo();
   // generate code for the intermediate tensors
   void GenerateIntermediateTensorInfo();
   // generate code for the dynamic tensors
   void GenerateDynamicTensorInfo();
   // generate code for declarations needed by operators
   void GenerateOperatorDeclarations();
   // generate code for inference
   void GenerateOutput();
   // generate code for initializing memory pool for intermediate tensors
   void GenerateIntermediateMemoryPool();
   // Generate all session code
   void GenerateSessionCode();
   void CheckAndFuseOperators();

public:
   const std::vector<std::string> &GetInputTensorNames() const { return fInputTensorNames; }
   const std::vector<std::string> &GetOutputTensorNames() const { return fOutputTensorNames; }

   void ReadInitializedTensorsFromFile(long);
   long WriteInitializedTensorsToFile(std::string filename = "");

   void PrintIntermediateTensors();
   void PrintOutputTensors();
   void OutputGenerated(std::string filename = "", bool append = false);
   std::vector<std::string> GetOutputTensorNames() { return fOutputTensorNames; }
   void SetFilename(std::string filename) { fName = filename; }

   /*
      template <typename T>
      void AddInitializedTensor(std::string tensor_name, RTensor<T> new_tensor){
         //a view only
         T obj;
         if (fInitializedTensors.find(tensor_name) != fInitializedTensors.end()){
            throw std::runtime_error("TMVA-SOFIE: initialized tensor with name " + tensor_name + " already exists \n");
         }
         InitializedTensor new_tensor_ {GetTemplatedType(obj), new_tensor.GetShape() ,
      static_cast<void>(new_tensor.GetData())}; fInitializedTensors[tensor_name] = new_tensor_;
      }
   */

   void PrintRequiredInputTensors();
   void PrintInitializedTensors();
   void PrintDynamicTensors();
   void HeadInitializedTensors(std::string name, int n_print = 50);

   bool UseSession() const { return fUseSession; }

   // Use the ClassDef macro to allow definition of custom streaming
   ClassDefNV(RModel, 3);
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_RMODEL
