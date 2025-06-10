#ifndef TMVA_SOFIE_ROPERATOR
#define TMVA_SOFIE_ROPERATOR

#include <vector>
#include <set>
#include <memory>

#include "TMVA/SOFIE_common.hxx"
//#include "RModel.hxx"



namespace TMVA{
namespace Experimental{
namespace SOFIE{

class RModel;

enum class OperatorKind {
   GEMM = 0,
   LAYERNORM = 1,
   RELU = 2,
   CONSTANT = 3,
   CONSTANTOFSHAPE = 4,
   UNDEFINED = 5,
   CONV=6,
   BATCHNORM=7
};

inline const char* toString(OperatorKind kind) {
   switch (kind) {
       case OperatorKind::GEMM:       return "GEMM";
       case OperatorKind::LAYERNORM:  return "LAYERNORM";
       case OperatorKind::RELU:       return "RELU";
       case OperatorKind::CONSTANT:       return "CONSTANT";
       case OperatorKind::CONSTANTOFSHAPE:       return "CONSTANTOFSHAPE";
              case OperatorKind::BATCHNORM:       return "batchnorm";  
                   case OperatorKind::CONV:       return "conv";
       case OperatorKind::UNDEFINED:  return "UNDEFINED";
       default:                       return "UNKNOWN";
   }
}
inline std::set<OperatorKind> FusableKinds = { OperatorKind::RELU, OperatorKind::LAYERNORM, OperatorKind::BATCHNORM};

class ROperator{


public:
   virtual std::vector<std::string> GetBlasRoutines() { return {}; }
   virtual std::vector<std::string> GetStdLibs() { return {}; }
   virtual std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>>) { return {}; };
   virtual std::vector<ETensorType> TypeInference(std::vector<ETensorType>) { return {}; };
   virtual void Initialize(RModel&) = 0;
   virtual std::string Generate(std::string OpName) = 0;  //expect unique opName for each operator within the same RModel
   // generate initialization code for session constructor
   virtual std::string GenerateInitCode() { return "";}
   // generate some specific declaration code for Session
   virtual std::string GenerateDeclCode() { return "";}
   // generate session data members specific to operator
   virtual std::string GenerateSessionMembersCode(std::string /*opName*/) { return ""; }
   virtual std::string Header() { return "";}
   virtual std::string GetFusableOutputTensorName() { return "";}
   virtual void UpdateFusableTensorName(std::string, const std::function<void(const std::string&)>& removal_func){ return;};


   //virtual void Forward_reference() = 0;
   //virtual void Forward_blas() = 0;
   virtual ~ROperator(){}

protected:
   OperatorKind fKind = OperatorKind::UNDEFINED;
   size_t fOpOrder = 0;
   const std::string SP = "   ";    ///< space used to correctly indent the generated C++ code
   bool fUseSession = false;        ///< flag to identify if using the session class
   bool fIsOutputConstant = false;  ///< flag to identify if operator has a constant output (no need to generate code)
   
   mutable std::vector<std::string> fInputTensorNames;
   mutable std::vector<std::string> fOutputTensorNames;

public:
   std::span<const std::string> GetOpInputTensors() const {
      return fInputTensorNames;
   }

   std::span<const std::string> GetOpOutputTensors() const {
      return fOutputTensorNames;
   }

   OperatorKind GetOpKind(){
            return fKind;
   }
   void RegisterOperatorOrder(const size_t ord){
      fOpOrder = ord;
   }
   size_t GetOpOrder(){
      return fOpOrder;
   }
    
};



}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_OPERATOR
