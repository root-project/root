#ifndef TMVA_SOFIE_ROPERATOR_Custom
#define TMVA_SOFIE_ROPERATOR_Custom


#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{


template<typename T>
class ROperator_Custom final : public ROperator
{

private:
    std::string fOpName;
    std::vector<std::vector<std::size_t>> fOutputShapes;
    std::string fHeaderName;

public:
    ROperator_Custom(){}
    ROperator_Custom(std::string OpName, std::vector<std::string>Inputs, std::vector<std::string>Outputs, std::vector<std::vector<std::size_t>> OutputShapes, std::string HeaderName){
        fOpName = OpName;
        fOutputShapes = OutputShapes;
        fHeaderName = HeaderName;
        for(auto& it:Inputs){
            fInputTensorNames.emplace_back(UTILITY::Clean_name(it));
        }
        for(auto& it:Outputs){
            fOutputTensorNames.emplace_back(UTILITY::Clean_name(it));
        }
    }

    std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>>) {return {{}};};
    std::vector<ETensorType> TypeInference(std::vector<ETensorType>){ return {};};

   void Initialize(RModel& model){
      model.AddNeededCustomHeader(fHeaderName);
      for(auto& it:fInputTensorNames){
        if (model.CheckIfTensorAlreadyExist(std::string(it)) == false){
         throw std::runtime_error("TMVA SOFIE Custom " + fOpName + " Op Input Tensor " + std::string(it) + " is not found in model");
        }
      }

      if(fOutputTensorNames.size() != fOutputShapes.size()){
        throw std::runtime_error("TMVA SOFIE Custom "+ fOpName + " Op was not intialized with the names/shapes of all the output tensors");
      }

      for(long unsigned int i=0; i<fOutputTensorNames.size(); ++i){
        model.AddIntermediateTensor(std::string(fInputTensorNames[i]), ETensorType::FLOAT, fOutputShapes[i]);
      }

      auto convertToStringVec = [](const std::vector<std::string_view>& vec) {
          return std::vector<std::string>(vec.begin(), vec.end());
      };

      model.UpdateOutputTensorList(convertToStringVec(fInputTensorNames), convertToStringVec(fOutputTensorNames));

      if (model.Verbose()) {
         std::cout << "Custom operator using " << fHeaderName;
         for (auto & i : fInputTensorNames) std::cout << " " << i;
         std::cout << " ---> ";
         for (auto & i : fOutputTensorNames) std::cout << " " << i;
         std::cout << "\n";
      }
   }

    std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      std::stringstream out;
      out << "\n//------ "<<fOpName<<" \n";
      std::string args;
      for(long unsigned int i = 0; i<fInputTensorNames.size(); ++i){
        args+="fTensor_"+std::string(fInputTensorNames[i])+",";
      }

      for(long unsigned int i = 0; i<fOutputTensorNames.size(); ++i){
        args+="fTensor_"+std::string(fOutputTensorNames[i])+",";
      }
      args.pop_back();
      out << SP << fOpName<<"::Compute("+args+");\n";
      return out.str();
   }

};


}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Custom
