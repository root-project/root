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
    std::vector<std::string> fInputNames;
    std::vector<std::string> fOutputNames;
    std::vector<std::vector<std::size_t>> fOutputShapes;
    std::vector<std::size_t> fInputSizes;
    std::string fHeaderName;
    ETensorType fInputType;

public:
    ROperator_Custom(){}
    ROperator_Custom(std::string OpName, std::vector<std::string>Inputs, std::vector<std::string>Outputs, std::vector<std::vector<std::size_t>> OutputShapes, std::string HeaderName){
        fOpName = OpName;
        fOutputShapes = OutputShapes;
        fHeaderName = HeaderName;
        for(auto& it:Inputs){
            fInputNames.emplace_back(UTILITY::Clean_name(it));
            fInputTensorNames.emplace_back(fInputNames.back());
        }
        for(auto& it:Outputs){
            fOutputNames.emplace_back(UTILITY::Clean_name(it));
            fOutputTensorNames.emplace_back(fOutputNames.back());
        }
    }

    std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>>) {return {{}};};
    std::vector<ETensorType> TypeInference(std::vector<ETensorType>){ return {};};

   void Initialize(RModel& model) override {
      model.AddNeededCustomHeader(fHeaderName);
      fInputType = model.GetTensorType(fInputNames[0]);

      for(auto& it:fInputNames){
        if (model.CheckIfTensorAlreadyExist(it) == false){
         throw std::runtime_error("TMVA SOFIE Custom " + fOpName + " Op Input Tensor " + it + " is not found in model");
        }
        fInputSizes.push_back(ConvertShapeToLength(model.GetTensorShape(it)));
      }

      if(fOutputNames.size() != fOutputShapes.size()){
        throw std::runtime_error("TMVA SOFIE Custom "+ fOpName + " Op was not intialized with the names/shapes of all the output tensors");
      }

      for(long unsigned int i=0; i<fOutputNames.size(); ++i){
        model.AddIntermediateTensor(std::string(fOutputNames[i]), ETensorType::FLOAT, fOutputShapes[i]);
      }


      model.UpdateOutputTensorList(fInputNames, fOutputNames);

      if (model.Verbose()) {
         std::cout << "Custom operator using " << fHeaderName;
         for (auto & i : fInputNames) std::cout << " " << i;
         std::cout << " ---> ";
         for (auto & i : fOutputNames) std::cout << " " << i;
         std::cout << "\n";
      }
      model.AddNeededCustomHeader("ROOT/RSpan.hxx");
   }

    std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      std::stringstream out;
      out << "\n//------ "<<fOpName<<" \n";
      std::string args;
      for(long unsigned int i = 0; i<fInputNames.size(); ++i){
        args+="std::span<const "+ConvertTypeToString(fInputType)+">(tensor_"+std::string(fInputNames[i])+", "+fInputSizes[i]+"),";
      }

      for(long unsigned int i = 0; i<fOutputNames.size(); ++i){
        args+="std::span<"+TensorType<T>::Name()+">(tensor_"+std::string(fOutputNames[i])+", "+ConvertShapeToLength(fOutputShapes[i])+"),";
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
