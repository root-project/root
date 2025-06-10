#ifndef TMVA_SOFIE_ROPERATOR_SubGraph
#define TMVA_SOFIE_ROPERATOR_SubGraph

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

   // operator dealing with subgraphs (such as If , Loop, etc..)

class ROperator_If final : public ROperator
{

private:

   std::string fNX;
   ETensorType fType = ETensorType::UNDEFINED;  // output type (support only one common type)
   std::vector<std::string> fNYs;
   std::shared_ptr<RModel> fModel_then;
   std::shared_ptr<RModel> fModel_else;
   std::string fInputSignature_modelThen;
   std::string fInputSignature_modelElse;

public:
   ROperator_If(){}
   ROperator_If(const std::string & nameX, const std::vector<std::string> & nameYs, std::unique_ptr<RModel> model_then, std::unique_ptr<RModel> model_else):
      fNX(UTILITY::Clean_name(nameX)), fNYs(nameYs), fModel_then(std::move(model_then)), fModel_else(std::move(model_else))
      {
         for (auto & n : fNYs)
            n = UTILITY::Clean_name(n);

         fInputTensorNames = { fNX };
         std::transform(fNYs.begin(), fNYs.end(), fOutputTensorNames.begin(),
                   [](const std::string& s) -> std::string { return s; });
      }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model) override {
       //input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
        throw std::runtime_error("TMVA SOFIE If Op Input Tensor is not found in model");
      }
      //add the subgraph model to parent RModel and initialize them
      model.InitializeSubGraph(fModel_then);
      model.InitializeSubGraph(fModel_else);

      // generate input string signature for subgraphs
      fInputSignature_modelThen = fModel_then->GenerateInferSignature(false);
      fInputSignature_modelElse = fModel_else->GenerateInferSignature(false);

      // add the outputs
      for (size_t i = 0; i < fNYs.size(); i++) {
         // assume shape of then tensor is same of else tensor
         // if not need to make a parametric tensor output (tbd)
         auto soutput_name = fModel_then->GetOutputTensorNames()[i];
         auto shape = fModel_then->GetTensorShape(soutput_name);
         auto type = fModel_then->GetTensorType(soutput_name);
         if (i == 0)
            fType = type;
         else {
            if (type != fType)
               throw std::runtime_error("TMVA SOFIE If Op supports only all outputs of the same type");
         }
         model.AddIntermediateTensor(fNYs[i], fType, shape );
      }

   }


   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fType == ETensorType::UNDEFINED) {
         throw std::runtime_error("TMVA If operator called to Generate without being initialized first");
      }
      std::stringstream out;
      //size_t length = ConvertShapeToLength(fShape);
      std::string typeName = ConvertTypeToString(fType);
      out << "\n//------ If operator\n";
      out << SP << "std::vector<std::vector<" << typeName << ">> outputs_" << opName << ";\n";
      // use the std::vector since is a boolean
      out << SP << "if (fTensor_" << fNX << "[0] ) { \n";
      // then branch
      out << SP << SP << "outputs_" << opName << " = "
         << "fSession_" <<  fModel_then->GetName() << ".infer(" << fInputSignature_modelThen << ");\n";
       // else branch
      out << SP << "} else {\n";
      out << SP << SP << "outputs_" << opName << " = "
         << "fSession_" + fModel_else->GetName() + ".infer(" << fInputSignature_modelElse << ");\n";
      out << SP << "}\n";
      // copy the outputs
      out << SP << "if (outputs_" << opName << ".size() != " << fNYs.size() << ")\n";
      out << SP << SP << "throw std::runtime_error(\" If operator: invalid output size!\");\n\n";
      for (size_t i = 0; i < fNYs.size(); i++) {
         out << SP << "std::copy(outputs_" << opName << "[" << i << "].begin(), outputs_" << opName << "[" << i << "].end(), fTensor_" << fNYs[i] << ".begin());\n";
      }
      return out.str();
   }



};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Tanh
