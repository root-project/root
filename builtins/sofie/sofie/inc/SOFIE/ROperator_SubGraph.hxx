#ifndef SOFIE_ROPERATOR_SubGraph
#define SOFIE_ROPERATOR_SubGraph

#include "SOFIE/common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>

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
         fOutputTensorNames.resize(fNYs.size());
         std::transform(fNYs.begin(), fNYs.end(), fOutputTensorNames.begin(),
                   [](const std::string& s) -> std::string_view { return s; });
      }

   void Initialize(RModel& model) override {
       //input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
        throw std::runtime_error("SOFIE If Op Input Tensor is not found in model");
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
               throw std::runtime_error("SOFIE If Op supports only all outputs of the same type");
         }
         model.AddIntermediateTensor(fNYs[i], fType, shape );
      }

   }


   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fType == ETensorType::UNDEFINED) {
         throw std::runtime_error("SOFIE If operator called to Generate without being initialized first");
      }
      std::stringstream out;
      out << "\n//------ If operator\n";
      // the subgraphs write directly into the output tensors of the If operator
      std::string outputs;
      for (size_t i = 0; i < fNYs.size(); i++) {
         if (i > 0)
            outputs += ", ";
         outputs += "tensor_" + fNYs[i];
      }
      auto callBranch = [&](const std::shared_ptr<RModel> &model, const std::string &signature) {
         return "doInfer(session.fSession_" + model->GetName() + ", " + signature + ", " + outputs + ");\n";
      };
      // the condition is a boolean tensor, only its first element is used
      out << SP << "if (tensor_" << fNX << "[0]) {\n";
      out << SP << SP << callBranch(fModel_then, fInputSignature_modelThen);
      out << SP << "} else {\n";
      out << SP << SP << callBranch(fModel_else, fInputSignature_modelElse);
      out << SP << "}\n";
      return out.str();
   }



};

}//SOFIE


#endif //SOFIE_ROPERATOR_Tanh
