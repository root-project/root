#ifndef TMVA_SOFIE_ROPERATOR_SELU
#define TMVA_SOFIE_ROPERATOR_SELU

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <iomanip>
#include <limits>
#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Selu final : public ROperator
{

private:
   float fAlpha = 1.67326319217681884765625;
   float fGamma = 1.05070102214813232421875;
   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;

public:
   ROperator_Selu(){}
   ROperator_Selu(float alpha, float gamma, std::string nameX, std::string nameY)
      : fAlpha(alpha), fGamma(gamma), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames = {fNX};
      fOutputTensorNames = {fNY};
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Selu Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()){
         throw std::runtime_error("TMVA SOFIE Operator Selu called to Generate without being initialized first");
      }
      std::stringstream out;
      int length = 1;
      for(auto& i: fShape){
         length *= i;
      }
      out << std::setprecision(std::numeric_limits<float>::max_digits10);
      out << "\t" << "float " << OpName << "_alpha = " << fAlpha << ";\n";
      out << "\t" << "float " << OpName << "_gamma = " << fGamma << ";\n";
      out << "\t" << "for (int id = 0; id < " << length << " ; id++){\n";
      out << "\t\t" << "tensor_" << fNY << "[id] = " << OpName << "_gamma * (std::max(float(0.0), tensor_" << fNX
          << "[id]) + std::min(float(0.0), " << OpName << "_alpha * (std::exp(tensor_" << fNX << "[id]) - 1)));\n";
      out << "\t}\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") };}
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_SELU
