#ifndef SOFIE_ROPERATOR_Swish
#define SOFIE_ROPERATOR_Swish

#include "SOFIE/common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <iomanip>
#include <limits>
#include <sstream>

namespace SOFIE{

class ROperator_Swish final : public ROperator
{

private:
   float fAlpha = 1.0;
   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;

public:
   ROperator_Swish(){}
   ROperator_Swish(float alpha, std::string nameX, std::string nameY)
      : fAlpha(alpha), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames = {fNX};
      fOutputTensorNames = {fNY};
   }
   ROperator_Swish(std::string nameX, std::string nameY) : ROperator_Swish(1.0f, nameX, nameY) {}

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("SOFIE Swish Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()){
         throw std::runtime_error("SOFIE Operator Swish called to Generate without being initialized first");
      }
      std::stringstream out;
      int length = 1;
      for(auto& i: fShape){
         length *= i;
      }
      out << std::setprecision(std::numeric_limits<float>::max_digits10);
      out << "\t" << "float " << OpName << "_alpha = " << fAlpha << ";\n";
      out << "\t" << "for (int id = 0; id < " << length << " ; id++){\n";
      out << "\t\t" << "tensor_" << fNY << "[id] = tensor_" << fNX << "[id] / (1.0f + std::exp(- " << OpName
          << "_alpha * tensor_" << fNX << "[id]));\n";
      out << "\t}\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") };}
};

}//SOFIE


#endif //SOFIE_ROPERATOR_Swish
