#ifndef TMVA_SOFIE_ROPERATOR_THRESHOLDEDRELU
#define TMVA_SOFIE_ROPERATOR_THRESHOLDEDRELU

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_ThresholdedRelu final : public ROperator
{

private:
   float fAlpha = 1.0;
   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;

public:
   ROperator_ThresholdedRelu(){}
   ROperator_ThresholdedRelu(float alpha, std::string nameX, std::string nameY):
      fAlpha(alpha), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
      }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
         throw std::runtime_error("TMVA SOFIE ThresholdedRelu Op Input Tensor " + fNX + " is not found in model");
      }

      fShape = model.GetDimTensorShape(fNX);

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      if (model.Verbose()) {
         std::cout << "ThresholdedRelu : " << fNX << " -> " << fNY << " " << ConvertDimShapeToString(fShape) << std::endl;
      }
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator ThresholdedRelu called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDimShapeToLength(fShape);
      out << "\n//------ THRESHOLDED RELU\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = ((tensor_" << fNX << "[id] > " << fAlpha << "f )? tensor_" << fNX << "[id] : 0.0f);\n";
      out << SP << "}\n";
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_ROPERATOR_THRESHOLDEDRELU

