#ifndef SOFIE_ROPERATOR_THRESHOLDEDRELU
#define SOFIE_ROPERATOR_THRESHOLDEDRELU

#include "SOFIE/common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <iomanip>
#include <limits>
#include <sstream>

namespace SOFIE {

template <typename T>
class ROperator_ThresholdedRelu final : public ROperator {

private:
   float fAlpha = 1.0;
   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;

public:
   ROperator_ThresholdedRelu() {}
   ROperator_ThresholdedRelu(float alpha, std::string nameX, std::string nameY)
      : fAlpha(alpha), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames = {fNX};
      fOutputTensorNames = {fNY};
   }

   void Initialize(RModel &model) override
   {
      if (model.CheckIfTensorAlreadyExist(fNX) == false) {
         throw std::runtime_error("SOFIE ThresholdedRelu Op Input Tensor " + fNX + " is not found in model");
      }

      fShape = model.GetDimTensorShape(fNX);

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      if (model.Verbose()) {
         std::cout << "ThresholdedRelu : " << fNX << " -> " << fNY << " " << ConvertDimShapeToString(fShape)
                   << std::endl;
      }
   }

   std::string Generate(std::string OpName) override
   {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error(
            "SOFIE Operator ThresholdedRelu called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDimShapeToLength(fShape);

      out << SP << "constexpr float " << OpName
          << "_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << fAlpha << ";\n";

      out << "\n//------ THRESHOLDED RELU\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = ((tensor_" << fNX << "[id] > " << OpName << "_alpha) ? tensor_"
          << fNX << "[id] : 0.0f);\n";
      out << SP << "}\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_THRESHOLDEDRELU
