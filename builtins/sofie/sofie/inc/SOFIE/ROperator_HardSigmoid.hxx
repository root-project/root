#ifndef SOFIE_ROPERATOR_HARDSIGMOID
#define SOFIE_ROPERATOR_HARDSIGMOID

#include "SOFIE/common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>

namespace SOFIE {

template <typename T>
class ROperator_HardSigmoid final : public ROperator {

private:
   float fAlpha = 0.2;
   float fBeta = 0.5;
   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;
   std::string fType;

public:
   ROperator_HardSigmoid() {}
   ROperator_HardSigmoid(float alpha, float beta, std::string nameX, std::string nameY)
      : fAlpha(alpha), fBeta(beta), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
      if (std::is_same<T, float>::value) {
         fType = "float";
      } else {
         throw std::runtime_error("SOFIE Encountered unsupported type parsing a HardSigmoid operator");
      }

      fInputTensorNames = {fNX};
      fOutputTensorNames = {fNY};
   }

   void Initialize(RModel &model) override
   {
      if (model.CheckIfTensorAlreadyExist(fNX) == false) {
         throw std::runtime_error("SOFIE HardSigmoid Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }

   std::string Generate(std::string OpName) override
   {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Operator HardSigmoid called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShape);

      out << SP << "constexpr float " << OpName
          << "_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << fAlpha << ";\n";
      out << SP << "constexpr float " << OpName
          << "_beta = " << std::setprecision(std::numeric_limits<float>::max_digits10) << fBeta << ";\n";

      out << "\n//------ HardSigmoid\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = std::max(0.0f, std::min(1.0f, " << OpName << "_alpha * tensor_"
          << fNX << "[id] + " << OpName << "_beta));\n";
      out << SP << "}\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override { return {std::string("algorithm")}; }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_HARDSIGMOID
