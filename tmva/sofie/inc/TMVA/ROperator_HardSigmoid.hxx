#ifndef TMVA_SOFIE_ROPERATOR_HARDSIGMOID
#define TMVA_SOFIE_ROPERATOR_HARDSIGMOID

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T>
class ROperator_HardSigmoid final : public ROperator {

private:
   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;
   float fAlpha;
   float fBeta;

public:
   ROperator_HardSigmoid() {}
   ROperator_HardSigmoid(std::string nameX, std::string nameY, float alpha, float beta)
      : fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)), fAlpha(alpha), fBeta(beta)
   {
      fInputTensorNames = {fNX};
      fOutputTensorNames = {fNY};
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override { return input; }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   {
      auto ret = input; // suggest copy to compiler
      return ret;
   }

   void Initialize(RModel &model) override
   {
      // input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNX) == false) {
         throw std::runtime_error("TMVA SOFIE HardSigmoid Op Input Tensor " + fNX + " is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }

   std::string Generate(std::string OpName) override
   {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE HardSigmoid operator called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShape);

      out << "\n//------ HardSigmoid\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = std::fmax(0x0p+0f, std::fmin(0x1p+0f, " << fAlpha << "f * tensor_"
          << fNX << "[id] + " << fBeta << "f));\n";
      out << SP << "}\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override { return {std::string("cmath")}; }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROPERATOR_HARDSIGMOID