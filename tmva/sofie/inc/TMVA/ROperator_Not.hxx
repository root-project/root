#ifndef TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_BASIC_UNARY
#define TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_BASIC_UNARY

#include <TMVA/ROperator.hxx>
#include <TMVA/RModel.hxx>
#include <TMVA/SOFIE_common.hxx>

namespace TMVA {
namespace Experimental {
namespace SOFIE {


class ROperator_Not final : public ROperator {
private:
   std::string fNX;
   std::string fNY;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;

public:
   ROperator_Not() {}

   ROperator_Not(std::string nameX, std::string nameY)
      : fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
         fInputTensorNames =  { fNX };
         fOutputTensorNames = { fNY };
   }


   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA::SOFIE - Tensor " + fNX + " not found.");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      fShapeY = fShapeX;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
   }

   std::string Generate(std::string opName) override
   {
      opName = "op_" + opName;
      std::stringstream out;

      out << SP << "\n//---- Operator Not  " << opName << "\n";
      auto length = ConvertDimShapeToLength(fShapeX);
      out << SP << "for (size_t i = 0; i < " << length << "; i++) {\n";
      out << SP << SP << "tensor_" << fNY << "[i] = !tensor_" + fNX + "[i];\n";
      out << SP << "}\n";
      return out.str();
   }

};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
