#ifndef TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_BASIC_IS
#define TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_BASIC_IS

#include <TMVA/ROperator.hxx>
#include <TMVA/RModel.hxx>
#include <TMVA/SOFIE_common.hxx>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

enum class EBasicIsOperator { kIsInf, kIsInfPos, kIsInfNeg, kIsNaN };

template <EBasicIsOperator Op>
struct IsOpTraits {
};
template<>
struct IsOpTraits<EBasicIsOperator::kIsInf> {
   static std::string Name() { return "IsInf"; }
   static std::string Op(const std::string &x) { return "std::isinf(" + x + ")"; }
};
template<>
struct IsOpTraits<EBasicIsOperator::kIsInfPos> {
   static std::string Name() { return "IsInfPos"; }
   static std::string Op(const std::string &x) { return "(std::isinf(" + x + ") && " + x + "> 0)"; }
};
template<>
struct IsOpTraits<EBasicIsOperator::kIsInfNeg> {
   static std::string Name() { return "IsInfNeg"; }
   static std::string Op(const std::string &x) { return "(std::isinf(" + x + ") && " + x + "< 0)"; }
};
template<>
struct IsOpTraits<EBasicIsOperator::kIsNaN> {
   static std::string Name() { return "IsInf"; }
   static std::string Op(const std::string &x) { return "std::isnan(" + x + ")"; }
};



template <EBasicIsOperator Op>
class ROperator_Basic_Is final : public ROperator {
private:
   std::string fNX;
   std::string fNY;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;

public:
   ROperator_Basic_Is() {}

   ROperator_Basic_Is(std::string nameX, std::string nameY)
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

      out << SP << "\n//---- Operator" << IsOpTraits<Op>::Name() << " " << opName << "\n";
      auto length = ConvertDimShapeToLength(fShapeX);
      out << SP << "for (size_t i = 0; i < " << length << "; i++) {\n";
      out << SP << SP << "tensor_" << fNY << "[i] = " << IsOpTraits<Op>::Op("tensor_" + fNX + "[i]") << ";\n";
      out << SP << "}\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override {
      return { std::string("cmath") };
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
