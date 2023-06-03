#ifndef TMVA_SOFIE_ROPERATOR_Range
#define TMVA_SOFIE_ROPERATOR_Range

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <cmath>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T>
class ROperator_Range final : public ROperator {

private:
   std::string fNStart;
   std::string fNLimit;
   std::string fNDelta;
   std::size_t num_of_elem;
   std::string fNY;
   std::vector<size_t> fShapeY;

public:
   ROperator_Range() {}

   ROperator_Range(std::string nameStart, std::string nameLimit, std::string nameDelta, std::string nameY)
      : fNStart(nameStart), fNLimit(nameLimit), fNDelta(nameDelta), fNY(UTILITY::Clean_name(nameY))
   {
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) { return input; }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input)
   {
      auto ret = input; // suggest copy to compiler
      return ret;
   }
   void Initialize(RModel &model)
   {
      // input must be a scalar
      double lim = std::stod(fNLimit);
      double start = std::stod(fNStart);
      double delta = std::stod(fNDelta);
      num_of_elem = std::max(std::ceil((lim - start) / delta), 0.0);
      fShapeY = {num_of_elem};
      model.AddIntermediateTensor(fNY, ETensorType::DOUBLE, fShapeY);
   }

   std::string Generate(std::string OpName)
   {
      OpName = "op_" + OpName;
      if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Range operator called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShapeY);
      out << "\n//------ Range\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = " << fNStart << " + (id * " << fNDelta << ");\n";
      out << SP << "}\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() { return {std::string("cmath")}; }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROPERATOR_Range
