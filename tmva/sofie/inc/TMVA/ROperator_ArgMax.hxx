#ifndef TMVA_SOFIE_ROPERATOR_ARGMAX
#define TMVA_SOFIE_ROPERATOR_ARGMAX

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class ROperator_ArgMax final : public ROperator {
private:
   /* Attributes */
   int64_t fAttrAxis = 0;
   int fKeepdims = 1;
   int fSelectLastIndex = 0;

   /* Tensor names */
   std::string fNX; // input
   std::string fNY; // output

   /* Tensor shapes */
   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;

public:
   ROperator_ArgMax() {}
   ROperator_ArgMax(int64_t axis, int keepdims, int selectLastIndex, std::string nameX, std::string nameY)
      : fAttrAxis(axis),
        fKeepdims(keepdims),
        fSelectLastIndex(selectLastIndex),
        fNX(UTILITY::Clean_name(nameX)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames = {fNX};
      fOutputTensorNames = {fNY};
   }

   void Initialize(RModel &model) override
   {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA SOFIE ArgMax Op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      int64_t rank = static_cast<int64_t>(fShapeX.size());

      // negative axis counts from the end (e.g. -1 is the last dimension)
      if (fAttrAxis < 0) {
         fAttrAxis += rank;
      }
      // valid range is [-rank, rank-1], i.e. [0, rank-1] after normalization
      if (fAttrAxis < 0 || fAttrAxis >= rank) {
         throw std::runtime_error("TMVA SOFIE ArgMax Op - invalid axis value " + std::to_string(fAttrAxis) +
                                  " for input tensor " + fNX + " of rank " + std::to_string(rank));
      }

      // compute output shape: reduced axis becomes 1 (keepdims) or is removed
      fShapeY = fShapeX;
      if (fKeepdims != 0) {
         fShapeY[fAttrAxis] = Dim{1};
      } else {
         fShapeY.erase(fShapeY.begin() + fAttrAxis);
      }
      // output contains indices, so its type is always INT64 regardless of the input type
      model.AddIntermediateTensor(fNY, ETensorType::INT64, fShapeY);
   }

   std::string Generate(std::string opName) override
   {
      // split the input shape around the axis: [outer..., axisDim, inner...]
      std::string outer = ConvertDimShapeToLength(std::vector<Dim>(fShapeX.begin(), fShapeX.begin() + fAttrAxis));
      std::string axisDim = fShapeX[fAttrAxis].GetVal();
      std::string inner = ConvertDimShapeToLength(std::vector<Dim>(fShapeX.begin() + fAttrAxis + 1, fShapeX.end()));

      // element (o, k, i) of the input is at o * axisDim * inner + k * inner + i
      std::string stride = "(" + axisDim + ") * (" + inner + ")";
      // with '>' ties keep the first index, with '>=' they move to the last one
      std::string cmp = (fSelectLastIndex != 0) ? ">=" : ">";

      std::stringstream out;
      out << "\n//------ ArgMax  " << opName << "\n";
      out << SP << "for (size_t o = 0; o < " << outer << "; o++) {\n";
      out << SP << SP << "for (size_t i = 0; i < " << inner << "; i++) {\n";
      out << SP << SP << SP << "size_t base = o * " << stride << " + i;\n";
      out << SP << SP << SP << "auto greatest = tensor_" << fNX << "[base];\n";
      out << SP << SP << SP << "int64_t bestIndex = 0;\n";
      out << SP << SP << SP << "for (size_t k = 1; k < " << axisDim << "; k++) {\n";
      out << SP << SP << SP << SP << "auto value = tensor_" << fNX << "[base + k * (" << inner << ")];\n";
      out << SP << SP << SP << SP << "if (value " << cmp << " greatest) {\n";
      out << SP << SP << SP << SP << SP << "greatest = value;\n";
      out << SP << SP << SP << SP << SP << "bestIndex = static_cast<int64_t>(k);\n";
      out << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << "}\n";
      out << SP << SP << SP << "tensor_" << fNY << "[o * (" << inner << ") + i] = bestIndex;\n";
      out << SP << SP << "}\n";
      out << SP << "}\n";
      return out.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROPERATOR_ARGMAX
