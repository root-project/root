#ifndef TMVA_SOFIE_ROPERATOR_PRelu
#define TMVA_SOFIE_ROPERATOR_PRelu

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T>
class ROperator_PRelu final : public ROperator {
private:
   /* Attributes */
   std::string fNX;
   std::string fNSlope;
   std::string fNY;
   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeSlope;
   std::string fType;

public:
   ROperator_PRelu() {}
   ROperator_PRelu(std::string nameX, std::string nameSlope, std::string nameY)
      : fNX(UTILITY::Clean_name(nameX)), fNSlope(UTILITY::Clean_name(nameSlope)), fNY(UTILITY::Clean_name(nameY))
   {
      if (std::is_same<T, float>::value) {
         fType = "float";
      } else {
         throw std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a PRelu operator");
      }
      fInputTensorNames = {fNX, fNSlope};
      fOutputTensorNames = {fNY};
   }
   void Initialize(RModel &model) override
   {
      if (model.CheckIfTensorAlreadyExist(fNX) == false || model.CheckIfTensorAlreadyExist(fNSlope) == false) {
         throw std::runtime_error("TMVA SOFIE PRelu Op Input Tensor is not found in model");
      }
      fShapeX = model.GetTensorShape(fNX);
      fShapeSlope = model.GetTensorShape(fNSlope);

      // check that slope is unidirectionally broadcastable to X (throws otherwise);
      // this also prepends 1s to fShapeSlope so that it has the same rank as X
      try {
         UTILITY::UnidirectionalBroadcastShape(fShapeSlope, fShapeX);
      } catch (const std::runtime_error &e) {
         throw std::runtime_error("TMVA SOFIE PRelu Op slope tensor " + fNSlope +
                                  " cannot be broadcast to input tensor " + fNX + ": " + e.what());
      }

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeX);
   }
   std::string Generate(std::string OpName) override
   {
      OpName = "op_" + OpName;

      if (fShapeX.empty() || fShapeSlope.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator PRelu called to Generate without being initialized first");
      }

      auto stridesX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto stridesSlope = UTILITY::ComputeStrideFromShape(fShapeSlope);

      std::string slopeIdx;
      if (UTILITY::AreSameShape(fShapeX, fShapeSlope)) {
         slopeIdx = "id";
      } else {
         for (size_t k = 0; k < fShapeSlope.size(); k++) {
            if (fShapeSlope[k] == 1)
               continue;
            if (!slopeIdx.empty())
               slopeIdx += " + ";
            if (stridesX[k] == 1)
               slopeIdx += "id % " + std::to_string(fShapeX[k]);
            else
               slopeIdx += "(id / " + std::to_string(stridesX[k]) + ") % " + std::to_string(fShapeX[k]);
            if (stridesSlope[k] != 1)
               slopeIdx += " * " + std::to_string(stridesSlope[k]);
         }
         if (slopeIdx.empty())
            slopeIdx = "0";
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShapeX);
      out << "\n//------ PRELU\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = ((tensor_" << fNX << "[id] >= 0 )? tensor_" << fNX
          << "[id] : tensor_" << fNSlope << "[" << slopeIdx << "] * tensor_" << fNX << "[id]);\n";
      out << SP << "}\n";
      return out.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROPERATOR_PRelu
