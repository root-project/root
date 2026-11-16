#ifndef TMVA_SOFIE_ROPERATOR_RMSNORMALIZATION
#define TMVA_SOFIE_ROPERATOR_RMSNORMALIZATION

#include "TMVA/RModel.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/SOFIE_common.hxx"

#include <sstream>
#include <string>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

/// ONNX RMSNormalization (opset 23):
///    Y = X / sqrt(mean(X^2 over the axes [axis, ..., rank-1]) + epsilon) * scale
/// where scale is unidirectionally broadcastable to the normalized shape X[axis, ..., rank-1].
template <typename T>
class ROperator_RMSNormalization final : public ROperator {
private:
   int fAttrAxis;
   float fAttrEpsilon;

   std::string fNX;
   std::string fNScale;
   std::string fNY;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeScale; // scale shape left-padded with 1 to the rank of X

   size_t fAxis = 0; // axis in [0, rank)
   size_t fSize = 0; // rank of X
   std::string fNormalizedLength;

public:
   ROperator_RMSNormalization() {}

   ROperator_RMSNormalization(int axis, float epsilon, const std::string &nameX, const std::string &nameScale,
                              const std::string &nameY)
      : fAttrAxis(axis),
        fAttrEpsilon(epsilon),
        fNX(UTILITY::Clean_name(nameX)),
        fNScale(UTILITY::Clean_name(nameScale)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames = {fNX, fNScale};
      fOutputTensorNames = {fNY};
   }

   void Initialize(RModel &model) override
   {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA::SOFIE - RMSNormalization - Tensor " + fNX + " not found.");
      }
      if (!model.CheckIfTensorAlreadyExist(fNScale)) {
         throw std::runtime_error("TMVA::SOFIE - RMSNormalization - Scale tensor " + fNScale + " not found.");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      fSize = fShapeX.size();
      if (fSize == 0) {
         throw std::runtime_error("TMVA::SOFIE - RMSNormalization - Input tensor " + fNX + " must have rank >= 1.");
      }
      const int rank = static_cast<int>(fSize);
      if (fAttrAxis < -rank || fAttrAxis >= rank) {
         throw std::runtime_error("TMVA::SOFIE - RMSNormalization - axis " + std::to_string(fAttrAxis) +
                                  " is out of range for an input of rank " + std::to_string(rank));
      }
      fAxis = (fAttrAxis < 0) ? fSize + fAttrAxis : fAttrAxis;
      fNormalizedLength = ConvertDimShapeToLength(std::vector<Dim>(fShapeX.begin() + fAxis, fShapeX.end()));

      // The scale must broadcast to the normalized shape X[axis, ..., rank-1]: left-pad it with 1 up to the
      // rank of X, then every non-1 dimension must match X and lie inside the normalized axes.
      fShapeScale = model.GetDimTensorShape(fNScale);
      if (fShapeScale.size() > fSize - fAxis) {
         throw std::runtime_error("TMVA::SOFIE - RMSNormalization - Scale tensor has shape " +
                                  ConvertDimShapeToString(fShapeScale) + " of higher rank than the normalized shape");
      }
      fShapeScale.insert(fShapeScale.begin(), fSize - fShapeScale.size(), Dim{1});
      for (size_t i = fAxis; i < fSize; i++) {
         if (fShapeScale[i].dim != 1 && fShapeScale[i] != fShapeX[i])
            throw std::runtime_error("TMVA::SOFIE - RMSNormalization - Scale tensor has shape " +
                                     ConvertDimShapeToString(model.GetDimTensorShape(fNScale)) +
                                     " which cannot be broadcast to the normalized shape of input " +
                                     ConvertDimShapeToString(fShapeX));
      }

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeX);
      model.AddNeededStdLib("cmath");
      if (model.Verbose()) {
         std::cout << "RMSNormalization : " << fNX << " -> " << fNY << " shape " << ConvertDimShapeToString(fShapeX)
                   << " axis " << fAxis << " scale shape " << ConvertDimShapeToString(fShapeScale) << std::endl;
      }
   }

   std::string Generate(std::string opName) override
   {
      opName = "op_" + opName;
      if (fShapeX.empty()) {
         throw std::runtime_error("TMVA::SOFIE RMSNormalization operator " + opName +
                                  " called to generate without being initialized first.");
      }

      const std::string type = TensorType<T>::Name();

      // flat index into X and Y from the loop variables axis_0, ..., axis_{rank-1}
      auto strides = UTILITY::ComputeStrideFromShape(fShapeX);
      std::string inputIndex;
      for (size_t i = 0; i < fSize; i++) {
         if (!inputIndex.empty())
            inputIndex += " + ";
         inputIndex += "axis_" + std::to_string(i);
         if (i < fSize - 1)
            inputIndex += " * " + strides[i].GetVal();
      }
      // flat index into the scale, skipping its broadcast (size 1) axes
      auto scaleStrides = UTILITY::ComputeStrideFromShape(fShapeScale);
      std::string scaleIndex;
      for (size_t i = fAxis; i < fSize; i++) {
         if (fShapeScale[i].dim != 1) {
            if (!scaleIndex.empty())
               scaleIndex += " + ";
            scaleIndex += "axis_" + std::to_string(i);
            if (scaleStrides[i].dim != 1)
               scaleIndex += " * " + scaleStrides[i].GetVal();
         }
      }
      if (scaleIndex.empty())
         scaleIndex = "0";

      auto openNormalizedLoops = [&](std::stringstream &out) {
         for (size_t j = fAxis; j < fSize; j++) {
            std::string jIdx = "axis_" + std::to_string(j);
            out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << fShapeX[j].GetVal() << "; "
                << jIdx << "++) {\n";
         }
      };
      auto closeNormalizedLoops = [&](std::stringstream &out) {
         for (size_t j = fAxis; j < fSize; j++)
            out << SP << SP << "}\n";
      };

      std::stringstream out;
      out << "//---- RMSNormalization operator " << opName << "\n";
      out << SP << "{\n";
      // loop over the outer axes [0, axis)
      for (size_t i = 0; i < fAxis; i++) {
         std::string iIdx = "axis_" + std::to_string(i);
         out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << fShapeX[i].GetVal() << "; " << iIdx
             << "++) {\n";
      }
      // mean of the squares over the normalized axes, accumulated in float (stash_type = 1)
      out << SP << SP << "float sumSq = 0.;\n";
      openNormalizedLoops(out);
      out << SP << SP << SP << "const float x = tensor_" << fNX << "[" << inputIndex << "];\n";
      out << SP << SP << SP << "sumSq += x * x;\n";
      closeNormalizedLoops(out);
      out << SP << SP << "const float invRms = 1.f / std::sqrt(sumSq / float(" << fNormalizedLength << ") + "
          << ConvertValToString(fAttrEpsilon) << "f);\n";
      // Y = X * invRms * scale
      openNormalizedLoops(out);
      out << SP << SP << SP << "tensor_" << fNY << "[" << inputIndex << "] = " << type << "(tensor_" << fNX << "["
          << inputIndex << "] * invRms * tensor_" << fNScale << "[" << scaleIndex << "]);\n";
      closeNormalizedLoops(out);
      for (size_t i = 0; i < fAxis; i++)
         out << SP << "}\n";
      out << SP << "}\n";

      return out.str();
   }

   std::vector<std::string> GetStdLibs() override { return {std::string("cmath")}; }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
