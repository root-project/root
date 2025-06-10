#ifndef TMVA_SOFIE_ROPERATOR_LAYERNORMALIZATION
#define TMVA_SOFIE_ROPERATOR_LAYERNORMALIZATION

#include "TMVA/RModel.hxx"
#include "TMVA/SOFIE_common.hxx"

#include <sstream>
#include <string>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T>
class ROperator_LayerNormalization : public ROperator {
private:
   int fAttrAxis;
   float fAttrEpsilon;
   size_t fAttrStashType;

   std::string fNX;
   std::string fNScale;
   std::string fNB;
   std::string fNY;
   std::string fNMean;
   std::string fNInvStdDev;

   std::string fNCastedX;
   std::string fNNormalizedX;
   std::string fNBroadcastedB;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeScale;
   std::vector<size_t> fShapeB;  // shape of input Bias (B) is assumed to be fully defined
   std::vector<Dim> fShapeY;
   std::vector<Dim> fShapeMean;
   std::vector<Dim> fShapeInvStdDev;

   size_t fAxis; // axis in [0, size)
   size_t fSize; // Size of the input
   // size_t fAxisDim;

   std::vector<Dim> fNormalizedShape;
   std::vector<Dim> fAxesShape;
   // lengths in string format
   std::string fLength; // Length of the input
   std::string fNormalizedLength;
   std::string fAxesLength;

   std::string fType;

public:
   ROperator_LayerNormalization() {}

   ROperator_LayerNormalization(int axis, float epsilon, size_t stashType, const std::string &nameX,
                                const std::string &nameScale, const std::string &nameB, const std::string &nameY,
                                const std::string &nameMean, const std::string &nameInvStdDev)
      : fAttrAxis(axis), fAttrEpsilon(epsilon), fAttrStashType(stashType), fNX(UTILITY::Clean_name(nameX)),
        fNScale(UTILITY::Clean_name(nameScale)), fNB(UTILITY::Clean_name(nameB)),
        fNY(UTILITY::Clean_name(nameY)), fNMean(UTILITY::Clean_name(nameMean)), fNInvStdDev(UTILITY::Clean_name(nameInvStdDev))
   {     
         fKind = OperatorKind::LAYERNORM;
         fInputTensorNames = { fNX, fNScale };
         if (!fNB.empty()){
            fInputTensorNames.emplace_back(fNB);
         }

         fOutputTensorNames = { fNY };
         if (!fNMean.empty()){
            fOutputTensorNames.emplace_back(fNMean);
         }
         if (!fNInvStdDev.empty()){
            fOutputTensorNames.emplace_back(fNInvStdDev);
         }
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override { return input; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override { return input; }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA::SOFIE - Tensor " + fNX + " not found.");
      }
      bool isDynamic = model.IsDynamicTensor(fNX);
      fShapeX = model.GetDynamicTensorShape(fNX);
      fShapeY = fShapeX;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      // Type of the output
      fType = ConvertTypeToString(model.GetTensorType(fNX));
      // Size of the input
      fSize = fShapeX.size();
      // Axis in [0, size)
      fAxis = (fAttrAxis < 0) ? fSize + fAttrAxis : fAttrAxis;
      // Shape of fShapeX[0, ..., fAxis)
      fAxesShape = std::vector<Dim>(fShapeX.begin(), fShapeX.begin() + fAxis);
      // Length of the axes
      fAxesLength = ConvertDynamicShapeToLength(fAxesShape);
      // Shape of fShapeX[fAxis, ..., fSize)
      fNormalizedShape = std::vector<Dim>(fShapeX.begin() + fAxis, fShapeX.end());
      // Length of the normalized axis
      fNormalizedLength = ConvertDynamicShapeToLength(fNormalizedShape);
      // length of the input
      fLength = ConvertDynamicShapeToLength(fShapeX);
      // Type of mean and std
      ETensorType type = (fAttrStashType == 1) ? ETensorType::FLOAT : model.GetTensorType(fNX);
      // Mean
      if (fNMean.empty()) {
         fNMean = "Mean" + fNX;
         // cannot use initializer list with one element since it is ambiguous
         if (isDynamic)
            // add size_t(-1) to indicate that shape is an expression
            model.AddIntermediateTensor(fNMean, type, std::vector<Dim>(1,Dim{fAxesLength,std::size_t(-1)}));
         else
            model.AddIntermediateTensor(fNMean, type, std::vector<size_t>(1,std::stoi(fAxesLength)));
      }
      // Inverse Standard Deviation
      if (fNInvStdDev.empty()) {
         fNInvStdDev = "InvStdDev" + fNX;
         if (isDynamic)
            model.AddIntermediateTensor(fNInvStdDev, type, std::vector<Dim>(1,Dim{fAxesLength,std::size_t(-1)}));
         else
            model.AddIntermediateTensor(fNInvStdDev, type, std::vector<size_t>(1,std::stoi(fAxesLength)));
      }
      // Cast X to float
      if (fAttrStashType == 1 && model.GetTensorType(fNX) != ETensorType::FLOAT) {
         fNCastedX = "Casted" + fNX;
         model.AddIntermediateTensor(fNCastedX, ETensorType::FLOAT, fShapeX);
         fNNormalizedX = "Normalized" + fNX;
         model.AddIntermediateTensor(fNNormalizedX, ETensorType::FLOAT, fShapeX);
      }
      // Broadcast the bias
      if (!fNB.empty()) {
         fShapeB = model.GetTensorShape(fNB);
         size_t lengthB = ConvertShapeToLength(fShapeB);
         if (isDynamic || lengthB < static_cast<size_t>(std::stoi(fLength))) {
            fNBroadcastedB = "Broadcasted" + fNB;
            model.AddIntermediateTensor(fNBroadcastedB, ConvertStringToType(fType), fShapeX);
         }
      }
      model.AddNeededStdLib("cmath");
   }

   std::string GenerateInitCode() override
   {
      std::stringstream out;
      if (!fNBroadcastedB.empty()) {
         out << SP << "// Broadcasting the bias of LayerNormalization op\n";
         out << SP << "{\n";
         out << SP << SP << "float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_";
         out << fNB << ", " << ConvertShapeToString(fShapeB) << ", " << ConvertDynamicShapeToString(fShapeX) << ");\n";
         out << SP << "std::copy(data, data + " << fLength << ", tensor_" << fNBroadcastedB << ");\n";
         out << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
      return out.str();
   }

   std::string Generate(std::string opName) override
   {
      opName = "op_" + opName;
      if (fShapeX.empty()) {
         throw std::runtime_error("TMVA::SOFIE LayerNormalization operator " + opName +
                                  " called to generate without being initialized first.");
      }
      if (fShapeX.size() > 5) {
         throw std::runtime_error("TMVA::SOFIE LayerNormalization operator not "
                                  "implemented for input tensor of size > 5.");
      }

      std::stringstream out;

      out << "//---- Layer Normalization  operator " << opName << "\n";

      // Loop over all the normalized axes i.e. [axis, ..., size)
      std::vector<std::string> inputShape(fSize);

      for (size_t i = 0; i < fSize; i++) {
         inputShape[i] = fShapeX[i].GetVal();
      }

      auto strides = UTILITY::ComputeStrideFromShape(fShapeX);
      std::string InputIndex = "axis_0 * " + strides[0].GetVal();
      for (size_t i = 1; i < fSize; i++) {
         InputIndex += " + axis_" + std::to_string(i) + " * " + strides[i].GetVal();
      }

      auto axesStrides = UTILITY::ComputeStrideFromShape(fAxesShape);
      std::string axesIndex = "axis_" + std::to_string(0) + " * " + axesStrides[0].GetVal();
      for (size_t i = 1; i < fAxis; i++) {
         axesIndex += " + axis_" + std::to_string(i) + " * " + axesStrides[i].GetVal();
      }

      auto normalizedStrides = UTILITY::ComputeStrideFromShape(fNormalizedShape);
      std::string normalizedIndex = "axis_" + std::to_string(fAxis) + " * " + normalizedStrides[0].GetVal();
      for (size_t i = fAxis + 1; i < fSize; i++) {
         normalizedIndex += " + axis_" + std::to_string(i) + " * " + normalizedStrides[i - fAxis].GetVal();
      }

      if (!fNCastedX.empty()) {
         // Cast X to float
         out << SP << "for (size_t i = 0; i < " << fLength << "; i++) {\n";
         out << SP << SP << "tensor_" << fNCastedX << "[i] = " << "static_cast<float>(tensor_" << fNX;
         out << "[i]);\n";
         out << SP << "}\n";
      }

      out << SP << "// Compute the mean\n";
      // Loop over the normalized dimensions
      for (size_t i = 0; i < fAxis; i++) {
         std::string iIdx = "axis_" + std::to_string(i);
         out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << inputShape[i]
                      << "; " << iIdx << "++) {\n";
      }
      out << SP << SP << fType << " sum = 0.;\n";
      // loop over all the dims in [0, fAxis)
      for (size_t j = fAxis; j < fSize; j++) {
         std::string jIdx = "axis_" + std::to_string(j);
         out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape[j]
                         << "; " << jIdx << "++) {\n";
      }
      out << SP << SP << SP << "sum += tensor_" << fNX << "[" << InputIndex << "];\n";
      for (size_t j = fAxis; j < fSize; j++) {
         out << SP << SP << "}\n";
      }
      out << SP << SP << "tensor_" << fNMean << "[" << axesIndex << "] = sum / " << fType << "(";
      out << fNormalizedLength << ");\n";
      for (size_t i = fAxis; i < fSize; i++) {
         out << SP << "}\n";
      }

      out << SP << "// Compute the inverse Standard Deviation\n";
      // Loop over the normalized dimensions
      for (size_t i = 0; i < fAxis; i++) {
         std::string iIdx = "axis_" + std::to_string(i);
         out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << inputShape[i]
                   << "; " << iIdx << "++){\n";
      }
      // Set sum = 0
      out << SP << SP << fType << " sum = 0.;\n";
      // loop over all the dims in [0, fAxis)
      for (size_t j = fAxis; j < fSize; j++) {
         std::string jIdx = "axis_" + std::to_string(j);
         out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape[j]
                          << "; " << jIdx << "++){\n";
      }
      out << SP << SP << SP << "float tmp = tensor_" << fNX << "[" << InputIndex << "] - tensor_"
                            << fNMean << "[" << axesIndex << "];\n";
      out << SP << SP << SP << "sum += tmp*tmp;\n";
      for (size_t j = fAxis; j < fSize; j++) {
         out << SP << SP << "}\n";
      }
      out << SP << SP << "tensor_" << fNInvStdDev << "[" << axesIndex << "] = 1 / std::sqrt(";
      out << "sum / " << fType << "(" << fNormalizedLength << ") + " << fAttrEpsilon << ");\n";
      for (size_t i = 0; i < fAxis; i++) {
         out << SP << "}\n";
      }

      if (!fNCastedX.empty()) {
         out << "// NormalizedX = InvStdDev * (CastedX - Mean)\n";
         for (size_t i = 0; i < fAxis; i++) {
            std::string iIdx = "axis_" + std::to_string(i);
            out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << inputShape[i]
                          << "; " << iIdx << "++){\n";
         }
         for (size_t j = fAxis; j < fSize; j++) {
            std::string jIdx = "axis_" + std::to_string(j);
            out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape[j]
                             << "; " << jIdx << "++){\n";
         }
         out << SP << SP << SP << "tensor_" << fNNormalizedX << "[" << InputIndex << "] = tensor_";
         out << fNInvStdDev << "[" << axesIndex << "] * (tensor_" << fNCastedX << "[" << InputIndex;
         out << "] - tensor_" << fNMean << "[" << axesIndex << "])\n";
         for (size_t j = fAxis; j < fSize; j++) {
            out << SP << SP << "}\n";
         }
         for (size_t i = fAxis; i < fSize; i++) {
            out << SP << "}\n";
         }
         out << "// Y = Scale o NormalizedX";
         for (size_t i = 0; i < fAxis; i++) {
            std::string iIdx = "axis_" + std::to_string(i);
            out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << inputShape[i]
                      << "; " << iIdx << "++){\n";
         }
         for (size_t j = fAxis; j < fSize; j++) {
            std::string jIdx = "axis_" + std::to_string(j);
            out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape[j]
                            << "; " << jIdx << "++){\n";
         }
         out << SP << SP << SP << "tensor_" << fNY << "[" << InputIndex << "] = tensor_" << fNScale;
         out << "[" << axesIndex << "] * static_cast<" << fType << ">(tensor_" << fNCastedX << "[" << InputIndex;
         out << "]);\n";
         for (size_t j = fAxis; j < fSize; j++) {
            out << SP << SP << "}\n";
         }
         for (size_t i = fAxis; i < fSize; i++) {
            out << SP << "}\n";
         }
      } else {
         out << SP << "// Y = Scale o InvStdDev (X - Mean)\n";
         for (size_t i = 0; i < fAxis; i++) {
            std::string iIdx = "axis_" + std::to_string(i);
            out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << inputShape[i]
                         << "; " << iIdx << "++){\n";
         }
         for (size_t j = fAxis; j < fSize; j++) {
            std::string jIdx = "axis_" + std::to_string(j);
            out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape[j]
                           << "; " << jIdx << "++){\n";
         }
         out << SP << SP << SP << "tensor_" << fNY << "[" << InputIndex << "] = tensor_" << fNScale;
         out << "[" << normalizedIndex << "] * tensor_" << fNInvStdDev << "[" << axesIndex;
         out << "] * (tensor_" << fNX << "[" << InputIndex << "] - tensor_" << fNMean << "[";
         out << axesIndex << "]);\n";
         for (size_t j = fAxis; j < fSize; j++) {
            out << SP << SP << "}\n";
         }
         for (size_t i = fAxis; i < fSize; i++) {
            out << SP << "}\n";
         }
      }

      if (!fNB.empty()) {
         std::string bias = "tensor_" + (fNBroadcastedB.empty() ? fNB : fNBroadcastedB);
         out << SP << "// Add the bias to Y\n";
         out << SP << "int " << opName << "_n = " << fLength << ";\n";
         out << SP << "float " << opName << "_alpha = 1.;\n";
         out << SP << "int " << opName << "_inc = 1;\n";
         out << SP << "BLAS::saxpy_(&" << opName << "_n, &" << opName << "_alpha, " << bias << ", &";
         out << opName << "_inc, " << "tensor_" << fNY << ", &" << opName << "_inc);\n";
      }

      return out.str();
   }

   std::vector<std::string> GetBlasRoutines() override { return { std::string("Axpy") }; }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") }; }

   std::string GetFusableOutputTensorName() override {
       return fNY;
   }
   
   void UpdateFusableTensorName(std::string fusable_tensor_name, const std::function<void(const std::string&)>& removal_func){
      removal_func(fNX);
      removal_func(fNY);
      fNX = fusable_tensor_name;
      fNY = fusable_tensor_name;
      fInputTensorNames[0] = fNX;
      fOutputTensorNames[0] = fNY;
   }

};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
