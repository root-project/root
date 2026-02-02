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
   bool fCastToFloat = false;  // flag to indicate if operation 1 are in floats (to be  impl)
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
   std::vector<Dim> fShapeB;
   std::vector<Dim> fShapeY;
   std::vector<Dim> fShapeMean;
   std::vector<Dim> fShapeInvStdDev;

   size_t fAxis; // axis in [0, size)
   size_t fSize; // Size of the input
   // size_t fAxisDim;

   std::vector<Dim> fNormalizedShape;  // shape from X[ axis,...,N-1]
   std::vector<Dim> fAxesShape;        // shape from X[0,..,axis-1]
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
         throw std::runtime_error("TMVA::SOFIE - LayerNormalization - Tensor " + fNX + " not found.");
      }
      bool isDynamic = model.IsDynamicTensor(fNX);
      fShapeX = model.GetDimTensorShape(fNX);
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
      fAxesLength = ConvertDimShapeToLength(fAxesShape);
      // Shape of fShapeX[fAxis, ..., fSize)
      fNormalizedShape = std::vector<Dim>(fShapeX.begin() + fAxis, fShapeX.end());
      // Length of the normalized axis
      fNormalizedLength = ConvertDimShapeToLength(fNormalizedShape);
      // length of the input
      fLength = ConvertDimShapeToLength(fShapeX);
      // Type of mean and std
      ETensorType type = (fAttrStashType == 1) ? ETensorType::FLOAT : model.GetTensorType(fNX);
      // Mean
      if (!fNMean.empty()) {
         // cannot use initializer list with one element since it is ambiguous
         if (isDynamic)
            // add size_t(-1) to indicate that shape is an expression
            model.AddIntermediateTensor(fNMean, type, std::vector<Dim>(1,Dim{fAxesLength,std::size_t(-1)}));
         else
            model.AddIntermediateTensor(fNMean, type, std::vector<size_t>(1,std::stoi(fAxesLength)));
      }
      // Inverse Standard Deviation
      if (!fNInvStdDev.empty()) {
         if (isDynamic)
            model.AddIntermediateTensor(fNInvStdDev, type, std::vector<Dim>(1,Dim{fAxesLength,std::size_t(-1)}));
         else
            model.AddIntermediateTensor(fNInvStdDev, type, std::vector<size_t>(1,std::stoi(fAxesLength)));
      }
      // if mean and stdev are not empty they are not defined in the output list
      // Cast X to float
      if (fAttrStashType == 1 && model.GetTensorType(fNX) != ETensorType::FLOAT) {
         fCastToFloat = true;
         fType = "float";
         // fNCastedX = "Casted" + fNX;
         // model.AddIntermediateTensor(fNCastedX, ETensorType::FLOAT, fShapeX);
         // fNNormalizedX = "Normalized" + fNX;
         // model.AddIntermediateTensor(fNNormalizedX, ETensorType::FLOAT, fShapeX);
      }
      // scale shape
      fShapeScale = model.GetDimTensorShape(fNScale);
      // appends 1 to scale shapes if missing
      size_t dimScale = fShapeScale.size();
      if (dimScale < fSize) {
         for (size_t i = 0; i < fSize-dimScale; i++)
            fShapeScale.insert(fShapeScale.begin(), Dim{1});
      }
      // check also shape if consistent now
      for (size_t i = 0; i < fSize; i++) {
         if (fShapeScale[i].dim != 1 && fShapeScale[i] != fShapeX[i])
            throw std::runtime_error("TMVA::SOFIE - LayerNormalization - Scale Tensor has invalid shape " + ConvertDimShapeToString(fShapeScale));
      }
      if (!fNB.empty()) {
         fShapeB = model.GetDimTensorShape(fNB);
         // appends 1 to bias shapes if missing
         size_t dimB = fShapeB.size();
         if (dimB < fShapeX.size()) {
            for (size_t i = 0; i < fSize-dimB; i++)
               fShapeB.insert(fShapeB.begin(), Dim{1});
         }
         for (size_t i = 0; i < fSize; i++) {
            if (fShapeB[i].dim != 1 && fShapeB[i] != fShapeX[i])
               throw std::runtime_error("TMVA::SOFIE - LayerNormalization - Bias Tensor has invalid shape " + ConvertDimShapeToString(fShapeScale));
         }
      }

      std::cout << "bias + scale " << ConvertDimShapeToString(fShapeB) << "  " << ConvertDimShapeToString(fShapeScale) << std::endl;

      // // Broadcast the bias
      // if (!fNB.empty()) {
      //    fShapeB = model.GetTensorShape(fNB);
      //    size_t lengthB = ConvertShapeToLength(fShapeB);
      //    if (isDynamic || lengthB < static_cast<size_t>(std::stoi(fLength))) {
      //       fNBroadcastedB = "Broadcasted" + fNB;
      //       model.AddIntermediateTensor(fNBroadcastedB, ConvertStringToType(fType), fShapeX);
      //    }
      // }
      model.AddNeededStdLib("cmath");
   }

   std::string GenerateInitCode() override
   {
      std::stringstream out;
      if (!fNBroadcastedB.empty()) {
         out << SP << "// Broadcasting the bias of LayerNormalization op\n";
         out << SP << "{\n";
         out << SP << SP << "float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_";
         out << fNB << ", " << ConvertShapeToString(fShapeB) << ", " << ConvertShapeToString(fShapeX) << ");\n";
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

      std::stringstream out;

      out << "//---- Layer Normalization  operator " << opName << "\n";

      // Loop over all the normalized axes i.e. [axis, ..., size)
      std::vector<std::string> inputShape(fSize);

      for (size_t i = 0; i < fSize; i++) {
         inputShape[i] = fShapeX[i].GetVal();
      }

      auto strides = UTILITY::ComputeStrideFromShape(fShapeX);
      std::string inputIndex = "axis_0 * " + strides[0].GetVal();
      for (size_t i = 1; i < fSize; i++) {
         inputIndex += " + axis_" + std::to_string(i);
         if (i < fSize-1) inputIndex += " * " + strides[i].GetVal();
      }
      auto scaleStrides = UTILITY::ComputeStrideFromShape(fShapeScale);
      std::string scaleIndex;
      for (size_t i = 0; i < fSize; i++) {
         if (fShapeScale[i].dim != 1) {
            if (!scaleIndex.empty()) scaleIndex += " + ";
            scaleIndex += "axis_" + std::to_string(i);
            if ( scaleStrides[i].dim != 1) scaleIndex +=  " * " + scaleStrides[i].GetVal();
         }
      }
      if (scaleIndex.empty()) scaleIndex = "0";

      auto biasStrides = UTILITY::ComputeStrideFromShape(fShapeB);
      std::string biasIndex;
      for (size_t i = 0; i < fSize; i++) {
         if (fShapeB[i].dim != 1) {
            if (!biasIndex.empty()) biasIndex += " + ";
            biasIndex += "axis_" + std::to_string(i);
            if ( biasStrides[i].dim != 1) biasIndex +=  " * " + biasStrides[i].GetVal();
         }
      }
      if (biasIndex.empty()) biasIndex = "0";

      auto axesStrides = UTILITY::ComputeStrideFromShape(fAxesShape);
      std::string axesIndex = "axis_" + std::to_string(0) + " * " + axesStrides[0].GetVal();
      for (size_t i = 1; i < fAxis; i++) {
         axesIndex += " + axis_" + std::to_string(i) + " * " + axesStrides[i].GetVal();
      }


      // compute mean and std-dev. Save in tensors if requested

      out << SP << "// Compute the mean\n";

      // Loop over all the outer dims in [0, fAxis)
      for (size_t i = 0; i < fAxis; i++) {
         std::string iIdx = "axis_" + std::to_string(i);
         out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << inputShape[i]
                      << "; " << iIdx << "++) {\n";
      }
      out << SP << SP << fType << " mean = 0.;\n";
      // loop over the normalized dimensions (fAxis,....,N-1)
      for (size_t j = fAxis; j < fSize; j++) {
         std::string jIdx = "axis_" + std::to_string(j);
         out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape[j]
                         << "; " << jIdx << "++) {\n";
      }
      out << SP << SP << SP << "mean += tensor_" << fNX << "[" << inputIndex << "];\n";
      for (size_t j = fAxis; j < fSize; j++) {
         out << SP << SP << "}\n";
      }
      out << SP << SP << "mean  /= " << fType << "(" << fNormalizedLength << ");\n";


      out << SP << "// Compute the inverse Standard Deviation\n";

      // Set sum = 0
      out << SP << SP << fType << " sum = 0.;\n";
      // loop over all the dims in [0, fAxis)
      for (size_t j = fAxis; j < fSize; j++) {
         std::string jIdx = "axis_" + std::to_string(j);
         out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape[j]
                          << "; " << jIdx << "++){\n";
      }
      out << SP << SP << SP << "float tmp = tensor_" << fNX << "[" << inputIndex << "] - mean;\n";
      out << SP << SP << SP << "sum += tmp*tmp;\n";
      for (size_t j = fAxis; j < fSize; j++) {
         out << SP << SP << "}\n";
      }
      out << SP << SP << fType << " invStdDev = 1 / std::sqrt(";
      out << "sum / " << fType << "(" << fNormalizedLength << ") + " << fAttrEpsilon << ");\n";


      // set output mean and invStdDev if requested
      if (!fNMean.empty())
         out << SP << SP <<  "tensor_" << fNMean << "[" << axesIndex << "] = mean;\n";
      if (!fNInvStdDev.empty())
         out << SP << SP <<  "tensor_" << fNInvStdDev << "[" << axesIndex << "] = invStdDev;\n";

      // scale and add bias

      out << SP << "// Y = Scale o InvStdDev (X - Mean)\n";

      for (size_t j = fAxis; j < fSize; j++) {
         std::string jIdx = "axis_" + std::to_string(j);
         out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape[j] << "; " << jIdx
             << "++){\n";
      }
      out << SP << SP << SP << "tensor_" << fNY << "[" << inputIndex << "] = tensor_" << fNScale;
      out << "[" << scaleIndex << "] * invStdDev * (tensor_" << fNX << "[" << inputIndex << "] - mean)";

      // add bias if needed
      if (!fNB.empty())
         // assume bias has index as scale
         out << " + tensor_" << fNB << "[" << biasIndex << "]";
      out << ";\n";

      // close loops on normalizing dim  [..,fAxis,...fSize-1]
      for (size_t j = fAxis; j < fSize; j++) {
         out << SP << SP << "}\n";
      }
      // close loops on the other dimensions [0,...,fAxis]
      for (size_t i = 0; i < fAxis; i++) {
         out << SP << "}\n";
      }

      return out.str();
   }

   std::vector<std::string> GetBlasRoutines() override { return { std::string("Axpy") }; }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") }; }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
