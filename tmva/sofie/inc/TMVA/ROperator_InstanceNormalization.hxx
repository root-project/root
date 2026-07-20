#ifndef TMVA_SOFIE_ROPERATOR_INSTANCENORMALIZATION
#define TMVA_SOFIE_ROPERATOR_INSTANCENORMALIZATION

#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"
#include "TMVA/SOFIE_common.hxx"

#include <sstream>
#include <stdexcept>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

/*! \class ROperator_InstanceNormalization
    \brief Implementation of the ONNX InstanceNormalization operator.

    The input X has shape (N, C, D1, ..., Dn). For every sample n and channel c,
    the elements over the spatial dimensions D1, ..., Dn are normalized to zero
    mean and unit variance, and then scaled and shifted with the per-channel
    scale and B tensors, which both have shape (C):

        Y[n, c, ...] = scale[c] * (X[n, c, ...] - mean[n, c]) / sqrt(var[n, c] + epsilon) + B[c]
*/
template <typename T>
class ROperator_InstanceNormalization final : public ROperator {
private:
   float fEpsilon;
   std::string fNInput;
   std::string fNScale;
   std::string fNBias;
   std::string fNOutput;
   std::vector<size_t> fShape;
   std::string fType;

public:
   ROperator_InstanceNormalization() : fEpsilon(1e-5) {}

   ROperator_InstanceNormalization(float epsilon, std::string nameInput, std::string nameScale, std::string nameBias,
                                   std::string nameOutput)
      : fEpsilon(epsilon),
        fNInput(UTILITY::Clean_name(nameInput)),
        fNScale(UTILITY::Clean_name(nameScale)),
        fNBias(UTILITY::Clean_name(nameBias)),
        fNOutput(UTILITY::Clean_name(nameOutput))
   {
      fInputTensorNames = {fNInput, fNScale, fNBias};
      fOutputTensorNames = {fNOutput};
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> inputShapes) override
   {
      return {inputShapes[0]};
   }

   void Initialize(RModel &model) override
   {
      for (const std::string &name : {fNInput, fNScale, fNBias}) {
         if (!model.CheckIfTensorAlreadyExist(name)) {
            throw std::runtime_error("TMVA::SOFIE - InstanceNormalization - Tensor " + name + " not found.");
         }
      }

      fShape = model.GetTensorShape(fNInput);
      if (fShape.size() < 3) {
         throw std::runtime_error("TMVA::SOFIE - InstanceNormalization - Input tensor " + fNInput + " has rank " +
                                  std::to_string(fShape.size()) + " but at least rank 3 (N, C, D1, ...) is required.");
      }

      // scale and B are 1D tensors of length C
      const size_t channels = fShape[1];
      for (const std::string &name : {fNScale, fNBias}) {
         auto shape = model.GetTensorShape(name);
         if (shape.size() != 1 || shape[0] != channels) {
            throw std::runtime_error("TMVA::SOFIE - InstanceNormalization - Tensor " + name + " has invalid shape " +
                                     ConvertShapeToString(shape) + ", expected " +
                                     ConvertShapeToString(std::vector<size_t>{channels}) + ".");
         }
      }

      fType = ConvertTypeToString(model.GetTensorType(fNInput));
      model.AddIntermediateTensor(fNOutput, model.GetTensorType(fNInput), fShape);
   }

   std::string Generate(std::string) override
   {
      if (fShape.empty()) {
         throw std::runtime_error("TMVA::SOFIE - InstanceNormalization called to generate without being initialized.");
      }

      const size_t batchSize = fShape[0];
      const size_t channels = fShape[1];
      size_t spatialSize = 1;
      for (size_t i = 2; i < fShape.size(); ++i)
         spatialSize *= fShape[i];

      std::stringstream out;
      out << "\n" << SP << "//---- InstanceNormalization " << fNOutput << "\n";
      out << SP << "for (size_t n = 0; n < " << batchSize << "; n++) {\n";
      out << SP << SP << "for (size_t c = 0; c < " << channels << "; c++) {\n";
      out << SP << SP << SP << "const size_t offset = n * " << channels * spatialSize << " + c * " << spatialSize
          << ";\n";

      // Compute the mean over the spatial dimensions
      out << SP << SP << SP << fType << " mean = 0.;\n";
      out << SP << SP << SP << "for (size_t i = 0; i < " << spatialSize << "; i++) {\n";
      out << SP << SP << SP << SP << "mean += tensor_" << fNInput << "[offset + i];\n";
      out << SP << SP << SP << "}\n";
      out << SP << SP << SP << "mean /= " << fType << "(" << spatialSize << ");\n";

      // Compute the inverse standard deviation from the deviations around the mean
      out << SP << SP << SP << fType << " sum = 0.;\n";
      out << SP << SP << SP << "for (size_t i = 0; i < " << spatialSize << "; i++) {\n";
      out << SP << SP << SP << SP << fType << " tmp = tensor_" << fNInput << "[offset + i] - mean;\n";
      out << SP << SP << SP << SP << "sum += tmp * tmp;\n";
      out << SP << SP << SP << "}\n";
      out << SP << SP << SP << fType << " invStdDev = 1 / std::sqrt(sum / " << fType << "(" << spatialSize << ") + "
          << std::to_string(fEpsilon) << ");\n";

      // Y = scale o invStdDev (X - mean) + B
      out << SP << SP << SP << fType << " scale = tensor_" << fNScale << "[c];\n";
      out << SP << SP << SP << fType << " bias = tensor_" << fNBias << "[c];\n";
      out << SP << SP << SP << "for (size_t i = 0; i < " << spatialSize << "; i++) {\n";
      out << SP << SP << SP << SP << "tensor_" << fNOutput << "[offset + i] = scale * (tensor_" << fNInput
          << "[offset + i] - mean) * invStdDev + bias;\n";
      out << SP << SP << SP << "}\n";

      out << SP << SP << "}\n";
      out << SP << "}\n";
      return out.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
#endif
