#ifndef TMVA_SOFIE_ROPERATOR_INSTANCENORMALIZATION
#define TMVA_SOFIE_ROPERATOR_INSTANCENORMALIZATION

#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"
#include "TMVA/SOFIE_common.hxx"
#include <sstream>
#include <cmath>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

/*! \class ROperator_InstanceNormalization
    \brief Implementation of the ONNX InstanceNormalization operator.

    Normalizes the input tensor across spatial dimensions independently for each
    feature channel and each instance in a batch.
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
      if (!model.CheckIfTensorAlreadyExist(fNInput)) {
         throw std::runtime_error("TMVA SOFIE InstanceNormalization Op: Input Tensor not found");
      }
      fShape = model.GetTensorShape(fNInput);
      model.AddIntermediateTensor(fNOutput, model.GetTensorType(fNInput), fShape);
   }

   std::string Generate(std::string) override
   {
      std::stringstream out;
      size_t rank = fShape.size();
      size_t N = fShape[0];
      size_t C = fShape[1];
      size_t spatial_size = 1;
      for (size_t i = 2; i < rank; ++i)
         spatial_size *= fShape[i];

      out << "\t// InstanceNormalization\n";
      out << "\t{\n";
      out << "\t\tconst float* src = tensor_" << fNInput << ";\n";
      out << "\t\tconst float* scale = tensor_" << fNScale << ";\n";
      out << "\t\tconst float* bias = tensor_" << fNBias << ";\n";
      out << "\t\tfloat* dst = tensor_" << fNOutput << ";\n";
      out << "\t\tfloat eps = " << fEpsilon << ";\n";

      out << "\t\tfor (size_t n = 0; n < " << N << "; ++n) {\n";
      out << "\t\t\tfor (size_t c = 0; c < " << C << "; ++c) {\n";
      out << "\t\t\t\tfloat sum = 0;\n";
      out << "\t\t\t\tfloat sq_sum = 0;\n";
      out << "\t\t\t\tconst float* img_ptr = src + n * " << C * spatial_size << " + c * " << spatial_size << ";\n";
      out << "\t\t\t\tfloat* dst_ptr = dst + n * " << C * spatial_size << " + c * " << spatial_size << ";\n";

      out << "\t\t\t\tfor (size_t i = 0; i < " << spatial_size << "; ++i) {\n";
      out << "\t\t\t\t\tfloat val = img_ptr[i];\n";
      out << "\t\t\t\t\tsum += val;\n";
      out << "\t\t\t\t\tsq_sum += val * val;\n";
      out << "\t\t\t\t}\n";

      out << "\t\t\t\tfloat mean = sum / " << spatial_size << ";\n";
      out << "\t\t\t\tfloat var = (sq_sum / " << spatial_size << ") - (mean * mean);\n";
      out << "\t\t\t\tfloat inv_std = 1.0f / std::sqrt(var + eps);\n";
      out << "\t\t\t\tfloat s = scale[c];\n";
      out << "\t\t\t\tfloat b = bias[c];\n";

      out << "\t\t\t\tfor (size_t i = 0; i < " << spatial_size << "; ++i) {\n";
      out << "\t\t\t\t\tdst_ptr[i] = s * (img_ptr[i] - mean) * inv_std + b;\n";
      out << "\t\t\t\t}\n";
      out << "\t\t\t}\n";
      out << "\t\t}\n";
      out << "\t}\n";
      return out.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
#endif
