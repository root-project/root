#ifndef TMVA_SOFIE_ROPERATOR_BatchNormalization
#define TMVA_SOFIE_ROPERATOR_BatchNormalization

#include "SOFIE_common.hxx"
#include "ROperator.hxx"
#include "RModel.hxx"


#include <cmath>
#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_BatchNormalization final : public ROperator
{

private:

   /* Attributes */
   float fepsilon = 1e-05;
   float fmomentum = 0.9;
   std::size_t ftraining_mode = 0;

   std::string fNX;
   std::string fNScale;
   std::string fNB;
   std::string fNMean;
   std::string fNVar;
   std::string fNY;
   EActivationType fActivation;
   std::string fNFusedScale;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;

   std::string fType;

public:
   ROperator_BatchNormalization() = delete;

   /* Constructor */
   ROperator_BatchNormalization( float epsilon, float momentum, std::size_t training_mode,
   std::string nameX, std::string nameScale, std::string nameB,
   std::string nameMean, std::string nameVar, std::string nameY, EActivationType activation=EActivationType::UNDEFINED):
   fepsilon(epsilon), fmomentum(momentum), ftraining_mode(training_mode),
   fNX(UTILITY::Clean_name(nameX)), fNScale(UTILITY::Clean_name(nameScale)),
   fNB(UTILITY::Clean_name(nameB)), fNMean(UTILITY::Clean_name(nameMean)),
   fNVar(UTILITY::Clean_name(nameVar)), fNY(UTILITY::Clean_name(nameY)), fActivation(activation)
   {
      fInputTensorNames = { fNX };
      fOutputTensorNames = { fNY };
      fNFusedScale = fNScale + "_fused_inv_std_dev";

      if(std::is_same<T, float>::value){
         fType = "float";
      }
      else{
	      throw
		      std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a BatchNormalization operator");
      }
   }


   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      ETensorType out = input[0];
      return {out};
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      if (input.size() != 5 ) {
         throw
         std::runtime_error("TMVA SOFIE BatchNormalization Op Shape inference need 5 input tensors");
      }
      for(size_t i = 0; i < input.size(); i++) {
         if (input[i].size() != 4) {
            throw
            std::runtime_error("TMVA SOFIE BatchNormalization Op Shape inference only accept tensor with 4 dimensions");
         }
      }

      auto ret = input;
      return ret;
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw
            std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNX + " fnx is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNScale)) {
	     throw
            std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNScale + " fns is not found in model");
      }
	  if (!model.CheckIfTensorAlreadyExist(fNB)) {
         throw
            std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNB + " fnb is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNMean)) {
         throw
            std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNMean + " fnm is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNVar)) {
         throw
            std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNVar + " fnv is not found in model");
      }

      fShapeX = model.GetDimTensorShape(fNX);

      if (fShapeX.size() <  2 || fShapeX.size() > 4) {
         throw
            std::runtime_error("TMVA SOFIE BatchNormalization Op input tensor " + fNX + " fnx has wrong shape : " + ConvertShapeToString(fShapeX));
      }

      fShapeY = fShapeX;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);

      auto original_S = model.GetInitializedTensorData(fNScale);
      auto original_V = model.GetInitializedTensorData(fNVar);

      auto shape_S = model.GetTensorShape(fNScale);
      if (shape_S.size() != 1) {
          throw std::runtime_error("TMVA SOFIE BatchNormalization 'scale' tensor must be 1D (per-channel).");
      }
      size_t channels = shape_S[0];

      if (fType == "float") {
         float *original_scale_ptr = static_cast<float *>(original_S.get());
         float *original_var_ptr = static_cast<float *>(original_V.get());
         float *fused_scale_data = new float[channels];

         for (size_t i = 0; i < channels; i++) {
            // Calculate scale * (1 / sqrt(variance + epsilon))
            fused_scale_data[i] = original_scale_ptr[i] / std::sqrt(original_var_ptr[i] + fepsilon);
         }

         std::shared_ptr<void> fused_scale_ptr(fused_scale_data, std::default_delete<float[]>());
         model.AddInitializedTensor(fNFusedScale, model.GetTensorType(fNScale), {channels}, fused_scale_ptr);
      }
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeX.empty()){
         throw std::runtime_error("TMVA SOFIE Batch Normalization called to Generate without being initialized first");
      }

      std::stringstream out;
      //// Batch Norm op
      auto batchSize = fShapeX[0].GetVal();
      auto channels = fShapeX[1].GetVal();
      std::string spatial_dim = "1";
      if (fShapeX.size() > 2) {
         auto spatialShape = fShapeX;
         spatialShape.erase(spatialShape.begin(), spatialShape.begin()+2);
         spatial_dim = ConvertDimShapeToLength( spatialShape);
      }

      out << "\n\n//---- BatchNorm" << (fActivation == EActivationType::RELU ? " + ReLU" : "") << "\n";
      out << SP << "{\n";
      out << SP << "   size_t i = 0;\n";
      out << SP << "   for (size_t n = 0; n < " << batchSize << "; ++n) {\n";
      out << SP << "      for (size_t c = 0; c < " << channels << "; ++c) {\n";
      out << SP << "         const float mean_val = tensor_" << fNMean << "[c];\n";
      out << SP << "         const float fused_scale_val = tensor_" << fNFusedScale << "[c];\n";
      out << SP << "         const float bias_val = tensor_" << fNB << "[c];\n";
      out << SP << "         for (size_t sp = 0; sp < " << spatial_dim << "; ++sp) {\n";
      out << SP << "            float val = (tensor_" << fNX << "[i] - mean_val) * fused_scale_val + bias_val;\n";

      if (fActivation == EActivationType::RELU) {
         out << SP << "            tensor_" << fNY << "[i] = (val > 0.0f) ? val : 0.0f;\n";
      } else {
         out << SP << "            tensor_" << fNY << "[i] = val;\n";
      }
      out << SP << "            i++;\n";
      out << SP << "         }\n";
      out << SP << "      }\n";
      out << SP << "   }\n";
      out << SP << "}\n";

      return out.str();
   }

   std::vector<std::string> GetBlasRoutines() override { return {}; }
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_BatchNormalization
