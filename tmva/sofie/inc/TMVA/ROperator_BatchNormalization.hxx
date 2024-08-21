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
   bool fIsDynamic = false;

   std::string fNX;
   std::string fNScale;
   std::string fNB;
   std::string fNMean;
   std::string fNVar;
   std::string fNY;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeScale;
   std::vector<Dim> fShapeB;
   std::vector<Dim> fShapeMean;
   std::vector<Dim> fShapeVar;
   std::vector<Dim> fShapeY;

   std::string fType;

public:
   ROperator_BatchNormalization() = delete;

   /* Constructor */
   ROperator_BatchNormalization( float epsilon, float momentum, std::size_t training_mode,
   std::string nameX, std::string nameScale, std::string nameB,
   std::string nameMean, std::string nameVar, std::string nameY):
   fepsilon(epsilon), fmomentum(momentum), ftraining_mode(training_mode),
   fNX(UTILITY::Clean_name(nameX)), fNScale(UTILITY::Clean_name(nameScale)),
   fNB(UTILITY::Clean_name(nameB)), fNMean(UTILITY::Clean_name(nameMean)),
   fNVar(UTILITY::Clean_name(nameVar)), fNY(UTILITY::Clean_name(nameY))
   {
      if(std::is_same<T, float>::value){
      fType = "float";
      }
      else{
	      throw
		      std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a BatchNormalization operator");
      }
   }


   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) {
      ETensorType out = input[0];
      return {out};
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) {
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

   void Initialize(RModel& model){
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

      if (model.IsDynamicTensor(fNX) || model.IsInputTensor(fNX)) {
         fShapeX = model.GetDynamicTensorShape(fNX);
         fIsDynamic = true;
      } else {
         fShapeX = ConvertShapeToDim(model.GetTensorShape(fNX));
      }

      if (fShapeX.size() <  2 || fShapeX.size() > 4) {
         throw
            std::runtime_error("TMVA SOFIE BatchNormalization Op input tensor " + fNX + " fnx has wrong shape : " + ConvertDynamicShapeToString(fShapeX));
      }
      if (model.IsDynamicTensor(fNScale) || model.IsInputTensor(fNScale)) {
         fShapeScale = model.GetDynamicTensorShape(fNScale);
      } else {
         fShapeScale = ConvertShapeToDim(model.GetTensorShape(fNScale));
      }

      if (model.IsDynamicTensor(fNB) || model.IsInputTensor(fNB)) {
         fShapeB = model.GetDynamicTensorShape(fNB);
      } else {
         fShapeB = ConvertShapeToDim(model.GetTensorShape(fNB));
      }

      if (model.IsDynamicTensor(fNMean) || model.IsInputTensor(fNMean)) {
         fShapeMean = model.GetDynamicTensorShape(fNMean);
      } else {
         fShapeMean = ConvertShapeToDim(model.GetTensorShape(fNMean));
      }

      if (model.IsDynamicTensor(fNVar) || model.IsInputTensor(fNVar)) {
         fShapeVar = model.GetDynamicTensorShape(fNVar);
      } else {
         fShapeVar = ConvertShapeToDim(model.GetTensorShape(fNVar));
      }

      fShapeY = fShapeX;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);

      if (!fIsDynamic && fShapeB.size() == 1) {
      // Broadcast scale, bias, input_mean and input_var to shape_X
      auto original_B = model.GetInitializedTensorData(fNB);
      auto original_S = model.GetInitializedTensorData(fNScale);
      auto original_M = model.GetInitializedTensorData(fNMean);
      auto original_V = model.GetInitializedTensorData(fNVar);
      size_t batchSize = fShapeX[0].dim;
      size_t channels = fShapeX[1].dim;
      size_t height = (fShapeX.size() > 2) ? fShapeX[2].dim : 1;
      size_t width = (fShapeX.size() > 3) ? fShapeX[3].dim : 1;
      size_t n = batchSize * channels * height * width;
         if (fType == "float") {
            float *original_bias = static_cast<float*>(original_B.get());
            float *original_scale = static_cast<float*>(original_S.get());
            float *original_mean = static_cast<float*>(original_M.get());
            float *original_var = static_cast<float*>(original_V.get());
            float *new_bias = new float[n];
            float *new_scale = new float[n];
            float *new_mean = new float[n];
            float *new_var = new float[n];
            size_t bs = 0, ch = 0, h = 0, w = 0;
            for(ch=0; ch<channels; ch++){
    	       for(h=0; h<height; h++){
         	      for(w=0; w<width; w++){
			         new_bias[bs*channels*height*width + ch*height*width + h*width + w] = original_bias[ch];
			         new_scale[bs*channels*height*width + ch*height*width + h*width + w] = original_scale[ch];
			         new_mean[bs*channels*height*width + ch*height*width + h*width + w] = original_mean[ch];
			         new_var[bs*channels*height*width + ch*height*width + h*width + w] = original_var[ch];
         	      }
               }
            }
            size_t Batchoffset = channels*height*width;
            for(bs = 1; bs<batchSize; bs++){
               std::copy(new_bias, new_bias+Batchoffset, new_bias+(bs*Batchoffset));
               std::copy(new_scale, new_scale+Batchoffset, new_scale+(bs*Batchoffset));
               std::copy(new_mean, new_mean+Batchoffset, new_mean+(bs*Batchoffset));
               std::copy(new_var, new_var+Batchoffset, new_var+(bs*Batchoffset));
            }
            //// new_var =1. / sqrt(input_var + fepsilon)
		    for(size_t i=0; i<n; i++){
		       new_var[i] = 1./sqrt(new_var[i] + fepsilon);
		    }
            std::vector<size_t> new_bias_shape = {batchSize,channels,height,width};
            std::shared_ptr<void> new_bias_ptr(new_bias, std::default_delete<float[]>());
            std::shared_ptr<void> new_scale_ptr(new_scale, std::default_delete<float[]>());
            std::shared_ptr<void> new_mean_ptr(new_mean, std::default_delete<float[]>());
            std::shared_ptr<void> new_var_ptr(new_var, std::default_delete<float[]>());
            model.UpdateInitializedTensor(fNB, model.GetTensorType(fNB), new_bias_shape, new_bias_ptr);
            model.UpdateInitializedTensor(fNScale, model.GetTensorType(fNScale), new_bias_shape, new_scale_ptr);
            model.UpdateInitializedTensor(fNMean, model.GetTensorType(fNMean), new_bias_shape, new_mean_ptr);
            model.UpdateInitializedTensor(fNVar, model.GetTensorType(fNVar), new_bias_shape, new_var_ptr);
            fShapeB = model.GetDynamicTensorShape(fNB);
            fShapeScale = model.GetDynamicTensorShape(fNScale);
            fShapeMean = model.GetDynamicTensorShape(fNMean);
            fShapeVar = model.GetDynamicTensorShape(fNVar);
         }
      }
   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShapeX.empty()){
         throw std::runtime_error("TMVA SOFIE Batch Normalization called to Generate without being initialized first");
      }
      std::stringstream out;

      // Batch Norm op
      auto n = ConvertDynamicShapeToLength(fShapeX);
      if(fIsDynamic && fShapeB.size()==1) {
        out << SP << "const size_t batchSize = " << fShapeX[0].param << ";\n";
        out << SP << "const size_t channels = " << fShapeX[1].param << ";\n";
        out << SP << "const size_t height = " << (fShapeX.size() > 2 ? fShapeX[2].param : std::to_string(1)) << ";\n";
        out << SP << "const size_t width = " << (fShapeX.size() > 3 ? fShapeX[3].param : std::to_string(1)) << ";\n";
        out << SP << "const size_t n = batchSize * channels * height * width;\n";

        if (fType == "float") {
            out << SP << "float *new_bias = new float[n];\n";
            out << SP << "float *new_scale = new float[n];\n";
            out << SP << "float *new_mean = new float[n];\n";
            out << SP << "float *new_var = new float[n];\n";
            
            out << SP << "fTensor_" + fNMean + ".resize(n);\n";
            out << SP << "tensor_" << fNMean << " = fTensor_" << fNMean << ".data();\n";
            out << SP << "fTensor_" + fNB + ".resize(n);\n";
            out << SP << "tensor_" << fNB << " = fTensor_" << fNB << ".data();\n";
            out << SP << "fTensor_" + fNScale + ".resize(n);\n";
            out << SP << "tensor_" << fNScale << " = fTensor_" << fNScale << ".data();\n";
            out << SP << "fTensor_" + fNVar + ".resize(n);\n";
            out << SP << "tensor_" << fNVar << " = fTensor_" << fNVar << ".data();\n";

            out << SP << "for (size_t bs = 0; bs < batchSize; bs++) {\n";
            out << SP << SP << "for (size_t ch = 0; ch < channels; ch++) {\n";
            out << SP << SP << SP << "for (size_t h = 0; h < height; h++) {\n";
            out << SP << SP << SP << SP << "for (size_t w = 0; w < width; w++) {\n";
            out << SP << SP << SP << SP << SP << "new_bias[bs * channels * height * width + ch * height * width + h * width + w] = tensor_" << fNB << "[ch];\n";
            out << SP << SP << SP << SP << SP << "new_scale[bs * channels * height * width + ch * height * width + h * width + w] = tensor_" << fNScale << "[ch];\n";
            out << SP << SP << SP << SP << SP << "new_mean[bs * channels * height * width + ch * height * width + h * width + w] = tensor_" << fNMean << "[ch];\n";
            out << SP << SP << SP << SP << SP << "new_var[bs * channels * height * width + ch * height * width + h * width + w] = tensor_" << fNVar << "[ch];\n";
            out << SP << SP << SP << SP << "}\n";
            out << SP << SP << SP << "}\n";
            out << SP << SP << "}\n";
            out << SP << "}\n";
            out << SP << "size_t Batchoffset = channels * height * width;\n";
            out << SP << "for (size_t bs = 1; bs < batchSize; bs++) {\n";
            out << SP << SP << "std::copy(new_bias, new_bias + Batchoffset, new_bias + (bs * Batchoffset));\n";
            out << SP << SP << "std::copy(new_scale, new_scale + Batchoffset, new_scale + (bs * Batchoffset));\n";
            out << SP << SP << "std::copy(new_mean, new_mean + Batchoffset, new_mean + (bs * Batchoffset));\n";
            out << SP << SP << "std::copy(new_var, new_var + Batchoffset, new_var + (bs * Batchoffset));\n";
            out << SP << "}\n";
 
            out << SP << "for (size_t i = 0; i < n; i++) {\n";
            out << SP << SP << "new_var[i] = 1.0 / sqrt(new_var[i] + " + std::to_string(fepsilon) + ");\n";
            out << SP << "}\n";

            out << SP << "std::copy(new_bias, new_bias + n, tensor_" << fNB << ");\n";
            out << SP << "std::copy(new_scale, new_scale + n, tensor_" << fNScale << ");\n";
            out << SP << "std::copy(new_mean, new_mean + n, tensor_" << fNMean << ");\n";
            out << SP << "std::copy(new_var, new_var + n, tensor_" << fNVar << ");\n";
            out << SP << "delete[] new_bias;\n";
            out << SP << "delete[] new_scale;\n";
            out << SP << "delete[] new_mean;\n";
            out << SP << "delete[] new_var;\n";
        }
      }
      size_t batchSize = fShapeX[0].dim;
      size_t channels = fShapeX[1].dim;
      size_t height = (fShapeX.size() > 2) ? fShapeX[2].dim : 1;
      size_t width = (fShapeX.size() > 3) ? fShapeX[3].dim : 1;
      // size_t n = batchSize * channels * height * width;  

      //// copy X into Y
      if(fIsDynamic)
         out << SP << "const int " << OpName << "_N = batchSize * channels * height * width;\n";
      else
         out << SP << "constexpr int " << OpName << "_N =" << batchSize * channels * height * width << ";\n";
      out << SP << "constexpr int "<<OpName<< "_incx = 1;\n";
      out << SP << "constexpr int "<<OpName<< "_incy = 1;\n";

      out << SP << "BLAS::scopy_(&" << OpName << "_N, " << "tensor_" << fNX << ", &" << OpName << "_incx," << "tensor_" << fNY << ", &" << OpName << "_incy);\n\n";
      //// blas saxpy (Y = -Bmean + Y)

      out << SP << "float "<<OpName<< "_alpha = -1;\n";
      out << SP << SP << "BLAS::saxpy_(&" << OpName << "_N, &" << OpName << "_alpha, " << "tensor_" << fNMean << ", &" << OpName << "_incx,"
         << "tensor_" << fNY <<", &" << OpName << "_incy);\n\n ";
         
      //// Y *= scale*var 
      out << SP << "for (size_t i = 0; i < " << n << "; i++) {\n";
      out << SP << SP << "tensor_" << fNY << "[i] *= tensor_" << fNScale << "[i] * tensor_" << fNVar << "[i]; \n";
      out << SP << "}\n";

      //// blas saxpy (Y = Bbias + Y)
      out << SP <<OpName<< "_alpha = 1;\n";
      out << SP << "BLAS::saxpy_(&" << OpName << "_N, &" << OpName << "_alpha, " << "tensor_" << fNB << ", &" << OpName << "_incx, "
         << "tensor_" << fNY << ", &" << OpName << "_incy);\n\n";

      return out.str();
   }

   std::vector<std::string> GetStdLibs() override {
         return { std::string("cmath") };
   }

   std::vector<std::string> GetBlasRoutines() { return { std::string("Copy"), std::string("Axpy") }; }
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_BatchNormalization