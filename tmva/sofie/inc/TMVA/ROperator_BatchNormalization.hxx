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

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeScale;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeMean;
   std::vector<size_t> fShapeVar;
   std::vector<size_t> fShapeY;

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

      fShapeX = model.GetTensorShape(fNX);

      if (fShapeX.size() <  2 || fShapeX.size() > 4) {
         throw
            std::runtime_error("TMVA SOFIE BatchNormalization Op input tensor " + fNX + " fnx has wrong shape : " + ConvertShapeToString(fShapeX));
      }

      fShapeScale = model.GetTensorShape(fNScale);
      fShapeB = model.GetTensorShape(fNB);
      fShapeMean = model.GetTensorShape(fNMean);
      fShapeVar = model.GetTensorShape(fNVar);
      fShapeY = fShapeX;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);

      if (fShapeB.size() == 1) {
         // Broadcast scale, bias, input_mean and input_var to shape_X
         auto original_B = model.GetInitializedTensorData(fNB);
         auto original_S = model.GetInitializedTensorData(fNScale);
         auto original_M = model.GetInitializedTensorData(fNMean);
         auto original_V = model.GetInitializedTensorData(fNVar);
         size_t batchSize = fShapeX[0];
         size_t channels = fShapeX[1];
         size_t height = (fShapeX.size() > 2) ? fShapeX[2] : 1;
         size_t width = (fShapeX.size() > 3) ? fShapeX[3] : 1;
         size_t n = batchSize * channels * height * width;
         if (fType == "float") {
            float *original_bias = static_cast<float *>(original_B.get());
            float *original_scale = static_cast<float *>(original_S.get());
            float *original_mean = static_cast<float *>(original_M.get());
            float *original_var = static_cast<float *>(original_V.get());
            float *new_bias = new float[n];
            float *new_scale = new float[n];
            float *new_mean = new float[n];
            float *new_var = new float[n];
            size_t bs = 0, ch = 0, h = 0, w = 0;
            for (ch = 0; ch < channels; ch++) {
               for (h = 0; h < height; h++) {
                  for (w = 0; w < width; w++) {
                     new_bias[bs * channels * height * width + ch * height * width + h * width + w] = original_bias[ch];
                     new_scale[bs * channels * height * width + ch * height * width + h * width + w] =
                        original_scale[ch];
                     new_mean[bs * channels * height * width + ch * height * width + h * width + w] = original_mean[ch];
                     new_var[bs * channels * height * width + ch * height * width + h * width + w] = original_var[ch];
                  }
               }
            }
            size_t Batchoffset = channels * height * width;
            for (bs = 1; bs < batchSize; bs++) {
               std::copy(new_bias, new_bias + Batchoffset, new_bias + (bs * Batchoffset));
               std::copy(new_scale, new_scale + Batchoffset, new_scale + (bs * Batchoffset));
               std::copy(new_mean, new_mean + Batchoffset, new_mean + (bs * Batchoffset));
               std::copy(new_var, new_var + Batchoffset, new_var + (bs * Batchoffset));
            }
            //// new_var =1. / sqrt(input_var + fepsilon)
            for (size_t i = 0; i < n; i++) {
               new_var[i] = 1. / sqrt(new_var[i] + fepsilon);
               new_scale[i] *= new_var[i]; // include var in new scale
            }
            std::vector<size_t> new_bias_shape = {batchSize, channels, height, width};
            std::shared_ptr<void> new_bias_ptr(new_bias, std::default_delete<float[]>());
            std::shared_ptr<void> new_scale_ptr(new_scale, std::default_delete<float[]>());
            std::shared_ptr<void> new_mean_ptr(new_mean, std::default_delete<float[]>());
            std::shared_ptr<void> new_var_ptr(new_var, std::default_delete<float[]>());
            model.UpdateInitializedTensor(fNB, model.GetTensorType(fNB), new_bias_shape, new_bias_ptr);
            model.UpdateInitializedTensor(fNScale, model.GetTensorType(fNScale), new_bias_shape, new_scale_ptr);
            model.UpdateInitializedTensor(fNMean, model.GetTensorType(fNMean), new_bias_shape, new_mean_ptr);
            model.UpdateInitializedTensor(fNVar, model.GetTensorType(fNVar), new_bias_shape, new_var_ptr);
            fShapeB = model.GetTensorShape(fNB);
            fShapeScale = model.GetTensorShape(fNScale);
            fShapeMean = model.GetTensorShape(fNMean);
            fShapeVar = model.GetTensorShape(fNVar);
         }
      }
   }

   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShapeX.empty()){
         throw std::runtime_error("TMVA SOFIE Batch Normalization called to Generate without being initialized first");
      }

      std::stringstream out;
      //// Batch Norm op
      size_t batchSize = fShapeX[0];
      size_t channels = fShapeX[1];
      size_t height = (fShapeX.size() > 2) ? fShapeX[2] : 1;
      size_t width = (fShapeX.size() > 3) ? fShapeX[3] : 1;
      size_t n = batchSize * channels * height * width;

      //// copy X into Y
      out << SP << "constexpr int " << OpName << "_N =" << batchSize * channels * height * width << ";\n";
      out << SP << "constexpr int "<<OpName<< "_incx = 1;\n";
      out << SP << "constexpr int "<<OpName<< "_incy = 1;\n";
      out << SP << "BLAS::scopy_(&" << OpName << "_N, " << "tensor_" << fNX << ", &" << OpName << "_incx," << "tensor_" << fNY << ", &" << OpName << "_incy);\n\n";

      //// blas saxpy (Y = -Bmean + Y)
      out << SP << "float "<<OpName<< "_alpha = -1;\n";
      out << SP << "BLAS::saxpy_(&" << OpName << "_N, &" << OpName << "_alpha, " << "tensor_" << fNMean << ", &" << OpName << "_incx,"
         << "tensor_" << fNY <<", &" << OpName << "_incy);\n\n ";

      //// Y *= scale*var
      out << SP << "for (size_t i = 0; i < " << n << "; i++) {\n";
      // scale tensor contains already the var
      out << SP << SP << "tensor_" << fNY << "[i] *= tensor_" << fNScale << "[i]; \n";
      out << SP << "}\n";

      //// blas saxpy (Y = Bbias + Y)
      out << SP <<OpName<< "_alpha = 1;\n";
      out << SP << "BLAS::saxpy_(&" << OpName << "_N, &" << OpName << "_alpha, " << "tensor_" << fNB << ", &" << OpName << "_incx, "
         << "tensor_" << fNY << ", &" << OpName << "_incy);\n\n";

      return out.str();
   }

   std::string GenerateGPU(std::string OpName, std::string gemm, std::string copy, 
   std::string axpy, std::string transpose, std::string nontrans, std::string trans, std::string copy_batch, std::string scal) {
      OpName = "op_" + OpName;
      

      if (fShapeX.empty()){
         throw std::runtime_error("TMVA SOFIE Batch Normalization called to Generate without being initialized first");
      }

      std::stringstream out;
      //// Batch Norm op
      size_t batchSize = fShapeX[0];
      size_t channels = fShapeX[1];
      size_t height = (fShapeX.size() > 2) ? fShapeX[2] : 1;
      size_t width = (fShapeX.size() > 3) ? fShapeX[3] : 1;
      size_t n = batchSize * channels * height * width;

      // Copy X into Y
      out << SP*3 << "constexpr int " << OpName << "_N =" << batchSize * channels * height * width << ";\n";
      out << SP*3 << "constexpr int "<<OpName<< "_incx = 1;\n";
      out << SP*3 << "constexpr int "<<OpName<< "_incy = 1;\n";

      if (gpu_blas == MKLBLAS) {
         out << SP*3 << copy << OpName << "_N, " << "buf_tensor_" << fNX;
         out << ", " << OpName << "_incx, buf_tensor_" << fNY << ", " << OpName << "_incy);\n\n";
      }
      else {
         out << SP*3 << copy << OpName << "_N, " << "blas::BufferIterator(buf_tensor_" << fNX;
         out << "), " << OpName << "_incx, blas::BufferIterator(buf_tensor_" << fNY << "), " << OpName << "_incy);\n\n";
      }

      // blas axpy (Y = -Bmean + Y = X - Bmean)
      out << SP*3 << "float "<<OpName<< "_alpha = -1;\n"; 
      if (gpu_blas == MKLBLAS) {
         out << SP*3 << axpy << OpName << "_N, " << OpName << "_alpha, buf_tensor_" << fNMean;
         out << ", " << OpName << "_incx, buf_tensor_" << fNY << ", " << OpName << "_incy);\n\n";
      }
      else {
         out << SP*3 << axpy << OpName << "_N, " << OpName << "_alpha, blas::BufferIterator(buf_tensor_" << fNMean;
         out << "), " << OpName << "_incx, blas::BufferIterator(buf_tensor_" << fNY << "), " << OpName << "_incy);\n\n";
      }

      out << SP*3 << "q.submit([&](cl::sycl::handler& cgh){\n";
      out << SP*4 << "auto acc_tensor_" << fNVar << " = cl::sycl::accessor{buf_tensor_" << fNVar;
      out << ", cgh, cl::sycl::read_only};\n";
      out << SP*4 << "auto acc_tensor_" << fNScale << " = cl::sycl::accessor{buf_tensor_" << fNScale;
      out << ", cgh, cl::sycl::read_only};\n";
      out << SP*4 << "auto acc_tensor_" << fNY << " = cl::sycl::accessor{buf_tensor_" << fNY;
      out << ", cgh, cl::sycl::read_write};\n";

      out << SP*4 << "cgh.parallel_for<class " << OpName << ">(cl::sycl::range<1>(" << n;
      out << "), [=](cl::sycl::id<1> id){\n";
      out << SP*5 << "acc_tensor_" << fNY << "[id] *= acc_tensor_" << fNVar << "[id] * acc_tensor_" << fNScale << "[id];\n";
      out << SP*4 << "});\n";
      out << SP*3 << "});\n";

      // blas axpy Y = Bbias + Y = (X - Bmean) * Scale * Var + Bbias
      out << SP*3 << OpName << "_alpha = 1;\n";

      if (gpu_blas == MKLBLAS) {
         out << SP*3 << axpy << OpName << "_N, " << OpName << "_alpha, ";
         out << "buf_tensor_" << fNB << ", " << OpName << "_incx, buf_tensor_" << fNY << ", " << OpName << "_incy);\n\n";
      }
      else {
         out << SP*3 << axpy << OpName << "_N, " << OpName << "_alpha, ";
         out << "blas::BufferIterator(buf_tensor_" << fNB << "), " << OpName << "_incx, blas::BufferIterator(buf_tensor_" << fNY << "), " << OpName << "_incy);\n\n";
      }
      
      return out.str();
   }

   std::vector<std::string> GetBlasRoutines() { return { std::string("Copy"), std::string("Axpy") }; }
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_BatchNormalization
