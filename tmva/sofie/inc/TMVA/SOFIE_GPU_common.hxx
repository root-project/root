#ifndef TMVA_SOFIE_GPU_SOFIE_COMMON
#define TMVA_SOFIE_GPU_SOFIE_COMMON

// #include "TMVA/RTensor.hxx"
// #include "TMVA/Types.h"

#include <stdexcept>
#include <type_traits>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <regex>
#include <sstream>
#include <iostream>
#include <CL/sycl.hpp>


namespace TMVA{
namespace Experimental{
namespace SOFIE_GPU{


//typedef RTensor tensor_t;

enum class ETensorType{
   UNDEFINED = 0, FLOAT = 1, UNINT8 = 2, INT8 = 3, UINT16 = 4, INT16 = 5, INT32 = 6, INT64 = 7, STRING = 8, BOOL = 9, //order sensitive
    FLOAT16 = 10, DOUBLE = 11, UINT32 = 12, UINT64 = 13, COMPLEX64 = 14, COMPLEX28 = 15, BFLOAT16 = 16
};

typedef std::int64_t int_t;

std::string ConvertTypeToString(ETensorType type);
ETensorType ConvertStringToType(std::string type);

struct Dim{
   bool isParam = false;
   size_t dim;
   std::string param;
};

std::vector<Dim> ConvertShapeToDim(std::vector<size_t> shape);


struct InputTensorInfo{
   ETensorType type;
   std::vector<Dim> shape;
};

struct TensorInfo{
   ETensorType type;
   std::vector<size_t> shape;
};

std::size_t ConvertShapeToLength(std::vector<size_t> shape);

std::string ConvertShapeToString(std::vector<size_t> shape);

struct InitializedTensor{
   ETensorType fType;
   std::vector<std::size_t> fShape;
   std::shared_ptr<void> fData;     //! Transient
   int fSize=1;
   char* fPersistentData=nullptr;   //[fSize] Persistent

   void CastSharedToPersistent(){
      for(auto item:fShape){
         fSize*=(int)item;
      }
      switch(fType){
         case ETensorType::FLOAT: fSize*=sizeof(float); break;
         case ETensorType::DOUBLE: fSize*=sizeof(double); break;
         case ETensorType::INT32: fSize*=sizeof(int32_t); break;
         case ETensorType::INT64: fSize*=sizeof(int64_t); break;
         default:
          throw std::runtime_error("TMVA::SOFIE doesn't yet supports serialising data-type " + ConvertTypeToString(fType));
      }
      fPersistentData=(char*)fData.get();
   }
   void CastPersistentToShared(){
      switch (fType) {
      case ETensorType::FLOAT: {
          std::shared_ptr<void> tData(malloc(fSize * sizeof(float)), free);
          std::memcpy(tData.get(), fPersistentData, fSize * sizeof(float));
          fData = tData;
          break;
      }
      case ETensorType::DOUBLE: {
          std::shared_ptr<void> tData(malloc(fSize * sizeof(double)), free);
          std::memcpy(tData.get(), fPersistentData, fSize * sizeof(double));
          fData = tData;
          break;
      }
      case ETensorType::INT32: {
          std::shared_ptr<void> tData(malloc(fSize * sizeof(int32_t)), free);
          std::memcpy(tData.get(), fPersistentData, fSize * sizeof(int32_t));
          fData = tData;
          break;
      }
      case ETensorType::INT64: {
          std::shared_ptr<void> tData(malloc(fSize * sizeof(int64_t)), free);
          std::memcpy(tData.get(), fPersistentData, fSize * sizeof(int64_t));
          fData = tData;
          break;
      }
      default: {
          throw std::runtime_error("TMVA::SOFIE doesn't yet supports serialising data-type " +
                                   ConvertTypeToString(fType));
      }
      }
   }
};

template <typename T>
ETensorType GetTemplatedType(T /*obj*/ ){
   if (std::is_same<T, float>::value) return ETensorType::FLOAT;
   if (std::is_same<T, uint8_t>::value) return ETensorType::UNINT8;
   if (std::is_same<T, int8_t>::value) return ETensorType::INT8;
   if (std::is_same<T, uint16_t>::value) return ETensorType::UINT16;
   if (std::is_same<T, int16_t>::value) return ETensorType::INT16;
   if (std::is_same<T, int32_t>::value) return ETensorType::INT32;
   if (std::is_same<T, int64_t>::value) return ETensorType::INT64;
   if (std::is_same<T, std::string>::value) return ETensorType::STRING;
   if (std::is_same<T, bool>::value) return ETensorType::BOOL;
   //float16 unimplemented
   if (std::is_same<T, double>::value) return ETensorType::DOUBLE;
   if (std::is_same<T, uint32_t>::value) return ETensorType::UINT32;
   if (std::is_same<T, uint64_t>::value) return ETensorType::UINT64;
   //complex 64, 28, bfloat 16 unimplemented
}

namespace UTILITY{
// Check if two shapes are equal
bool AreSameShape(const std::vector<size_t>&, const std::vector<size_t>&);

// Multidirectional broadcast a list of tensors to the same shape
std::vector<size_t> MultidirectionalBroadcastShape(std::vector<std::vector<size_t>>);

// Unidirectional broadcast two shapes to the same shape
std::vector<size_t> UnidirectionalBroadcastShape(std::vector<size_t>, std::vector<size_t>);

std::string Clean_name(std::string input_tensor_name);

template<typename T>
T* BroadcastConvBias(const T* data, const size_t channel, const std::vector<size_t>& targetShape) {
   size_t size = targetShape.size();
   if (targetShape[1] != channel) {
      std::stringstream ss;
      ss << "TMVA::SOFIE - Error broadcasting Conv Bias of shape {";
      ss << std::to_string(channel);
      ss << "} to ";
      ss << ConvertShapeToString(targetShape);
      throw
         std::runtime_error(ss.str());
   }

   size_t targetLength = ConvertShapeToLength(targetShape);
   T* newData = new T[targetLength];

   if (targetLength == channel) {
      std::copy(data, data + channel, newData);
      return newData;
   }

   // cStride = OutDepth * outHeight * outWidth
   size_t cStride = 1;
   for (size_t i = 2; i < size; i++)
      cStride *= targetShape[i];
   // Broadcast each element of the bias to a vector of size cStride and concatenate them
   // into a vector of size channel * cStride
   for (size_t i = 0; i < channel; i++) {
      std::fill(newData + i * cStride, newData + (i + 1) * cStride, data[i]);
   }
   // Broadcast newData[0...channel * cStride) to newData[0...batch * channel * cStride)
   size_t batch = targetShape[0];
   size_t bStride = channel * cStride;
   for (size_t i = 1; i < batch; i++) {
      std::copy(newData, newData + bStride, newData + i * bStride);
   }
   return newData;
}

// Broadcast a tensor from shape to targetShape according to numpy broadcasting rules
// See more at https://numpy.org/doc/stable/user/basics.broadcasting.html
// and https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md .
template<typename T>
T* BroadcastTensor(const T* data, const std::vector<size_t>& shape, const std::vector<size_t>& targetShape) {
   // Size of the shapes
   size_t size = shape.size();
   // Current length of the broadcasted tensor
   size_t curLength = ConvertShapeToLength(shape);
   size_t targetLength = ConvertShapeToLength(targetShape);
   // newShape is an aray of size equal to dimension along which we are broadcasting the tensor
   T* broadcastedData = new T[targetLength];
   std::copy(data, data + curLength, broadcastedData);
   // Product of the previous dimensions of targetShape
   size_t arrayNum = 1;
   // New broadcasted data
   std::vector<T> newData(targetLength);

   for (size_t idx = 0; idx < size; idx++) {
      size_t dim = shape[idx];
      size_t targetDim = targetShape[idx];
      if (dim == 1 && targetDim > 1) {
         // Set the new length of the data
         size_t newLength = curLength * targetDim;
         // View the data as a list of arrayNum arrays of size arrayLength
         size_t arrayLength = curLength / arrayNum;
         // Broadcast each array dim times
         if (arrayLength > 1) {
            // If each array has at least two elements
            for (size_t arrayIdx = 0; arrayIdx < arrayNum; arrayIdx++) {
               for (size_t targetIdx = 0; targetIdx < targetDim; targetIdx++) {
                  size_t offset = arrayIdx * arrayLength * targetDim + targetIdx * arrayLength;
                  std::copy(broadcastedData + arrayIdx * arrayLength,
                     broadcastedData + (arrayIdx + 1) * arrayLength,
                     newData.begin() + offset);
               }
            }
         } else {
            // If each array has one element
            for (size_t arrayIdx = 0; arrayIdx < arrayNum; arrayIdx++) {
               std::fill(newData.begin() + arrayIdx * targetDim,
                  newData.begin() + (arrayIdx + 1) * targetDim, broadcastedData[arrayIdx]);
            }
         }
         // Update current length
         curLength = newLength;
         // Update broadcasted data
         std::copy(newData.begin(), newData.begin() + newLength, broadcastedData);
      }
      // Update the number of arrays
      arrayNum *= targetDim;
   }
   return broadcastedData;
}

// Unidirectional broadcasting shape to targetShape
template<typename T>
T* UnidirectionalBroadcast(const T* data, const std::vector<size_t>& shape, const std::vector<size_t>& targetShape) {
   // Prepend shape with ones
   if (shape.size() < targetShape.size()) {
      size_t targetSize = targetShape.size();
      std::vector<size_t> newShape(targetSize, 1);
      size_t offset = targetSize - shape.size();
      std::copy(shape.begin(), shape.end(), newShape.begin() + offset);
      return BroadcastTensor<T>(data, newShape, targetShape);
   }
   return BroadcastTensor<T>(data, shape, targetShape);
}

/// compute stride of a tensor given its shape (assume layout is row-major)
std::vector<size_t> ComputeStrideFromShape(const std::vector<size_t> & shape);

/// function to check if a >> 0 and a < MAX using a single comparison
//// use trick casting to unsigned values so it becomes a single comparison
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
   return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}


/// im2col : efficient function to re-arrange input data of convolution to a matrix
/// that can be used by BLAS
/// Use trick to loop on each element of filtered region first and follow input data layout
/// By doing this reads and writes are of consecutive data in memory and one gains in efficiency
/// The resulting matrix will be already transposed and can be used directly in BLAS
/// since output will be a matrix : (channels*kernel_h*kernel_w , output_h*output_w)
/// Example: with an input matrix
///    a1 a2 a3
///    b1 b2 b3    and a 2x2 kernel    (k1,k2,k3,k4) and padding 1 :
///    c1 c2 c3
///     outpout will be a matrix (4 x 16)
///  the routine will follow output order :
//     first all elements which will be operated by k1 then k2 then k3
///  -> ( 0  0  0  0  0  a1 a2 a3 0  b1 b2 b3  0 c1 c2 c3  )    all elements for k1
///     ( 0  0  0  0  a1 a2 a3  0 b1 b2 b3  0 c1 c2 c3  0  )     for k2
///     ( 0  a1 a2 a3 0  b1 b2 b3 0  c1 c2 c3  0  0  0  0  )     for k3
///     ( a1 a2 a3 0  b1 b2 b3  0 c1 c2 c3  0  0  0  0  0  )     for k4
///

template <typename T, int dims>
void Im2col(cl::sycl::queue q, cl::sycl::buffer<T, dims> data_im, const int channels, const int height, const int width,
                        const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, 
                        const int stride_h, const int stride_w, const int dilation_h, const int dilation_w, cl::sycl::buffer<T, dims> data_col)
{
   const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
   const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
   const int channel_size = height * width;

   q.submit([&](cl::sycl::handler& cgh){
      auto acc_data_im = cl::sycl::accessor{data_im, cgh, cl::sycl::read_only};
      auto acc_data_col = cl::sycl::accessor{data_col, cgh, cl::sycl::write_only, cl::sycl::no_init};

      cgh.parallel_for<class im2col>(cl::sycl::range{static_cast<size_t>(channels * output_h * output_w)}, [=](cl::sycl::id<1> id){
         int w_out = id % output_w;
         int idx = id / output_w;

         int h_out = idx % output_h;
         int channel_in = idx / output_h;

         int channel_out = channel_in * kernel_h * kernel_w;
         int h_in = h_out * stride_h - pad_h;
         int w_in = w_out * stride_w - pad_w;

         int dest = (channel_out * output_h + h_out) * output_w + w_out;
         const int src = (channel_in * height + h_in) * width + w_in;

         for (int i=0; i<kernel_h; i++) {
            for (int j=0; j<kernel_w; j++) {
               int h = h_in + i * dilation_h;
               int w = w_in + j * dilation_w;

               if ( (h >= 0) && (w >= 0) && (h < height) && (w < width) ) {
                  acc_data_col[dest] = acc_data_im[src + i*dilation_h*width + j * dilation_w];
               }
               else {
                  acc_data_col[dest] = 0;
               }

               dest += output_h * output_w;
            }
         }
      });
   });
}

template <typename T, int dims>
void Im2col_3d(cl::sycl::queue q, cl::sycl::buffer<T, dims> data_im, const int channels, 
                     const int depth, const int height, const int width,
                     const int kernel_d, const int kernel_h, const int kernel_w, 
                     const int pad_d, const int pad_h, const int pad_w, 
                     const int stride_d, const int stride_h, const int stride_w, 
                     const int dilation_d, const int dilation_h, const int dilation_w, 
                     cl::sycl::buffer<T, dims> data_col)
{
   const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
   const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
   const int output_d = (depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1;
   const int channel_size = height * width * depth;

   q.submit([&](cl::sycl::handler& cgh){
      auto acc_data_im = cl::sycl::accessor{data_im, cgh, cl::sycl::read_only};
      auto acc_data_col = cl::sycl::accessor{data_col, cgh, cl::sycl::write_only, cl::sycl::no_init};

      cgh.parallel_for<class im2col_3d>(cl::sycl::range{static_cast<size_t>(channels * output_h * output_w * output_d)}, [=](cl::sycl::id<1> id){
         int w_out = id % output_w;
         id /= output_w;

         int h_out = id % output_h;
         id /= output_h;

         int d_out = id % output_d;
         int channel_in = id / output_d;
   
         int channel_out = channel_in * kernel_d * kernel_h * kernel_w;
         int d_in = d_out * stride_d - pad_d;
         int h_in = h_out * stride_h - pad_h;
         int w_in = w_out * stride_w - pad_w;

         int dest = ((channel_out * output_d + d_out) * output_h + h_out) *
                  output_w + w_out;
         const int src = ((channel_in * depth + d_in) * height + h_in) * 
                  width  + w_in;
         
         for (int i=0; i < kernel_d; i++) {
            for (int j=0; j < kernel_h; j++) {
               for (int k=0; k < kernel_w; k++) {
                  int d = d_in + i * dilation_d;
                  int h = h_in + j * dilation_h;
                  int w = w_in + k * dilation_w;

                  if ( (d >= 0) && (h >= 0) && (w >= 0) && (d < depth) && (h < height) && (w < width)) {
                     acc_data_col[dest] = acc_data_im[src + i * dilation_d * height * width + 
                     j * dilation_h * width + k * dilation_w];
                  }
                  else {
                     acc_data_col[dest] = 0;
                  }
                  dest += output_d * output_h * output_w;
               }
            }
         }
      });
   });
}

template <typename T, int dims>
void col2im(cl::sycl::queue q, cl::sycl::buffer<T, dims> data_im, const int channels, const int height, const int width,
                        const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, 
                        const int stride_h, const int stride_w, const int dilation_h, const int dilation_w, cl::sycl::buffer<T, dims> data_col)
{
   const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
   const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
   const int channel_size = height * width;

   q.submit([&](cl::sycl::handler& cgh){
      auto acc_data_im = cl::sycl::accessor{data_im, cgh, cl::sycl::write_only};
      auto acc_data_col = cl::sycl::accessor{data_col, cgh, cl::sycl::read_only};

      cgh.parallel_for<class im2col>(cl::sycl::range{static_cast<size_t>(channels * output_h * output_w)}, [=](cl::sycl::id<1> id){
         int w_out = id % output_w;
         int idx = id / output_w;

         int h_out = idx % output_h;
         int channel_in = idx / output_h;

         int channel_out = channel_in * kernel_h * kernel_w;
         int h_in = h_out * stride_h - pad_h;
         int w_in = w_out * stride_w - pad_w;

         int dest = (channel_out * output_h + h_out) * output_w + w_out;
         const int src = (channel_in * height + h_in) * width + w_in;

         for (int i=0; i<kernel_h; i++) {
            for (int j=0; j<kernel_w; j++) {
               int h = h_in + i * dilation_h;
               int w = w_in + j * dilation_w;

               if ( (h >= 0) && (w >= 0) && (h < height) && (w < width) ) {
                  acc_data_im[src + i*dilation_h*width + j * dilation_w] = acc_data_col[dest];
               }
               else {
                  dest += output_w;
               }

               dest += output_h * output_w;
            }
         }
      });
   });
}




}  // end namespace UTILITY

namespace BLAS{
extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
                       const float * beta, float * C, const int * ldc);
}//BLAS
}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RMODEL
