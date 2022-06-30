#ifndef TMVA_SOFIE_SOFIE_COMMON
#define TMVA_SOFIE_SOFIE_COMMON

// #include "TMVA/RTensor.hxx"
// #include "TMVA/Types.h"

#include <type_traits>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <regex>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

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
         default:
          throw std::runtime_error("TMVA::SOFIE doesn't yet supports serialising data-type " + ConvertTypeToString(fType));
      }
      fPersistentData=(char*)fData.get();
   }
   void CastPersistentToShared(){
     switch(fType){
       case ETensorType::FLOAT: {
      std::shared_ptr<void> tData(malloc(fSize * sizeof(float)), free);
      std::memcpy(tData.get(), fPersistentData,fSize * sizeof(float));
      fData=tData;
      break;
      }
      default: {
          throw std::runtime_error("TMVA::SOFIE doesn't yet supports serialising data-type " + ConvertTypeToString(fType));
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
template<typename T>
T* Unidirectional_broadcast(const T* original_data, const std::vector<size_t> original_shape, const std::vector<size_t> target_shape);
std::vector<size_t> Multidirectional_broadcast(std::vector<size_t> input1_shape, std::vector<size_t> input2_shape);
std::string Clean_name(std::string input_tensor_name);


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

template <typename T>
void Im2col(const T *data_im, const int channels, const int height, const int width, const int kernel_h,
                const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                const int dilation_h, const int dilation_w, T *data_col)
{
   const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
   const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
   const int channel_size = height * width;
   for (int channel = channels; channel--; data_im += channel_size) {
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
         for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
            int input_row = -pad_h + kernel_row * dilation_h;
            for (int output_rows = output_h; output_rows; output_rows--) {
               if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                  for (int output_cols = output_w; output_cols; output_cols--) {
                     *(data_col++) = 0;
                  }
               } else {
                  int input_col = -pad_w + kernel_col * dilation_w;
                  for (int output_col = output_w; output_col; output_col--) {
                     if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                        *(data_col++) = data_im[input_row * width + input_col];
                     } else {
                        *(data_col++) = 0;
                     }
                     input_col += stride_w;
                  }
               }
               input_row += stride_h;
            }
         }
      }
   }
}

/// 3d implementation
template <typename T>
void Im2col_3d(const T *data_im, const int channels, 
            const int depth, const int height, const int width, 
            const int kernel_d, const int kernel_h, const int kernel_w, 
            const int pad_d, const int pad_h, const int pad_w, 
            const int stride_d, const int stride_h, const int stride_w, 
            const int dilation_d, const int dilation_h,  const int dilation_w, T *data_col)
{
   const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
   const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
   const int output_d = (depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1;
   const int channel_size = height * width * depth;
   // assume data are c x d x h x w
   for (int channel = channels; channel--; data_im += channel_size) {
      for (int kernel_depth = 0; kernel_depth < kernel_d; kernel_depth++) {
         for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
               int input_dep = -pad_d + kernel_depth * dilation_d;
               for (int output_dep = output_d; output_dep; output_dep--) {
                  if (!is_a_ge_zero_and_a_lt_b(input_dep, depth)) {
                     for (int output_rows = output_h; output_rows; output_rows--) {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                           *(data_col++) = 0;
                        }
                     }
                  } else {
                     int input_row = -pad_h + kernel_row * dilation_h;
                     for (int output_rows = output_h; output_rows; output_rows--) {
                        if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                           for (int output_cols = output_w; output_cols; output_cols--) {
                              *(data_col++) = 0;
                           }  
                        } else {
                           int input_col = -pad_w + kernel_col * dilation_w;
                           for (int output_col = output_w; output_col; output_col--) {
                              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                 *(data_col++) = data_im[input_dep * width * height + input_row * width + input_col];
                              } else {
                                 *(data_col++) = 0;
                              }
                              input_col += stride_w;
                           }
                        }
                        input_row += stride_h;
                     }
                  }
                  input_dep += stride_d;
               }
            }
         }
      }
   }
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
