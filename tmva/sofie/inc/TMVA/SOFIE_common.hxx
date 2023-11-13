#ifndef TMVA_SOFIE_SOFIE_COMMON
#define TMVA_SOFIE_SOFIE_COMMON

#include "TMVA/RTensor.hxx"

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

template <typename Dtype>
void col2im(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_im) {
   // note that output data_im needs to be set to zero value!!!!
   std::fill(data_im, data_im + height * width * channels, 0.);
  //caffe_set(height * width * channels, Dtype(0), data_im);
  // data_im must be a zero vector
  //const Dtype * data_col_0 = data_col;
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                //assert(input_row*width+input_col < height * width * channels);
                //assert(data_col - data_col_0 < output_h*output_w*channels);
               //  std::cout << "COL2IM: input_row" << "  " << input_row << "  " << input_col
               //       << " <---- " << data_col - data_col_0 << " values:  "
               //       << data_im[input_row * width + input_col] << " <--- " << *data_col << std::endl;
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
  //std::cout << "finishing col2imp" << std::endl;
}



}  // end namespace UTILITY

namespace BLAS{
extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
                       const float * beta, float * C, const int * ldc);
}//BLAS


struct GNN_Data {
      RTensor<float> node_data;
      RTensor<float> edge_data;
      RTensor<float> global_data;

      std::vector<int> receivers;
      std::vector<int> senders;

      // need to have default constructor since RTensor has not one
      GNN_Data(): node_data(RTensor<float>({})), edge_data(RTensor<float>({})), global_data(RTensor<float>({})) {}

};

template<typename T>
TMVA::Experimental::RTensor<T> Concatenate( TMVA::Experimental::RTensor<T> & t1,  TMVA::Experimental::RTensor<T> & t2, int axis = 0)
{
   // concatenate tensor along axis. Shape must be the same except in the dimension of the concatenated axis
   if (t1.GetMemoryLayout() != t2.GetMemoryLayout())
      throw std::runtime_error("TMVA RTensor Concatenate - tensors have different memory layout");
   auto & shape1 = t1.GetShape();
   auto & shape2 = t2.GetShape();
   if (t1.GetSize()/shape1[axis] != t2.GetSize()/shape2[axis]) {
      std::cout << "axis " << axis << " sizes " << t1.GetSize() << " " << t2.GetSize() << "  ";
      std::cout << "shape 1 : " << ConvertShapeToString(t1.GetShape());
      std::cout << " shape 2 : " << ConvertShapeToString(t2.GetShape()) << std::endl;
      throw std::runtime_error("TMVA RTensor Concatenate - tensors have incompatible shapes");
   }
   std::vector<size_t> outShape = shape1;
   outShape[axis] = shape1[axis] + shape2[axis];
   TMVA::Experimental::RTensor<T> tout(outShape, t1.GetMemoryLayout());
   if (t1.GetMemoryLayout() == TMVA::Experimental::MemoryLayout::ColumnMajor) {
      throw std::runtime_error("TMVA RTensor Concatenate is not yet supported for column major tensors");
   }

   auto & stride1 = t1.GetStrides();
   auto & stride2 = t2.GetStrides();
   auto & outStride = tout.GetStrides();

   size_t s1 = (axis > 0) ? stride1[axis-1] : t1.GetSize();  // block size to copy from first tensor
   size_t s2 = (axis > 0) ? stride2[axis-1] : t2.GetSize();  // block size to copy from second tensor
   size_t sout = (axis > 0) ? outStride[axis-1] : tout.GetSize();
   size_t nb = t1.GetSize()/s1;
   for (size_t i = 0; i < nb; i++) {
      std::copy(t1.GetData() + i*s1, t1.GetData() + (i+1)*s1, tout.GetData() + i * sout );
      std::copy(t2.GetData() + i*s2, t2.GetData() + (i+1)*s2, tout.GetData() + i * sout + s1 );
   }

   return tout;
}


inline GNN_Data Concatenate(GNN_Data & data1, GNN_Data & data2, int axis = 0) {
   GNN_Data out;
   out.node_data = Concatenate(data1.node_data,data2.node_data, axis);
   out.edge_data = Concatenate(data1.edge_data,data2.edge_data, axis);
   out.global_data = Concatenate<float>(data1.global_data,data2.global_data, axis-1);
   // assume sender/receivers of data1 and data2 are the same
   if (data1.receivers != data2.receivers || data1.senders != data2.senders)
       throw std::runtime_error("GNN_Data Concatenate: data1 and data2 have different net structures");
   out.receivers = data1.receivers;
   out.senders = data1.senders;
   return out;
}

inline GNN_Data Copy(const GNN_Data & data) {
   GNN_Data out;
   out.node_data = RTensor<float>(data.node_data.GetShape());
   out.edge_data = RTensor<float>(data.edge_data.GetShape());
   out.global_data = RTensor<float>(data.global_data.GetShape());
   std::copy(data.node_data.GetData(), data.node_data.GetData()+ data.node_data.GetSize(), out.node_data.GetData());
   std::copy(data.edge_data.GetData(), data.edge_data.GetData()+ data.edge_data.GetSize(), out.edge_data.GetData());
   std::copy(data.global_data.GetData(), data.global_data.GetData()+ data.global_data.GetSize(), out.global_data.GetData());
   out.receivers = data.receivers;
   out.senders = data.senders;
   return out;
}

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RMODEL
