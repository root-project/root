#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <string>
#include "RtypesCore.h"
#include "TError.h"
#include "cuda.h"

// TODO: reused from RooBatchComputeTypes.h.
#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      Fatal((func + "(), " + file + ":" + std::to_string(line)).c_str(), "%s", cudaGetErrorString(error));
      throw std::bad_alloc();
   }
}


namespace CUDAHelperKernels {
   // See https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
   //     https://github.com/zchee/cuda-sample/blob/master/6_Advanced/reduction/reduction_kernel.cu
   template <UInt_t BlockSize, typename Op, typename ValType, Bool_t Overwrite=true>
   __global__ void ReductionKernel(ValType *g_idata, ValType *g_odata, UInt_t n, ValType init);

   template <typename T>
   __device__ Long64_t BinarySearchCUDA(Long64_t n, const T  *array, T value);
}

class CUDAHelper {
public:
   template <typename ValType, Bool_t Overwrite=true>
   void ReduceSum(ValType *input, ValType *output, UInt_t n);
};

#endif