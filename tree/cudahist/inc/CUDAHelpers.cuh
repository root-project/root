#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <string>
#include <thrust/binary_search.h>
#include <thrust/functional.h>

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

namespace CUDAHelpers {
   ////////////////////////////////////////////////////////////////////////////////
   /// Reduction operations.
   template <typename T = void>
   struct plus {
      // Function call operator. The return value is <tt>lhs + rhs</tt>.
      __host__ __device__ constexpr T operator()(const T &lhs, const T &rhs) const { return lhs + rhs; }
   };

   ////////////////////////////////////////////////////////////////////////////////
   /// CUDA Kernels

   template <UInt_t ABlockSize, typename AOp, typename AValType>
   __device__ inline void UnrolledReduce(AValType *sdata, UInt_t tid)
   {
      AOp operation;

      if (ABlockSize >= 512 && tid < 256) { sdata[tid] = operation(sdata[tid], sdata[tid + 256]); } __syncthreads();
      if (ABlockSize >= 256 && tid < 128) { sdata[tid] = operation(sdata[tid], sdata[tid + 128]); } __syncthreads();
      if (ABlockSize >= 128 && tid < 64)  { sdata[tid] = operation(sdata[tid], sdata[tid + 64]);  } __syncthreads();

      // Reduction within a warp
      if (ABlockSize >= 64 && tid < 32)  { sdata[tid] = operation(sdata[tid], sdata[tid + 32]); } __syncthreads();
      if (ABlockSize >= 32 && tid < 16)  { sdata[tid] = operation(sdata[tid], sdata[tid + 16]); } __syncthreads();
      if (ABlockSize >= 16 && tid < 8)   { sdata[tid] = operation(sdata[tid], sdata[tid + 8]);  } __syncthreads();
      if (ABlockSize >= 8  && tid < 4)   { sdata[tid] = operation(sdata[tid], sdata[tid + 4]);  } __syncthreads();
      if (ABlockSize >= 4  && tid < 2)   { sdata[tid] = operation(sdata[tid], sdata[tid + 2]);  } __syncthreads();
      if (ABlockSize >= 2  && tid < 1)   { sdata[tid] = operation(sdata[tid], sdata[tid + 1]);  } __syncthreads();
   }

   // See https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
   //     https://github.com/zchee/cuda-sample/blob/master/6_Advanced/reduction/reduction_kernel.cu
   template <UInt_t ABlockSize, typename AOp, typename AValType, Bool_t AOverwrite>
   __global__ void ReductionKernel(AValType *g_idata, AValType *g_odata, UInt_t n, AValType init) {
      extern __shared__ AValType sdata[];
      AOp operation;

      UInt_t tid = threadIdx.x;
      UInt_t i = blockIdx.x*(ABlockSize*2) + tid;
      UInt_t gridSize = (ABlockSize*2)*gridDim.x;

      // if (i == 0) {
      //    printf("blockdim:%d griddim:%d gridsize:%d\n", blockDim.x, gridDim.x, gridSize);
      // }

      // Operate on local var instead of sdata to avoid illegal memory accesses?
      AValType r = init;

      while (i < n) {
         r = operation(r, g_idata[i]);
         if (i + ABlockSize < n) r = operation(r, g_idata[i+ABlockSize]);
         i += gridSize;
      }
      sdata[tid] = r;
      __syncthreads();

      UnrolledReduce<ABlockSize, AOp, AValType>(sdata, tid);

      // The first thread of each block writes the sum of the block into the global device array.
      if (tid == 0) {
         if (AOverwrite) {
            g_odata[blockIdx.x] = sdata[0];
         } else {
            g_odata[blockIdx.x] = operation(g_odata[blockIdx.x], sdata[0]);
         }
      }
   }

   template <typename T>
   __device__ Long64_t BinarySearchCUDA(Long64_t n, const T  *array, T value)
   {
      const T* pind;

      pind = thrust::lower_bound(thrust::seq, array, array + n, value);
      // printf("%lld %f %f %f %f\n", n, array[0], array[n], value, pind[0]);

      if ( (pind != array + n) && (*pind == value) )
         return (pind - array);
      else
         return ( pind - array - 1);

      // return pind - array - !((pind != array + n) && (*pind == value)); // OPTIMIZATION: is this better?
   }
}

#endif
