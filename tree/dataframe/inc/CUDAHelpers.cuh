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
   // See https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
   //     https://github.com/zchee/cuda-sample/blob/master/6_Advanced/reduction/reduction_kernel.cu
   template <UInt_t BlockSize, typename Op, typename ValType, Bool_t Overwrite>
   __global__ void ReductionKernel(ValType *g_idata, ValType *g_odata, UInt_t n, ValType init) {
      extern __shared__ ValType sdata[];
      Op operation;

      UInt_t tid = threadIdx.x;
      UInt_t i = blockIdx.x*(BlockSize*2) + tid;
      UInt_t gridSize = (BlockSize*2)*gridDim.x;

      // if (i == 0) {
      //    printf("blockdim:%d griddim:%d gridsize:%d\n", blockDim.x, gridDim.x, gridSize);
      // }

      // Operate on local var instead of sdata to avoid illegal memory accesses?
      ValType r = init;

      while (i < n) {
         r = operation(r, g_idata[i]);
         if (i + BlockSize < n) r = operation(r, g_idata[i+BlockSize]);
         i += gridSize;
      }
      sdata[tid] = r;
      __syncthreads();

      if (BlockSize >= 512 && tid < 256) { sdata[tid] = operation(sdata[tid], sdata[tid + 256]); } __syncthreads();
      if (BlockSize >= 256 && tid < 128) { sdata[tid] = operation(sdata[tid], sdata[tid + 128]); } __syncthreads();
      if (BlockSize >= 128 && tid < 64)  { sdata[tid] = operation(sdata[tid], sdata[tid + 64]);  } __syncthreads();

      // WarpReduce
      if (BlockSize >= 64 && tid < 32)  { sdata[tid] = operation(sdata[tid], sdata[tid + 32]); } __syncthreads();
      if (BlockSize >= 32 && tid < 16)  { sdata[tid] = operation(sdata[tid], sdata[tid + 16]); } __syncthreads();
      if (BlockSize >= 16 && tid < 8)   { sdata[tid] = operation(sdata[tid], sdata[tid + 8]);  } __syncthreads();
      if (BlockSize >= 8  && tid < 4)   { sdata[tid] = operation(sdata[tid], sdata[tid + 4]);  } __syncthreads();
      if (BlockSize >= 4  && tid < 2)   { sdata[tid] = operation(sdata[tid], sdata[tid + 2]);  } __syncthreads();
      if (BlockSize >= 2  && tid < 1)   { sdata[tid] = operation(sdata[tid], sdata[tid + 1]);  } __syncthreads();

      if (tid == 0) {
         if (Overwrite) {
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

      // if ( (pind != array + n) && (*pind == value) )
      //    return (pind - array);
      // else
      //    return ( pind - array - 1);

      return pind - array - !((pind != array + n) && (*pind == value)); // OPTIMIZATION: is this better?
   }


   template <UInt_t BlockSize, typename Op, typename ValType, Bool_t Overwrite>
   void Reduce(ValType *input, ValType *output, UInt_t n)
   {
      Int_t smemSize = (BlockSize <= 32) ? 2 * BlockSize : BlockSize;
      UInt_t numBlocks = fmax(1, ceil(n / BlockSize / 2.)); // Number of blocks in grid is halved!

      ValType *intermediate = NULL;
      ERRCHECK(cudaMalloc((void **)&intermediate, numBlocks * sizeof(ValType)));

      CUDAHelpers::ReductionKernel<BlockSize, Op, ValType, Overwrite>
         <<<numBlocks, BlockSize, smemSize * sizeof(ValType)>>>(input, intermediate, n, 0.);
      ERRCHECK(cudaGetLastError());

      // TODO: in some cases the final reduction requires multiple passes?
      // OPTIMIZATION: final reduction on CPU under certain threshold?
      CUDAHelpers::ReductionKernel<BlockSize, Op, ValType, Overwrite>
         <<<1, BlockSize, smemSize * sizeof(ValType)>>>(intermediate, output, numBlocks, 0.);
      ERRCHECK(cudaGetLastError());

      ERRCHECK(cudaFree(intermediate));
   }

}

#endif
