#include "CUDAHelpers.cuh"

#include <thrust/binary_search.h>
#include <thrust/functional.h>

#include "RtypesCore.h"
#include "TError.h"
#include "cuda.h"


namespace CUDAHelpers {
   // See https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
   //     https://github.com/zchee/cuda-sample/blob/master/6_Advanced/reduction/reduction_kernel.cu
   template <unsigned int BlockSize, typename Op, typename ValType, bool Overwrite>
   __global__ void ReductionKernel(ValType *g_idata, ValType *g_odata, unsigned int n, ValType init) {
      extern __shared__ ValType sdata[];
      Op operation;

      unsigned int tid = threadIdx.x;
      unsigned int i = blockIdx.x*(BlockSize*2) + tid;
      unsigned int gridSize = (BlockSize*2)*gridDim.x;

      if (i == 0) {
         printf("blockdim:%d griddim:%d gridsize:%d\n", blockDim.x, gridDim.x, gridSize);
      }

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


   template <unsigned int BlockSize, typename Op, typename ValType, bool Overwrite>
   void Reduce(ValType *input, ValType *output, unsigned int n)
   {
      Int_t smemSize = (BlockSize <= 32) ? 2 * BlockSize : BlockSize;
      unsigned int numBlocks = fmax(1, ceil(n / BlockSize / 2.)); // Number of blocks in grid is halved!

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

   /*
    * Template declaration to allow linking.
    */

   // Binary search an array of doubles.
   template __device__ Long64_t BinarySearchCUDA<Double_t>(Long64_t n, const Double_t  *array, Double_t value);

   // Sum reduction with doubles.
   template void Reduce<512, thrust::plus<Double_t>, Double_t, false>(Double_t *input, Double_t *output, unsigned int n);

}
