#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <string>
#include <thrust/binary_search.h>
#include <thrust/functional.h>

#include "RtypesCore.h"
#include "TError.h"

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

// Dynamic shared memory needs to be declared as "extern" in CUDA. Having templated kernels with shared memory
// of different data types results in a redeclaration error if the name of the array is the same, so we use a
// proxy function to initialize shared memory arrays of different types with different names.

template <typename T>
__device__ T *shared_memory_proxy()
{
   Fatal("template <typename T> __device__ T *shared_memory_proxy()", "Unsupported shared memory type");
   return (T *)0;
};

template <>
__device__ int *shared_memory_proxy<int>()
{
   extern __shared__ int s_int[];
   return s_int;
}

template <>
__device__ double *shared_memory_proxy<double>()
{
   extern __shared__ double s_double[];
   return s_double;
}

template <>
__device__ float *shared_memory_proxy<float>()
{
   extern __shared__ float s_float[];
   return s_float;
}

////////////////////////////////////////////////////////////////////////////////
/// Reduction operations.
template <typename T = double>
struct Plus {
   // Function call operator. The return value is <tt>lhs + rhs</tt>.
   __host__ __device__ constexpr T operator()(const T &lhs, const T &rhs) const { return lhs + rhs; }
};

////////////////////////////////////////////////////////////////////////////////
/// CUDA Kernels

// clang-format off
template <unsigned int BlockSize, typename T, typename Op>
__device__ inline void UnrolledReduce(T *sdata, unsigned int tid, Op operation)
{
   // 1024 is the maximum number of threads per block in an NVIDIA GPU:
   // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
   if (BlockSize >= 1024 && tid < 512) { sdata[tid] = operation(sdata[tid], sdata[tid + 512]); } __syncthreads();
   if (BlockSize >= 512  && tid < 256) { sdata[tid] = operation(sdata[tid], sdata[tid + 256]); } __syncthreads();
   if (BlockSize >= 256  && tid < 128) { sdata[tid] = operation(sdata[tid], sdata[tid + 128]); } __syncthreads();
   if (BlockSize >= 128  && tid < 64)  { sdata[tid] = operation(sdata[tid], sdata[tid + 64]);  } __syncthreads();

   // Reduction within a warp
   if (BlockSize >= 64 && tid < 32)  { sdata[tid] = operation(sdata[tid], sdata[tid + 32]); } __syncthreads();
   if (BlockSize >= 32 && tid < 16)  { sdata[tid] = operation(sdata[tid], sdata[tid + 16]); } __syncthreads();
   if (BlockSize >= 16 && tid < 8)   { sdata[tid] = operation(sdata[tid], sdata[tid + 8]);  } __syncthreads();
   if (BlockSize >= 8  && tid < 4)   { sdata[tid] = operation(sdata[tid], sdata[tid + 4]);  } __syncthreads();
   if (BlockSize >= 4  && tid < 2)   { sdata[tid] = operation(sdata[tid], sdata[tid + 2]);  } __syncthreads();
   if (BlockSize >= 2  && tid < 1)   { sdata[tid] = operation(sdata[tid], sdata[tid + 1]);  } __syncthreads();
}
// clang-format on

// See https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//     https://github.com/zchee/cuda-sample/blob/master/6_Advanced/reduction/reduction_kernel.cu
template <unsigned int BlockSize, typename T, typename InitOp, typename MainOp>
inline __device__ void ReduceBase(T *in, T *out, unsigned int n, InitOp initOp, MainOp mainOp, T init)
{
   auto sdata = CUDAHelpers::shared_memory_proxy<T>();

   unsigned int local_tid = threadIdx.x;
   unsigned int i = blockIdx.x * (BlockSize * 2) + local_tid;
   unsigned int gridSize = (BlockSize * 2) * gridDim.x;

   T r = init;

   while (i < n) {
      r = initOp(i, r, in[i]);
      if (i + BlockSize < n) {
         r = initOp(i + BlockSize, r, in[i + BlockSize]);
      }
      i += gridSize;
   }
   sdata[local_tid] = r;
   __syncthreads();

   CUDAHelpers::UnrolledReduce<BlockSize, T>(sdata, local_tid, mainOp);

   // The first thread of each block writes the sum of the block into the global device array.
   if (local_tid == 0) {
      out[blockIdx.x] = mainOp(out[blockIdx.x], sdata[0]);
   }
}

template <unsigned int BlockSize, typename T>
__global__ void ReduceSumKernel(T *in, T *out, unsigned int n, T init)
{
   auto initOp = [](unsigned int i, T r, T in) { return r + in; };
   ReduceBase<BlockSize>(in, out, n, initOp, CUDAHelpers::Plus<T>(), init);
}

template <typename T = double>
void ReduceSum(int numBlocks, int blockSize, T *in, T *out, unsigned int n, T init = 0.)
{
   auto initOp = [](unsigned int i, T r, T in) { return r + in; };
   auto smemSize = (blockSize <= 32) ? 2 * blockSize : blockSize;

   if (blockSize == 2)
      ReduceSumKernel<2, T><<<numBlocks, 2, smemSize>>>(in, out, n, init);
   else if (blockSize == 4)
      ReduceSumKernel<4, T><<<numBlocks, 4, smemSize>>>(in, out, n, init);
   else if (blockSize == 8)
      ReduceSumKernel<8, T><<<numBlocks, 8, smemSize>>>(in, out, n, init);
   else if (blockSize == 16)
      ReduceSumKernel<16, T><<<numBlocks, 16, smemSize>>>(in, out, n, init);
   else if (blockSize == 32)
      ReduceSumKernel<32, T><<<numBlocks, 32, smemSize>>>(in, out, n, init);
   else if (blockSize == 64)
      ReduceSumKernel<64, T><<<numBlocks, 64, smemSize>>>(in, out, n, init);
   else if (blockSize == 128)
      ReduceSumKernel<128, T><<<numBlocks, 128, smemSize>>>(in, out, n, init);
   else if (blockSize == 256)
      ReduceSumKernel<256, T><<<numBlocks, 256, smemSize>>>(in, out, n, init);
   else if (blockSize == 512)
      ReduceSumKernel<512, T><<<numBlocks, 512, smemSize>>>(in, out, n, init);
   else if (blockSize == 1024)
      ReduceSumKernel<1024, T><<<numBlocks, 1024, smemSize>>>(in, out, n, init);
   else
      Error("ReduceSum", "Unsupported block size: %d", blockSize);
}

// CUDA version of TMath::BinarySearchCUDA
template <typename T>
__device__ Long64_t BinarySearchCUDA(Long64_t n, const T *array, T value)
{
   const T *pind;

   pind = thrust::lower_bound(thrust::seq, array, array + n, value);

   if ((pind != array + n) && (*pind == value))
      return (pind - array);
   else
      return (pind - array - 1);

   // return pind - array - !((pind != array + n) && (*pind == value)); // OPTIMIZATION: is this better?
}

} // namespace CUDAHelpers

#endif
