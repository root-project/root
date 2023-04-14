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
   if (BlockSize >= 512 && tid < 256) { sdata[tid] = operation(sdata[tid], sdata[tid + 256]); } __syncthreads();
   if (BlockSize >= 256 && tid < 128) { sdata[tid] = operation(sdata[tid], sdata[tid + 128]); } __syncthreads();
   if (BlockSize >= 128 && tid < 64)  { sdata[tid] = operation(sdata[tid], sdata[tid + 64]);  } __syncthreads();

   // Reduction within a warp
   if (BlockSize >= 64 && tid < 32)  { sdata[tid] = operation(sdata[tid], sdata[tid + 32]); } __syncthreads();
   if (BlockSize >= 32 && tid < 16)  { sdata[tid] = operation(sdata[tid], sdata[tid + 16]); } __syncthreads();
   if (BlockSize >= 16 && tid < 8)   { sdata[tid] = operation(sdata[tid], sdata[tid + 8]);  } __syncthreads();
   if (BlockSize >= 8  && tid < 4)   { sdata[tid] = operation(sdata[tid], sdata[tid + 4]);  } __syncthreads();
   if (BlockSize >= 4  && tid < 2)   { sdata[tid] = operation(sdata[tid], sdata[tid + 2]);  } __syncthreads();
   if (BlockSize >= 2  && tid < 1)   { sdata[tid] = operation(sdata[tid], sdata[tid + 1]);  } __syncthreads();
}
// clang-format on

template <unsigned int BlockSize = 512, typename T = double, typename InitOp, typename MainOp>
inline __device__ void ReduceBase(T *sdata, T *in, T *out, unsigned int n, InitOp initOp, MainOp mainOp, T init = 0.)
{
   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x * (BlockSize * 2) + tid;
   unsigned int gridSize = (BlockSize * 2) * gridDim.x;

   T r = 0;

   while (i < n) {
      r = initOp(i, r, in[i]);
      if (i + BlockSize < n) {
         r = initOp(i + BlockSize, r, in[i + BlockSize]);
      }
      i += gridSize;
   }
   sdata[tid] = r;
   __syncthreads();

   CUDAHelpers::UnrolledReduce<BlockSize, T>(sdata, tid, mainOp);

   // The first thread of each block writes the sum of the block into the global device array.
   if (tid == 0) {
      out[blockIdx.x] = mainOp(out[blockIdx.x], sdata[0]);
   }
}

// See https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//     https://github.com/zchee/cuda-sample/blob/master/6_Advanced/reduction/reduction_kernel.cu
template <unsigned int BlockSize = 512, typename T = double, Bool_t Overwrite = false, typename Op = Plus<double>()>
__global__ void ReductionKernel(T *in, T *out, unsigned int n, Op operation, T init)
{
   auto sdata = CUDAHelpers::shared_memory_proxy<T>();
   ReduceBase(
      sdata, in, out, n, [](unsigned int i, T r, T in) { return r + in; }, operation, init);
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
