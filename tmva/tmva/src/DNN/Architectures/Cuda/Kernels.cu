// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////////
// Implementation of the device kernels for the CUDA implementation of //
// the low-level interface.                                            //
/////////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "cuda.h"
#include "math.h"

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
__device__ CudaDouble_t atomicAdd(CudaDouble_t* address, CudaDouble_t val)
{
   unsigned long long int* address_as_ull = (unsigned long long int*)address;
   unsigned long long int old = *address_as_ull, assumed;
   do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
                      __double_as_longlong(val +
                                           __longlong_as_double(assumed)));
   } while (assumed != old);
   return __longlong_as_double(old);
}

//____________________________________________________________________________
__device__ void reduce_sum_vertical(CudaDouble_t *result, CudaDouble_t * sdata)
{
   // i,j are block row and column indices.
   int i = threadIdx.y;
   int j = threadIdx.x;
   int index = i * blockDim.x + j;

   __syncthreads();
   if ((blockDim.y > 512) && (i < 512)) {
      if ((i + 512) < blockDim.y) {
         sdata[index] += sdata[index + 512 * blockDim.x];
      }
   }

   __syncthreads();
   if ((blockDim.y > 256) && (i < 256)) {
      if ((i + 256) < blockDim.y) {
         sdata[index] += sdata[index + 256 * blockDim.x];
      }
   }
   __syncthreads();
   if ((blockDim.y > 128) && (i < 128)) {
      if ((i + 128) < blockDim.y) {
         sdata[index] += sdata[index + 128 * blockDim.x];
      }
   }
   __syncthreads();
   if ((blockDim.y > 64) && (i < 64)) {
      if ((i + 64) < blockDim.y) {
         sdata[index] += sdata[index + 64 * blockDim.x];
      }
   }
   __syncthreads();
   if ((blockDim.y > 32) && (i < 32)) {
      if ((i + 32) < blockDim.y) {
         sdata[index] += sdata[index + 32 * blockDim.x];
      }
   }
   __syncthreads();
   if ((blockDim.y > 16) && (i < 16)) {
      if ((i + 16) < blockDim.y) {
         sdata[index] += sdata[index + 16 * blockDim.x];
      }
   }
   __syncthreads();
   if ((blockDim.y > 8) && (i < 8)) {
      if ((i + 8) < blockDim.y) {
         sdata[index] += sdata[index + 8 * blockDim.x];
      }
   }
   __syncthreads();
   if ((blockDim.y > 4) && (i < 4)) {
      if ((i + 4) < blockDim.y) {
         sdata[index] += sdata[index + 4 * blockDim.x];
      }
   }
   __syncthreads();
   if ((blockDim.y > 2) && (i < 2)) {
      if ((i + 2) < blockDim.y) {
         sdata[index] += sdata[index + 2 * blockDim.x];
      }
   }
   __syncthreads();
   if ((blockDim.y > 1) && (i < 1)) {
      if ((i + 1) < blockDim.y) {
         sdata[index] += sdata[index + 1 * blockDim.x];
      }
   }
   __syncthreads();
   if (i == 0) {
      atomicAdd(result + j, sdata[index]);
   }
   __syncthreads();
}

//____________________________________________________________________________
__device__ void reduce_sum(CudaDouble_t *result, CudaDouble_t * sdata)
{
   int tid = threadIdx.x + threadIdx.y * blockDim.x;

   __syncthreads();
   if ((TDevice::BlockSize > 512) && (tid < 512)) {
      if ((tid + 512) < TDevice::BlockSize) {
         sdata[tid] += sdata[tid + 512];
      }
   }

   __syncthreads();
   if ((TDevice::BlockSize > 256) && (tid < 256)) {
      if ((tid + 256) < TDevice::BlockSize) {
         sdata[tid] += sdata[tid + 256];
      }
   }
   __syncthreads();
   if ((TDevice::BlockSize > 128) && (tid < 128)) {
      if ((tid + 128) < TDevice::BlockSize) {
         sdata[tid] += sdata[tid + 128];
      }
   }
   __syncthreads();
   if ((TDevice::BlockSize > 64) && (tid < 64)) {
      if ((tid + 64) < TDevice::BlockSize) {
         sdata[tid] += sdata[tid + 64];
      }
   }
   __syncthreads();
   if ((TDevice::BlockSize > 32) && (tid < 32)) {
      if ((tid + 32) < TDevice::BlockSize) {
         sdata[tid] += sdata[tid + 32];
      }
   }
   __syncthreads();
   if ((TDevice::BlockSize > 16) && (tid < 16)) {
      if ((tid + 16) < TDevice::BlockSize) {
         sdata[tid] += sdata[tid + 16];
      }
   }
   __syncthreads();
   if ((TDevice::BlockSize > 8) && (tid < 8)) {
      if ((tid + 8) < TDevice::BlockSize) {
         sdata[tid] += sdata[tid + 8];
      }
   }
   __syncthreads();
   if ((TDevice::BlockSize > 4) && (tid < 4)) {
      if ((tid + 4) < TDevice::BlockSize) {
         sdata[tid] += sdata[tid + 4];
      }
   }
   __syncthreads();
   if ((TDevice::BlockSize > 2) && (tid < 2)) {
      if ((tid + 2) < TDevice::BlockSize) {
         sdata[tid] += sdata[tid + 2];
      }
   }
   __syncthreads();
   if ((TDevice::BlockSize > 1) && (tid < 1)) {
      if ((tid + 1) < TDevice::BlockSize) {
         sdata[tid] += sdata[tid + 1];
      }
   }
   if (tid == 0) {
       atomicAdd(result, sdata[0]);
   }

   __syncthreads();
}

//____________________________________________________________________________
__global__ void add_row_wise(CudaDouble_t * W,
                             const CudaDouble_t * theta,
                             int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n))
       W[index] += theta[j];
}

//____________________________________________________________________________
__global__ void hadamard(CudaDouble_t * B,
                         const CudaDouble_t * A,
                         int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n))
       B[index] *= A[index];
}

//____________________________________________________________________________
__global__ void identity_derivative(CudaDouble_t * A,
                                    int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n))
       A[index] = 1.0;
}

//____________________________________________________________________________
__global__ void relu(CudaDouble_t * A,
                     int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      CudaDouble_t x = A[index];
      A[index] = (x < 0.0) ? 0.0 : x;
   }
}

//____________________________________________________________________________
__global__ void relu_derivative(CudaDouble_t * B,
                                const CudaDouble_t * A, int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      CudaDouble_t x = A[index];
      B[index] = (x < 0.0) ? 0.0 : 1.0;
   }
}

//____________________________________________________________________________
__global__ void sigmoid(CudaDouble_t * A,
                        int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      CudaDouble_t sig = 1.0 / (1.0 + exp(-A[index]));
      A[index] = sig;
   }
}

//____________________________________________________________________________
__global__ void sigmoid(CudaDouble_t * B,
                        const CudaDouble_t * A,
                        int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      CudaDouble_t sig = 1.0 / (1.0 + exp(-A[index]));
      B[index] = sig;
   }
}
//____________________________________________________________________________
__global__ void sigmoid_derivative(CudaDouble_t * B,
                                   const CudaDouble_t * A,
                                   int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      CudaDouble_t sig = 1.0 / (1.0 + exp(-A[index]));
      B[index] = sig * (1.0 - sig);
   }
}

//____________________________________________________________________________
__global__ void tanh(CudaDouble_t * A,
                        int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      CudaDouble_t t = ::tanh(A[index]);
      A[index] = t;
   }
}

//____________________________________________________________________________
__global__ void tanh_derivative(CudaDouble_t * B,
                                   const CudaDouble_t * A,
                                   int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      CudaDouble_t t = ::tanh(A[index]);
      B[index] = 1 - t*t;
   }
}

//____________________________________________________________________________
__global__ void symmetric_relu(CudaDouble_t * A,
                               int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      A[index] = abs(A[index]);
   }
}

//____________________________________________________________________________
__global__ void symmetric_relu_derivative(CudaDouble_t * B,
                                          const CudaDouble_t * A,
                                          int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      B[index] = (A[index] < 0.0) ? -1.0 : 1.0;
   }
}

//____________________________________________________________________________
__global__ void soft_sign(CudaDouble_t * A,
                          int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      CudaDouble_t x = A[index];
      A[index] = x / (1.0 + abs(x));
   }
}

//____________________________________________________________________________
__global__ void soft_sign_derivative(CudaDouble_t * B,
                                     const CudaDouble_t * A,
                                     int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      CudaDouble_t x = 1.0 + fabs(A[index]);
      B[index] = 1 / (x * x);
   }
}

//____________________________________________________________________________
__global__ void gauss(CudaDouble_t * A,
                      int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      CudaDouble_t x = A[index];
      A[index] = exp(- x * x);
   }
}

//____________________________________________________________________________
__global__ void gauss_derivative(CudaDouble_t * B,
                                 const CudaDouble_t * A,
                                 int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      CudaDouble_t x = A[index];
      B[index] = - 2.0 * x * exp(- x * x);
   }
}

//____________________________________________________________________________
__global__ void mean_squared_error(CudaDouble_t * result,
                                   const CudaDouble_t * Y,
                                   const CudaDouble_t * output,
                                   int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int tid   = blockDim.x * threadIdx.y + threadIdx.x;
   int index = j * m + i;

   __shared__ CudaDouble_t sdata[TDevice::BlockSize];

   if ((i < m) && (j < n)) {
       CudaDouble_t norm = 1 / ((CudaDouble_t) (m * n));
       CudaDouble_t e   = Y[index] - output[index];
       sdata[tid] = norm * e * e;
   } else {
       sdata[tid] = 0.0;
   }
   reduce_sum(result, sdata);
}

//____________________________________________________________________________
__global__ void squared_sum(CudaDouble_t * result,
                            const CudaDouble_t * A,
                            int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int tid   = blockDim.x * threadIdx.y + threadIdx.x;
   int index = j * m + i;

   __shared__ CudaDouble_t sdata[TDevice::BlockSize];

   if ((i < m) && (j < n)) {
       CudaDouble_t e = A[index];
       sdata[tid] = e * e;
   } else {
       sdata[tid] = 0.0;
   }
   reduce_sum(result, sdata);
}

//____________________________________________________________________________
__global__ void absolute_sum(CudaDouble_t * result,
                             const CudaDouble_t * A,
                             int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int tid   = blockDim.x * threadIdx.y + threadIdx.x;
   int index = j * m + i;

   __shared__ CudaDouble_t sdata[TDevice::BlockSize];

   if ((i < m) && (j < n)) {
       sdata[tid] = abs(A[index]);
   } else {
       sdata[tid] = 0.0;
   }
   reduce_sum(result, sdata);
}

//____________________________________________________________________________
__global__ void mean_squared_error_gradients(CudaDouble_t * dY,
                                             const CudaDouble_t * Y,
                                             const CudaDouble_t * output,
                                             int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n))
       dY[index] = 2.0 / ((CudaDouble_t) (m * n)) * (output[index] - Y[index]);
}

//____________________________________________________________________________
__global__ void add_l1_regularization_gradients(CudaDouble_t * A,
                                                const CudaDouble_t * B,
                                                CudaDouble_t weightDecay,
                                                int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
       CudaDouble_t sign = (B[index] < 0.0) ? -1.0 : 1.0;
       A[index] += sign * weightDecay;
   }
}

//____________________________________________________________________________
__global__ void add_l2_regularization_gradients(CudaDouble_t * A,
                                                const CudaDouble_t * B,
                                                CudaDouble_t weightDecay,
                                                int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
       A[index] += 2.0 * weightDecay * B[index];
   }
}

//____________________________________________________________________________
__global__ void cross_entropy(CudaDouble_t * result,
                              const CudaDouble_t * Y,
                              const CudaDouble_t * output,
                              int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int tid   = blockDim.x * threadIdx.y + threadIdx.x;
   int index = j * m + i;

   __shared__ CudaDouble_t sdata[TDevice::BlockSize];

   if ((i < m) && (j < n)) {
       CudaDouble_t norm = 1 / ((CudaDouble_t) (m * n));
       CudaDouble_t sig  = 1.0 / (1.0 + exp(-output[index]));
       CudaDouble_t ce   = Y[index] * log(sig) + (1.0 - Y[index]) * log(1.0 - sig);
       sdata[tid]        = - norm * ce;
   } else {
       sdata[tid] = 0.0;
   }

   reduce_sum(result, sdata);
}

//____________________________________________________________________________
__global__ void cross_entropy_gradients(CudaDouble_t * dY,
                                        const CudaDouble_t * Y,
                                        const CudaDouble_t * output,
                                        int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      CudaDouble_t norm = 1 / ((CudaDouble_t) (m * n));
      CudaDouble_t y = Y[index];
      CudaDouble_t sig = 1.0 / (1.0 + exp(-output[index]));
      dY[index] = norm * (sig - y);
   }
}

//____________________________________________________________________________
__global__ void reduce_matrix(CudaDouble_t *result,
                              const CudaDouble_t *A,
                              int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int tid = threadIdx.y * blockDim.x + threadIdx.x;
   int index = j * m + i;

   __shared__ CudaDouble_t smem[TDevice::BlockSize];
   if ((i < m) && (j < n))
       smem[tid] = A[index];
   else
       smem[tid] = 0.0;

   reduce_sum(result, smem);
}

//____________________________________________________________________________
__global__ void sum_columns(CudaDouble_t *B,
                            const CudaDouble_t *A,
                            int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int matrixIndex = j * m + i;
   int blockIndex  = blockDim.x * threadIdx.y + threadIdx.x;


   __shared__ CudaDouble_t smem[TDevice::BlockSize];

   if ((i < m) && (j < n)) {
       smem[blockIndex] = A[matrixIndex];
   } else {
       smem[blockIndex] = 0.0;
   }

   reduce_sum_vertical(B + blockDim.x * blockIdx.x, smem);
}

//____________________________________________________________________________
__global__ void initialize_curand_states(unsigned long long seed,
                                   curandState_t *state)
{
   int i   = blockDim.y * blockIdx.y + threadIdx.y;
   int j   = blockDim.x * blockIdx.x + threadIdx.x;
   int tid = i * gridDim.x + j;
   curand_init(seed, tid, 0, state + tid);
}

//____________________________________________________________________________
__global__ void dropout(CudaDouble_t *A,
                        int m, int n,
                        CudaDouble_t dropoutProbability,
                        curandState_t *state)
{
   int i   = blockDim.y * blockIdx.y + threadIdx.y;
   int j   = blockDim.x * blockIdx.x + threadIdx.x;
   int tid = i * gridDim.x + j;
   if ((i < m) && (j < n)) {
      float r = curand_uniform(state + tid);
      if (r > dropoutProbability) {
         A[j * m + i] = 0.0;
      } else {
         A[j * m + i] /= dropoutProbability;
      }
   }
}

} // namespace DNN
} // namespace TMVA
