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

#ifndef TMVA_DNN_ARCHITECTURES_CUDA_KERNELS
#define TMVA_DNN_ARCHITECTURES_CUDA_KERNELS

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "cuda.h"
#include "math.h"

namespace TMVA {
namespace DNN  {
namespace Cuda {

//____________________________________________________________________________
template<typename AFloat>
__device__ AFloat AtomicAdd(AFloat* address, AFloat val);

template<>
__device__ double AtomicAdd(double* address, double val)
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

template<>
__device__ float AtomicAdd(float* address, float val)
{
   return atomicAdd(address, val);
}

//____________________________________________________________________________
template<typename AFloat>
__device__ void ReduceSumVertical(AFloat *result,
                                  AFloat * sdata,
                                  int n)
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
   if ((i == 0) && ((blockIdx.x * blockDim.x + threadIdx.x) < n)) {
      AtomicAdd(result + j, sdata[index]);
   }
   __syncthreads();
}

//____________________________________________________________________________
template<typename AFloat>
__device__ void ReduceSum(AFloat *result, AFloat * sdata)
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
       AtomicAdd(result, sdata[0]);
   }

   __syncthreads();
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void AddRowWise(AFloat * W,
                           const AFloat * theta,
                           int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n))
       W[index] += theta[j];
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void Hadamard(AFloat * B,
                         const AFloat * A,
                         int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n))
       B[index] *= A[index];
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void IdentityDerivative(AFloat * A,
                                   int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n))
       A[index] = 1.0;
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void Relu(AFloat * A,
                     int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      AFloat x = A[index];
      A[index] = (x < 0.0) ? 0.0 : x;
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void ReluDerivative(AFloat * B,
                               const AFloat * A, int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      AFloat x = A[index];
      B[index] = (x < 0.0) ? 0.0 : 1.0;
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void Sigmoid(AFloat * A,
                        int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      AFloat sig = 1.0 / (1.0 + exp(-A[index]));
      A[index] = sig;
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void Sigmoid(AFloat * B,
                        const AFloat * A,
                        int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      AFloat sig = 1.0 / (1.0 + exp(-A[index]));
      B[index] = sig;
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void SigmoidDerivative(AFloat * B,
                                  const AFloat * A,
                                  int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      AFloat sig = 1.0 / (1.0 + exp(-A[index]));
      B[index] = sig * (1.0 - sig);
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void Softmax(AFloat * B,
                        const AFloat * A,
                        int m, int n)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;

   if (i < m) {
      AFloat sum = 0.0;
      for (int j = 0; j < n; j++) {
         sum += exp(A[i + j * n]);
      }
      for (int j = 0; j < n; j++) {
         B[i + j * n] = exp(A[i * n + j]) / sum;
      }
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void Tanh(AFloat * A,
                     int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      AFloat t = ::tanh(A[index]);
      A[index] = t;
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void TanhDerivative(AFloat * B,
                               const AFloat * A,
                               int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      AFloat t = ::tanh(A[index]);
      B[index] = 1 - t*t;
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void SymmetricRelu(AFloat * A,
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
template<typename AFloat>
__global__ void SymmetricReluDerivative(AFloat * B,
                                        const AFloat * A,
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
template<typename AFloat>
__global__ void SoftSign(AFloat * A,
                          int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      AFloat x = A[index];
      A[index] = x / (1.0 + abs(x));
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void SoftSignDerivative(AFloat * B,
                                   const AFloat * A,
                                   int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      AFloat x = 1.0 + fabs(A[index]);
      B[index] = 1 / (x * x);
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void Gauss(AFloat * A,
                      int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      AFloat x = A[index];
      A[index] = exp(- x * x);
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void GaussDerivative(AFloat * B,
                                const AFloat * A,
                                int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      AFloat x = A[index];
      B[index] = - 2.0 * x * exp(- x * x);
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void MeanSquaredError(AFloat * result,
                                 const AFloat * Y,
                                 const AFloat * output,
                                 const AFloat * weights,
                                 int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int tid   = blockDim.x * threadIdx.y + threadIdx.x;
   int index = j * m + i;

   __shared__ AFloat sdata[TDevice::BlockSize];

   if ((i < m) && (j < n)) {
       AFloat w = weights[i];
       AFloat norm = 1 / ((AFloat) (m * n));
       AFloat e   = Y[index] - output[index];
       sdata[tid] = w * norm * e * e;
   } else {
       sdata[tid] = 0.0;
   }
   ReduceSum(result, sdata);
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void SquaredSum(AFloat * result,
                           const AFloat * A,
                           int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int tid   = blockDim.x * threadIdx.y + threadIdx.x;
   int index = j * m + i;

   __shared__ AFloat sdata[TDevice::BlockSize];

   if ((i < m) && (j < n)) {
       AFloat e = A[index];
       sdata[tid] = e * e;
   } else {
       sdata[tid] = 0.0;
   }
   ReduceSum(result, sdata);
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void AbsoluteSum(AFloat * result,
                            const AFloat * A,
                            int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int tid   = blockDim.x * threadIdx.y + threadIdx.x;
   int index = j * m + i;

   __shared__ AFloat sdata[TDevice::BlockSize];

   if ((i < m) && (j < n)) {
       sdata[tid] = abs(A[index]);
   } else {
       sdata[tid] = 0.0;
   }
   ReduceSum(result, sdata);
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void MeanSquaredErrorGradients(AFloat * dY,
                                          const AFloat * Y,
                                          const AFloat * output,
                                          const AFloat * weights,
                                          int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
       dY[index] = weights[i] * 2.0 / ((AFloat) (m * n)) * (output[index] - Y[index]);
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void AddL1RegularizationGradients(AFloat * A,
                                             const AFloat * B,
                                             AFloat weightDecay,
                                             int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
       AFloat sign = (B[index] < 0.0) ? -1.0 : 1.0;
       A[index] += sign * weightDecay;
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void AddL2RegularizationGradients(AFloat * A,
                                             const AFloat * B,
                                             AFloat weightDecay,
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
template<typename AFloat>
__global__ void CrossEntropy(AFloat * result,
                             const AFloat * Y,
                             const AFloat * output,
                             const AFloat * weights,
                             int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int tid   = blockDim.x * threadIdx.y + threadIdx.x;
   int index = j * m + i;

   __shared__ AFloat sdata[TDevice::BlockSize];

   if ((i < m) && (j < n)) {
       AFloat norm = 1 / ((AFloat) (m * n));
       AFloat sig  = 1.0 / (1.0 + exp(-output[index]));
       AFloat ce   = Y[index] * log(sig) + (1.0 - Y[index]) * log(1.0 - sig);
       sdata[tid]  = -weights[i] * norm * ce;
   } else {
       sdata[tid] = 0.0;
   }

   ReduceSum(result, sdata);
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void CrossEntropyGradients(AFloat * dY,
                                      const AFloat * Y,
                                      const AFloat * output,
                                      const AFloat * weights,
                                      int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      AFloat norm = 1 / ((AFloat) (m * n));
      AFloat y = Y[index];
      AFloat sig = 1.0 / (1.0 + exp(-output[index]));
      dY[index] = weights[i] * norm * (sig - y);
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void SoftmaxCrossEntropy(AFloat * result,
                                    const AFloat * Y,
                                    const AFloat * output,
                                    const AFloat * weights,
                                    int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int tid = threadIdx.y;

   __shared__ AFloat sdata[TDevice::BlockSize];
   AFloat norm = 1.0 / ((AFloat) m);

   sdata[tid] = 0.0;
   if (i < m) {
      AFloat sum  = 0.0;
      for (int j = 0; j < n; j++) {
         sum  += exp(output[i + j * m]);
      }
      for (int j = 0; j < n; j++) {
         sdata[tid] += Y[i + j * m] * log(exp(output[i + j * m]) / sum);
      }
      sdata[tid] *= -weights[i] *  norm;
   } else {
      sdata[tid] = 0.0;
   }

   ReduceSum(result, sdata);
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void SoftmaxCrossEntropyGradients(AFloat * dY,
                                             const AFloat * Y,
                                             const AFloat * output,
                                             const AFloat * weights,
                                             int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   AFloat norm = 1.0 / ((AFloat) m);

   if (i < m) {
      AFloat sum  = 0.0;
      AFloat sumY = 0.0;
      for (int j = 0; j < n; j++) {
         sum  += exp(output[i + j * m]);
         sumY += Y[i + j * m];
      }
      for (int j = 0; j < n; j++) {
         dY[i + j * m] =  sumY * exp(output[i + j * m]) / sum - Y[i + j * m];
         dY[i + j * m] *= weights[i] * norm;
      }
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void ReduceMatrix(AFloat *result,
                             const AFloat *A,
                             int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int tid = threadIdx.y * blockDim.x + threadIdx.x;
   int index = j * m + i;

   __shared__ AFloat smem[TDevice::BlockSize];
   if ((i < m) && (j < n))
       smem[tid] = A[index];
   else
       smem[tid] = 0.0;

   ReduceSum(result, smem);
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void SumColumns(AFloat *B,
                            const AFloat *A,
                            int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int matrixIndex = j * m + i;
   int blockIndex  = blockDim.x * threadIdx.y + threadIdx.x;


   __shared__ AFloat smem[TDevice::BlockSize];

   if ((i < m) && (j < n)) {
       smem[blockIndex] = A[matrixIndex];
   } else {
       smem[blockIndex] = 0.0;
   }

   ReduceSumVertical(B + blockDim.x * blockIdx.x, smem, n);
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void Dropout(AFloat *A,
                        int m, int n,
                        AFloat dropoutProbability,
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

} // namespace Cuda
} // namespace DNN
} // namespace TMVA

#endif
