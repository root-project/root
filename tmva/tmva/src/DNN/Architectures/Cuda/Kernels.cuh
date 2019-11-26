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

template<typename AFloat>
__device__ AFloat max(AFloat x, AFloat y)
{
    if (x < y) return y;
    return x;
}

////////////////////////////////////////////////////////////////////////////////////
/// \brief Calculate the dimension of an output volume, given the sliding parameters
///        and the input shape.
/// \param[in] imgDim The size of the input tensor in a spatial dimension.
/// \param[in] fltDim The size of the sliding filter in the same dimension.
/// \param[in] padding Number of zeroes to pad the input with.
/// \param[in] stride Number of pixels the kernel is sliding in each iteration.
/// \returns   The output dimension.
///
/// Note that no checks are performed to assert validity of the input parameters.
/// We are allowed to assume them valid because those checks have already been
/// performed prior to the invocation of the kernel.
////////////////////////////////////////////////////////////////////////////////////
__device__ int calculateDimension(int imgDim, int fltDim, int padding, int stride)
{
   // Parameters passed at this point are guaranteed to be valid - skip checks.
   return ((imgDim - fltDim + 2 * padding) / stride) + 1;
}

////////////////////////////////////////////////////////////////////////////////////
/// \brief A kernel that re-arranges image regions of the input matrix \B, into
///        column vectors in matrix \A.
///
/// \param[out] A The output matrix. Each row corresponds to a receptive field.
/// \param[in] B The input matrix. Each row corresponds to a row in the image view.
/// \param[in] depth The depth of the input tensor.
/// \param[in] imgHeight The height of the input tensor.
/// \param[in] imgWidth The output of the input tensor
/// \param[in] fltHeight Height of the filter.
/// \param[in] fltWidth Width of the filter.
/// \param[in] strideRows stride size in the horizontal dimension.
/// \param[in] strideCols stride size in the vertical dimension.
/// \param[in] zeroPaddingHeight The padding in the horizontal dimension.
/// \param[in] zeroPaddingWidth The padding in the vertical dimension.
///
/// The kernel should be invoked with one thread per output element. Note that
/// matrices \A and \B have different shapes. Each thread in this kernel is
/// responsible for filling one cell of the output matrix \A. It does so by computing
/// the correct element to copy from the input matrix \B. We therefore never need to
/// block. When reading this kernel it is important to keep in mind that TCudaMatrix
/// objects are saved in column major order for compatibility with cuBLAS.
////////////////////////////////////////////////////////////////////////////////////
template<typename AFloat>
__global__ void Im2Col(AFloat * A,
                       const AFloat * B,
                       int depth,
                       int imgHeight,
                       int imgWidth,
                       int fltHeight,
                       int fltWidth,
                       int strideRows,
                       int strideCols,
                       int zeroPaddingHeight,
                       int zeroPaddingWidth)
{
    // The row of the output matrix.
    int i = blockDim.y * blockIdx.y + threadIdx.y;

    // The column of the output matrix.
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    // Number of column in matrix A.
    int NLocalViewPixels = fltHeight * fltWidth * depth;

    // Number of rows in matrix A.
    int NLocalViews = calculateDimension(imgWidth, fltWidth, zeroPaddingWidth, strideCols) *
                      calculateDimension(imgHeight, fltHeight, zeroPaddingHeight, strideRows);

    if (i >= NLocalViews || j >= NLocalViewPixels) return;

    int index = j * NLocalViews + i;

    int numSlidesPerRow = calculateDimension(imgWidth, fltWidth, zeroPaddingWidth, strideCols);

    // Which image channel of B?
    int bz = j / (fltHeight * fltWidth);

    // Which row in matrix B?
    int by = (i / numSlidesPerRow) * strideRows - zeroPaddingHeight + (j - bz * fltHeight * fltWidth) / fltWidth;

    // Which column in matrix B?
    int bx = (i % numSlidesPerRow) * strideCols - zeroPaddingWidth + (j - bz * fltHeight * fltWidth) % fltWidth;

    if (bx < 0 || by < 0 || bx >= imgWidth || by >= imgHeight) {
        // This is a padding element.
        A[index] = 0;
    }
    else {
        A[index] = B[(bx + by * imgWidth) * depth + bz];
    }
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
__global__ void ConstAdd(AFloat * A, AFloat beta,
                     int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      A[index] = A[index] + beta;
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void ConstMult(AFloat * A, AFloat beta,
                     int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      A[index] = A[index] * beta;
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void ReciprocalElementWise(AFloat * A,
                     int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      A[index] = 1.0 / A[index];
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void SquareElementWise(AFloat * A,
                     int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      A[index] = A[index] * A[index];
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void SqrtElementWise(AFloat * A,
                     int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      A[index] = sqrt(A[index]);
   }
}


/// optimizer kernel functions

//____________________________________________________________________________
template<typename AFloat>
__global__ void AdamUpdate(AFloat * A, const AFloat * M, const AFloat * V,
                           int m, int n, AFloat alpha, AFloat eps)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      A[index] = A[index] - alpha * M[index]/( sqrt(V[index]) + eps);
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void AdamUpdateFirstMom(AFloat * A, const AFloat * B,
                           int m, int n, AFloat beta)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      A[index] = beta * A[index] + (1.-beta) * B[index];
   }
}

//____________________________________________________________________________
template<typename AFloat>
__global__ void AdamUpdateSecondMom(AFloat * A, const AFloat * B,
                           int m, int n, AFloat beta)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int index = j * m + i;

   if ((i < m) && (j < n)) {
      A[index] = beta * A[index] + (1.-beta) * B[index] * B[index];
   }
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
       if (Y[index] == 0)
          sdata[tid] = -weights[i] * norm * log(1.0 - sig);
       else if (Y[index] == 1.0)
          sdata[tid] = -weights[i] * norm * log(sig);
       else {
          AFloat ce  = Y[index] * log(sig) + (1.0 - Y[index]) * log(1.0 - sig);
          sdata[tid] = -weights[i] * norm * ce;
       }
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

template<typename AFloat>
__global__ void AlmostEquals(bool * result, const AFloat * A, const AFloat * B, double epsilon, int m, int n)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;

   if (i >= m || j >= n) return;
   int matrixIndex = j * m + i;

   // This is a race condition but still thread safe: If many threads find inequality I don't care
   // if they overwrite each other, the result is still going to be false.
   if(fabs(A[matrixIndex] - B[matrixIndex]) > epsilon) result[0] = false;
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

//____________________________________________________________________________
//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Downsampling kernel used as the forward propagation step of a
///        Max-Pooling layer.
///
/// \param[out] A The output matrix. Each row corresponds to a slice and each element
///             is the max within a receptive field.
/// \param[out] B The winning indices matrix. Each element is the index of the max element.
/// \param[in] C The input matrix. Each row is a slice.
/// \param[in] imgHeight The heigh of the input.
/// \param[in] imgWidth The output of the input.
/// \param[in] fltHeight Height of the kernel.
/// \param[in] fltWidth Width of the kernel.
/// \param[in] strideRows stride size in the horizontal dimension.
/// \param[in] strideCols stride size in the vertical dimension.
///
/// Each output element is the maximum of the receptive field. The caller launches one thread
/// per output element in order to eliminate shared write access.
///////////////////////////////////////////////////////////////////////////////////////////////
template<typename AFloat>
__global__ void Downsample(AFloat * output, AFloat * indexMatrix, const AFloat * input, int depth, int imgHeight,
                           int imgWidth, int fltHeight, int fltWidth, int strideRows, int strideCols)
{
   // The row of the output matrix.
   int i = blockDim.y * blockIdx.y + threadIdx.y;

   // The column of the output matrix.
   int j = blockDim.x * blockIdx.x + threadIdx.x;

   // Number of columns in matrix A.
   int NLocalViews = calculateDimension(imgWidth, fltWidth, 0, strideCols) *
                     calculateDimension(imgHeight, fltHeight, 0, strideRows);

   if (i >= depth || j >= NLocalViews) return;

   int outputIndex = j * depth + i;

   int numSlidesPerRow = calculateDimension(imgWidth, fltWidth, 0, strideCols);

   int rowMin = (j / numSlidesPerRow) * strideRows;  // First row of B that this thread should look at.
   int colMin = (j % numSlidesPerRow) * strideCols;  // First column of B that this thread should look at.
   int bz = i;                                       // Slice of B that this thread should look at.

   AFloat value = 0;
   AFloat maxIndex = 0;
   bool first = true; // The first element should write to `value` no matter what.

   for (size_t by = rowMin; by < rowMin + fltHeight; by++) {
      for (size_t bx = colMin; bx < colMin + fltWidth; bx++) {
         int inputIndex = (bx + by * imgWidth) * depth + bz;
         if (input[inputIndex] > value || first) {
            first = false;
            maxIndex = bx + by * imgWidth;
            value = input[inputIndex];
         }
      }
   }
   indexMatrix[outputIndex] = maxIndex;
   output[outputIndex] = value;

}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Back-propagate the gradients through a max-pooling layer.
///
/// \param[out] gradientsBackward The gradients to be written. One gradient for each neuron at the layers's input.
/// \param[in] gradients The gradients coming from the next layer. One gradient for each receptive field.
/// \param[in] indexMatrix Winning indices. One index for each receptive field.
/// \param[in] depth The depth of the input tensor.
/// \param[in] imgHeight The height of the input tensor.
/// \param[in] imgWidth The output of the input tensor
/// \param[in] fltHeight Height of the filter.
/// \param[in] fltWidth Width of the filter.
/// \param[in] strideRows stride size in the horizontal dimension.
/// \param[in] strideCols stride size in the vertical dimension.
/////////////////////////////////////////////////////////////////////////////////////////////////
template<typename AFloat>
__global__ void MaxPoolBackward(AFloat * activationGradientsBackward,
                                const AFloat * activationGradients,
                                const AFloat * indexMatrix,
                                int depth, int imgHeight, int imgWidth, int fltHeight, int fltWidth,
                                int strideRows, int strideCols)
{
    int slice = blockDim.y * blockIdx.y + threadIdx.y;   // row of the gradientsBackward matrix.
    int j = blockDim.x * blockIdx.x + threadIdx.x;       // column of the gradientsBackward matrix.

    if (slice >= depth || j >= imgHeight * imgWidth) return;

    int height = calculateDimension(imgHeight, fltHeight, 0, strideRows);
    int width = calculateDimension(imgWidth, fltWidth, 0, strideCols);

    // Which gradientsBackward element should this thread write to?
    int backRow = j % imgHeight;
    int backCol = j / imgHeight;

    // Which gradient and indexMatrix elements should this thread read?
    int nextRowMin = floor((backRow - fltHeight) / (AFloat) strideRows) + 1;
    int nextColMin = floor((backCol - fltWidth) / (AFloat) strideCols) + 1;

    int outputIndex = 0;
    AFloat grad = 0;

    // Iterate over all output elements that were the outcome of receptive fields I was part of.
    for (int row = nextRowMin; row <= nextRowMin + fltHeight - strideRows; row++) {
        for (int col = nextColMin; col <= nextColMin + fltWidth - strideCols; col++) {

            if (row >= height || col >= width || col < 0 || row < 0) continue;

            outputIndex = (row * width + col) * depth + slice;

            // Was I the winning index within this receptive field?
            if (indexMatrix[outputIndex] == backCol + backRow * imgWidth) {
                grad += activationGradients[outputIndex];
            }
        }
    }
    activationGradientsBackward[(backCol + backRow * imgWidth) * depth + slice] = grad;
}

template<typename AFloat>
__global__ void RotateWeights(AFloat * A, const AFloat * B, int filterDepth, int filterHeight, int filterWidth,
                              int numFilters)
{
   int i   = blockDim.y * blockIdx.y + threadIdx.y;
   int j   = blockDim.x * blockIdx.x + threadIdx.x;

   if (i >= numFilters || j > filterDepth * filterHeight * filterWidth) return;

   int jump = filterHeight * filterWidth;
   int row = j / jump;
   int col = i * jump + jump - j % jump - 1;

   A[col * filterDepth + row] = B[j * numFilters + i];
}

template<typename AFloat>
__global__ void AddBiases(AFloat * A, const AFloat * B, int nRows, int nCols)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   if (i >= nRows || j >= nCols) return;

   A[i + j * nRows] += B[i];
}

template<typename AFloat>
__global__ void UpdateWeights(AFloat * A, const AFloat ** B, int batchSize, int nRows, int nCols)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= nRows || j >= nCols) return;

    for (size_t event = 0; event < batchSize; event++) {
        size_t index = i * nCols + j;
        A[index] += B[event][index];
    }
}

template<typename AFloat>
__global__ void Reshape(AFloat * A, const AFloat * B, int nRowsA, int nColsA, int nRowsB, int nColsB)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nRowsA || j >= nColsA) return;

    size_t indexA = j * nRowsA + i;

    size_t nElem = i * nColsA + j;
    size_t indexB = (nElem % nColsB) * nRowsB + nElem / nColsB;

    A[indexA] = B[indexB];
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Flatten an array of 2D-arrays into a single 2D-array.
///
/// \param[out] A Output 2D-array saved in column major order.
/// \param[in] B Input array of 2D-arrays. Each element is a matrix to be concatenated.
/// \param[in] size Number of 2D-arrays in the input.
/// \param[in] nRows Number of rows in each matrix of the input.
/// \param[in] nCols Number of columns on each matrix of the input.
///
/// B is a pointer to `size` raw `TCudaMatrix` pointers. Each of those contains
/// elements saved on column major order. However the concatenation is performed
/// row wise. Each thread writes a single output element by locating the
/// appropriate input index.
//////////////////////////////////////////////////////////////////////////////////
template<typename AFloat>
__global__ void Flatten(AFloat * A, const AFloat ** B, int size, int nRows, int nCols)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;

   int nColsA = nRows * nCols;
   if (i >= size || j >= nColsA) return;

   // Get a transposed view on matrix B[i].
   int row = j / nCols;
   int col = j % nCols;
   AFloat element = B[i][col * nRows + row];

   size_t index = j * size + i;
   A[index] = element;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Deflatten a 2D-array into an array of 2D-arrays.
///
/// \param[out] A Output array of 2D-arrays, each of which is column-major.
/// \param[in] B Input 2D-array to be split into `size` parts.
/// \param[in] size Number of 2D-arrays in the output.
/// \param[in] nRows Number of rows in each matrix of the output.
/// \param[in] nCols Number of columns on each matrix of the output.
///
/// A is a pointer to `size` raw `TCudaMatrix` pointers. Each of those will
/// contain elements saved on column major order. However the concatenation
/// is performed row wise. Each thread writes a single output element
/// by locating the appropriate input index.
//////////////////////////////////////////////////////////////////////////////////
template<typename AFloat>
__global__ void Deflatten(AFloat ** A, const AFloat * B, int size, int nRows, int nCols)
{
   int i = blockDim.y * blockIdx.y + threadIdx.y;
   int j = blockDim.x * blockIdx.x + threadIdx.x;

   int nColsB = nRows * nCols;
   if (i >= size || j >= nColsB) return;

   AFloat element = B[j * size + i];

   // Get a transposed view on matrix A[i].
   int row = j / nCols;
   int col = j % nCols;
   A[i][col * nRows + row] = element;
}

} // namespace Cuda
} // namespace DNN
} // namespace TMVA

#endif
