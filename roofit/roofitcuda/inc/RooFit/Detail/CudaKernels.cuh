/*
 * Project: RooFit
 * Author:
 *   Jonas Rembser, CERN 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_Detail_CudaKernels_cuh
#define RooFit_Detail_CudaKernels_cuh

namespace RooFit {
namespace Detail {

/// Namespace for templated CUDA kernels.
namespace CudaKernels {

/// The type for array size parameters.
using Size_t = int;

/// Dedicated namespace for reduction kernels.
namespace Reducers {

/**
 * Performs a multi-block sum reduction on the input array `arr`. The input
 * array can either be scalar, or a flattened vector array with inner dimension
 * `ElemDim_n`.
 *
 * The output array is of shape `[ gridDim.x, ElemDim_n ]` (row-major flattened, meaning
 * when iterating over the bins, the elements are contiguous).
 *
 * @tparam ElemDim_n    Inner dimension of the flattened input array. Set to
 *                      `1` when summing scalar values.
 * @tparam BlockSize_n  Needs to be identical to the number of thread blocks.
 *                      Has to be known statically for the size of the shared memory.
 * @tparam Elem_t       Data type of the input array elements.
 * @tparam Sum_t        Data type of the output and shared memory array elements.
 * @param[in] nElems    Number of elements in the input array.
 * @param[in] arr       Input array containing data to be summed.
 * @param[out] output   Output array containing partial sums for each block.
 *                      (Each block's sum is stored in 'output[ElemDim_n * blockIdx.x]').
 */
template <int BlockSize_n, int ElemDim_n, class Elem_t, class Sum_t>
__global__ void SumVectors(Size_t nElems, const Elem_t *__restrict__ arr, Sum_t *__restrict__ output)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * BlockSize_n;
   int arraySize = nElems * ElemDim_n;
   const int nThreadsTotal = BlockSize_n * gridDim.x;
   // Each thread calculates its partial sum and writes the result to memory
   // that is shared by all threads in the block.
   Sum_t sum[ElemDim_n];
   // Don't forget the zero-initialization!
   for (int i = 0; i < ElemDim_n; i += 1) {
      sum[i] = 0;
   }

   for (int i = gthIdx; i < arraySize; i += nThreadsTotal) {
      sum[i % ElemDim_n] += arr[i];
   }
   __shared__ Sum_t shArr[ElemDim_n * BlockSize_n];
   for (int i = 0; i < ElemDim_n; i += 1) {
      shArr[i * BlockSize_n + thIdx] = sum[i];
   }
   __syncthreads();
   // Sum within the thread blocks
   for (int size = BlockSize_n / 2; size > 0; size /= 2) { // uniform
      if (thIdx < size) {
         for (int i = 0; i < ElemDim_n; i += 1) {
            shArr[i * BlockSize_n + thIdx] += shArr[i * BlockSize_n + thIdx + size];
         }
      }
      __syncthreads();
   }
   if (thIdx == 0)
      for (int i = 0; i < ElemDim_n; i += 1) {
         output[ElemDim_n * blockIdx.x + i] = shArr[i * BlockSize_n];
      }
}

/**
 * Computes bin-wise sum and count of elements from the 'arr' array into separate output arrays
 * based on indices provided in 'x1' and 'x2' arrays, using a 2D grid-stride loop approach.
 *
 * The output arrays are of shape `[ gridDim.x, Bins1_n, Bins2_n ]` (row-major
 * flattened, meaning when iterating over the bins, the elements are
 * contiguous).
 *
 * @tparam Bins1_n             Number of bins in the first dimension.
 * @tparam Bins2_n             Number of bins in the second dimension.
 * @tparam BlockSize_n         Needs to be identical to the number of thread blocks.
 *                             Has to be known statically for the size of the shared memory.
 * @tparam Idx_t               Data type of index arrays `x1` and `x2`.
 * @tparam Elem_t              Data type of the input array `arr` elements.
 * @tparam Sum_t               Data type for bin-wise sum output.
 * @tparam Counts_t            Data type for bin-wise count output.
 * @param[in] arraySize        Size of the input arrays `x1`, `x2`, and `arr`.
 * @param[in] x1               Input array containing bin indices for the first dimension.
 * @param[in] x2               Input array containing bin indices for the second dimension.
 * @param[in] arr              Input array containing elements to be summed.
 * @param[out] outputSum       Output array for storing bin-wise sum of elements.
 * @param[out] outputCounts    Output array for storing bin-wise count of elements.
 */
template <int BlockSize_n, int Bins1_n, int Bins2_n, class Idx_t, class Elem_t, class Sum_t, class Counts_t>
__global__ void
SumBinwise2D(int arraySize, Idx_t const *__restrict__ x1, Idx_t const *__restrict__ x2, const Elem_t *__restrict__ arr,
             Sum_t *__restrict__ outputSum, Counts_t *__restrict__ outputCounts)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * BlockSize_n;
   const int nThreadsTotal = BlockSize_n * gridDim.x;
   constexpr int nBins = Bins1_n * Bins2_n;
   float sum[nBins];
   int cnt[nBins];

   // We need to do zero-initialization.
   for (int i = 0; i < nBins; i += 1) {
      sum[i] = 0;
      cnt[i] = 0;
   }

   for (int i = gthIdx; i < arraySize; i += nThreadsTotal) {
      int idx = Bins2_n * x1[i] + x2[i];
      sum[idx] += arr[i];
      cnt[idx] += 1;
   }
   __shared__ float shArrSum[nBins * BlockSize_n];
   __shared__ int shArrCnt[nBins * BlockSize_n];

   for (int i = 0; i < nBins; i += 1) {
      shArrSum[i * BlockSize_n + thIdx] = sum[i];
      shArrCnt[i * BlockSize_n + thIdx] = cnt[i];
   }

   __syncthreads();
   for (int size = BlockSize_n / 2; size > 0; size /= 2) { // uniform
      if (thIdx < size) {
         for (int i = 0; i < nBins; i += 1) {
            shArrSum[i * BlockSize_n + thIdx] += shArrSum[i * BlockSize_n + thIdx + size];
            shArrCnt[i * BlockSize_n + thIdx] += shArrCnt[i * BlockSize_n + thIdx + size];
         }
      }
      __syncthreads();
   }
   if (thIdx == 0)
      for (int i = 0; i < nBins; i += 1) {
         outputSum[nBins * blockIdx.x + i] = shArrSum[i * BlockSize_n];
         outputCounts[nBins * blockIdx.x + i] = shArrCnt[i * BlockSize_n];
      }
}

/**
 * Computes the covariance, variance of 'x', and variance of 'y' for a 2D data set.

 * The output array is of shape `[ gridDim.x, 3 ]` (row-major flattened,
 * meaning when iterating over the elements of the covariance matrix, the
 * elements are contiguous). Only three output elements are needed to store the
 * symmetric 2-by-2 covariance matrix.
 *
 * @tparam BlockSize_n    Needs to be identical to the number of thread blocks.
 *                        Has to be known statically for the size of the shared memory.
 * @param[in] arraySize   Size of the input arrays 'x' and 'y'.
 * @param[in] x           Input array containing 'x' data.
 * @param[in] y           Input array containing 'y' data.
 * @param[in] xMean       Mean of the 'x' data.
 * @param[in] yMean       Mean of the 'y' data.
 * @param[out] output     Output array to store computed covariance and variances.
 *                        (Three values are stored per block: variance of 'x', covariance, variance of 'y'.)
 */
template <int BlockSize_n>
__global__ void Covariance2D(Size_t arraySize, float const *__restrict__ x, const float *__restrict__ y, double xMean,
                             double yMean, double *__restrict__ output)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * BlockSize_n;
   const int nThreadsTotal = BlockSize_n * gridDim.x;
   double sumX2 = 0;
   double sumCov = 0;
   double sumY2 = 0;
   for (int i = gthIdx; i < arraySize; i += nThreadsTotal) {
      const double argx = double(x[i]) - xMean;
      const double argy = double(y[i]) - yMean;
      sumX2 += argx * argx;
      sumCov += argx * argy;
      sumY2 += argy * argy;
   }
   __shared__ double shArrCov[BlockSize_n];
   __shared__ double shArrX2[BlockSize_n];
   __shared__ double shArrY2[BlockSize_n];
   shArrX2[thIdx] = sumX2;
   shArrCov[thIdx] = sumCov;
   shArrY2[thIdx] = sumY2;
   __syncthreads();
   for (int size = BlockSize_n / 2; size > 0; size /= 2) { // uniform
      if (thIdx < size) {
         shArrX2[thIdx] += shArrX2[thIdx + size];
         shArrCov[thIdx] += shArrCov[thIdx + size];
         shArrY2[thIdx] += shArrY2[thIdx + size];
      }
      __syncthreads();
   }
   if (thIdx == 0) {
      output[3 * blockIdx.x + 0] = shArrX2[0] / arraySize;
      output[3 * blockIdx.x + 1] = shArrCov[0] / arraySize;
      output[3 * blockIdx.x + 2] = shArrY2[0] / arraySize;
   }
}

} // namespace Reducers

/**
 * Divides elements of the `num` array in-place by corresponding elements of
 * the `den` array.
 *
 * @tparam Num_t          Data type of the 'num' array elements.
 * @tparam Den_t          Data type of the 'den' array elements.
 * @param[in] arraySize   Size of the input arrays.
 * @param[in,out] num     Numerator array containing values to be divided.
 * @param[in] den         Denominator array containing divisor values.
 */
template <class Num_t, class Den_t>
__global__ void DivideBy(Size_t arraySize, Den_t *__restrict__ num, Num_t const *__restrict__ den)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * blockDim.x;
   const int nThreadsTotal = blockDim.x * gridDim.x;

   for (int i = gthIdx; i < arraySize; i += nThreadsTotal) {
      num[i] = num[i] / den[i];
   }
}

/**
 * Fills an output array 'output' with values from a row-wise flattened
 * two-dimensional lookup table `lut` based on input arrays `x1` and `x2`:
 *
 * `output[ i ] = lut[ n2 * x1[i] + x2[i] ]`.
 *
 * Each thread processes a portion of the input arrays using grid-stride looping.
 *
 * @tparam Idx_t              Data type of the indices.
 * @tparam Elem_t             Data type of the lookup table and output array elements.
 * @param[in] arraySize       Size of the input and output arrays.
 * @param[in] n2              Size of the second dimension of the lookup table.
 * @param[in] x1              Input array containing indices for the first dimension of the lookup table.
 * @param[in] x2              Input array containing indices for the second dimension of the lookup table.
 * @param[in] lut             Lookup table containing values.
 * @param[out] output         Output array to be filled with values from the lookup table.
 */
template <class Idx_t, class Elem_t>
__global__ void Lookup2D(Size_t arraySize, Idx_t n2, Idx_t const *__restrict__ x1, Idx_t const *__restrict__ x2,
                         Elem_t const *__restrict__ lut, Elem_t *__restrict__ output)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * blockDim.x;
   const int nThreadsTotal = blockDim.x * gridDim.x;

   for (int i = gthIdx; i < arraySize; i += nThreadsTotal) {
      output[i] = lut[n2 * x1[i] + x2[i]];
   }
}

} // namespace CudaKernels

} // namespace Detail
} // namespace RooFit

#endif
