#include "RHnCUDA.h"

#include "CUDAHelpers.cuh"
#include "TError.h"
#include "TMath.h"

#include <thrust/functional.h>
#include <array>
#include <vector>
#include <iostream>

namespace ROOT {
namespace Experimental {
////////////////////////////////////////////////////////////////////////////////
/// CUDA kernels

__device__ inline int FindFixBin(double x, const double *binEdges, int nBins, double xMin, double xMax)
{
   int bin;

   // OPTIMIZATION: can this be done with less branching?
   if (x < xMin) { // underflow
      bin = 0;
   } else if (!(x < xMax)) { // overflow  (note the way to catch NaN)
      bin = nBins + 1;
   } else {
      if (binEdges == NULL) { // fix bins
         bin = 1 + int(nBins * (x - xMin) / (xMax - xMin));
      } else { // variable bin sizes
         bin = 1 + CUDAHelpers::BinarySearch(nBins + 1, binEdges, x);
      }
   }

   return bin;
}

// Use Horner's method to calculate the bin in an n-Dimensional array.
template <unsigned int Dim>
__device__ inline int GetBin(int i, AxisDescriptor *axes, double *coords, int *bins)
{
   auto *x = &coords[i * Dim];

   auto bin = 0;
   for (int d = Dim - 1; d >= 0; d--) {
      auto binD = FindFixBin(x[d], axes[d].kBinEdges, axes[d].fNbins - 2, axes[d].fMin, axes[d].fMax);
      bins[i * Dim + d] = binD;

      if (binD < 0) {
         return -1;
      }

      bin = bin * axes[d].fNbins + binD;
   }

   return bin;
}

///////////////////////////////////////////
/// Device kernels for incrementing a bin.

template <typename T>
__device__ inline void AddBinContent(T *histogram, int bin, double weight)
{
   atomicAdd(&histogram[bin], (T)weight);
}

// TODO:
// template <>
// __device__ inline void AddBinContent(char *histogram, int bin, char weight)
// {
//    int newVal = histogram[bin] + int(weight);
//    if (newVal > -128 && newVal < 128) {
//       atomicExch(&histogram[bin], (char) newVal);
//       return;
//    }
//    if (newVal < -127)
//       atomicExch(&histogram[bin], (char) -127);
//    if (newVal > 127)
//       atomicExch(&histogram[bin], (char) 127);
// }

template <>
__device__ inline void AddBinContent(short *histogram, int bin, double weight)
{
   // There is no CUDA atomicCAS for short so we need to operate on integers... (Assumes little endian)
   short *addr = &histogram[bin];
   int *addrInt = (int *)((char *)addr - ((size_t)addr & 2));
   int old = *addrInt, assumed, newVal, overwrite;

   do {
      assumed = old;

      if ((size_t)addr & 2) {
         newVal = (assumed >> 16) + (int)weight; // extract short from upper 16 bits
         overwrite = assumed & 0x0000ffff;       // clear upper 16 bits
         if (newVal > -32768 && newVal < 32768)
            overwrite |= (newVal << 16); // Set upper 16 bits to newVal
         else if (newVal < -32767)
            overwrite |= 0x80010000; // Set upper 16 bits to min short (-32767)
         else
            overwrite |= 0x7fff0000; // Set upper 16 bits to max short (32767)
      } else {
         newVal = (((assumed & 0xffff) << 16) >> 16) + (int)weight; // extract short from lower 16 bits + sign extend
         overwrite = assumed & 0xffff0000;                          // clear lower 16 bits
         if (newVal > -32768 && newVal < 32768)
            overwrite |= (newVal & 0xffff); // Set lower 16 bits to newVal
         else if (newVal < -32767)
            overwrite |= 0x00008001; // Set lower 16 bits to min short (-32767)
         else
            overwrite |= 0x00007fff; // Set lower 16 bits to max short (32767)
      }

      old = atomicCAS(addrInt, assumed, overwrite);
   } while (assumed != old);
}

template <>
__device__ inline void AddBinContent(int *histogram, int bin, double weight)
{
   int old = histogram[bin], assumed;
   long newVal;

   do {
      assumed = old;
      newVal = max(long(-INT_MAX), min(assumed + long(weight), long(INT_MAX)));
      old = atomicCAS(&histogram[bin], assumed, newVal);
   } while (assumed != old); // Repeat on failure/when the bin was already updated by another thread
}

///////////////////////////////////////////
/// Histogram filling kernels

template <typename T, unsigned int Dim>
__global__ void HistoKernel(T *histogram, AxisDescriptor *axes, int nBins, double *coords, int *bins, double *weights,
                            unsigned int bufferSize)
{
   auto sMem = CUDAHelpers::shared_memory_proxy<T>();
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int localTid = threadIdx.x;
   unsigned int stride = blockDim.x * gridDim.x; // total number of threads

   // Initialize a local per-block histogram
   for (auto i = localTid; i < nBins; i += blockDim.x) {
      sMem[i] = 0;
   }
   __syncthreads();

   // Fill local histogram
   for (auto i = tid; i < bufferSize; i += stride) {
      auto bin = GetBin<Dim>(i, axes, coords, bins);
      if (bin >= 0)
         AddBinContent<T>(sMem, bin, weights[i]);
   }
   __syncthreads();

   // Merge results in global histogram
   for (auto i = localTid; i < nBins; i += blockDim.x) {
      AddBinContent<T>(histogram, i, sMem[i]);
   }
}

// Slower histogramming, but requires less memory.
// OPTIMIZATION: consider sorting the coords array.
template <typename T, unsigned int Dim>
__global__ void HistoKernelGlobal(T *histogram, AxisDescriptor *axes, int nBins, double *coords, int *bins,
                                  double *weights, unsigned int bufferSize)
{
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x;

   // Fill histogram
   for (auto i = tid; i < bufferSize; i += stride) {
      auto bin = GetBin<Dim>(i, axes, coords, bins);
      if (bin >= 0)
         AddBinContent<T>(histogram, bin, weights[i]);
   }
}

///////////////////////////////////////////
/// Statistics calculation kernels

template <unsigned int BlockSize>
__global__ void GetSumW(double *weights, unsigned int nCoords, double *fDIntermediateStats)
{
   // Tsumw
   CUDAHelpers::ReduceBase<BlockSize>(
      weights, fDIntermediateStats, nCoords, [](unsigned int i, double r, double w) { return r + w; },
      CUDAHelpers::Plus<double>(), 0.);
}

// Calculates Tsumw2
template <unsigned int BlockSize>
__global__ void GetSumW2(double *weights, unsigned int nCoords, double *fDIntermediateStats)
{
   CUDAHelpers::ReduceBase<BlockSize>(
      weights, &fDIntermediateStats[gridDim.x], nCoords, [](unsigned int i, double r, double w) { return r + w * w; },
      CUDAHelpers::Plus<double>(), 0.);
}

// Multiplies weight with coordinate of current axis. E.g., for Dim = 2 this computes Tsumwx and Tsumwy
template <unsigned int Dim, unsigned int BlockSize>
__global__ void
GetSumWAxis(int axis, int is_offset, double *coords, double *weights, unsigned int nCoords, double *fDIntermediateStats)
{
   CUDAHelpers::ReduceBase<BlockSize>(
      weights, &fDIntermediateStats[is_offset], nCoords,
      [&coords, &axis](unsigned int i, double r, double w) { return r + w * coords[i * Dim + axis]; },
      CUDAHelpers::Plus<double>(), 0.);
}

// Multiplies weight with coordinate of current axis. E.g., for Dim = 2 this computes Tsumwx and Tsumwy
template <unsigned int Dim, unsigned int BlockSize>
__global__ void GetSumWAxis2(int axis, int is_offset, double *coords, double *weights, unsigned int nCoords,
                             double *fDIntermediateStats)
{
   CUDAHelpers::ReduceBase<BlockSize>(
      weights, &fDIntermediateStats[is_offset], nCoords,
      [&coords, &axis](unsigned int i, double r, double w) {
         return r + w * coords[i * Dim + axis] * coords[i * Dim + axis];
      },
      CUDAHelpers::Plus<double>(), 0.);
}

// Multiplies coordinate of current axis with the "previous" axis. E.g., for Dim = 2 this computes Tsumwxy
template <unsigned int Dim, unsigned int BlockSize>
__global__ void GetSumWAxisAxis(int axis1, int axis2, int is_offset, double *coords, double *weights,
                                unsigned int nCoords, double *fDIntermediateStats)
{
   CUDAHelpers::ReduceBase<BlockSize>(
      weights, &fDIntermediateStats[is_offset], nCoords,
      [&coords, &axis1, &axis2](unsigned int i, double r, double w) {
         return r + w * coords[i * Dim + axis1] * coords[i * Dim + axis2];
      },
      CUDAHelpers::Plus<double>(), 0.);
}

// Nullify weights of under/overflow bins to exclude them from stats
template <unsigned int Dim, unsigned int BlockSize>
__global__ void ExcludeUOverflowKernel(int *bins, double *weights, unsigned int nCoords, AxisDescriptor *axes)
{
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x;

   for (auto i = tid; i < nCoords * Dim; i += stride) {
      if (bins[i] <= 0 || bins[i] >= axes[i % Dim].fNbins - 1) {
         weights[i / Dim] = 0.;
      }
   }
}

///////////////////////////////////////////
/// RHnCUDA

template <typename T, unsigned int Dim, unsigned int BlockSize>
RHnCUDA<T, Dim, BlockSize>::RHnCUDA(std::array<int, Dim> ncells, std::array<double, Dim> xlow,
                                    std::array<double, Dim> xhigh, const double **binEdges)
   : kNStats([]() {
        // Sum of weights (squared) + sum of weight * bin (squared) per axis + sum of weight * binAx1 * binAx2 for
        // all axis combinations
        return Dim > 1 ? 2 + 2 * Dim + TMath::Binomial(Dim, 2) : 2 + 2 * Dim;
     }()),
     kStatsSmemSize((BlockSize <= 32) ? 2 * BlockSize * sizeof(double) : BlockSize * sizeof(double))
{
   fBufferSize = 10000;

   fNbins = 1;
   fEntries = 0;
   fDIntermediateStats = NULL;
   fDStats = NULL;
   fDAxes = NULL;
   fHCoords.reserve(Dim * fBufferSize);
   fHWeights.reserve(fBufferSize);

   // Initialize axis descriptors.
   for (auto i = 0; i < Dim; i++) {
      AxisDescriptor axis;
      axis.fNbins = ncells[i];
      axis.fMin = xlow[i];
      axis.fMax = xhigh[i];
      if (binEdges != NULL)
         axis.kBinEdges = binEdges[i];
      else
         axis.kBinEdges = NULL;

      fHAxes[i] = axis;
      fNbins *= ncells[i];
   }

   cudaDeviceProp prop;
   ERRCHECK(cudaGetDeviceProperties(&prop, 0));
   fMaxSmemSize = prop.sharedMemPerBlock;
   fHistoSmemSize = fNbins * sizeof(T);

   AllocateBuffers();
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
RHnCUDA<T, Dim, BlockSize>::~RHnCUDA()
{
   ERRCHECK(cudaFree(fDHistogram));
   ERRCHECK(cudaFree(fDAxes));
   ERRCHECK(cudaFree(fDCoords));
   ERRCHECK(cudaFree(fDWeights));
   ERRCHECK(cudaFree(fDBins));
   ERRCHECK(cudaFree(fDIntermediateStats));
   ERRCHECK(cudaFree(fDStats));
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::AllocateBuffers()
{
   // Allocate histogram on GPU
   ERRCHECK(cudaMalloc((void **)&fDHistogram, fNbins * sizeof(T)));
   ERRCHECK(cudaMemset(fDHistogram, 0, fNbins * sizeof(T)));

   // Allocate weights array on GPU
   ERRCHECK(cudaMalloc((void **)&fDWeights, fBufferSize * sizeof(double)));

   // Allocate array of coords to fill on GPU
   ERRCHECK(cudaMalloc((void **)&fDCoords, Dim * fBufferSize * sizeof(double)));

   // Allocate array of bins corresponding to the coords.
   ERRCHECK(cudaMalloc((void **)&fDBins, Dim * fBufferSize * sizeof(int)));

   // Allocate axes on the GPU
   ERRCHECK(cudaMalloc((void **)&fDAxes, Dim * sizeof(AxisDescriptor)));
   ERRCHECK(cudaMemcpy(fDAxes, fHAxes.data(), Dim * sizeof(AxisDescriptor), cudaMemcpyHostToDevice));
   for (auto i = 0; i < Dim; i++) {
      // Allocate memory for BinEdges array.
      if (fHAxes[i].kBinEdges != NULL) {
         double *deviceBinEdges;
         ERRCHECK(cudaMalloc((void **)&deviceBinEdges, fHAxes[i].fNbins * sizeof(double)));
         ERRCHECK(
            cudaMemcpy(deviceBinEdges, fHAxes[i].kBinEdges, fHAxes[i].fNbins * sizeof(double), cudaMemcpyHostToDevice));
         ERRCHECK(cudaMemcpy(&fDAxes[i].kBinEdges, &deviceBinEdges, sizeof(double *), cudaMemcpyHostToDevice));
      }
   }

   // Allocate array with (intermediate) results of the stats for each block.
   ERRCHECK(cudaMalloc((void **)&fDIntermediateStats, ceil(fBufferSize / BlockSize / 2.) * kNStats * sizeof(double)));
   ERRCHECK(cudaMalloc((void **)&fDStats, kNStats * sizeof(double)));
   ERRCHECK(cudaMemset(fDStats, 0, kNStats * sizeof(double)));
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::Fill(const std::array<double, Dim> &coords, double w)
{
   auto bufferIdx = fEntries % fBufferSize;
   std::copy(coords.begin(), coords.end(), &fHCoords[bufferIdx * Dim]);
   fHWeights[bufferIdx] = w;

   // Only execute when a certain number of values are buffered to increase the GPU workload and decrease the
   // frequency of kernel launches.
   fEntries++;
   if (fEntries % fBufferSize == 0) {
      ExecuteCUDAHisto();
   }
}

unsigned int nextPow2(unsigned int x)
{
   --x;
   x |= x >> 1;
   x |= x >> 2;
   x |= x >> 4;
   x |= x >> 8;
   x |= x >> 16;
   return ++x;
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::GetStats(unsigned int size)
{
   // Number of blocks in grid is halved, because each thread loads two elements from global memory.
   int numBlocks = fmax(1, ceil(size / BlockSize / 2.));

   ExcludeUOverflowKernel<Dim, BlockSize><<<fmax(1, size / BlockSize), BlockSize>>>(fDBins, fDWeights, size, fDAxes);
   ERRCHECK(cudaPeekAtLastError());

   double *resultArray;
   if (numBlocks > 1) {
      ERRCHECK(cudaMemset(fDIntermediateStats, 0, numBlocks * kNStats * sizeof(double)));
      resultArray = fDIntermediateStats;
   } else {
      resultArray = fDStats;
   }

   // OPTIMIZATION: interleave/change order of computation of different stats? or parallelize via
   // streams. Need to profile first.
   GetSumW<BlockSize><<<numBlocks, BlockSize, kStatsSmemSize>>>(fDWeights, size, resultArray);
   ERRCHECK(cudaPeekAtLastError());
   GetSumW2<BlockSize><<<numBlocks, BlockSize, kStatsSmemSize>>>(fDWeights, size, resultArray);
   ERRCHECK(cudaPeekAtLastError());

   auto is_offset = 2 * numBlocks;
   for (auto d = 0; d < Dim; d++) {
      // Multiply weight with coordinate of current axis. E.g., for Dim = 2 this computes Tsumwx and Tsumwy
      GetSumWAxis<Dim, BlockSize>
         <<<numBlocks, BlockSize, kStatsSmemSize>>>(d, is_offset, fDCoords, fDWeights, size, resultArray);
      ERRCHECK(cudaPeekAtLastError());
      is_offset += numBlocks;

      // Squares coodinate per axis. E.g., for Dim = 2 this computes Tsumwx2 and Tsumwy2
      GetSumWAxis2<Dim, BlockSize>
         <<<numBlocks, BlockSize, kStatsSmemSize>>>(d, is_offset, fDCoords, fDWeights, size, resultArray);
      ERRCHECK(cudaPeekAtLastError());
      is_offset += numBlocks;

      for (auto prev_d = 0; prev_d < d; prev_d++) {
         // Multiplies coordinate of current axis with the "previous" axis. E.g., for Dim = 2 this computes Tsumwxy
         GetSumWAxisAxis<Dim, BlockSize>
            <<<numBlocks, BlockSize, kStatsSmemSize>>>(d, prev_d, is_offset, fDCoords, fDWeights, size, resultArray);
         ERRCHECK(cudaPeekAtLastError());
         is_offset += numBlocks;
      }
   }

   if (numBlocks > 1) {
      // fDintermediateStats stores the result of the sum for each block, per statistic. We need to perform another
      // reduction to merge the per-block sums to get the total sum for each statistic.
      // OPTIMIZATION: perform final reductions for a small number of blocks on CPU?
      // TODO: the max blocksize is 1024, so for numBlocks > 1024 the final reduction has to be performed in multiple
      // stages?
      for (auto i = 0; i < kNStats; i++) {
         CUDAHelpers::ReduceSum<double>(1, nextPow2(numBlocks) / 2, &fDIntermediateStats[i * numBlocks], &fDStats[i],
                                        numBlocks, 0.);
         ERRCHECK(cudaPeekAtLastError());
      }
   }
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::ExecuteCUDAHisto()
{
   unsigned int size = (fEntries - 1) % fBufferSize + 1;
   int numBlocks = size % BlockSize == 0 ? size / BlockSize : size / BlockSize + 1;

   fEntries += size;

   ERRCHECK(cudaMemcpy(fDCoords, fHCoords.data(), Dim * size * sizeof(double), cudaMemcpyHostToDevice));
   ERRCHECK(cudaMemcpy(fDWeights, fHWeights.data(), size * sizeof(double), cudaMemcpyHostToDevice));

   if (fHistoSmemSize > fMaxSmemSize) {
      HistoKernelGlobal<T, Dim>
         <<<numBlocks, BlockSize>>>(fDHistogram, fDAxes, fNbins, fDCoords, fDBins, fDWeights, size);
   } else {
      HistoKernel<T, Dim>
         <<<numBlocks, BlockSize, fHistoSmemSize>>>(fDHistogram, fDAxes, fNbins, fDCoords, fDBins, fDWeights, size);
   }
   ERRCHECK(cudaPeekAtLastError());

   GetStats(size);

   fHCoords.clear();
   fHWeights.clear();
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::RetrieveResults(T *histResult, double *statsResult)
{
   // Fill the histogram with remaining values in the buffer.
   if (fEntries % fBufferSize != 0) {
      ExecuteCUDAHisto();
   }

   // Copy back results from GPU to CPU.
   ERRCHECK(cudaMemcpy(histResult, fDHistogram, fNbins * sizeof(T), cudaMemcpyDeviceToHost));
   ERRCHECK(cudaMemcpy(statsResult, fDStats, kNStats * sizeof(double), cudaMemcpyDeviceToHost));
}

#include "RHnCUDA-impl.cu"
} // namespace Experimental
} // namespace ROOT
