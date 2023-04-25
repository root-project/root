#include "RHnCUDA.h"

#include "CUDAHelpers.cuh"
#include "RtypesCore.h"
#include "TError.h"
#include "TMath.h"

#include <thrust/functional.h>
#include <array>
#include <vector>
#include <utility>
#include <iostream>

namespace CUDAhist {

////////////////////////////////////////////////////////////////////////////////
/// CUDA Histogram Kernels

__device__ inline int FindFixBin(double x, const double *binEdges, int nBins, double xMin, double xMax)
{
   int bin;

   // OPTIMIZATION: can this be done with less branching?
   if (x < xMin) { //*-* underflow
      bin = 0;
   } else if (!(x < xMax)) { //*-* overflow  (note the way to catch NaN)
      bin = nBins + 1;
   } else {
      if (binEdges == NULL) { //*-* fix bins
         bin = 1 + int(nBins * (x - xMin) / (xMax - xMin));
      } else { //*-* variable bin sizes
         bin = 1 + CUDAHelpers::BinarySearchCUDA(nBins + 1, binEdges, x);
      }
   }

   return bin;
}

// Use Horner's method to calculate the bin in an n-Dimensional array.
template <unsigned int Dim>
__device__ inline int GetBin(int i, CUDAhist::RAxis *axes, double *coords, int *bins)
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

template <typename T, unsigned int Dim>
__global__ void HistoKernel(T *histogram, CUDAhist::RAxis *axes, int nBins, double *coords, int *bins, double *weights,
                            unsigned int bufferSize)
{
   auto smem = CUDAHelpers::shared_memory_proxy<T>();
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int local_tid = threadIdx.x;
   unsigned int stride = blockDim.x * gridDim.x;

   // Initialize a local per-block histogram
   for (auto i = local_tid; i < nBins; i += blockDim.x) {
      smem[local_tid] = 0;
   }
   __syncthreads();

   // Fill local histogram
   for (auto i = tid; i < bufferSize; i += stride) {
      auto bin = GetBin<Dim>(i, axes, coords, bins);
      if (bin >= 0)
         atomicAdd(&smem[bin], (T)weights[i]);
   }
   __syncthreads();

   // Merge results in global histogram
   for (auto i = local_tid; i < nBins; i += blockDim.x) {
      atomicAdd(&histogram[i], smem[i]);
   }
}

// Slower histogramming, but requires less memory.
// OPTIMIZATION: consider sorting the coords array.
template <typename T, unsigned int Dim>
__global__ void HistoKernelGlobal(T *histogram, CUDAhist::RAxis *axes, int nBins, double *coords, int *bins,
                                  double *weights, unsigned int bufferSize)
{
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x;

   // Fill histogram
   for (auto i = tid; i < bufferSize; i += stride) {
      auto bin = GetBin<Dim>(i, axes, coords, bins);
      if (bin >= 0)
         atomicAdd(&histogram[bin], (T)weights[i]);
   }
}

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
__global__ void ExcludeUOverflowKernel(int *bins, double *weights, unsigned int nCoords, CUDAhist::RAxis *axes)
{
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x;

   for (auto i = tid; i < nCoords; i += stride) {
      for (auto d = 0; d < Dim; d++) {
         if (bins[i * Dim + d] <= 0 || bins[i * Dim + d] >= axes[d].fNbins - 1) {
            weights[i] = 0.;
         }
      }
   }
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
RHnCUDA<T, Dim, BlockSize>::RHnCUDA(int *ncells, double *xlow, double *xhigh, const double **binEdges)
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
      RAxis axis;
      axis.fNbins = ncells[i];
      axis.fMin = xlow[i];
      axis.fMax = xhigh[i];
      axis.kBinEdges = binEdges[i];
      fHAxes[i] = axis;

      fNbins *= ncells[i];
   }

   cudaDeviceProp prop;
   ERRCHECK(cudaGetDeviceProperties(&prop, 0));
   fMaxSmemSize = prop.sharedMemPerBlock;
   fHistoSmemSize = fNbins * sizeof(T);
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::AllocateH1D()
{
   // Allocate histogram on GPU
   ERRCHECK(cudaMalloc((void **)&fDeviceHisto, fNbins * sizeof(double)));
   ERRCHECK(cudaMemset(fDeviceHisto, 0, fNbins * sizeof(double)));

   // Allocate weights array on GPU
   ERRCHECK(cudaMalloc((void **)&fDWeights, fBufferSize * sizeof(double)));

   // Allocate array of coords to fill on GPU
   ERRCHECK(cudaMalloc((void **)&fDCoords, Dim * fBufferSize * sizeof(double)));

   // Allocate array of bins corresponding to the coords.
   ERRCHECK(cudaMalloc((void **)&fDBins, Dim * fBufferSize * sizeof(int)));

   // Allocate axes on the GPU
   ERRCHECK(cudaMalloc((void **)&fDAxes, fBufferSize * sizeof(RAxis)));
   ERRCHECK(cudaMemcpy(fDAxes, fHAxes.data(), Dim * sizeof(RAxis), cudaMemcpyHostToDevice));
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
   ERRCHECK(cudaMalloc((void **)&fDStats, kNStats * sizeof(double)));
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::Fill(const std::array<T, Dim> &coords, double w)
{
   fHCoords.insert(fHCoords.end(), coords.begin(), coords.end());
   fHWeights.push_back(w);

   // Only execute when a certain number of values are buffered to increase the GPU workload and decrease the
   // frequency of kernel launches.
   if (fHWeights.size() == fBufferSize) {
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
      if (fDIntermediateStats == NULL)
         ERRCHECK(cudaMalloc((void **)&fDIntermediateStats, numBlocks * kNStats * sizeof(double)));
      else
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
   unsigned int size = fmin(fBufferSize, fHWeights.size());
   int numBlocks = size % BlockSize == 0 ? size / BlockSize : size / BlockSize + 1;

   fEntries += size;

   ERRCHECK(cudaMemcpy(fDCoords, fHCoords.data(), Dim * size * sizeof(double), cudaMemcpyHostToDevice));
   ERRCHECK(cudaMemcpy(fDWeights, fHWeights.data(), size * sizeof(double), cudaMemcpyHostToDevice));

   if (fHistoSmemSize > fMaxSmemSize) {
      HistoKernelGlobal<T, Dim>
         <<<numBlocks, BlockSize>>>(fDeviceHisto, fDAxes, fNbins, fDCoords, fDBins, fDWeights, size);
   } else {
      HistoKernel<T, Dim>
         <<<numBlocks, BlockSize, fHistoSmemSize>>>(fDeviceHisto, fDAxes, fNbins, fDCoords, fDBins, fDWeights, size);
   }
   ERRCHECK(cudaPeekAtLastError());

   GetStats(size);

   fHCoords.clear();
   fHWeights.clear();
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
int RHnCUDA<T, Dim, BlockSize>::RetrieveResults(double *histResult, double *statsResult)
{
   // Fill the histogram with remaining values in the buffer.
   if (fHWeights.size() > 0) {
      ExecuteCUDAHisto();
   }

   // Copy back results from GPU to CPU.
   ERRCHECK(cudaMemcpy(histResult, fDeviceHisto, fNbins * sizeof(double), cudaMemcpyDeviceToHost));
   ERRCHECK(cudaMemcpy(statsResult, fDStats, kNStats * sizeof(double), cudaMemcpyDeviceToHost));

   // TODO: Free device pointers?

   return fEntries;
}

#include "RHnCUDA-impl.cu"

} // namespace CUDAhist
