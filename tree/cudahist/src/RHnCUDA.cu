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

      // printf("Dim:%d  bin:%d x:%f ncells:%d min:%f max:%f\n", d, bin, x[d], axes[d].fNbins, axes[d].fMin,
      // axes[d].fMax);
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
      // printf("%d: add %f to bin %d\n", tid, weights[i], bin);

      // TODO: check for datatype under/overflow
      if (bin >= 0)
         atomicAdd(&smem[bin], (T)weights[i]);
   }
   __syncthreads();

   // Merge results in global histogram
   for (auto i = local_tid; i < nBins; i += blockDim.x) {
      // printf("%d: merge %f into bin %d\n", tid, smem[i], i);
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

// TODO: worked for 1 Dimension, need to adapt to n-Dimensional case.
// OPTIMIZATION: interleave/change order of computation of different stats to improve coalescing?
template <unsigned int Dim, unsigned int BlockSize>
__global__ void GetStatsKernel(double *coords, int *bins, double *weights, unsigned int nCoords, CUDAhist::RAxis *axes,
                               double *fDIntermediateStats, const int nStats)
{
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x;
   auto sdata = CUDAHelpers::shared_memory_proxy<double>();

   // if (tid == 0 && Dim == 2)
   //    printf("ncoords:%i nbinsX:%i nbinsY:%d\n", nCoords, axes[0].fNbins, axes[1].fNbins);

   // Exclude under/overflow bins from stats
   for (auto i = tid; i < nCoords; i += stride) {
      for (auto d = 0; d < Dim; d++) {
         if (bins[i * Dim + d] <= 0 || bins[i * Dim + d] >= axes[d].fNbins - 1) {
            // if (Dim == 2)
            // printf("cuda SKIP? bin:%d val:%f i:%d\n", bins[i * Dim + d], coords[i * Dim + d], i * Dim + d);

            weights[i] = 0.;
            // continue;
         }
      }
      // printf("%d\n", i);
   }

   // Tsumw
   CUDAHelpers::ReduceBase<BlockSize>(
      sdata, weights, fDIntermediateStats, nCoords, [](unsigned int i, double r, double w) { return r + w; },
      CUDAHelpers::Plus<double>());

   if (tid == 0) {
      printf("sdara1:\n");
      for (int i = 0; i <  BlockSize * nStats / 2; i++) {
         if (i % ((BlockSize <= 32) ? 2 * BlockSize * nStats : BlockSize * nStats)  == 0)
            printf("\n\n");
         printf("%i%f \n", sdata[i]);
      }
      printf("\n\n");
   }

   // Tsumw2
   unsigned int sdata_offset = blockDim.x;
   unsigned int is_offset = gridDim.x;
   CUDAHelpers::ReduceBase<BlockSize>(
      &sdata[sdata_offset], weights, &fDIntermediateStats[is_offset], nCoords,
      [](unsigned int i, double r, double w) { return r + w * w; }, CUDAHelpers::Plus<double>());

   for (auto d = 0; d < Dim; d++) {
      // Multiply weight with coordinate of current axis. E.g., for Dim = 2 this computes Tsumwx and Tsumwy
      sdata_offset += blockDim.x;
      is_offset += gridDim.x;
      CUDAHelpers::ReduceBase<BlockSize>(
         &sdata[sdata_offset], weights, &fDIntermediateStats[is_offset], nCoords,
         [&coords, &d](unsigned int i, double r, double w) { return r + w * coords[i * Dim + d]; },
         CUDAHelpers::Plus<double>());

      // Squares coodinate per axis. E.g., for Dim = 2 this computes Tsumw2 and Tsumwy2
      sdata_offset += blockDim.x;
      is_offset += gridDim.x;
      CUDAHelpers::ReduceBase<BlockSize>(
         &sdata[sdata_offset], weights, &fDIntermediateStats[is_offset], nCoords,
         [&coords, &d](unsigned int i, double r, double w) {
            return r + w * coords[i * Dim + d] * coords[i * Dim + d];
         },
         CUDAHelpers::Plus<double>());

      for (auto prev_d = d - 1; prev_d >= 0; prev_d--) {
         // Multiplies coordinate of current axis with the "previous" axis. E.g., for Dim = 2 this computes Tsumwxy
         sdata_offset += blockDim.x;
         is_offset += gridDim.x;
         CUDAHelpers::ReduceBase<BlockSize>(
            &sdata[sdata_offset], weights, &fDIntermediateStats[is_offset], nCoords,
            [&coords, &prev_d, &d](unsigned int i, double r, double w) {
               return r + w * coords[i * Dim + prev_d] * coords[i * Dim + d];
            },
            CUDAHelpers::Plus<double>());
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
     kStatsSmemSize((BlockSize <= 32) ? 2 * BlockSize * kNStats * sizeof(double) : BlockSize * kNStats * sizeof(double))
// template <typename T, unsigned int Dim, unsigned int BlockSize>
// RHnCUDA<T, Dim, BlockSize>::Initialize(int *ncells, double *xlow, double *xhigh, const double **binEdges)
// RHnCUDA::Initialize(int *ncells, double *xlow, double *xhigh, const double **binEdges)
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
   for (int i = 0; i < Dim; i++) {
      RAxis axis;
      axis.fNbins = ncells[i];
      axis.fMin = xlow[i];
      axis.fMax = xhigh[i];
      axis.kBinEdges = binEdges[i];
      fHAxes[i] = axis;

      fNbins *= ncells[i];
      if (getenv("DBG"))
         printf("\t axis %d -- ncells:%d min:%f max:%f\n", i, axis.fNbins, axis.fMin, axis.fMax);
   }

   cudaDeviceProp prop;
   ERRCHECK(cudaGetDeviceProperties(&prop, 0));
   fMaxSmemSize = prop.sharedMemPerBlock;
   fHistoSmemSize = fNbins * sizeof(T);

   if (getenv("DBG"))
      printf("nbins:%d Dim:%d nstats:%d maxsmem:%d\n", fNbins, Dim, kNStats, fMaxSmemSize);
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

// TODO: ref to array
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

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::GetStats(unsigned int size)
{
   // TODO: move this to the constructor
   // Number of blocks in grid is halved, because each thread loads two elements from global memory.
   int numBlocks = fmax(1, ceil(size / BlockSize / 2.));

   if (getenv("DBG") && atoi(getenv("DBG")) > 0)
      printf("STATS -- size:%d smemsize: %lu numblocks: %d blocksize %d\n", size, kStatsSmemSize / sizeof(double),
             numBlocks, BlockSize);

   if (fDIntermediateStats == NULL)
      ERRCHECK(cudaMalloc((void **)&fDIntermediateStats, numBlocks * kNStats * sizeof(double)));
   else
      ERRCHECK(cudaMemset(fDIntermediateStats, 0, numBlocks * kNStats * sizeof(double)));

   GetStatsKernel<Dim, BlockSize><<<numBlocks, BlockSize, kStatsSmemSize>>>(fDCoords, fDBins, fDWeights, size, fDAxes,
                                                                            fDIntermediateStats, kNStats);
   ERRCHECK(cudaPeekAtLastError());

   // OPTIMIZATION: final reduction in a single kernel?
   for (auto i = 0; i < kNStats; i++) {
      CUDAHelpers::ReductionKernel<BlockSize, double, false><<<1, BlockSize, kStatsSmemSize>>>(
         &fDIntermediateStats[i * numBlocks], &fDStats[i], numBlocks, CUDAHelpers::Plus<double>(), 0.);
      ERRCHECK(cudaPeekAtLastError());
   }
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::ExecuteCUDAHisto()
{
   unsigned int size = fmin(fBufferSize, fHWeights.size());
   int numBlocks = size % BlockSize == 0 ? size / BlockSize : size / BlockSize + 1;

   if (getenv("DBG") && atoi(getenv("DBG")) > 2) {
      printf("HISTO -- cellsize:%lu buffersize:%d Size:%d nCells:%d nBlocks:%d smemsize:%u\n", fHCoords.size(),
             fBufferSize, size, fNbins, numBlocks, fHistoSmemSize);
   }

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
