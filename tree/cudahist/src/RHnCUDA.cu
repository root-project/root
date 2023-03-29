#include <vector>
#include <stdio.h>
#include <thrust/functional.h>

#include "RHnCUDA.h"
#include "CUDAHelpers.cuh"
#include "TMath.h"

using namespace std;

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

// Use Horner's method to calculate the bin in an n-dimensional array.
__device__ inline int GetBin(int i, int dim, RHnCUDA::RAxis *axes, double *cells)
{
   auto *x = &cells[i * dim];

   auto d = dim-1;
   auto bin = FindFixBin(x[d], axes[d].kBinEdges, axes[d].fNcells - 2, axes[d].fMin, axes[d].fMax);
   // printf("dim:%d  bin:%d x:%f ncells:%d min:%f max:%f\n", d, bin, x[d], axes[d].fNcells, axes[d].fMin, axes[d].fMax);

   for (d--; d >= 0; d--) {
      auto binD = FindFixBin(x[d], axes[d].kBinEdges, axes[d].fNcells - 2, axes[d].fMin, axes[d].fMax);
      if (binD < 0) return -1;
      // printf("dim:%d  bin:%d x:%f ncells:%d min:%f max:%f\n", d, bin, x[d], axes[d].fNcells, axes[d].fMin, axes[d].fMax);
      bin = bin * axes[d].fNcells + binD;
   }

   return bin;
}

__global__ void HistoKernel(double *histogram, int dim, RHnCUDA::RAxis *axes, int nBins, double *cells,
                            double *w, unsigned int bufferSize, RHnCUDA::Stats *stats)
{
   extern __shared__ double smem[];
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   int local_tid = threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   // Initialize a local per-block histogram
   if (local_tid < nBins)
      smem[local_tid] = 0;
   __syncthreads();

   // Fill local histogram
   for (int i = tid; i < bufferSize; i += stride) {
      auto bin = GetBin(i, dim, axes, cells);
      // printf("%d: add %f to bin %d\n", tid, w[i], bin);
      if (bin < 0) continue;
      atomicAdd(&smem[bin], w[i]);

      // Don't include u/overflow bins in stats.
      // TODO: maybe not very clean to be modifying the input weights array.
      if (bin == 0 || bin > nBins)
         w[i] = 0;
   }
   __syncthreads();

   // Merge results in global histogram
   if (local_tid < nBins) {
      atomicAdd(&histogram[local_tid], smem[local_tid]);
   }
}

// TODO: worked for 1 dimension, need to adapt to n-dimensional case.
// template <unsigned int BlockSize, typename ValType>
// __global__ void GetStatsKernel(ValType *cells, double *weights, double *oSumw, double *oSumw2, double
// *oSumwx,
//                                double *oSumwx2, unsigned int n)
// {
//    extern __shared__ double sdata[];

//    unsigned int tid = threadIdx.x;
//    unsigned int i = blockIdx.x * (BlockSize * 2) + tid;
//    unsigned int gridSize = (BlockSize * 2) * gridDim.x;

//    // Only one shared memory buffer can be declared so we index with an offset to differentiate multiple arrays.
//    double *sdataSumw = &sdata[0];
//    double *sdataSumw2 = &sdata[blockDim.x];
//    double *sdataSumwx = &sdata[2 * blockDim.x];
//    double *sdataSumwx2 = &sdata[3 * blockDim.x];

//    // if (i == 0) {
//    //    printf("blockdim:%d griddim:%d gridsize:%d\n", blockDim.x, gridDim.x, gridSize);
//    // }

//    // Operate on local var instead of sdata to avoid illegal memory accesses?
//    double rsumw = 0, rsumw2 = 0, rsumwx = 0, rsumwx2 = 0.;

//    while (i < n) {
//       rsumw += weights[i];
//       rsumw2 = weights[i] * weights[i];
//       rsumwx = weights[i] * cells[i];
//       rsumwx2 = weights[i] * cells[i] * cells[i];

//       if (i + BlockSize < n) {
//          rsumw += weights[i + BlockSize];
//          rsumw2 += weights[i + BlockSize] * weights[i + BlockSize];
//          rsumwx += weights[i + BlockSize] * cells[i + BlockSize];
//          rsumwx2 += weights[i + BlockSize] * cells[i + BlockSize] * cells[i + BlockSize];
//       }

//       i += gridSize;
//    }
//    sdataSumw[tid] = rsumw;
//    sdataSumw2[tid] = rsumw2;
//    sdataSumwx[tid] = rsumwx;
//    sdataSumwx2[tid] = rsumwx2;
//    __syncthreads();

//    CUDAHelpers::UnrolledReduce<BlockSize, CUDAHelpers::plus<double>, double>(sdataSumw, tid);
//    CUDAHelpers::UnrolledReduce<BlockSize, CUDAHelpers::plus<double>, double>(sdataSumw2, tid);
//    CUDAHelpers::UnrolledReduce<BlockSize, CUDAHelpers::plus<double>, double>(sdataSumwx, tid);
//    CUDAHelpers::UnrolledReduce<BlockSize, CUDAHelpers::plus<double>, double>(sdataSumwx2, tid);

//    // The first thread of each block writes the sum of the block into the global device array.
//    if (tid == 0) {
//       oSumw[blockIdx.x] = sdataSumw[0];
//       oSumw2[blockIdx.x] = sdataSumw2[0];
//       oSumwx[blockIdx.x] = sdataSumwx[0];
//       oSumwx2[blockIdx.x] = sdataSumwx2[0];
//    }
// }

// template __global__ void GetStatsKernel<512, double>(double *cells, double *weights, double *oSumw,
//                                                        double *oSumw2, double *oSumwx, double *oSumwx2, unsigned int
//                                                        n);

// __global__ void H1DKernelGlobal(double *histogram, double *binEdges, double xMin, double xMax, int nCells,
//                                 double *cells, double *w, unsigned int bufferSize)
// {
//    int tid = threadIdx.x + blockDim.x * blockIdx.x;
//    int stride = blockDim.x * gridDim.x;

//    // Fill histogram
//    for (int i = tid; i < bufferSize; i += stride) {
//       auto bin = FindFixBin(cells[i], binEdges, nCells, xMin, xMax);
//       // printf("%d: add %f to bin %d\n", tid, w[i], bin);
//       atomicAdd(&histogram[bin], w[i]);
//    }
// }

////////////////////////////////////////////////////////////////////////////////
/// RHnCUDA constructor

RHnCUDA::RHnCUDA(int dim, int *ncells, double *xlow, double *xhigh, const double **binEdges) : kDim(dim)
{
   fThreadBlockSize = 512;
   fBufferSize = 10000;

   fNbins = 1;
   fEntries = 0;
   fDeviceStats = NULL;
   fDeviceAxes = NULL;
   fCells.reserve(dim * fBufferSize);
   fWeights.reserve(fBufferSize);

   // Initialize axis descriptors.
   for (int i = 0; i < dim; i++) {
      RAxis axis;
      axis.fNcells = ncells[i];
      axis.fMin = xlow[i];
      axis.fMax = xhigh[i];
      axis.kBinEdges = binEdges[i];
      fAxes.push_back(axis);

      fNbins *= ncells[i];
      if (getenv("DBG")) printf("ncells:%d min:%f max:%f ", axis.fNcells, axis.fMin, axis.fMax);
   }
   if (getenv("DBG")) printf("nbins:%d dim:%d\n", fNbins, kDim);
}

// Allocate buffers for histogram on GPU
void RHnCUDA::AllocateH1D()
{
   // Allocate histogram on GPU
   ERRCHECK(cudaMalloc((void **)&fDeviceHisto, fNbins * sizeof(double)));
   ERRCHECK(cudaMemset(fDeviceHisto, 0, fNbins * sizeof(double)));

   // Allocate weights array on GPU
   ERRCHECK(cudaMalloc((void **)&fDeviceWeights, fBufferSize * sizeof(double)));

   // Allocate array of cells to fill on GPU
   ERRCHECK(cudaMalloc((void **)&fDeviceCells, kDim * fBufferSize * sizeof(double)));

   // Allocate axes on the GPU
   ERRCHECK(cudaMalloc((void **)&fDeviceAxes, fBufferSize * sizeof(RAxis)));
   ERRCHECK(cudaMemcpy(fDeviceAxes, fAxes.data(), kDim * sizeof(RAxis), cudaMemcpyHostToDevice));
   for (int i = 0; i < kDim; i++) {
      // Allocate memory for BinEdges array.
      if (fAxes[i].kBinEdges != NULL) {
         // ERRCHECK(cudaMalloc((void **)fDeviceAxes[i].fBinEdges, fAxes[i].fNcells * sizeof(double)));
         // ERRCHECK(cudaMemcpy(&fDeviceAxes[i].fBinEdges, fAxes[i].fBinEdges, fAxes[i].fNcells * sizeof(double),
         //                     cudaMemcpyHostToDevice));

         double *deviceBinEdges;
         ERRCHECK(cudaMalloc((void **)&deviceBinEdges, fAxes[i].fNcells * sizeof(double)));
         ERRCHECK(cudaMemcpy(deviceBinEdges, fAxes[i].kBinEdges, fAxes[i].fNcells * sizeof(double),
                             cudaMemcpyHostToDevice));
         ERRCHECK(cudaMemcpy(&fDeviceAxes[i].kBinEdges, &deviceBinEdges, sizeof(double *), cudaMemcpyHostToDevice));
      }
   }

   ERRCHECK(cudaMalloc((void **)&fDeviceStats, sizeof(RHnCUDA::Stats)));
   ERRCHECK(cudaMemset(fDeviceStats, 0, 5 * sizeof(double))); // set the first 5 variables in the struct to 0.
}

// TODO: worked for 1 dimension, need to adapt to n-dimensional case.
void RHnCUDA::GetStats(unsigned int size)
{
   // const unsigned int blockSize = 512;

   // int smemSize = (blockSize <= 32) ? 2 * blockSize : blockSize;
   // unsigned int numBlocks = fmax(1, ceil(size / blockSize / 2.)); // Number of blocks in grid is halved!

   // double *intermediate_sumw = NULL;
   // double *intermediate_sumw2 = NULL;
   // double *intermediate_sumwx = NULL;
   // double *intermediate_sumwx2 = NULL;
   // ERRCHECK(cudaMalloc((void **)&intermediate_sumw, numBlocks * sizeof(double)));
   // ERRCHECK(cudaMalloc((void **)&intermediate_sumw2, numBlocks * sizeof(double)));
   // ERRCHECK(cudaMalloc((void **)&intermediate_sumwx, numBlocks * sizeof(double)));
   // ERRCHECK(cudaMalloc((void **)&intermediate_sumwx2, numBlocks * sizeof(double)));

   // GetStatsKernel<blockSize, double><<<numBlocks, blockSize, 4 * smemSize * sizeof(double)>>>(
   //    fDeviceCells, fDeviceWeights, intermediate_sumw, intermediate_sumw2, intermediate_sumwx, intermediate_sumwx2,
   //    size);
   // ERRCHECK(cudaGetLastError());
   // // OPTIMIZATION: final reduction in a single kernel?
   // CUDAHelpers::ReductionKernel<blockSize, CUDAHelpers::plus<double>, double, false>
   //    <<<1, blockSize, smemSize * sizeof(double)>>>(intermediate_sumw, &(fDeviceStats->fTsumw), numBlocks, 0.);
   // ERRCHECK(cudaGetLastError());
   // CUDAHelpers::ReductionKernel<blockSize, CUDAHelpers::plus<double>, double, false>
   //    <<<1, blockSize, smemSize * sizeof(double)>>>(intermediate_sumw2, &(fDeviceStats->fTsumw2), numBlocks, 0.);
   // ERRCHECK(cudaGetLastError());
   // CUDAHelpers::ReductionKernel<blockSize, CUDAHelpers::plus<double>, double, false>
   //    <<<1, blockSize, smemSize * sizeof(double)>>>(intermediate_sumwx, &(fDeviceStats->fTsumwx), numBlocks, 0.);
   // ERRCHECK(cudaGetLastError());
   // CUDAHelpers::ReductionKernel<blockSize, CUDAHelpers::plus<double>, double, false>
   //    <<<1, blockSize, smemSize * sizeof(double)>>>(intermediate_sumwx2, &(fDeviceStats->fTsumwx2), numBlocks, 0.);
   // ERRCHECK(cudaGetLastError());

   // ERRCHECK(cudaFree(intermediate_sumw));
   // ERRCHECK(cudaFree(intermediate_sumw2));
   // ERRCHECK(cudaFree(intermediate_sumwx));
   // ERRCHECK(cudaFree(intermediate_sumwx2));
}

void RHnCUDA::ExecuteCUDAH1D()
{
   unsigned int size = fmin(fBufferSize, fWeights.size());
   // printf("cellsize:%lu buffersize:%f Size:%f nCells:%d\n", fCells.size(), fBufferSize, size, fNcells);

   fEntries += size;

   ERRCHECK(cudaMemcpy(fDeviceCells, fCells.data(), kDim * size * sizeof(double), cudaMemcpyHostToDevice));
   ERRCHECK(cudaMemcpy(fDeviceWeights, fWeights.data(), size * sizeof(double), cudaMemcpyHostToDevice));

   // TODO: this kernel fails with invalid argument when  fNbins * sizeof(double) exceeds max shared mem size.
   HistoKernel<<<size / fThreadBlockSize + 1, fThreadBlockSize, fNbins * sizeof(double)>>>(
      fDeviceHisto, kDim, fDeviceAxes, fNbins, fDeviceCells, fDeviceWeights, size, fDeviceStats);
   ERRCHECK(cudaGetLastError());
   GetStats(size);

   fCells.clear();
   fWeights.clear();
}

void RHnCUDA::Fill(std::vector<double> coords, double w)
{
   // Fill expects kDim number of coordinates.
   if (coords.size() != kDim)
      return;

   fCells.insert(fCells.end(), coords.begin(), coords.end());
   fWeights.push_back(w);

   // Only execute when a certain number of values are buffered to increase the GPU workload and decrease the
   // frequency of kernel launches.
   if (fWeights.size() == fBufferSize) {
      ExecuteCUDAH1D();
   }
}

void RHnCUDA::Fill(std::vector<double> coords)
{
   Fill(coords, 1.0);
}

// Copy back results on GPU to CPU.
int RHnCUDA::RetrieveResults(double *histResult, double *statsResult)
{
   // Fill the histogram with remaining values in the buffer.
   if (fCells.size() > 0) {
      ExecuteCUDAH1D();
   }

   // Copy back results from GPU to CPU.
   Stats stats;
   ERRCHECK(cudaMemcpy(histResult, fDeviceHisto, fNbins * sizeof(double), cudaMemcpyDeviceToHost));
   ERRCHECK(cudaMemcpy(&stats, fDeviceStats, sizeof(Stats), cudaMemcpyDeviceToHost));
   statsResult[0] = stats.fTsumw;
   statsResult[1] = stats.fTsumw2;
   statsResult[2] = stats.fTsumwx;
   statsResult[3] = stats.fTsumwx2;

   // // TODO: Free device pointers?

   return fEntries;
}
