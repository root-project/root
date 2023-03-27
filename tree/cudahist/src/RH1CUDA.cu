#include <vector>
#include <stdio.h>
#include <thrust/functional.h>

#include "RH1CUDA.h"
#include "CUDAHelpers.cuh"
#include "TMath.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
/// Find bin number corresponding to abscissa x. NOTE: this method does not work with alphanumeric bins !!!
///
/// If x is underflow or overflow, attempt to extend the axis if TAxis::kCanExtend is true.
/// Otherwise, return 0 or fNbins+1.

// Int_t TAxis::FindBin(Double_t x)
// {
//    Int_t bin;
//    // NOTE: This should not be allowed for Alphanumeric histograms,
//    // but it is heavily used (legacy) in the TTreePlayer to fill alphanumeric histograms.
//    // but in case of alphanumeric do-not extend the axis. It makes no sense
//    if (IsAlphanumeric() && gDebug) Info("FindBin","Numeric query on alphanumeric axis - Sorting the bins or extending
//    the axes / rebinning can alter the correspondence between the label and the bin interval."); if (x < fXmin) {
//    //*-* underflow
//       bin = 0;
//       if (fParent == 0) return bin;
//       if (!CanExtend() || IsAlphanumeric() ) return bin;
//       ((TH1*)fParent)->ExtendAxis(x,this);
//       return FindFixBin(x);
//    } else  if ( !(x < fXmax)) {     //*-* overflow  (note the way to catch NaN)
//       bin = fNbins+1;
//       if (fParent == 0) return bin;
//       if (!CanExtend() || IsAlphanumeric() ) return bin;
//       ((TH1*)fParent)->ExtendAxis(x,this);
//       return FindFixBin(x);
//    } else {
//       if (!fXbins.fN) {        //*-* fix bins
//          bin = 1 + int (fNbins*(x-fXmin)/(fXmax-fXmin) );
//       } else {                  //*-* variable bin sizes
//          //for (bin =1; x >= fXbins.fArray[bin]; bin++);
//          bin = 1 + TMath::BinarySearch(fXbins.fN,fXbins.fArray,x);
//       }
//    }
//    return bin;

////////////////////////////////////////////////////////////////////////////////
/// CUDA Histogram Kernels

__device__ inline Int_t FindFixBin(Double_t x, Double_t *binEdges, Int_t nBins, Double_t xMin, Double_t xMax)
{
   Int_t bin;
   Int_t nCells = nBins - 2; // number of bins excluding U/Overflow

   // TODO: optimization -> can this be done without branching?
   if (x < xMin) { //*-* underflow
      bin = 0;
   } else if (!(x < xMax)) { //*-* overflow  (note the way to catch NaN)
      bin = nCells + 1;
   } else {
      if (binEdges == NULL) { //*-* fix bins
         bin = 1 + int(nCells * (x - xMin) / (xMax - xMin));
      } else { //*-* variable bin sizes
         bin = 1 + CUDAHelpers::BinarySearchCUDA(nBins - 1, binEdges, x);
      }
   }

   return bin;
}

__global__ void HistoKernel(Double_t *histogram, Double_t *binEdges, Double_t xMin, Double_t xMax, Int_t nCells,
                            Double_t *cells, Double_t *w, UInt_t bufferSize, HistStats *stats)
{
   extern __shared__ Double_t smem[];
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   int local_tid = threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   // Initialize a local per-block histogram
   if (local_tid < nCells)
      smem[local_tid] = 0;
   __syncthreads();

   // Fill local histogram
   for (int i = tid; i < bufferSize; i += stride) {
      auto bin = FindFixBin(cells[i], binEdges, nCells, xMin, xMax);
      printf("bin:%d x:%f ncells:%d min:%f max:%f\n", bin, cells[i], nCells, xMin, xMax);
      atomicAdd(&smem[bin], w[i]);

      // Don't include u/overflow bins in stats.
      // TODO: maybe not very clean to be modifying the input weights array.
      if (bin == 0 || bin > nCells)
         w[i] = 0;
   }
   __syncthreads();

   // Merge results in global histogram
   if (local_tid < nCells) {
      atomicAdd(&histogram[local_tid], smem[local_tid]);
   }
}

template <UInt_t BlockSize, typename ValType>
__global__ void GetStatsKernel(ValType *cells, Double_t *weights, Double_t *oSumw, Double_t *oSumw2, Double_t *oSumwx,
                               Double_t *oSumwx2, UInt_t n)
{
   extern __shared__ Double_t sdata[];

   UInt_t tid = threadIdx.x;
   UInt_t i = blockIdx.x * (BlockSize * 2) + tid;
   UInt_t gridSize = (BlockSize * 2) * gridDim.x;

   // Only one shared memory buffer can be declared so we index with an offset to differentiate multiple arrays.
   Double_t *sdataSumw = &sdata[0];
   Double_t *sdataSumw2 = &sdata[blockDim.x];
   Double_t *sdataSumwx = &sdata[2 * blockDim.x];
   Double_t *sdataSumwx2 = &sdata[3 * blockDim.x];

   // if (i == 0) {
   //    printf("blockdim:%d griddim:%d gridsize:%d\n", blockDim.x, gridDim.x, gridSize);
   // }

   // Operate on local var instead of sdata to avoid illegal memory accesses?
   Double_t rsumw = 0, rsumw2 = 0, rsumwx = 0, rsumwx2 = 0.;

   while (i < n) {
      rsumw += weights[i];
      rsumw2 = weights[i] * weights[i];
      rsumwx = weights[i] * cells[i];
      rsumwx2 = weights[i] * cells[i] * cells[i];

      if (i + BlockSize < n) {
         rsumw += weights[i + BlockSize];
         rsumw2 += weights[i + BlockSize] * weights[i + BlockSize];
         rsumwx += weights[i + BlockSize] * cells[i + BlockSize];
         rsumwx2 += weights[i + BlockSize] * cells[i + BlockSize] * cells[i + BlockSize];
      }

      i += gridSize;
   }
   sdataSumw[tid] = rsumw;
   sdataSumw2[tid] = rsumw2;
   sdataSumwx[tid] = rsumwx;
   sdataSumwx2[tid] = rsumwx2;
   __syncthreads();

   CUDAHelpers::UnrolledReduce<BlockSize, CUDAHelpers::plus<Double_t>, Double_t>(sdataSumw, tid);
   CUDAHelpers::UnrolledReduce<BlockSize, CUDAHelpers::plus<Double_t>, Double_t>(sdataSumw2, tid);
   CUDAHelpers::UnrolledReduce<BlockSize, CUDAHelpers::plus<Double_t>, Double_t>(sdataSumwx, tid);
   CUDAHelpers::UnrolledReduce<BlockSize, CUDAHelpers::plus<Double_t>, Double_t>(sdataSumwx2, tid);

   // The first thread of each block writes the sum of the block into the global device array.
   if (tid == 0) {
      oSumw[blockIdx.x] = sdataSumw[0];
      oSumw2[blockIdx.x] = sdataSumw2[0];
      oSumwx[blockIdx.x] = sdataSumwx[0];
      oSumwx2[blockIdx.x] = sdataSumwx2[0];
   }
}

template __global__ void GetStatsKernel<512, Double_t>(Double_t *cells, Double_t *weights, Double_t *oSumw,
                                                       Double_t *oSumw2, Double_t *oSumwx, Double_t *oSumwx2, UInt_t n);

__global__ void H1DKernelGlobal(Double_t *histogram, Double_t *binEdges, Double_t xMin, Double_t xMax, Int_t nCells,
                                Double_t *cells, Double_t *w, UInt_t bufferSize)
{
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   int stride = blockDim.x * gridDim.x;

   // Fill histogram
   for (int i = tid; i < bufferSize; i += stride) {
      auto bin = FindFixBin(cells[i], binEdges, nCells, xMin, xMax);
      // printf("%d: add %f to bin %d\n", tid, w[i], bin);
      atomicAdd(&histogram[bin], w[i]);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// RH1CUDA constructors

RH1CUDA::RH1CUDA()
{
   fThreadBlockSize = 512;
   fBufferSize = 10000;

   fNcells = 0;
   fXmin = 0;
   fXmax = 1;
   fEntries = 0;

   fDeviceCells = NULL;
   fDeviceWeights = NULL;
   fDeviceBinEdges = NULL;
   fDeviceStats = NULL;
   fBinEdges = NULL;
}

// RH1CUDA::RH1CUDA(Int_t _ncells) : RH1CUDA() {
RH1CUDA::RH1CUDA(Int_t _nCells, Double_t _xLow, Double_t _xHigh, const Double_t *_binEdges) : RH1CUDA()
{
   fNcells = _nCells;
   fXmin = _xLow;
   fXmax = _xHigh;
   fBinEdges = _binEdges;
}

// Allocate buffers for histogram on GPU
void RH1CUDA::AllocateH1D()
{
   // Allocate histogram on GPU
   ERRCHECK(cudaMalloc((void **)&fDeviceHisto, fNcells * sizeof(Double_t)));
   ERRCHECK(cudaMemset(fDeviceHisto, 0, fNcells * sizeof(Double_t)));

   // Allocate weights array on GPU
   ERRCHECK(cudaMalloc((void **)&fDeviceWeights, fBufferSize * sizeof(Double_t)));

   // Allocate array of cells to fill on GPU
   ERRCHECK(cudaMalloc((void **)&fDeviceCells, fBufferSize * sizeof(Double_t)));

   if (fBinEdges != NULL) {
      ERRCHECK(cudaMalloc((void **)&fDeviceBinEdges, (fNcells - 1) * sizeof(Double_t)));
      ERRCHECK(cudaMemcpy(fDeviceBinEdges, fBinEdges, (fNcells - 1) * sizeof(Double_t), cudaMemcpyHostToDevice));
      printf("fBinEdges\n:");
      for (int j = 0; j < fNcells - 1; ++j) {
         printf("%f ", fBinEdges[j]);
      }
      printf("\n");
   }

   ERRCHECK(cudaMalloc((void **)&fDeviceStats, sizeof(HistStats)));
   ERRCHECK(cudaMemset(fDeviceStats, 0, 5 * sizeof(Double_t))); // set the first 5 variables in the struct to 0.
}

void RH1CUDA::GetStats(UInt_t size)
{
   const UInt_t blockSize = 512;

   Int_t smemSize = (blockSize <= 32) ? 2 * blockSize : blockSize;
   UInt_t numBlocks = fmax(1, ceil(size / blockSize / 2.)); // Number of blocks in grid is halved!

   Double_t *intermediate_sumw = NULL;
   Double_t *intermediate_sumw2 = NULL;
   Double_t *intermediate_sumwx = NULL;
   Double_t *intermediate_sumwx2 = NULL;
   ERRCHECK(cudaMalloc((void **)&intermediate_sumw, numBlocks * sizeof(Double_t)));
   ERRCHECK(cudaMalloc((void **)&intermediate_sumw2, numBlocks * sizeof(Double_t)));
   ERRCHECK(cudaMalloc((void **)&intermediate_sumwx, numBlocks * sizeof(Double_t)));
   ERRCHECK(cudaMalloc((void **)&intermediate_sumwx2, numBlocks * sizeof(Double_t)));

   GetStatsKernel<blockSize, Double_t><<<numBlocks, blockSize, 4 * smemSize * sizeof(Double_t)>>>(
      fDeviceCells, fDeviceWeights, intermediate_sumw, intermediate_sumw2, intermediate_sumwx, intermediate_sumwx2,
      size);
   ERRCHECK(cudaGetLastError());
   // OPTIMIZATION: final reduction in a single kernel?
   CUDAHelpers::ReductionKernel<blockSize, CUDAHelpers::plus<Double_t>, Double_t, false>
      <<<1, blockSize, smemSize * sizeof(Double_t)>>>(intermediate_sumw, &(fDeviceStats->fTsumw), numBlocks, 0.);
   ERRCHECK(cudaGetLastError());
   CUDAHelpers::ReductionKernel<blockSize, CUDAHelpers::plus<Double_t>, Double_t, false>
      <<<1, blockSize, smemSize * sizeof(Double_t)>>>(intermediate_sumw2, &(fDeviceStats->fTsumw2), numBlocks, 0.);
   ERRCHECK(cudaGetLastError());
   CUDAHelpers::ReductionKernel<blockSize, CUDAHelpers::plus<Double_t>, Double_t, false>
      <<<1, blockSize, smemSize * sizeof(Double_t)>>>(intermediate_sumwx, &(fDeviceStats->fTsumwx), numBlocks, 0.);
   ERRCHECK(cudaGetLastError());
   CUDAHelpers::ReductionKernel<blockSize, CUDAHelpers::plus<Double_t>, Double_t, false>
      <<<1, blockSize, smemSize * sizeof(Double_t)>>>(intermediate_sumwx2, &(fDeviceStats->fTsumwx2), numBlocks, 0.);
   ERRCHECK(cudaGetLastError());

   ERRCHECK(cudaFree(intermediate_sumw));
   ERRCHECK(cudaFree(intermediate_sumw2));
   ERRCHECK(cudaFree(intermediate_sumwx));
   ERRCHECK(cudaFree(intermediate_sumwx2));
}

void RH1CUDA::ExecuteCUDAH1D()
{
   UInt_t size = fmin(fBufferSize, fCells.size());
   // printf("cellsize:%lu buffersize:%f Size:%f nCells:%d\n", fCells.size(), fBufferSize, size, fNcells);

   fEntries += size;

   ERRCHECK(cudaMemcpy(fDeviceCells, fCells.data(), size * sizeof(Double_t), cudaMemcpyHostToDevice));
   ERRCHECK(cudaMemcpy(fDeviceWeights, fWeights.data(), size * sizeof(Double_t), cudaMemcpyHostToDevice));

   HistoKernel<<<size / fThreadBlockSize + 1, fThreadBlockSize, fNcells * sizeof(Double_t)>>>(
      fDeviceHisto, fDeviceBinEdges, fXmin, fXmax, fNcells, fDeviceCells, fDeviceWeights, size, fDeviceStats);
   ERRCHECK(cudaGetLastError());
   GetStats(size);

   fCells.clear();
   fWeights.clear();
}

void RH1CUDA::Fill(Double_t x, Double_t w)
{
   fCells.push_back(x);
   fWeights.push_back(w);

   // Only execute when a certain number of values are buffered to increase the GPU workload and decrease the
   // frequency of kernel launches.
   if (fCells.size() == fBufferSize) {
      ExecuteCUDAH1D();
   }
}

void RH1CUDA::Fill(Double_t x)
{
   Fill(x, 1.0);
}

void RH1CUDA::Fill(const char *namex, Double_t w)
{
   Fatal(":Fill(const char *namex, Double_t w)", "Cuda version not implemented yet");
}

// Copy back results on GPU to CPU.
Int_t RH1CUDA::RetrieveResults(Double_t *histResult, Double_t *stats)
{
   // Fill the histogram with remaining values in the buffer.
   if (fCells.size() > 0) {
      ExecuteCUDAH1D();
   }

   ERRCHECK(cudaMemcpy(histResult, fDeviceHisto, fNcells * sizeof(Double_t), cudaMemcpyDeviceToHost));
   ERRCHECK(cudaMemcpy(&fStats, fDeviceStats, sizeof(HistStats), cudaMemcpyDeviceToHost));
   stats[0] = fStats.fTsumw;
   stats[1] = fStats.fTsumw2;
   stats[2] = fStats.fTsumwx;
   stats[3] = fStats.fTsumwx2;

   // TODO: Free device pointers?

   return fEntries;
}
