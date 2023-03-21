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
// }

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
      atomicAdd(&smem[bin], w[i]);

      // if (bin == 0 || bin > nCells) continue;
      // Double_t z= w;
      // fTsumw   += z;
      // fTsumw2  += z*z;
   }
   __syncthreads();

   // Merge results in global histogram
   if (local_tid < nCells) {
      atomicAdd(&histogram[local_tid], smem[local_tid]);
   }
}

// - Int_t bin;
// - fEntries++;
// - bin =fXaxis.FindBin(namex);
// if (bin <0) return -1;
// if (!fSumw2.fN && w != 1.0 && !TestBit(TH1::kIsNotW))  Sumw2();
// if (fSumw2.fN) fSumw2.fArray[bin] += w*w;
// AddBinContent(bin, w);
// if (bin == 0 || bin > fXaxis.GetNbins()) return -1;
// Double_t z= w;
// fTsumw   += z;
// fTsumw2  += z*z;
// // this make sense if the histogram is not expanding (the x axis cannot be extended)
// if (!fXaxis.CanExtend() || !fXaxis.IsAlphanumeric()) {
//    Double_t x = fXaxis.GetBinCenter(bin);
//    fTsumwx  += z*x;
//    fTsumwx2 += z*x*x;
// }
// return bin;

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

// Default constructor
RH1CUDA::RH1CUDA()
{
   fThreadBlockSize = 512;
   fBufferSize = 10000;

   fNcells = 0;
   fXmin = 0;
   fXmax = 1;

   fDeviceCells = NULL;
   fDeviceWeights = NULL;
   fDeviceBinEdges = NULL;
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
   // ERRCHECK(cudaMemset(fDeviceStats, 0, 5 * sizeof(Double_t))); // set the first 5 variables in the struct to 0.
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
   CUDAHelpers::Reduce<512, thrust::plus<Double_t>, Double_t, false>(fDeviceWeights, &(fDeviceStats->fTsumw), size);
   ERRCHECK(cudaGetLastError());

   fCells.clear();
   fWeights.clear();
}

void RH1CUDA::Fill(Double_t x, Double_t w)
{
   fCells.push_back(x);
   fWeights.push_back(w);

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

void RH1CUDA::RetrieveResults(Double_t *histResult, HistStats stats)
{
   // Fill remaning values in the histogram.
   if (fCells.size() > 0) {
      ExecuteCUDAH1D();
   }

   ERRCHECK(cudaMemcpy(histResult, fDeviceHisto, fNcells * sizeof(Double_t), cudaMemcpyDeviceToHost));
   ERRCHECK(cudaMemcpy(&fStats, fDeviceStats, sizeof(HistStats), cudaMemcpyDeviceToHost));
   // Free device pointers?
}
