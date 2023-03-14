#include <cuda.h>
#include <vector>
#include <stdio.h>
#include <string>

#include "HistCUDA.h"
#include "TError.h"
#include "TAxis.h" // TODO: this is terrible


using namespace std;


// TODO: reuse from RooBatchCompute.
#ifdef __CUDACC__
#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      Fatal((func + "(), " + file + ":" + std::to_string(line)).c_str(), "%s", cudaGetErrorString(error));
      throw std::bad_alloc();
   }
}
#endif

__rooglobal__ void
H1DKernel(Double_t *histogram, Int_t nbins, Int_t *bins, Double_t *w, Size_t bufferSize)
{
   extern __shared__ Double_t block_histogram[];
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   int local_tid = threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   // Initialize a local per-block histogram
   if (local_tid < nbins)
      block_histogram[local_tid] = 0;
   __syncthreads();

   // // Fill local histogram
   for (int i = tid; i < bufferSize; i += stride) {
      atomicAdd(&block_histogram[bins[i]], w[i]);
   }
   __syncthreads();

   // Merge results in global histogram
   if (local_tid < nbins) {
      atomicAdd(&histogram[local_tid], block_histogram[local_tid]);
      // printf("%d: Add %f to %f\n", local_tid, histogram[local_tid], block_histogram[local_tid]);
   }
}

__rooglobal__ void
H1DKernelGlobal(Double_t *histogram, Int_t nbins, Int_t *bins, Double_t *w, Size_t bufferSize)
{
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   int stride = blockDim.x * gridDim.x;

   // Fill histogram
   for (int i = tid; i < bufferSize; i += stride) {
      // printf("%d: add %f to bin %d\n", tid, w[i], bins[i]);
      atomicAdd(&histogram[bins[i]], w[i]);
   }
}

// Default constructor
HistCUDA::HistCUDA() {
   threadBlockSize = 512;
   bufferSize = 10000;
   deviceHisto = NULL;
   deviceCells = NULL;
   deviceWeights = NULL;
   ncells = 0;
   fXaxis = NULL;
   fYaxis = NULL;
}

// HistCUDA::HistCUDA(Int_t _ncells) : HistCUDA() {
HistCUDA::HistCUDA(Int_t _ncells, TAxis *_xaxis, TAxis *_yaxis) : HistCUDA() {
   ncells = _ncells;
   fXaxis = _xaxis;
   fYaxis = _yaxis;
}

// Allocate buffers for histogram on GPU
void HistCUDA::AllocateH1D()
{
   // Allocate histogram on GPU
   ERRCHECK(cudaMalloc((void **)&deviceHisto, ncells * sizeof(Double_t)));
   ERRCHECK(cudaMemset(deviceHisto, 0, ncells * sizeof(Double_t)));

   // Allocate weights array on GPU
   ERRCHECK(cudaMalloc((void **)&deviceWeights, bufferSize * sizeof(Double_t)));

   // Allocate bins array on GPU
   ERRCHECK(cudaMalloc((void **)&deviceCells, bufferSize * sizeof(Int_t)));

   // copy the original vectors to the GPU
   // ERRCHECK(cudaMemcpy(deviceVals, vals, numVals * sizeof(Double_t), cudaMemcpyHostToDevice));
}

void HistCUDA::ExecuteCUDAH1D()
{
   Size_t size = fmin(bufferSize, cells.size());
   // printf("cellsize:%lu buffersize:%f Size:%f nCells:%d\n", cells.size(), bufferSize, size, ncells);

   ERRCHECK(cudaMemcpy(deviceCells, cells.data(), size * sizeof(Int_t), cudaMemcpyHostToDevice));
   ERRCHECK(cudaMemcpy(deviceWeights, weights.data(), size * sizeof(Double_t), cudaMemcpyHostToDevice));

   H1DKernel<<<size / threadBlockSize + 1, threadBlockSize, ncells * sizeof(Double_t)>>>(deviceHisto, ncells, deviceCells, deviceWeights, size);
   ERRCHECK(cudaGetLastError());

   cells.clear();
   weights.clear();
}

// template <<typename... ValTypes, std::enable_if_t<!Disjunction<IsDataContainer<ValTypes>...>::value, int> = 0>
void HistCUDA::AddBinCUDA(Int_t bin, Double_t w)
{
   if (bin < 0) return;

   cells.push_back(bin);
   weights.push_back(w);

   if (cells.size() == bufferSize) {
      ExecuteCUDAH1D();
   }
}

void HistCUDA::AddBinCUDA(Int_t bin)
{
   AddBinCUDA(bin, 1.0);
}


void HistCUDA::AddBinCUDA(Double_t x, Double_t y)
{
   Int_t binx, biny, bin;
   binx = fXaxis->FindBin(x);
   biny = fYaxis->FindBin(y);
   if (binx <0 || biny <0) return;
   bin  = biny*(fXaxis->GetNbins()+2) + binx;
   AddBinCUDA(bin);
   return;
}


void HistCUDA::RetrieveResults(Double_t *result)
{
   // Fill remaning values in the histogram.
   if (cells.size() > 0) {
      ExecuteCUDAH1D();
   }

   ERRCHECK(cudaMemcpy(result, deviceHisto, ncells * sizeof(Double_t), cudaMemcpyDeviceToHost));
   // ERRCHECK(cudaFree(deviceVals));
   // ERRCHECK(cudaFree(deviceHisto));
}

