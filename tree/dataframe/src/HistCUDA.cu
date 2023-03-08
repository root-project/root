#include <cuda.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <iterator>
#include <vector>
#include <sstream>
#include <string>

#include "HistCUDA.h"
#include "TError.h"


using namespace std;


// TODO: reuse from RooBatchCompute.
#ifdef __CUDACC__
#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      // Fatal((func + "(), " + file + ":" + std::to_string(line)).c_str(), "%s", cudaGetErrorString(error));
      fprintf(stderr, "cuda error!! %s %s", std::to_string(line).c_str(), cudaGetErrorString(error));
      throw std::bad_alloc();
   }
}
#endif

__rooglobal__ void
Hist1DKernel(Double_t *histogram, Int_t bin, Double_t *w)
{
   extern __shared__ Double_t block_histogram[];
   // int tid = threadIdx.x + blockDim.x * blockIdx.x;
   // int local_tid = threadIdx.x;
   // int stride = blockDim.x * gridDim.x;

   // Initialize a local per-block histogram
   // if (local_tid < nbins)
   //    block_histogram[local_tid] = 0;
   // __syncthreads();

   // // Fill local histogram
   // for (int i = tid; i < numVals; i += stride) {
   //    int bin = (vals[i] - xlow) / binSize;
   //    atomicAdd(&block_histogram[bin], 1);
   // }
   // __syncthreads();

   // Merge results in global histogram
   // if (local_tid < nbins)
   //    atomicAdd(&histogram[local_tid], block_histogram[local_tid]);

   // if (tid == bin) {
      // printf("adding %f to bin %d on GPU!!\n", w[0], bin);
      atomicAdd(&histogram[bin], w[0]);
   // }
}

// __rooglobal__ void
// Hist1DKernel(Double_t *vals, int numVals, UInt_t *histogram, Double_t xlow, int binSize, int nbins)
// {
//    extern __shared__ UInt_t block_histogram[];
//    int tid = threadIdx.x + blockDim.x * blockIdx.x;
//    int local_tid = threadIdx.x;
//    int stride = blockDim.x * gridDim.x;

//    // Initialize a local per-block histogram
   // if (local_tid < nbins)
   //    block_histogram[local_tid] = 0;
   // __syncthreads();

//    // Fill local histogram
//    for (int i = tid; i < numVals; i += stride) {
//       int bin = (vals[i] - xlow) / binSize;
//       atomicAdd(&block_histogram[bin], 1);
//    }
//    __syncthreads();

//    // Merge results in global histogram
//    if (local_tid < nbins)
//       atomicAdd(&histogram[local_tid], block_histogram[local_tid]);
// }

// Default constructor
HistCUDA::HistCUDA() {
   deviceHisto = NULL;
   threadBlockSize = 512;
   nbins = 0;
   xlow = 0;
}

HistCUDA::HistCUDA(Int_t _nbins, Double_t _xlow, Double_t _xhigh) {
   deviceHisto = NULL;
   deviceW = NULL;
   threadBlockSize = 512;
   nbins = _nbins;
   xlow = _xlow;
   xhigh = _xhigh;
}

// Allocate buffers for histogram on GPU
void HistCUDA::AllocateH1D()
{
   // histogram = (Double_t *)calloc(nbins, sizeof(Double_t));

   // allocate the vectors on the GPU
   // ERRCHECK(cudaMalloc((void **)&deviceVals, numVals * sizeof(Double_t)));
   ERRCHECK(cudaMalloc((void **)&deviceHisto, nbins * sizeof(Double_t)));
   ERRCHECK(cudaMalloc((void **)&deviceW, sizeof(Double_t)));

   // copy the original vectors to the GPU
   // ERRCHECK(cudaMemcpy(deviceVals, vals, numVals * sizeof(Double_t), cudaMemcpyHostToDevice));
}

void HistCUDA::ExecuteCUDAHist1D(Double_t *vals, Int_t numVals)
{
   // Hist1DKernel<<<numVals / threadBlockSize + 1, threadBlockSize, nbins * sizeof(Double_t)>>>(
   //    deviceVals, numVals, deviceHisto, xlow, binSize, nbins);
   // ERRCHECK(cudaGetLastError());
}

// template <<typename... ValTypes, std::enable_if_t<!Disjunction<IsDataContainer<ValTypes>...>::value, int> = 0>
void HistCUDA::AddBinCUDA(Int_t bin, Double_t w)
{
   // cout << "received in AddBinCUDA: " << bin << "  " << w << endl;
   ERRCHECK(cudaMemcpy(deviceW, &w, sizeof(Double_t), cudaMemcpyHostToDevice));
   Hist1DKernel<<<1, 1, nbins * sizeof(Double_t)>>>(deviceHisto, bin, deviceW);
   ERRCHECK(cudaGetLastError());
}

void HistCUDA::RetrieveResults(Double_t *result)
{
   ERRCHECK(cudaMemcpy(result, deviceHisto, nbins * sizeof(Double_t), cudaMemcpyDeviceToHost));
   // ERRCHECK(cudaFree(deviceVals));
   ERRCHECK(cudaFree(deviceHisto));
}

