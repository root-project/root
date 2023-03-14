#ifndef HIST_CUDA
#define HIST_CUDA

#include "RtypesCore.h"
#include "TAxis.h"
#include <vector>

// TODO: reuse from RioBatchComputeTypes.h.
#ifdef __CUDACC__
#define __roodevice__ __device__
#define __roohost__ __host__
#define __rooglobal__ __global__
#else
#define __roodevice__
#define __roohost__
#define __rooglobal__
struct cudaEvent_t;
struct cudaStream_t;
#endif // #indef __CUDACC__

class HistCUDA {
private:
   Int_t                  threadBlockSize;

   Double_t              *deviceHisto;      // Pointer to histogram buffer on the GPU.
   Int_t                  ncells;            // Number of bins(1D)

   Size_t                 bufferSize;       // Number of bins to buffer.
   std::vector<Int_t>     cells;             // Buffer of bins to fill. TODO: vector or just int*?
   std::vector<Double_t>  weights;          // Buffer of weigths for each bin.
   Int_t                 *deviceCells;       // Pointer to array of bins to fill on the GPU.
   Double_t              *deviceWeights;    // Pointer to array of weights on the GPU.

   TAxis                 *fXaxis, *fYaxis;

public:
   HistCUDA();

   // HistCUDA(Int_t _nbins);
   HistCUDA(Int_t _ncells, TAxis *_xaxis, TAxis *_yaxis);

   void AllocateH1D();
   void RetrieveResults(Double_t *result);

   void AddBinCUDA(Int_t bin, Double_t w);
   void AddBinCUDA(Int_t bin);
   void AddBinCUDA(Double_t x, Double_t y);

   template <typename... ValTypes>
   // template <typename... ValTypes, std::enable_if_t<std::conjunction<std::is_floating_point<ValTypes>...>::value, bool> = true>
   void AddBinCUDA(const ValTypes &...x)
   {

   }

protected:
   void ExecuteCUDAH1D();
};

#endif