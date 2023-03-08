#ifndef HIST_CUDA
#define HIST_CUDA

#include "RtypesCore.h"

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
   Int_t threadBlockSize;
   Double_t *deviceHisto;
   Double_t *deviceW;
   Double_t *histogram;
   Int_t nbins;
   Int_t binSize;
   Double_t xlow, xhigh;

public:
   HistCUDA();

   HistCUDA(Int_t _nbins, Double_t _xlow, Double_t _xup);

   void AllocateH1D();
   void ExecuteCUDAHist1D(Double_t *vals, Int_t nbins);
   void AddBinCUDA(Int_t bin, Double_t w);
   void RetrieveResults(Double_t *result);

   template <typename... ValTypes>
   // template <typename... ValTypes, std::enable_if_t<std::conjunction<std::is_floating_point<ValTypes>...>::value, bool> = true>
   void AddBinCUDA( const ValTypes &...x)
   {

   }
};

#endif