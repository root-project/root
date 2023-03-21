#ifndef RH1CUDA_H
#define RH1CUDA_H

#include "RtypesCore.h"
#include "TError.h"

#include <vector>
#include <utility>
#include <iostream>

struct HistStats {
   Double_t      fTsumw;           ///<  Total Sum of weights
   Double_t      fTsumw2;          ///<  Total Sum of squares of weights
   Double_t      fTsumwx;          ///<  Total Sum of weight*X
   Double_t      fTsumwx2;         ///<  Total Sum of weight*X*X
   Double_t     *fSumw2;           ///<  Array of sum of squares of weights
};

class RH1CUDA  {
private:
   Int_t                  fThreadBlockSize;

   Double_t              *fDeviceHisto;      // Pointer to histogram buffer on the GPU.
   Double_t              *fDeviceBinEdges;       // Pointer to bin edges array on the GPU.
   Int_t                  fNcells;           // Number of bins(1D)
   const Double_t        *fBinEdges;         // Bin edges array.
   Double_t               fXmin;             // Low edge of first bin
   Double_t               fXmax;             // Upper edge of last bin

   Double_t               fEntries;          /// Number of entries
   HistStats              fStats;
   HistStats             *fDeviceStats;

   UInt_t                 fBufferSize;       // Number of bins to buffer.
   std::vector<Double_t>  fCells;            // Buffer of bins to fill. TODO: vector or just int*?
   std::vector<Double_t>  fWeights;          // Buffer of weigths for each bin.
   Double_t              *fDeviceCells;      // Pointer to array of bins to fill on the GPU.
   Double_t              *fDeviceWeights;    // Pointer to array of weights on the GPU.

public:
   RH1CUDA();

   // RH1CUDA(Int_t _nbins);
   RH1CUDA(Int_t ncells, Double_t xlow, Double_t xhigh, const Double_t *binEdges);

   void AllocateH1D();
   void RetrieveResults(Double_t *histResult, HistStats stats);

   void Fill(Double_t x);
   void Fill(Double_t x, Double_t w);
   void Fill(const char *namex, Double_t w);

   template <typename... ValTypes>
   void Fill(const ValTypes &...x)
   {
      // ( (std::cout << ", " << x), ...) << std::endl;
      Fatal("Fill", "Cuda version not implemented yet");
   }

protected:
   void ExecuteCUDAH1D();
};

#endif