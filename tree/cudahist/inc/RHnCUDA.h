#ifndef RHnCUDA_H
#define RHnCUDA_H

#include "RtypesCore.h"
#include "TError.h"

#include <vector>
#include <utility>
#include <iostream>



class RHnCUDA  {
public:
   struct Stats {
      Double_t      fTsumw;           ///<  Total Sum of weights
      Double_t      fTsumw2;          ///<  Total Sum of squares of weights
      Double_t      fTsumwx;          ///<  Total Sum of weight*X
      Double_t      fTsumwx2;         ///<  Total Sum of weight*X*X
      Double_t     *fSumw2;           ///<  Array of sum of squares of weights
   };

   struct RAxis {
      Int_t                  fNcells;           ///< Number of bins(1D) WITH u/overflow
      Double_t               fMin;              ///< Low edge of first bin
      Double_t               fMax;              ///< Upper edge of last bin

      const Double_t        *kBinEdges;         ///< Bin edges array.
   };

private:
   Double_t              *fDeviceHisto;         ///< Pointer to histogram buffer on the GPU.
   Int_t                  fNbins;               ///< Total number of bins in the histogram

   Int_t                  fThreadBlockSize;
   const Int_t            kDim;
   std::vector<RAxis>     fAxes;
   RAxis                 *fDeviceAxes;

   std::vector<Double_t>  fCells;               ///< Buffer of bins to fill. TODO: vector or just int*?
   std::vector<Double_t>  fWeights;             ///< Buffer of weigths for each bin.
   Double_t              *fDeviceCells;         ///< Pointer to array of bins to fill on the GPU.
   Double_t              *fDeviceWeights;       ///< Pointer to array of weights on the GPU.

   Double_t               fEntries;             ///< Number of entries
   Stats                 *fDeviceStats;
   UInt_t                 fBufferSize;          ///< Number of bins to buffer.


public:
   RHnCUDA() = delete;

   // RHnCUDA(Int_t _nbins);
   RHnCUDA(Int_t dim, Int_t *ncells, Double_t *xlow, Double_t *xhigh, const Double_t **binEdges);

   void AllocateH1D();
   Int_t RetrieveResults(Double_t *histResult, Double_t *stats);

   void Fill(std::vector<Double_t> x);
   void Fill(std::vector<Double_t> x, Double_t w);

   // TODO: how to distinguish between different THn fills...
   void Fill(Double_t a, Double_t b) {
      if (kDim == 1) {
         Fill(std::vector<Double_t> {a}, (Double_t) b);
      } else if (kDim == 2) {
         Fill(std::vector<Double_t> {a, b}, 1.0);
      }
   }

   void Fill(Double_t x) { Fill( std::vector<Double_t> {x}); }

   void Fill(Float_t x, Double_t w) { Fill((Double_t)x, (Double_t) w); }
   void Fill(const char *namex, Double_t w);

   template <typename... ValTypes>
   void Fill(const ValTypes &...x)
   {
   // ( (std::cout << ", " << x), ...) << std::endl;
      Fatal("Fill", "Cuda version not implemented yet");
   }

protected:
   void GetStats(UInt_t size);
   void ExecuteCUDAH1D();
};


#endif