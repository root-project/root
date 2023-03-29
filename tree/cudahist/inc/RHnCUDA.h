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
      double      fTsumw;           ///<  Total Sum of weights
      double      fTsumw2;          ///<  Total Sum of squares of weights
      double      fTsumwx;          ///<  Total Sum of weight*X
      double      fTsumwx2;         ///<  Total Sum of weight*X*X
      double     *fSumw2;           ///<  Array of sum of squares of weights
   };

   struct RAxis {
      int                  fNcells;       ///< Number of bins(1D) WITH u/overflow
      double               fMin;          ///< Low edge of first bin
      double               fMax;          ///< Upper edge of last bin

      const double        *kBinEdges;     ///< Bin edges array, can be NULL
   };

private:
   double              *fDeviceHisto;     ///< Pointer to histogram buffer on the GPU.
   int                  fNbins;           ///< Total number of bins in the histogram WITH u/overflow

   int                  fThreadBlockSize; ///< Block size used in CUDA kernels
   const int            kDim;             ///< Dimension of the histogram
   std::vector<RAxis>   fAxes;            ///< Vector of kDim axis descriptors
   RAxis               *fDeviceAxes;      ///< Pointer to axis descriptors on the GPU.

   std::vector<double>  fCells;           ///< 1D buffer with bufferSize number of kDim-dimensional coordinates to fill.
   std::vector<double>  fWeights;         ///< Buffer of weigths for each bin.
   double              *fDeviceCells;     ///< Pointer to array of bins to fill on the GPU.
   double              *fDeviceWeights;   ///< Pointer to array of weights on the GPU.

   double               fEntries;         ///< Number of entries that have been filled.
   Stats               *fDeviceStats;     ///< Pointer to statistics on the GPU.
   unsigned int         fBufferSize;      ///< Number of bins to buffer.


public:
   RHnCUDA() = delete;

   RHnCUDA(int dim, int *ncells, double *xlow, double *xhigh, const double **binEdges);

   void AllocateH1D();
   int RetrieveResults(double *histResult, double *stats);

   void Fill(const std::vector<double> x);
   void Fill(const std::vector<double> x, double w);

   // Temporary catch-all
   // template <typename... ValTypes>
   // void Fill(const ValTypes &...x)
   // {
   //    Fatal("Fill", "Cuda version not implemented yet");
   // }

protected:
   void GetStats(unsigned int size);
   void ExecuteCUDAH1D();
};


#endif