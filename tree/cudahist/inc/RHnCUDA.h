#ifndef RHnCUDA_H
#define RHnCUDA_H

#include "RtypesCore.h"
#include "TMath.h"

#include <vector>
#include <array>
#include <utility>

namespace CUDAhist {

struct RAxis {
   int fNbins;   ///< Number of bins(1D) WITH u/overflow
   double fMin;  ///< Low edge of first bin
   double fMax;  ///< Upper edge of last bin

   const double *kBinEdges; ///< Bin edges array, can be NULL
};

template <typename T, unsigned int Dim, unsigned int BlockSize = 256>
class RHnCUDA {
   // clang-format off
private:
   T                       *fDeviceHisto;        ///< Pointer to histogram buffer on the GPU.
   int                      fNbins;              ///< Total number of bins in the histogram WITH under/overflow

   const int                kNStats;             ///< Number of statistics.
   std::array<RAxis, Dim>   fHAxes;              ///< Vector of Dim axis descriptors
   RAxis                   *fDAxes;              ///< Pointer to axis descriptors on the GPU.

   std::vector<double>      fHCoords;            ///< 1D buffer with bufferSize #Dim-dimensional coordinates to fill.
   std::vector<double>      fHWeights;           ///< Buffer of weigths for each bin on the Host.
   double                  *fDCoords;            ///< Pointer to array of coordinates to fill on the GPU.
   int                     *fDBins;              ///< Pointer to array of bins (corresponding to the coordinates) to fill on the GPU.
   double                  *fDWeights;           ///< Pointer to array of weights on the GPU.

   double                   fEntries;            ///< Number of entries that have been filled.
   double                  *fDIntermediateStats; ///< Pointer to statistics buffer on GPU.
   double                  *fDStats;             ///< Pointer to statistics buffer on GPU.

   // Kernel size parameters
   unsigned int             fNumBlocks;          ///< Number of blocks used in CUDA kernels
   unsigned int             fBufferSize;         ///< Number of coordinates to buffer.
   unsigned int             fMaxSmemSize;        ///< Maximum shared memory size per block on device 0.
   unsigned int             kStatsSmemSize;      ///< Size of shared memory per block in GetStatsKernel
   unsigned int             fHistoSmemSize;      ///< Size of shared memory per block in HistoKernel
   // clang-format on

public:
   RHnCUDA() = delete;

   RHnCUDA(int *ncells, double *xlow, double *xhigh, const double **binEdges);

   void AllocateH1D();

   int RetrieveResults(double *histResult, double *statsResult);

   void Fill(const std::array<T, Dim> &coords, double w = 1.);

protected:
   void GetStats(unsigned int size);

   void ExecuteCUDAHisto();
};

} // namespace CUDAhist
#endif
