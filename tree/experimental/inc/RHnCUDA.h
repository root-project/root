#ifndef RHnCUDA_H
#define RHnCUDA_H

#include <vector>
#include <array>
#include "AxisDescriptor.h"

namespace ROOT {
namespace Experimental {

template <typename T, unsigned int Dim, unsigned int BlockSize = 256>
class RHnCUDA {
   // clang-format off
private:
   T                                *fDHistogram;         ///< Pointer to histogram buffer on the GPU.
   int                               fNbins;              ///< Total number of bins in the histogram WITH under/overflow

   const int                         kNStats;             ///< Number of statistics.
   std::array<AxisDescriptor, Dim>   fHAxes;              ///< Vector of Dim axis descriptors
   AxisDescriptor                   *fDAxes;              ///< Pointer to axis descriptors on the GPU.

   std::vector<double>               fHCoords;            ///< 1D buffer with bufferSize #Dim-dimensional coordinates to fill.
   std::vector<double>               fHWeights;           ///< Buffer of weigths for each bin on the Host.
   double                           *fDCoords;            ///< Pointer to array of coordinates to fill on the GPU.
   double                           *fDWeights;           ///< Pointer to array of weights on the GPU.
   int                              *fDBins;              ///< Pointer to array of bins (corresponding to the coordinates) to fill on the GPU.

   int                               fEntries;            ///< Number of entries that have been filled.
   double                           *fDIntermediateStats; ///< Buffer for storing intermediate results of stat reduction on GPU.
   double                           *fDStats;             ///< Pointer to statistics array on GPU.

   // Kernel size parameters
   unsigned int                      fNumBlocks;          ///< Number of blocks used in CUDA kernels
   unsigned int                      fBufferSize;         ///< Number of coordinates to buffer.
   unsigned int                      fMaxSmemSize;        ///< Maximum shared memory size per block on device 0.
   unsigned int                      kStatsSmemSize;      ///< Size of shared memory per block in GetStatsKernel
   unsigned int                      fHistoSmemSize;      ///< Size of shared memory per block in HistoKernel
   // clang-format on

public:
   RHnCUDA() = delete;

   RHnCUDA(std::array<int, Dim> ncells, std::array<double, Dim> xlow, std::array<double, Dim> xhigh,
           const double **binEdges = NULL);

   ~RHnCUDA();

   RHnCUDA(const RHnCUDA &) = delete;
   RHnCUDA &operator=(const RHnCUDA &) = delete;

   int GetEntries() { return fEntries; }

   void AllocateBuffers();

   void RetrieveResults(T *histResult, double *statsResult);

   void Fill(const std::array<double, Dim> &coords, double w = 1.);

protected:
   void GetStats(unsigned int size);

   void ExecuteCUDAHisto();
};

} // namespace Experimental
} // namespace ROOT
#endif
