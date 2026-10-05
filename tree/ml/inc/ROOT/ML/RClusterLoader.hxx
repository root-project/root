// Author: Dante Niewenhuis, VU Amsterdam 07/2023
// Author: Kristupas Pranckietis, Vilnius University 05/2024
// Author: Nopphakorn Subsa-Ard, King Mongkut's University of Technology Thonburi (KMUTT) (TH) 08/2024
// Author: Vincenzo Eduardo Padulano, CERN 10/2024
// Author: Silia Taider, CERN 03/2026

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_INTERNAL_ML_RCLUSTERLOADER
#define ROOT_INTERNAL_ML_RCLUSTERLOADER

#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ROOT/ML/RFlat2DMatrix.hxx"
#include "ROOT/ML/RFlat2DMatrixOperators.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDFHelpers.hxx"
#include "ROOT/RDF/Utils.hxx"

namespace ROOT::Experimental::Internal::ML {

/**
 * \struct RClusterRange
 * \brief Describes a contiguous range of entries within a single RDataFrame,
 * corresponding to one TTree/RNTuple cluster boundary.
 *
 * For filtered RDataFrames the \p numEntries field may be smaller than `end - start`
 * because it tracks the number of entries that actually pass the filter,
 * discovered and set lazily during the first epoch.
 */
struct RClusterRange {
   std::size_t rdfIdx;                  // which rdf this cluster belongs to
   std::uint64_t start;                 // first raw entry (incl)
   std::uint64_t end;                   // one-past-last entry (excl)
   std::size_t numEntries{
      static_cast<std::size_t>(end - start)}; // number of entries in the cluster (that pass filters, if any)

   std::size_t GetNumEntries() const { return numEntries; }
   void SetNumEntries(std::size_t num) { numEntries = num; }
};

/**
 * \class ROOT::Experimental::Internal::ML::RClusterLoader
 * \brief Loads TTree/RNTuple clusters from one or more RDataFrames into RFlat2DMatrix
 *        buffers for ML training and validation.
 *
 * ### Overview
 * At construction the loader scans the cluster boundaries of every
 * provided RDataFrame and stores them as a flat list of \ref RClusterRange objects.
 * SplitDataset() then partitions those ranges into training and validation sets according to \p validationSplit.
 *
 * ### The split strategy depends on whether shuffling is enabled or not
 * - **Unshuffled**: one cut is made so that the first `(1 - validationSplit)`
 * fraction of entries goes to training. At most one cluster is split at the boundary.
 * - **Shuffled**: each cluster is split proportionally (according to `validationSplit`)
 * so both sets draw entries from every part of the dataset. ShuffleTrainingClusters()
 * and ShuffleValidationClusters() re-order the cluster lists at the start of each epoch.
 * A second shuffling step, at the entries level, happens inside LoadTrainingClusterInto()
 * and LoadValidationClusterInto() when loading the data into the tensors.
 *
 * ### Filtered RDataFrames
 * When any RDataFrame carries a filter, the true entry count is not known
 * until the computation graph is executed. In this case SplitDataset() is a
 * no-op and the split is discovered lazily inside LoadTrainingClusterInto()
 * during the first epoch.
 * After the first epoch FinaliseSplitDiscovery() marks the split as stable and
 * all subsequent epochs use the same pre-computed ranges.
 */
class RClusterLoader {
private:
   std::vector<ROOT::RDF::RNode> &fRdfs;
   std::vector<std::size_t> fRdfSizes;
   std::vector<std::string> fCols;
   std::vector<std::size_t> fVecSizes;
   float fVecPadding;
   float fValidationSplit;
   bool fShuffle;
   std::size_t fSetSeed;

   std::size_t fNumCols;
   std::size_t fSumVecSizes;
   std::size_t fNumChunkCols;

   std::vector<RClusterRange> fAllClusters;
   std::vector<RClusterRange> fTrainingClusters;
   std::vector<RClusterRange> fValidationClusters;

   std::size_t fTotalEntries{0};
   std::size_t fNumTrainingEntries{0};
   std::size_t fNumValidationEntries{0};

   bool fIsFiltered{false};
   bool fSplitDiscovered{false};
   std::size_t fAccumulatedFilteredForTrain{0};
   std::size_t fAccumulatedFilteredForVal{0};

public:
   RClusterLoader(std::vector<ROOT::RDF::RNode> &rdfs, const std::vector<std::string> &cols,
                  const std::vector<std::size_t> &vecSizes, float vecPadding, float validationSplit, bool shuffle,
                  std::size_t setSeed);

   void SplitDataset();

   void ShuffleTrainingClusters(std::size_t epochIdx);

   void ShuffleValidationClusters(std::size_t epochIdx);

   void LoadClusterInto(RFlat2DMatrix &dest, std::size_t rdfIdx, std::uint64_t startRow, std::uint64_t endRow,
                        std::size_t rowOffset = 0);

   std::size_t LoadTrainingClusterInto(RFlat2DMatrix &dest, std::size_t rdfIdx, std::uint64_t startRow,
                                       std::uint64_t endRow, std::size_t rowOffset = 0);

   void LoadValidationClusterInto(RFlat2DMatrix &dest, std::size_t rdfIdx, std::uint64_t startRow, std::uint64_t endRow,
                                  std::size_t rowOffset = 0);

   void FinaliseSplitDiscovery();

   bool IsSplitDiscovered() const { return !fIsFiltered || fSplitDiscovered; }

   //////////////////////////////////////////////////////////////////////////
   // Accessors
   std::size_t GetNumTrainingEntries() const { return fNumTrainingEntries; }
   std::size_t GetNumValidationEntries() const { return fNumValidationEntries; }
   std::size_t GetNumChunkCols() const { return fNumChunkCols; }

   const std::vector<RClusterRange> &GetTrainingClusters() const
   {
      return (fIsFiltered && !fSplitDiscovered) ? fAllClusters : fTrainingClusters;
   }
   const std::vector<RClusterRange> &GetValidationClusters() const { return fValidationClusters; }

   std::size_t GetNumTrainingClusters() const
   {
      return (fIsFiltered && !fSplitDiscovered) ? fAllClusters.size() : fTrainingClusters.size();
   }
   std::size_t GetNumValidationClusters() const { return fValidationClusters.size(); }
   std::size_t GetNmTotalClusters() const { return fAllClusters.size(); }
};

} // namespace ROOT::Experimental::Internal::ML
#endif // ROOT_INTERNAL_ML_RCLUSTERLOADER
