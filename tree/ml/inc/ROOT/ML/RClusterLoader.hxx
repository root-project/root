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
 * \class ROOT::Experimental::Internal::ML::RClusterLoaderFunctor
 * \brief Functor invoked by RDataFrame::Foreach to fill one row of an RFlat2DMatrix.
 *
 */

template <typename... ColTypes>
class RClusterLoaderFunctor {
   std::size_t fOffset{};
   std::size_t fVecSizeIdx{};
   float fVecPadding{};
   std::vector<std::size_t> fMaxVecSizes{};
   RFlat2DMatrix &fChunkTensor;

   std::size_t fNumChunkCols;

   int fI;
   int fNumColumns;

   //////////////////////////////////////////////////////////////////////////
   /// \brief \brief Copy the content of a column into the current tensor when the column consists of vectors
   template <typename T, std::enable_if_t<ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &vec, int i, int numColumns)
   {
      std::size_t max_vec_size = fMaxVecSizes[fVecSizeIdx++];
      std::size_t vec_size = vec.size();

      float *dst = fChunkTensor.GetData() + fOffset + numColumns * i;
      if (vec_size < max_vec_size) // Padding vector column to max_vec_size with fVecPadding
      {
         std::copy(vec.begin(), vec.end(), dst);
         std::fill(dst + vec_size, dst + max_vec_size, fVecPadding);
      } else // Copy only max_vec_size length from vector column
      {
         std::copy(vec.begin(), vec.begin() + max_vec_size, dst);
      }
      fOffset += max_vec_size;
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Copy the content of a column into the current tensor when the column consists of scalar values
   template <typename T, std::enable_if_t<!ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &val, int i, int numColumns)
   {
      fChunkTensor.GetData()[fOffset + numColumns * i] = val;
      fOffset++;
   }

public:
   RClusterLoaderFunctor(RFlat2DMatrix &chunkTensor, std::size_t numColumns,
                         const std::vector<std::size_t> &maxVecSizes, float vecPadding, int i,
                         std::size_t rowOffset = 0)
      : fChunkTensor(chunkTensor),
        fMaxVecSizes(maxVecSizes),
        fVecPadding(vecPadding),
        fI(i),
        fNumColumns(numColumns),
        fOffset(rowOffset * numColumns)
   {
   }

   void operator()(const ColTypes &...cols)
   {
      fVecSizeIdx = 0;
      (AssignToTensor(cols, fI, fNumColumns), ...);
   }
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
template <typename... Args>
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

public:
   RClusterLoader(std::vector<ROOT::RDF::RNode> &rdfs, const std::vector<std::string> &cols,
                  const std::vector<std::size_t> &vecSizes, float vecPadding, float validationSplit, bool shuffle,
                  std::size_t setSeed)
      : fRdfs(rdfs),
        fCols(cols),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding),
        fValidationSplit(validationSplit),
        fShuffle(shuffle),
        fSetSeed(setSeed)
   {
      fNumCols = fCols.size();
      fSumVecSizes = std::accumulate(fVecSizes.begin(), fVecSizes.end(), 0UL);
      fNumChunkCols = fNumCols + fSumVecSizes - fVecSizes.size();

      for (auto &rdf : fRdfs) {
         // TODO(staider) We need a better API in RDF to detect generically whether there's a filter or not
         if (!rdf.GetFilterNames().empty()) {
            fIsFiltered = true;
            break;
         }
      }

      fRdfSizes.resize(fRdfs.size(), 0);

      // scan cluster boundaries across files
      // TODO(staider) Add progress bar to inform the user about this potentially long operation
      for (std::size_t rdfIdx = 0; rdfIdx < fRdfs.size(); ++rdfIdx) {
         for (const auto &r : ROOT::Internal::RDF::GetDatasetGlobalClusterBoundaries(fRdfs[rdfIdx])) {
            fAllClusters.push_back({rdfIdx, r.first, r.second});
            auto numEntries = r.second - r.first;
            fRdfSizes[rdfIdx] += numEntries;
            fTotalEntries += numEntries;
         }
      }
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Distribute the clusters into training and validation datasets
   /// No-op for filtered RDataFrames, the split is discovered lazily during the first epoch.
   void SplitDataset()
   {
      if (fAllClusters.empty())
         throw std::runtime_error("RClusterLoader::SplitDataset: no clusters found.");

      if (fIsFiltered) {
         return;
      }

      if (fShuffle) {
         // --- Shuffled path
         // Every cluster contributes a prefix to training and a suffix to validation.
         // Cost: Each cluster is read twice per epoch, only when validation split is more than 0.
         // TODO(staider) Swicth between prefix or suffix for validation randomly per cluster
         for (const RClusterRange &c : fAllClusters) {
            const std::size_t sz = c.GetNumEntries();
            const std::size_t trainSz = static_cast<std::size_t>((1.0f - fValidationSplit) * sz);
            const std::size_t valSz = sz - trainSz;

            if (trainSz > 0) {
               fTrainingClusters.push_back({c.rdfIdx, c.start, c.start + static_cast<std::uint64_t>(trainSz)});
               fNumTrainingEntries += trainSz;
            }
            if (valSz > 0) {
               fValidationClusters.push_back({c.rdfIdx, c.start + static_cast<std::uint64_t>(trainSz), c.end});
               fNumValidationEntries += valSz;
            }
         }
      } else {
         // --- Unshuffled path
         // Contiguous split: first (1 - validationSplit) fraction of entries go to
         // training, the remainder to validation. At most one cluster is split at
         // the boundary.
         const std::size_t targetTraining = fTotalEntries - static_cast<std::size_t>(fValidationSplit * fTotalEntries);

         std::size_t accumulated = 0;
         std::size_t splitIdx = 0;
         for (; splitIdx < fAllClusters.size(); ++splitIdx) {
            const std::size_t sz = fAllClusters[splitIdx].GetNumEntries();
            if (accumulated + sz > targetTraining) {
               break;
            }
            accumulated += sz;
         }

         // Assign whole train/val clusters
         fTrainingClusters.assign(fAllClusters.begin(), fAllClusters.begin() + splitIdx);
         fNumTrainingEntries = accumulated;

         if (splitIdx < fAllClusters.size() && accumulated < targetTraining) {
            // Split the boundary cluster
            const RClusterRange &boundary = fAllClusters[splitIdx];
            const std::uint64_t splitPoint = boundary.start + static_cast<std::uint64_t>(targetTraining - accumulated);

            fTrainingClusters.push_back({boundary.rdfIdx, boundary.start, splitPoint});
            fValidationClusters.push_back({boundary.rdfIdx, splitPoint, boundary.end});
            fValidationClusters.insert(fValidationClusters.end(), fAllClusters.begin() + splitIdx + 1,
                                       fAllClusters.end());

            fNumTrainingEntries += splitPoint - boundary.start;
         } else {
            fValidationClusters.assign(fAllClusters.begin() + splitIdx, fAllClusters.end());
         }

         fNumValidationEntries = fTotalEntries - fNumTrainingEntries;
      }

      if (fTrainingClusters.empty())
         throw std::runtime_error("RClusterLoader::SplitDataset: no entries for training after split. "
                                  "Reduce validation_split.");

      if (fValidationSplit > 0.0f && fValidationClusters.empty())
         throw std::runtime_error("RClusterLoader::SplitDataset: no entries for validation after split. "
                                  "Increase validation_split.");
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Re-order training clusters for the upcoming epoch
   void ShuffleTrainingClusters(std::size_t epochIdx)
   {
      if (!fShuffle) {
         return;
      }

      std::mt19937 g(fSetSeed == 0 ? std::random_device{}() : fSetSeed ^ epochIdx);
      std::shuffle(fTrainingClusters.begin(), fTrainingClusters.end(), g);
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Re-order validation clusters for the upcoming epoch
   void ShuffleValidationClusters(std::size_t epochIdx)
   {
      if (!fShuffle) {
         return;
      }
      std::mt19937 g(fSetSeed == 0 ? std::random_device{}() : fSetSeed ^ epochIdx);
      std::shuffle(fValidationClusters.begin(), fValidationClusters.end(), g);
   }

   void LoadClusterInto(RFlat2DMatrix &dest, std::size_t rdfIdx, std::uint64_t startRow, std::uint64_t endRow,
                        std::size_t rowOffset = 0)
   {
      ROOT::RDF::RNode &rdf = fRdfs[rdfIdx];
      ROOT::Internal::RDF::ChangeBeginAndEndEntries(rdf, startRow, endRow);
      RClusterLoaderFunctor<Args...> func(dest, fNumChunkCols, fVecSizes, fVecPadding, 0, rowOffset);
      rdf.Foreach(func, fCols);
      ROOT::Internal::RDF::ChangeBeginAndEndEntries(rdf, 0, fRdfSizes[rdfIdx]);
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Load one training cluster and return the number of rows written.
   ///
   /// **Unfiltered**: delegates directly to `LoadClusterInto()`
   /// **Filtered**, epoch 1 (!fSplitDiscovered):
   ///  - On the first call, Count() is called across all RDFs to obtain
   ///  the total filtered entry count, fNumTrainingEntries and
   ///  fNumValidationEntries are set as targets.
   ///  - A single Foreach on the full raw cluster range loads data and captures
   ///  rdfentry_ simultaneously. The real train/val boundary is computed from
   ///  the accumulated filtered count vs the target, then the train sub-range
   ///  is pushed to fTrainingClusters and the val sub-range to fValidationClusters.
   ///  - Only the train rows are written into \p dest.
   ///  -All subsequent epochs: delegates directly to `LoadClusterInto()`
   std::size_t LoadTrainingClusterInto(RFlat2DMatrix &dest, std::size_t rdfIdx, std::uint64_t startRow,
                                       std::uint64_t endRow, std::size_t rowOffset = 0)
   {
      if (fIsFiltered && !fSplitDiscovered) {
         // First call: discover total filtered count and set split targets.
         if (fAccumulatedFilteredForTrain == 0 && fNumTrainingEntries == 0) {
            std::vector<ROOT::RDF::RResultPtr<ULong64_t>> counts;
            counts.reserve(fRdfs.size());
            for (auto &rdf : fRdfs) {
               counts.push_back(rdf.Count());
            }
            ROOT::RDF::RunGraphs({counts.begin(), counts.end()});

            std::size_t totalFiltered = 0;
            for (auto &c : counts) {
               totalFiltered += c.GetValue();
            }
            fNumTrainingEntries = static_cast<std::size_t>(totalFiltered * (1.0f - fValidationSplit));
            fNumValidationEntries = totalFiltered - fNumTrainingEntries;
         }

         ROOT::RDF::RNode &rdf = fRdfs[rdfIdx];

         // Fill data and collect raw entry indices that pass the filter
         std::vector<ULong64_t> rdfEntries;
         rdfEntries.reserve(endRow - startRow);

         RClusterLoaderFunctor<Args...> loader(dest, fNumChunkCols, fVecSizes, fVecPadding, 0, rowOffset);
         ROOT::Internal::RDF::ChangeBeginAndEndEntries(rdf, startRow, endRow);

         std::vector<std::string> colsWithEntry;
         colsWithEntry.reserve(fCols.size() + 1);
         colsWithEntry.push_back("rdfentry_");
         colsWithEntry.insert(colsWithEntry.end(), fCols.begin(), fCols.end());

         rdf.Foreach(
            [&](ULong64_t entry, const Args &...cols) {
               rdfEntries.push_back(entry);
               loader(cols...);
            },
            colsWithEntry);

         ROOT::Internal::RDF::ChangeBeginAndEndEntries(rdf, 0, fRdfSizes[rdfIdx]);

         const std::size_t totalFiltered = rdfEntries.size();
         if (totalFiltered == 0) {
            return 0;
         }
         std::sort(rdfEntries.begin(), rdfEntries.end());

         const std::size_t trainRemaining = fNumTrainingEntries - fAccumulatedFilteredForTrain;
         const std::size_t trainCount =
            std::min(static_cast<std::size_t>(totalFiltered * (1.0f - fValidationSplit)), trainRemaining);
         const std::size_t valCount = totalFiltered - trainCount;

         // The boundary is the raw entry index of the first entry assigned to validation.
         // Stable across epochs since the same filter always produces the same ordered entries.
         const std::uint64_t boundary = (valCount > 0) ? rdfEntries[trainCount] : endRow;

         if (trainCount > 0)
            fTrainingClusters.push_back({rdfIdx, startRow, boundary, trainCount});
         if (valCount > 0)
            fValidationClusters.push_back({rdfIdx, boundary, endRow, valCount});

         fAccumulatedFilteredForTrain += trainCount;
         return trainCount;
      }

      LoadClusterInto(dest, rdfIdx, startRow, endRow, rowOffset);
      return endRow - startRow;
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Load one validation cluster into \p dest starting at \p rowOffset
   void LoadValidationClusterInto(RFlat2DMatrix &dest, std::size_t rdfIdx, std::uint64_t startRow, std::uint64_t endRow,
                                  std::size_t rowOffset = 0)
   {
      LoadClusterInto(dest, rdfIdx, startRow, endRow, rowOffset);
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Mark the train/val split as finalised after the first epoch
   void FinaliseSplitDiscovery()
   {
      if (fIsFiltered)
         fSplitDiscovered = true;
   }

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
