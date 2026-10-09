#include "ROOT/ML/RClusterLoader.hxx"

namespace ROOT::Experimental::Internal::ML {

RClusterLoader::RClusterLoader(std::vector<ROOT::RDF::RNode> &rdfs, const std::vector<std::string> &cols,
                               const std::vector<std::size_t> &vecSizes, float vecPadding, float validationSplit,
                               bool shuffle, std::size_t setSeed)
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
void RClusterLoader::SplitDataset()
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
      // We generate a random boolean value to decide whether the training set gets the prefix
      // or suffix of each cluster to ensure better shuffling across runs when splitting.
      std::mt19937 g(fSetSeed);
      std::uniform_int_distribution<int> coin(0, 1);

      std::size_t cumulativeEntries = 0;
      std::size_t currentCumulativeTrain = 0;
      // We iterate over clusters and accumulate the entry counts to assign training and validation sizes
      // proportionally to the cluster size. Filtered clusters have varying sizes, so instead of calculating
      // the training size as a fraction of each cluster's size independently, we take into account
      // the cumulative counts of previous clusters in each calculation.
      for (const RClusterRange &c : fAllClusters) {
         const std::size_t sz = c.GetNumEntries();
         cumulativeEntries += sz;
         const std::size_t targetCumulativeTrain =
            static_cast<std::size_t>(cumulativeEntries * (1.0f - fValidationSplit));
         const std::size_t trainSz = targetCumulativeTrain - currentCumulativeTrain;
         currentCumulativeTrain = targetCumulativeTrain;
         const std::size_t valSz = sz - trainSz;

         // Randomly assign prefix or suffix to training
         bool trainIsPrefix = coin(g);
         const uint64_t trainStart = trainIsPrefix ? c.start : c.start + static_cast<std::uint64_t>(valSz);
         const uint64_t valStart = trainIsPrefix ? c.start + static_cast<std::uint64_t>(trainSz) : c.start;

         if (trainSz > 0) {
            fTrainingClusters.push_back({c.rdfIdx, trainStart, trainStart + static_cast<std::uint64_t>(trainSz)});
            fNumTrainingEntries += trainSz;
         }
         if (valSz > 0) {
            fValidationClusters.push_back({c.rdfIdx, valStart, valStart + static_cast<std::uint64_t>(valSz)});
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
         fValidationClusters.insert(fValidationClusters.end(), fAllClusters.begin() + splitIdx + 1, fAllClusters.end());

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
void RClusterLoader::ShuffleTrainingClusters(std::size_t epochIdx)
{
   if (!fShuffle) {
      return;
   }

   std::mt19937 g(fSetSeed == 0 ? std::random_device{}() : fSetSeed ^ epochIdx);
   std::shuffle(fTrainingClusters.begin(), fTrainingClusters.end(), g);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Re-order validation clusters for the upcoming epoch
void RClusterLoader::ShuffleValidationClusters(std::size_t epochIdx)
{
   if (!fShuffle) {
      return;
   }
   std::mt19937 g(fSetSeed == 0 ? std::random_device{}() : fSetSeed ^ epochIdx);
   std::shuffle(fValidationClusters.begin(), fValidationClusters.end(), g);
}

void RClusterLoader::LoadClusterInto(RFlat2DMatrix &dest, std::size_t rdfIdx, std::uint64_t startRow,
                                     std::uint64_t endRow, std::size_t rowOffset)
{
   ROOT::RDF::RNode &rdf = fRdfs[rdfIdx];
   ROOT::Internal::RDF::ChangeBeginAndEndEntries(rdf, startRow, endRow);
   auto values = ROOT::Internal::RDF::LoadCustomValues(rdf, fCols, fVecSizes, fVecPadding);
   std::copy(values->begin(), values->end(), dest.GetData() + rowOffset * fNumChunkCols);
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
std::size_t RClusterLoader::LoadTrainingClusterInto(RFlat2DMatrix &dest, std::size_t rdfIdx, std::uint64_t startRow,
                                                    std::uint64_t endRow, std::size_t rowOffset)
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

      std::vector<ULong64_t> rdfEntries;
      rdfEntries.reserve(endRow - startRow);

      ROOT::Internal::RDF::ChangeBeginAndEndEntries(rdf, startRow, endRow);
      rdf.Foreach([&](ULong64_t entry) { rdfEntries.push_back(entry); }, {"rdfentry_"});
      ROOT::Internal::RDF::ChangeBeginAndEndEntries(rdf, 0, fRdfSizes[rdfIdx]);

      const std::size_t totalFiltered = rdfEntries.size();
      if (totalFiltered == 0) {
         return 0;
      }
      std::sort(rdfEntries.begin(), rdfEntries.end());

      const std::size_t cumulativeFiltered = fAccumulatedFilteredForTrain + fAccumulatedFilteredForVal + totalFiltered;
      const std::size_t targetCumulativeTrain =
         std::min(static_cast<std::size_t>(cumulativeFiltered * (1.0f - fValidationSplit)), fNumTrainingEntries);
      const std::size_t trainCount = targetCumulativeTrain - fAccumulatedFilteredForTrain;
      const std::size_t valCount = totalFiltered - trainCount;

      bool trainIsPrefix = true;
      if (fShuffle) {
         // If shuffling is enabled, we generate a random boolean value to decide whether the training set
         // gets the prefix or suffix of each cluster to ensure better shuffling across runs when splitting.
         std::mt19937 g(fSetSeed + fAccumulatedFilteredForTrain); // vary per cluster
         std::uniform_int_distribution<int> coin(0, 1);
         trainIsPrefix = coin(g);
      }

      // The boundary is the raw entry index that splits train and val sub-ranges within the
      // cluster. Stable across epochs since the same filter always produces the same ordered
      // entries. When one side has no filtered entries we fall back to the cluster endpoint that
      // collapses that side to an empty range, avoiding an out-of-bounds access into rdfEntries
      // (whose size is totalFiltered, so rdfEntries[totalFiltered] is OOB and trips libstdc++
      // hardened-mode assertions).
      std::uint64_t boundary;
      if (trainIsPrefix) {
         // train = [startRow, boundary), val = [boundary, endRow)
         boundary = (trainCount < totalFiltered) ? rdfEntries[trainCount] : endRow;
      } else {
         // train = [boundary, endRow), val = [startRow, boundary)
         boundary = (valCount < totalFiltered) ? rdfEntries[valCount] : endRow;
      }

      const std::uint64_t trainStart = trainIsPrefix ? startRow : boundary;
      const std::uint64_t trainEnd = trainIsPrefix ? boundary : endRow;
      const std::uint64_t valStart = trainIsPrefix ? boundary : startRow;
      const std::uint64_t valEnd = trainIsPrefix ? endRow : boundary;

      if (trainCount > 0)
         fTrainingClusters.push_back({rdfIdx, trainStart, trainEnd, trainCount});
      if (valCount > 0)
         fValidationClusters.push_back({rdfIdx, valStart, valEnd, valCount});

      fAccumulatedFilteredForTrain += trainCount;
      fAccumulatedFilteredForVal += valCount;

      if (trainCount > 0)
         LoadClusterInto(dest, rdfIdx, trainStart, trainEnd, rowOffset);

      return trainCount;
   }

   LoadClusterInto(dest, rdfIdx, startRow, endRow, rowOffset);
   return endRow - startRow;
}

//////////////////////////////////////////////////////////////////////////
/// \brief Load one validation cluster into \p dest starting at \p rowOffset
void RClusterLoader::LoadValidationClusterInto(RFlat2DMatrix &dest, std::size_t rdfIdx, std::uint64_t startRow,
                                               std::uint64_t endRow, std::size_t rowOffset)
{
   LoadClusterInto(dest, rdfIdx, startRow, endRow, rowOffset);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Mark the train/val split as finalised after the first epoch
void RClusterLoader::FinaliseSplitDiscovery()
{
   if (fIsFiltered)
      fSplitDiscovered = true;
}

} // namespace ROOT::Experimental::Internal::ML
