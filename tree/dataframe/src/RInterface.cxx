// Author: Vincenzo Eduardo Padulano, Axel Naumann, Enrico Guiraud CERN 02/2023

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RInterface.hxx"
#include <ROOT/InternalTreeUtils.hxx>
#include "ROOT/RNTupleDS.hxx"
#include "ROOT/RTTreeDS.hxx"
#ifdef R__USE_IMT
#include <ROOT/TThreadExecutor.hxx>
#endif

void ROOT::Internal::RDF::ChangeEmptyEntryRange(const ROOT::RDF::RNode &node,
                                                std::pair<ULong64_t, ULong64_t> &&newRange)
{
   R__ASSERT(newRange.second >= newRange.first && "end is less than begin in the passed entry range!");
   node.GetLoopManager()->SetEmptyEntryRange(std::move(newRange));
}

void ROOT::Internal::RDF::ChangeBeginAndEndEntries(const ROOT::RDF::RNode &node, Long64_t begin, Long64_t end)
{
   R__ASSERT(end >= begin && "end is less than begin in the passed entry range!");
   node.GetLoopManager()->ChangeBeginAndEndEntries(begin, end);
}

/**
 * \brief Changes the input dataset specification of an RDataFrame.
 *
 * \param node Any node of the computation graph.
 * \param spec The new specification.
 */
void ROOT::Internal::RDF::ChangeSpec(const ROOT::RDF::RNode &node, ROOT::RDF::Experimental::RDatasetSpec &&spec)
{
   node.GetLoopManager()->ChangeSpec(std::move(spec));
}

/**
 * \brief Retrieve the cluster boundaries for each cluster in the dataset,
 * across files, with a global offset.
 *
 * \param node Any node of the computation graph.
 * \return A vector of [begin, end) entry pairs for each cluster in the dataset.
 *
 * \note When IMT is enabled, files are processed in parallel using a thread pool.
 */
std::vector<std::pair<std::uint64_t, std::uint64_t>>
ROOT::Internal::RDF::GetDatasetGlobalClusterBoundaries(const ROOT::RDF::RNode &node)
{
   std::vector<std::pair<std::uint64_t, std::uint64_t>> boundaries{};

   auto *lm = node.GetLoopManager();
   auto *ds = lm->GetDataSource();

   if (!ds) {
      throw std::runtime_error("Cannot retrieve cluster boundaries: no data source available.");
   }

   std::string datasetName;
   std::vector<std::string> fileNames;
   bool isTTree = false;

   if (auto *ttreeds = dynamic_cast<RTTreeDS *>(ds)) {
      auto *tree = ttreeds->GetTree();
      assert(tree && "The internal TTree is not available, something went wrong.");
      datasetName = tree->GetName();
      fileNames = ROOT::Internal::TreeUtils::GetFileNamesFromTree(*tree);
      isTTree = true;
   } else if (auto *rntupleds = dynamic_cast<RNTupleDS *>(ds)) {
      datasetName = rntupleds->fNTupleName;
      fileNames = rntupleds->fFileNames;
      isTTree = false;
   } else {
      throw std::runtime_error("Cannot retrieve cluster boundaries: unsupported data source type.");
   }

   if (fileNames.empty()) {
      return boundaries;
   }

   const auto nFiles = fileNames.size();

   // For each file retrieve the cluster boundaries + the number of entries
   using FileResult = std::pair<std::vector<std::pair<std::uint64_t, std::uint64_t>>, std::uint64_t>;
   std::vector<FileResult> perFileResults(nFiles);

   // Function to process a single file and return its cluster boundaries + entry count
   auto processFile = [&datasetName, isTTree](const std::string &fileName) -> FileResult {
      std::vector<std::pair<std::uint64_t, std::uint64_t>> clusters;
      std::uint64_t nEntries = 0;

      if (isTTree) {
         // TTree
         auto [clusterBoundaries, entries] = ROOT::Internal::TreeUtils::GetClustersAndEntries(datasetName, fileName);
         nEntries = entries;
         // [0, 10, 20, ...] --> [(0,10), (10,20), ...]
         for (std::size_t i = 0; i + 1 < clusterBoundaries.size(); ++i) {
            clusters.emplace_back(clusterBoundaries[i], clusterBoundaries[i + 1]);
         }
      } else {
         // RNTuple
         auto [clusterBoundaries, entries] = GetClustersAndEntries(datasetName, fileName);
         nEntries = entries;
         for (const auto &cluster : clusterBoundaries) {
            clusters.emplace_back(cluster.fFirstEntry, cluster.fLastEntryPlusOne);
         }
      }

      return {clusters, nEntries};
   };

#ifdef R__USE_IMT
   ROOT::TThreadExecutor pool;
   // Distribute the processing of files in parallel across the thread pool,
   // each thread takes a file and its index in the fileNames vector as input
   // and fills the corresponding position in the perFileResults vector
   pool.Foreach([&perFileResults, &fileNames,
                 &processFile](std::size_t idx) { perFileResults[idx] = processFile(fileNames[idx]); },
                ROOT::TSeq<std::size_t>(nFiles));
#else
   // Process files sequentially as a fallback
   for (std::size_t idx = 0; idx < nFiles; ++idx) {
      perFileResults[idx] = processFile(fileNames[idx]);
   }
#endif
   // Now that we have the cluster boundaries and entry counts for each file,
   // we can compute the global boundaries with offsets (sequentially)
   std::uint64_t offset = 0;
   for (const auto &[clusters, nEntries] : perFileResults) {
      for (const auto &[start, end] : clusters) {
         boundaries.emplace_back(offset + start, offset + end);
      }
      offset += nEntries;
   }

   return boundaries;
}

/**
 * \brief Trigger the execution of an RDataFrame computation graph.
 * \param[in] node A node of the computation graph (not a result).
 *
 * This function calls the RLoopManager::Run method on the \p fLoopManager data
 * member of the input argument. It is intended for internal use only.
 */
void ROOT::Internal::RDF::TriggerRun(ROOT::RDF::RNode node)
{
   node.fLoopManager->Run();
}

std::string ROOT::Internal::RDF::GetDataSourceLabel(const ROOT::RDF::RNode &node)
{
   if (auto ds = node.GetDataSource()) {
      return ds->GetLabel();
   } else {
      return "EmptyDS";
   }
}

void ROOT::Internal::RDF::SetTTreeLifeline(ROOT::RDF::RNode &node, std::any lifeline)
{
   node.GetLoopManager()->SetTTreeLifeline(std::move(lifeline));
}
