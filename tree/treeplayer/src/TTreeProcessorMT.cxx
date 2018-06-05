// @(#)root/thread:$Id$
// Authors: Enric Tejedor, Enrico Guiraud CERN  05/06/2018

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class ROOT::TTreeProcessorMT
    \ingroup Parallelism
    \brief A class to process the entries of a TTree in parallel.

By means of its Process method, ROOT::TTreeProcessorMT provides a way to process the
entries of a TTree in parallel. When invoking TTreeProcessor::Process, the user
passes a function whose only parameter is a TTreeReader. The function iterates
on a subrange of entries by using that TTreeReader.

The implementation of ROOT::TTreeProcessorMT parallelizes the processing of the subranges,
each corresponding to a cluster in the TTree. This is possible thanks to the use
of a ROOT::TThreadedObject, so that each thread works with its own TFile and TTree
objects.
*/

#include "TROOT.h"
#include "ROOT/TTreeProcessorMT.hxx"
#include "ROOT/TThreadExecutor.hxx"

using namespace ROOT;

namespace ROOT {
namespace Internal {
////////////////////////////////////////////////////////////////////////
/// Return a vector of cluster boundaries for the given tree and files.
ClustersAndEntries
MakeClusters(const std::string &treeName, const std::vector<std::string> &fileNames)
{
   // Note that as a side-effect of opening all files that are going to be used in the
   // analysis once, all necessary streamers will be loaded into memory.
   TDirectory::TContext c;
   std::vector<std::vector<EntryCluster>> clustersPerFile;
   std::vector<Long64_t> entriesPerFile;
   const auto nFileNames = fileNames.size();
   Long64_t offset = 0ll;
   for (auto i = 0u; i < nFileNames; ++i) {
      std::unique_ptr<TFile> f(TFile::Open(fileNames[i].c_str())); // need TFile::Open to load plugins if need be
      TTree *t = nullptr; // not a leak, t will be deleted by f
      f->GetObject(treeName.c_str(), t);
      auto clusterIter = t->GetClusterIterator(0);
      Long64_t start = 0ll, end = 0ll;
      const Long64_t entries = t->GetEntries();
      // Iterate over the clusters in the current file
      std::vector<EntryCluster> clusters;
      while ((start = clusterIter()) < entries) {
         end = clusterIter.GetNextEntry();
         // Add the current file's offset to start and end to make them (chain) global
         clusters.emplace_back(EntryCluster{start + offset, end + offset});
      }
      offset += entries;
      clustersPerFile.emplace_back(std::move(clusters));
      entriesPerFile.emplace_back(entries);
   }

   return std::make_pair(std::move(clustersPerFile), std::move(entriesPerFile));
}

////////////////////////////////////////////////////////////////////////
/// Return a vector containing the number of entries of each file of each friend TChain
std::vector<std::vector<Long64_t>> GetFriendEntries(const std::vector<std::pair<std::string, std::string>> &friendNames,
                                                    const std::vector<std::vector<std::string>> &friendFileNames)
{
   std::vector<std::vector<Long64_t>> friendEntries;
   const auto nFriends = friendNames.size();
   for (auto i = 0u; i < nFriends; ++i) {
      std::vector<Long64_t> nEntries;
      const auto &thisFriendName = friendNames[i].first;
      const auto &thisFriendFiles = friendFileNames[i];
      for (const auto &fname : thisFriendFiles) {
         std::unique_ptr<TFile> f(TFile::Open(fname.c_str()));
         TTree *t = nullptr; // owned by TFile
         f->GetObject(thisFriendName.c_str(), t);
         nEntries.emplace_back(t->GetEntries());
      }
      friendEntries.emplace_back(std::move(nEntries));
   }

   return friendEntries;
}
}
}

////////////////////////////////////////////////////////////////////////
/// Constructor based on a file name.
/// \param[in] filename Name of the file containing the tree to process.
/// \param[in] treename Name of the tree to process. If not provided,
///                     the implementation will automatically search for a
///                     tree in the file.
TTreeProcessorMT::TTreeProcessorMT(std::string_view filename, std::string_view treename) : treeView(filename, treename) {}

////////////////////////////////////////////////////////////////////////
/// Constructor based on a collection of file names.
/// \param[in] filenames Collection of the names of the files containing the tree to process.
/// \param[in] treename Name of the tree to process. If not provided,
///                     the implementation will automatically search for a
///                     tree in the collection of files.
TTreeProcessorMT::TTreeProcessorMT(const std::vector<std::string_view> &filenames, std::string_view treename) : treeView(filenames, treename) {}

////////////////////////////////////////////////////////////////////////
/// Constructor based on a TTree.
/// \param[in] tree Tree or chain of files containing the tree to process.
TTreeProcessorMT::TTreeProcessorMT(TTree &tree) : treeView(tree) {}

////////////////////////////////////////////////////////////////////////
/// Constructor based on a TTree and a TEntryList.
/// \param[in] tree Tree or chain of files containing the tree to process.
/// \param[in] entries List of entry numbers to process.
TTreeProcessorMT::TTreeProcessorMT(TTree &tree, TEntryList &entries) : treeView(tree, entries) {}

//////////////////////////////////////////////////////////////////////////////
/// Process the entries of a TTree in parallel. The user-provided function
/// receives a TTreeReader which can be used to iterate on a subrange of
/// entries
/// ~~~{.cpp}
/// TTreeProcessorMT::Process([](TTreeReader& readerSubRange) {
///                            // Select branches to read
///                            while (readerSubRange.next()) {
///                                // Use content of current entry
///                            }
///                         });
/// ~~~
/// The user needs to be aware that each of the subranges can potentially
/// be processed in parallel. This means that the code of the user function
/// should be thread safe.
///
/// \param[in] func User-defined function that processes a subrange of entries
void TTreeProcessorMT::Process(std::function<void(TTreeReader &)> func)
{
   // If an entry list or friend trees are present, we need to generate clusters with global entry numbers,
   // so we do it here for all files.
   const bool hasFriends = !treeView->GetFriendNames().empty();
   const bool hasEntryList = treeView->GetEntryList().GetN() > 0;
   const bool shouldRetrieveAllClusters = hasFriends || hasEntryList;
   const auto clustersAndEntries = shouldRetrieveAllClusters
                                      ? Internal::MakeClusters(treeView->GetTreeName(), treeView->GetFileNames())
                                      : Internal::ClustersAndEntries{};
   const auto &clusters = clustersAndEntries.first;
   const auto &entries = clustersAndEntries.second;

   // Retrieve number of entries for each file for each friend tree
   const auto friendEntries =
      hasFriends ? Internal::GetFriendEntries(treeView->GetFriendNames(), treeView->GetFriendFileNames())
                 : std::vector<std::vector<Long64_t>>{};

   TThreadExecutor pool;
   // Parent task, spawns tasks that process each of the entry clusters for each input file
   using Internal::EntryCluster;
   auto processFile = [&](std::size_t fileIdx) {

      // If cluster information is already present, build TChains with all input files and use global entry numbers
      // Otherwise get cluster information only for the file we need to process and use local entry numbers
      const bool shouldUseGlobalEntries = hasFriends || hasEntryList;
      // theseFiles contains either all files or just the single file to process
      const auto &theseFiles = shouldUseGlobalEntries ? treeView->GetFileNames()
                                                      : std::vector<std::string>({treeView->GetFileNames()[fileIdx]});
      // Evaluate clusters (with local entry numbers) and number of entries for this file, if needed
      const auto theseClustersAndEntries = shouldUseGlobalEntries
                                              ? Internal::ClustersAndEntries{}
                                              : Internal::MakeClusters(treeView->GetTreeName(), theseFiles);

      // All clusters for the file to process, either with global or local entry numbers
      const auto &thisFileClusters = shouldUseGlobalEntries ? clusters[fileIdx] : theseClustersAndEntries.first[0];

      // Either all number of entries or just the ones for this file
      const auto &theseEntries =
         shouldUseGlobalEntries ? entries : std::vector<Long64_t>({theseClustersAndEntries.second[0]});

      auto processCluster = [&](const Internal::EntryCluster &c) {
         // This task will operate with the tree that contains start
         treeView->PushTaskFirstEntry(c.start);

         std::unique_ptr<TTreeReader> reader;
         std::unique_ptr<TEntryList> elist;
         std::tie(reader, elist) = treeView->GetTreeReader(c.start, c.end, theseFiles, theseEntries, friendEntries);
         func(*reader);

         // In case of task interleaving, we need to load here the tree of the parent task
         treeView->PopTaskFirstEntry();
      };

      pool.Foreach(processCluster, thisFileClusters);
   };

   std::vector<std::size_t> fileIdxs(treeView->GetFileNames().size());
   std::iota(fileIdxs.begin(), fileIdxs.end(), 0u);

   // Enable this IMT use case (activate its locks)
   Internal::TParTreeProcessingRAII ptpRAII;

   pool.Foreach(processFile, fileIdxs);
}
