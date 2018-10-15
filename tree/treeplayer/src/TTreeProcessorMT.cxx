// @(#)root/thread:$Id$
// Author: Enric Tejedor, CERN  12/09/2016

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
   std::vector<EntryCluster> clusters;
   std::vector<Long64_t> nEntries;
   const auto nFileNames = fileNames.size();
   Long64_t offset = 0ll;
   for (auto i = 0u; i < nFileNames; ++i) {
      std::unique_ptr<TFile> f(TFile::Open(fileNames[i].c_str())); // need TFile::Open to load plugins if need be
      TTree *t = nullptr; // not a leak, t will be deleted by f
      f->GetObject(treeName.c_str(), t);
      auto clusterIter = t->GetClusterIterator(0);
      Long64_t start = 0ll, end = 0ll;
      const Long64_t entries = t->GetEntries();
      nEntries.emplace_back(entries);
      // Iterate over the clusters in the current file
      while ((start = clusterIter()) < entries) {
         end = clusterIter.GetNextEntry();
         // Add the current file's offset to start and end to make them (chain) global
         clusters.emplace_back(EntryCluster{start + offset, end + offset});
      }
      offset += entries;
   }

   return std::make_pair(std::move(clusters), std::move(nEntries));
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

////////////////////////////////////////////////////////////////////////
/// Return the full path of the tree
static std::string GetTreeFullPath(const TTree &tree)
{
   // Case 1: this is a TChain: we get the name out of the first TChainElement
   if (0 == strcmp("TChain", tree.ClassName())) {
      auto &chain = dynamic_cast<const TChain&>(tree);
      auto files = chain.GetListOfFiles();
      if (files && 0 != files->GetEntries()) {
         return files->At(0)->GetName();
      }
   }

   // Case 2: this is a TTree: we get the full path of it
   if (auto motherDir = tree.GetDirectory()) {
      std::string fullPath(motherDir->GetPath());
      fullPath += "/";
      fullPath += tree.GetName();
      return fullPath;
   }

   // We do our best and return the name of the tree
   return tree.GetName();
}

TTreeView::TTreeView(TTree& tree) : fTreeName(GetTreeFullPath(tree))
{
   static const TClassRef clRefTChain("TChain");
   if (clRefTChain == tree.IsA()) {
      TObjArray* filelist = static_cast<TChain&>(tree).GetListOfFiles();
      if (filelist->GetEntries() > 0) {
         for (auto f : *filelist)
            fFileNames.emplace_back(f->GetTitle());
         StoreFriends(tree, false);
      }
      else {
         auto msg = "The provided chain of files is empty, cannot process tree " + fTreeName;
         throw std::runtime_error(msg);
      }
   }
   else {
      TFile *f = tree.GetCurrentFile();
      if (f) {
         fFileNames.emplace_back(f->GetName());
         StoreFriends(tree, true);
      }
      else {
         auto msg = "The specified TTree is not linked to any file, in-memory-only trees are not supported. Cannot process tree " + fTreeName;
         throw std::runtime_error(msg);
      }
   }
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
   // Enable this IMT use case (activate its locks)
   Internal::TParTreeProcessingRAII ptpRAII;

   const auto clustersAndEntries = ROOT::Internal::MakeClusters(treeView->GetTreeName(), treeView->GetFileNames());
   const auto &clusters = clustersAndEntries.first;
   const auto &entries = clustersAndEntries.second;

   const auto friendEntries =
      ROOT::Internal::GetFriendEntries(treeView->GetFriendNames(), treeView->GetFriendFileNames());

   auto mapFunction = [this, &func, &entries, &friendEntries](const ROOT::Internal::EntryCluster &c) {
      // This task will operate with the tree that contains start
      treeView->PushTaskFirstEntry(c.start);

      std::unique_ptr<TTreeReader> reader;
      std::unique_ptr<TEntryList> elist;
      std::tie(reader, elist) = treeView->GetTreeReader(c.start, c.end, entries, friendEntries);
      func(*reader);

      // In case of task interleaving, we need to load here the tree of the parent task
      treeView->PopTaskFirstEntry();
   };

   // Assume number of threads has been initialized via ROOT::EnableImplicitMT
   TThreadExecutor pool;
   pool.Foreach(mapFunction, clusters);
}
