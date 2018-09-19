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

////////////////////////////////////////////////////////////////////////////////
/// Get and store the names, aliases and file names of the friends of the tree.
/// \param[in] tree The main tree whose friends to 
///
/// Note that "friends of friends" and circular references in the lists of friends are not supported.
Internal::FriendInfo TTreeProcessorMT::GetFriendInfo(TTree &tree)
{
   std::vector<Internal::NameAlias> friendNames;
   std::vector<std::vector<std::string>> friendFileNames;

   const auto friends = tree.GetListOfFriends();
   if (!friends)
      return Internal::FriendInfo();

   for (auto fr : *friends) {
      const auto frTree = static_cast<TFriendElement *>(fr)->GetTree();

      // Check if friend tree has an alias
      const auto realName = frTree->GetName();
      const auto alias = tree.GetFriendAlias(frTree);
      if (alias) {
         friendNames.emplace_back(std::make_pair(realName, std::string(alias)));
      } else {
         friendNames.emplace_back(std::make_pair(realName, ""));
      }

      // Store the file names of the friend tree
      friendFileNames.emplace_back();
      auto &fileNames = friendFileNames.back();
      const bool isChain = tree.IsA() == TChain::Class();
      if (isChain) {
         const auto frChain = static_cast<TChain *>(frTree);
         for (auto f : *(frChain->GetListOfFiles())) {
            fileNames.emplace_back(f->GetTitle());
         }
      } else {
         const auto f = frTree->GetCurrentFile();
         if (!f)
            throw std::runtime_error("Friend trees with no associated file are not supported.");
         fileNames.emplace_back(f->GetName());
      }
   }

   return Internal::FriendInfo{std::move(friendNames), std::move(friendFileNames)};
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the name of the first TTree in the first input file, else throw.
std::string TTreeProcessorMT::FindTreeName()
{
   std::string treeName;

   if (fFileNames.empty())
      throw std::runtime_error("Empty list of files and no tree name provided");

   ::TDirectory::TContext ctxt(gDirectory);
   std::unique_ptr<TFile> f(TFile::Open(fFileNames[0].c_str()));
   TIter next(f->GetListOfKeys());
   while (TKey *key = (TKey *)next()) {
      const char *className = key->GetClassName();
      if (strcmp(className, "TTree") == 0) {
         treeName = key->GetName();
         break;
      }
   }
   if (treeName.empty())
      throw std::runtime_error("Cannot find any tree in file " + fFileNames[0]);

   return treeName;
}

////////////////////////////////////////////////////////////////////////
/// Constructor based on a file name.
/// \param[in] filename Name of the file containing the tree to process.
/// \param[in] treename Name of the tree to process. If not provided,
///                     the implementation will automatically search for a
///                     tree in the file.
TTreeProcessorMT::TTreeProcessorMT(std::string_view filename, std::string_view treename)
   : fFileNames({std::string(filename)}), fTreeName(treename.empty() ? FindTreeName() : treename), fFriendInfo() {}

std::vector<std::string> CheckAndConvert(const std::vector<std::string_view> & views)
{
   if (views.empty())
      throw std::runtime_error("The provided list of file names is empty");

   std::vector<std::string> strings;
   strings.reserve(views.size());
   for (const auto &v : views)
      strings.emplace_back(v);
   return strings;
}

////////////////////////////////////////////////////////////////////////
/// Constructor based on a collection of file names.
/// \param[in] filenames Collection of the names of the files containing the tree to process.
/// \param[in] treename Name of the tree to process. If not provided,
///                     the implementation will automatically search for a
///                     tree in the collection of files.
TTreeProcessorMT::TTreeProcessorMT(const std::vector<std::string_view> &filenames, std::string_view treename)
   : fFileNames(CheckAndConvert(filenames)), fTreeName(treename.empty() ? FindTreeName() : treename), fFriendInfo() {}

std::vector<std::string> GetFilesFromTree(TTree &tree)
{
   std::vector<std::string> filenames;

   const bool isChain = tree.IsA() == TChain::Class();
   if (isChain) {
      TObjArray *filelist = static_cast<TChain &>(tree).GetListOfFiles();
      const auto nFiles = filelist->GetEntries();
      if (nFiles == 0)
         throw std::runtime_error("The provided chain of files is empty");
      filenames.reserve(nFiles);
      for (auto f : *filelist)
         filenames.emplace_back(f->GetTitle());
   } else {
      TFile *f = tree.GetCurrentFile();
      if (!f) {
         const auto msg = "The specified TTree is not linked to any file, in-memory-only trees are not supported.";
         throw std::runtime_error(msg);
      }

      filenames.emplace_back(f->GetName());
   }

   return filenames;
}

////////////////////////////////////////////////////////////////////////
/// Constructor based on a TTree and a TEntryList.
/// \param[in] tree Tree or chain of files containing the tree to process.
/// \param[in] entries List of entry numbers to process.
TTreeProcessorMT::TTreeProcessorMT(TTree &tree, const TEntryList &entries)
   : fFileNames(GetFilesFromTree(tree)), fTreeName(tree.GetName()), fEntryList(entries),
     fFriendInfo(GetFriendInfo(tree)) {}

////////////////////////////////////////////////////////////////////////
/// Constructor based on a TTree.
/// \param[in] tree Tree or chain of files containing the tree to process.
TTreeProcessorMT::TTreeProcessorMT(TTree &tree) : TTreeProcessorMT(tree, TEntryList()) {}

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
   const std::vector<Internal::NameAlias> &friendNames = fFriendInfo.fFriendNames;
   const std::vector<std::vector<std::string>> &friendFileNames = fFriendInfo.fFriendFileNames;

   // If an entry list or friend trees are present, we need to generate clusters with global entry numbers,
   // so we do it here for all files.
   const bool hasFriends = !friendNames.empty();
   const bool hasEntryList = fEntryList.GetN() > 0;
   const bool shouldRetrieveAllClusters = hasFriends || hasEntryList;
   const auto clustersAndEntries =
      shouldRetrieveAllClusters ? Internal::MakeClusters(fTreeName, fFileNames) : Internal::ClustersAndEntries{};
   const auto &clusters = clustersAndEntries.first;
   const auto &entries = clustersAndEntries.second;

   // Retrieve number of entries for each file for each friend tree
   const auto friendEntries =
      hasFriends ? Internal::GetFriendEntries(friendNames, friendFileNames) : std::vector<std::vector<Long64_t>>{};

   TThreadExecutor pool;
   // Parent task, spawns tasks that process each of the entry clusters for each input file
   using Internal::EntryCluster;
   auto processFile = [&](std::size_t fileIdx) {

      // If cluster information is already present, build TChains with all input files and use global entry numbers
      // Otherwise get cluster information only for the file we need to process and use local entry numbers
      const bool shouldUseGlobalEntries = hasFriends || hasEntryList;
      // theseFiles contains either all files or just the single file to process
      const auto &theseFiles = shouldUseGlobalEntries ? fFileNames : std::vector<std::string>({fFileNames[fileIdx]});
      // Evaluate clusters (with local entry numbers) and number of entries for this file, if needed
      const auto theseClustersAndEntries =
         shouldUseGlobalEntries ? Internal::ClustersAndEntries{} : Internal::MakeClusters(fTreeName, theseFiles);

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
         std::tie(reader, elist) = treeView->GetTreeReader(c.start, c.end, fTreeName, theseFiles, fFriendInfo,
                                                           fEntryList, theseEntries, friendEntries);
         func(*reader);

         // In case of task interleaving, we need to load here the tree of the parent task
         treeView->PopTaskFirstEntry();
      };

      pool.Foreach(processCluster, thisFileClusters);
   };

   std::vector<std::size_t> fileIdxs(fFileNames.size());
   std::iota(fileIdxs.begin(), fileIdxs.end(), 0u);

   // Enable this IMT use case (activate its locks)
   Internal::TParTreeProcessingRAII ptpRAII;

   pool.Foreach(processFile, fileIdxs);
}
