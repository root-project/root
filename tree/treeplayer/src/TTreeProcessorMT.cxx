// @(#)root/thread:$Id$
// Authors: Enric Tejedor, Enrico Guiraud CERN  05/06/2018

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
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

using namespace ROOT;

namespace {

/// A cluster of entries
struct EntryCluster {
   Long64_t start;
   Long64_t end;
};

// note that this routine assumes global entry numbers
static bool ClustersAreSortedAndContiguous(const std::vector<std::vector<EntryCluster>> &cls)
{
   Long64_t last_end = 0ll;
   for (const auto &fcl : cls) {
      for (const auto &c : fcl) {
         if (last_end != c.start)
            return false;
         last_end = c.end;
      }
   }
   return true;
}

/// Take a vector of vectors of EntryClusters (a vector per file), filter the entries according to entryList, and
/// and return a new vector of vectors of EntryClusters where cluster start/end entry numbers have been converted to
/// TEntryList-local entry numbers.
///
/// This routine assumes that entry numbers in the TEntryList (and, if present, in the sub-entrylists) are in
/// ascending order, i.e., for n > m:
///   elist.GetEntry(n) + tree_offset_for_entry_from_elist(n) > elist.GetEntry(m) + tree_offset_for_entry_from_elist(m)
static std::vector<std::vector<EntryCluster>>
ConvertToElistClusters(std::vector<std::vector<EntryCluster>> &&clusters, TEntryList &entryList,
                       const std::vector<std::string> &treeNames, const std::vector<std::string> &fileNames,
                       const std::vector<Long64_t> &entriesPerFile)
{
   R__ASSERT(entryList.GetN() > 0); // wasteful to call this function if it has nothing to do
   R__ASSERT(ClustersAreSortedAndContiguous(clusters));

   const bool listHasGlobalEntryNumbers = entryList.GetLists() == nullptr;
   const auto nFiles = clusters.size();

   std::unique_ptr<TChain> chain;
   using NextFn_t = Long64_t (*)(Long64_t &, TEntryList &, TChain *);
   // A function that advances TEntryList and returns global entry numbers or -1 if we reached the end
   // (might or might not need a TChain depending on whether listHasGlobalEntryNumbers)
   NextFn_t Next;
   if (listHasGlobalEntryNumbers) {
      Next = [](Long64_t &elEntry, TEntryList &elist, TChain *) {
         ++elEntry;
         return elist.Next();
      };
   } else {
      // we need `chain` to be able to convert local entry numbers to global entry numbers in `Next`
      chain.reset(new TChain());
      for (auto i = 0u; i < nFiles; ++i)
         chain->Add((fileNames[i] + "/" + treeNames[i]).c_str(), entriesPerFile[i]);
      Next = [](Long64_t &elEntry, TEntryList &elist, TChain *ch) {
         ++elEntry;
         int treenum = -1;
         Long64_t localEntry = elist.GetEntryAndTree(elEntry, treenum);
         if (localEntry == -1ll)
            return localEntry;
         return localEntry + ch->GetTreeOffset()[treenum];
      };
   }

   // the call to GetEntry also serves the purpose to reset TEntryList::fLastIndexQueried,
   // so we can be sure TEntryList::Next will return the correct thing
   Long64_t elistEntry = 0ll;
   Long64_t entry = entryList.GetEntry(elistEntry);

   std::vector<std::vector<EntryCluster>> elistClusters;

   for (auto fileN = 0u; fileN < nFiles; ++fileN) {
      std::vector<EntryCluster> elistClustersForFile;
      for (const auto &c : clusters[fileN]) {
         if (entry >= c.end || entry == -1ll) // no entrylist entries in this cluster
            continue;
         R__ASSERT(entry >= c.start); // current entry should never come before the cluster we are looking at
         const Long64_t elistRangeStart = elistEntry;
         // advance entry list until the entrylist entry goes beyond the end of the cluster
         while (entry < c.end && entry != -1ll)
            entry = Next(elistEntry, entryList, chain.get());
         elistClustersForFile.emplace_back(EntryCluster{elistRangeStart, elistEntry});
      }
      elistClusters.emplace_back(std::move(elistClustersForFile));
   }

   R__ASSERT(elistClusters.size() == clusters.size()); // same number of files
   R__ASSERT(ClustersAreSortedAndContiguous(elistClusters));

   entryList.GetEntry(0ll); // reset TEntryList internal state, lest we incur in ROOT-10807
   return elistClusters;
}

// EntryClusters and number of entries per file
using ClustersAndEntries = std::pair<std::vector<std::vector<EntryCluster>>, std::vector<Long64_t>>;

////////////////////////////////////////////////////////////////////////
/// Return a vector of cluster boundaries for the given tree and files.
static ClustersAndEntries MakeClusters(const std::vector<std::string> &treeNames,
                                       const std::vector<std::string> &fileNames, const unsigned int maxTasksPerFile)
{
   // Note that as a side-effect of opening all files that are going to be used in the
   // analysis once, all necessary streamers will be loaded into memory.
   TDirectory::TContext c;
   const auto nFileNames = fileNames.size();
   std::vector<std::vector<EntryCluster>> clustersPerFile;
   std::vector<Long64_t> entriesPerFile;
   entriesPerFile.reserve(nFileNames);
   Long64_t offset = 0ll;
   for (auto i = 0u; i < nFileNames; ++i) {
      const auto &fileName = fileNames[i];
      const auto &treeName = treeNames[i];

      std::unique_ptr<TFile> f(TFile::Open(fileName.c_str())); // need TFile::Open to load plugins if need be
      if (!f || f->IsZombie()) {
         const auto msg = "TTreeProcessorMT::Process: an error occurred while opening file \"" + fileName + "\"";
         throw std::runtime_error(msg);
      }
      auto *t = f->Get<TTree>(treeName.c_str()); // t will be deleted by f

      if (!t) {
         const auto msg = "TTreeProcessorMT::Process: an error occurred while getting tree \"" + treeName +
                          "\" from file \"" + fileName + "\"";
         throw std::runtime_error(msg);
      }

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

   // Here we "fuse" clusters together if the number of clusters is too big with respect to
   // the number of slots, otherwise we can incur in an overhead which is big enough
   // to make parallelisation detrimental to performance.
   // For example, this is the case when, following a merging of many small files, a file
   // contains a tree with many entries and with clusters of just a few entries each.
   // Another problematic case is a high number of slots (e.g. 256) coupled with a high number
   // of files (e.g. 1000 files): the large amount of files might result in a large amount
   // of tasks, but the elevated concurrency level makes the little synchronization required by
   // task initialization very expensive. In this case it's better to simply process fewer, larger tasks.
   // Cluster-merging can help reduce the number of tasks down to a minumum of one task per file.
   //
   // The criterion according to which we fuse clusters together is to have around
   // TTreeProcessorMT::GetTasksPerWorkerHint() clusters per slot.
   // Concretely, for each file we will cap the number of tasks to ceil(GetTasksPerWorkerHint() * nWorkers / nFiles).

   std::vector<std::vector<EntryCluster>> eventRangesPerFile(clustersPerFile.size());
   auto clustersPerFileIt = clustersPerFile.begin();
   auto eventRangesPerFileIt = eventRangesPerFile.begin();
   for (; clustersPerFileIt != clustersPerFile.end(); clustersPerFileIt++, eventRangesPerFileIt++) {
      const auto clustersInThisFileSize = clustersPerFileIt->size();
      const auto nFolds = clustersInThisFileSize / maxTasksPerFile;
      // If the number of clusters is less than maxTasksPerFile
      // we take the clusters as they are
      if (nFolds == 0) {
         *eventRangesPerFileIt = std::move(*clustersPerFileIt);
         continue;
      }
      // Otherwise, we have to merge clusters, distributing the reminder evenly
      // onto the first clusters
      auto nReminderClusters = clustersInThisFileSize % maxTasksPerFile;
      const auto &clustersInThisFile = *clustersPerFileIt;
      for (auto i = 0ULL; i < clustersInThisFileSize; ++i) {
         const auto start = clustersInThisFile[i].start;
         // We lump together at least nFolds clusters, therefore
         // we need to jump ahead of nFolds-1.
         i += (nFolds - 1);
         // We now add a cluster if we have some reminder left
         if (nReminderClusters > 0) {
            i += 1U;
            nReminderClusters--;
         }
         const auto end = clustersInThisFile[i].end;
         eventRangesPerFileIt->emplace_back(EntryCluster({start, end}));
      }
   }

   return std::make_pair(std::move(eventRangesPerFile), std::move(entriesPerFile));
}

////////////////////////////////////////////////////////////////////////
/// Return a vector containing the number of entries of each file of each friend TChain
static std::vector<std::vector<Long64_t>> GetFriendEntries(const Internal::TreeUtils::RFriendInfo &friendInfo)
{

   const auto &friendNames = friendInfo.fFriendNames;
   const auto &friendFileNames = friendInfo.fFriendFileNames;
   const auto &friendChainSubNames = friendInfo.fFriendChainSubNames;

   std::vector<std::vector<Long64_t>> friendEntries;
   const auto nFriends = friendNames.size();
   for (auto i = 0u; i < nFriends; ++i) {
      std::vector<Long64_t> nEntries;
      const auto &thisFriendName = friendNames[i].first;
      const auto &thisFriendFiles = friendFileNames[i];
      const auto &thisFriendChainSubNames = friendChainSubNames[i];
      // If this friend has chain sub names, it means it's a TChain.
      // In this case, we need to traverse all files that make up the TChain,
      // retrieve the correct sub tree from each file and store the number of
      // entries for that sub tree.
      if (!thisFriendChainSubNames.empty()) {
         // Traverse together filenames and respective treenames
         for (auto fileidx = 0u; fileidx < thisFriendFiles.size(); ++fileidx) {
            std::unique_ptr<TFile> curfile(TFile::Open(thisFriendFiles[fileidx].c_str()));
            TTree *curtree = nullptr; // owned by TFile
            // thisFriendChainSubNames[fileidx] stores the name of the current
            // subtree in the TChain stored in the current file.
            curfile->GetObject(thisFriendChainSubNames[fileidx].c_str(), curtree);
            nEntries.emplace_back(curtree->GetEntries());
         }
         // Otherwise, if there are no sub names for the current friend, it means
         // it's a TTree. We can safely use `thisFriendName` as the name of the tree
         // to retrieve from the file in `thisFriendFiles`
      } else {
         for (const auto &fname : thisFriendFiles) {
            std::unique_ptr<TFile> f(TFile::Open(fname.c_str()));
            TTree *t = nullptr; // owned by TFile
            f->GetObject(thisFriendName.c_str(), t);
            nEntries.emplace_back(t->GetEntries());
         }
      }
      // Store the vector with entries for each file in the current tree/chain.
      friendEntries.emplace_back(std::move(nEntries));
   }

   return friendEntries;
}

} // anonymous namespace

namespace ROOT {

unsigned int TTreeProcessorMT::fgTasksPerWorkerHint = 10U;

namespace Internal {

////////////////////////////////////////////////////////////////////////////////
/// Construct fChain, also adding friends if needed and injecting knowledge of offsets if available.
/// \param[in] treeNames Name of the tree for each file in `fileNames`.
/// \param[in] fileNames Files to be opened.
/// \param[in] friendInfo Information about TTree friends, if any.
/// \param[in] nEntries Number of entries to be processed.
/// \param[in] friendEntries Number of entries in each friend. Expected to have same ordering as friendInfo.
void TTreeView::MakeChain(const std::vector<std::string> &treeNames, const std::vector<std::string> &fileNames,
                          const TreeUtils::RFriendInfo &friendInfo, const std::vector<Long64_t> &nEntries,
                          const std::vector<std::vector<Long64_t>> &friendEntries)
{

   const auto &friendNames = friendInfo.fFriendNames;
   const auto &friendFileNames = friendInfo.fFriendFileNames;
   const auto &friendChainSubNames = friendInfo.fFriendChainSubNames;

   fChain.reset(new TChain());
   const auto nFiles = fileNames.size();
   for (auto i = 0u; i < nFiles; ++i) {
      fChain->Add((fileNames[i] + "/" + treeNames[i]).c_str(), nEntries[i]);
   }
   fChain->ResetBit(TObject::kMustCleanup);

   fFriends.clear();
   const auto nFriends = friendNames.size();
   for (auto i = 0u; i < nFriends; ++i) {
      const auto &thisFriendNameAlias = friendNames[i];
      const auto &thisFriendName = thisFriendNameAlias.first;
      const auto &thisFriendAlias = thisFriendNameAlias.second;
      const auto &thisFriendFiles = friendFileNames[i];
      const auto &thisFriendChainSubNames = friendChainSubNames[i];
      const auto &thisFriendEntries = friendEntries[i];

      // Build a friend chain
      auto frChain = std::make_unique<TChain>(thisFriendName.c_str());
      const auto nFileNames = friendFileNames[i].size();
      // If there are no chain subnames, the friend was a TTree. It's safe
      // to add to the chain the filename directly.
      if (thisFriendChainSubNames.empty()) {
         for (auto j = 0u; j < nFileNames; ++j) {
            frChain->Add(thisFriendFiles[j].c_str(), thisFriendEntries[j]);
         }
         // Otherwise, the new friend chain needs to be built using the nomenclature
         // "filename/treename" as argument to `TChain::Add`
      } else {
         for (auto j = 0u; j < nFileNames; ++j) {
            frChain->Add((thisFriendFiles[j] + "/" + thisFriendChainSubNames[j]).c_str(), thisFriendEntries[j]);
         }
      }

      // Make it friends with the main chain
      fChain->AddFriend(frChain.get(), thisFriendAlias.c_str());
      fFriends.emplace_back(std::move(frChain));
   }
}

//////////////////////////////////////////////////////////////////////////
/// Get a TTreeReader for the current tree of this view.
std::unique_ptr<TTreeReader>
TTreeView::GetTreeReader(Long64_t start, Long64_t end, const std::vector<std::string> &treeNames,
                         const std::vector<std::string> &fileNames, const TreeUtils::RFriendInfo &friendInfo,
                         const TEntryList &entryList, const std::vector<Long64_t> &nEntries,
                         const std::vector<std::vector<Long64_t>> &friendEntries)
{
   const bool hasEntryList = entryList.GetN() > 0;
   const bool usingLocalEntries = friendInfo.fFriendNames.empty() && !hasEntryList;
   const bool needNewChain =
      fChain == nullptr || (usingLocalEntries && (fileNames[0] != fChain->GetListOfFiles()->At(0)->GetTitle() ||
                                                  treeNames[0] != fChain->GetListOfFiles()->At(0)->GetName()));
   if (needNewChain) {
      MakeChain(treeNames, fileNames, friendInfo, nEntries, friendEntries);
      if (hasEntryList) {
         fEntryList.reset(new TEntryList(entryList));
         if (fEntryList->GetLists() != nullptr) {
            // need to associate the TEntryList to the TChain for the latter to set entry the fTreeNumbers of the
            // sub-lists of the former...
            fChain->SetEntryList(fEntryList.get());
            fEntryList->ResetBit(TObject::kCanDelete); // ...but we want to retain ownership
         }
      }
   }
   auto reader = std::make_unique<TTreeReader>(fChain.get(), fEntryList.get());
   reader->SetEntriesRange(start, end);
   return reader;
}

} // namespace Internal
} // namespace ROOT

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Retrieve the names of the TTrees in each of the input files, throw if a TTree cannot be found.
std::vector<std::string> TTreeProcessorMT::FindTreeNames()
{
   std::vector<std::string> treeNames;

   if (fFileNames.empty()) // This can never happen
      throw std::runtime_error("Empty list of files and no tree name provided");

   ::TDirectory::TContext ctxt(gDirectory);
   for (const auto &fname : fFileNames) {
      std::string treeName;
      std::unique_ptr<TFile> f(TFile::Open(fname.c_str()));
      TIter next(f->GetListOfKeys());
      while (auto *key = static_cast<TKey *>(next())) {
         const char *className = key->GetClassName();
         if (strcmp(className, "TTree") == 0) {
            treeName = key->GetName();
            break;
         }
      }
      if (treeName.empty())
         throw std::runtime_error("Cannot find any tree in file " + fname);
      treeNames.emplace_back(std::move(treeName));
   }

   return treeNames;
}

////////////////////////////////////////////////////////////////////////
/// Constructor based on a file name.
/// \param[in] filename Name of the file containing the tree to process.
/// \param[in] treename Name of the tree to process. If not provided, the implementation will search
///            for a TTree key in the file and will use the first one it finds.
/// \param[in] nThreads Number of threads to create in the underlying thread-pool. The semantics of this argument are
///                     the same as for TThreadExecutor.
TTreeProcessorMT::TTreeProcessorMT(std::string_view filename, std::string_view treename, UInt_t nThreads)
   : fFileNames({std::string(filename)}),
     fTreeNames(treename.empty() ? FindTreeNames() : std::vector<std::string>{std::string(treename)}), fFriendInfo(),
     fPool(nThreads)
{
   ROOT::EnableThreadSafety();
}

std::vector<std::string> CheckAndConvert(const std::vector<std::string_view> &views)
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
/// \param[in] treename Name of the tree to process. If not provided, the implementation will
///                     search filenames for a TTree key and will use the first one it finds in each file.
/// \param[in] nThreads Number of threads to create in the underlying thread-pool. The semantics of this argument are
///                     the same as for TThreadExecutor.
///
/// If different files contain TTrees with different names and automatic TTree name detection is not an option
/// (for example, because some of the files contain multiple TTrees) please manually create a TChain and pass
/// it to the appropriate TTreeProcessorMT constructor.
TTreeProcessorMT::TTreeProcessorMT(const std::vector<std::string_view> &filenames, std::string_view treename,
                                   UInt_t nThreads)
   : fFileNames(CheckAndConvert(filenames)),
     fTreeNames(treename.empty() ? FindTreeNames()
                                 : std::vector<std::string>(fFileNames.size(), std::string(treename))),
     fFriendInfo(), fPool(nThreads)
{
   ROOT::EnableThreadSafety();
}

////////////////////////////////////////////////////////////////////////
/// Constructor based on a TTree and a TEntryList.
/// \param[in] tree Tree or chain of files containing the tree to process.
/// \param[in] entries List of entry numbers to process.
/// \param[in] nThreads Number of threads to create in the underlying thread-pool. The semantics of this argument are
///                     the same as for TThreadExecutor.
TTreeProcessorMT::TTreeProcessorMT(TTree &tree, const TEntryList &entries, UInt_t nThreads)
   : fFileNames(Internal::TreeUtils::GetFileNamesFromTree(tree)),
     fTreeNames(Internal::TreeUtils::GetTreeFullPaths(tree)), fEntryList(entries),
     fFriendInfo(Internal::TreeUtils::GetFriendInfo(tree)), fPool(nThreads)
{
   ROOT::EnableThreadSafety();
}

////////////////////////////////////////////////////////////////////////
/// Constructor based on a TTree.
/// \param[in] tree Tree or chain of files containing the tree to process.
/// \param[in] nThreads Number of threads to create in the underlying thread-pool. The semantics of this argument are
///                     the same as for TThreadExecutor.
TTreeProcessorMT::TTreeProcessorMT(TTree &tree, UInt_t nThreads) : TTreeProcessorMT(tree, TEntryList(), nThreads) {}

//////////////////////////////////////////////////////////////////////////////
/// Process the entries of a TTree in parallel. The user-provided function
/// receives a TTreeReader which can be used to iterate on a subrange of
/// entries
/// ~~~{.cpp}
/// TTreeProcessorMT::Process([](TTreeReader& readerSubRange) {
///                            // Select branches to read
///                            while (readerSubRange.Next()) {
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
   // compute number of tasks per file
   const unsigned int maxTasksPerFile =
      std::ceil(float(GetTasksPerWorkerHint() * fPool.GetPoolSize()) / float(fFileNames.size()));

   // If an entry list or friend trees are present, we need to generate clusters with global entry numbers,
   // so we do it here for all files.
   // Otherwise we can do it later, concurrently for each file, and clusters will contain local entry numbers.
   // TODO: in practice we could also find clusters per-file in the case of no friends and a TEntryList with
   // sub-entrylists.
   const bool hasFriends = !fFriendInfo.fFriendNames.empty();
   const bool hasEntryList = fEntryList.GetN() > 0;
   const bool shouldRetrieveAllClusters = hasFriends || hasEntryList;
   ClustersAndEntries clusterAndEntries{};
   if (shouldRetrieveAllClusters) {
      clusterAndEntries = MakeClusters(fTreeNames, fFileNames, maxTasksPerFile);
      if (hasEntryList)
         clusterAndEntries.first = ConvertToElistClusters(std::move(clusterAndEntries.first), fEntryList, fTreeNames,
                                                          fFileNames, clusterAndEntries.second);
   }

   const auto &clusters = clusterAndEntries.first;
   const auto &entries = clusterAndEntries.second;

   // Retrieve number of entries for each file for each friend tree
   const auto friendEntries = hasFriends ? GetFriendEntries(fFriendInfo) : std::vector<std::vector<Long64_t>>{};

   // Parent task, spawns tasks that process each of the entry clusters for each input file
   // TODO: for readability we should have two versions of this lambda, for shouldRetrieveAllClusters == true/false
   auto processFile = [&](std::size_t fileIdx) {
      // theseFiles contains either all files or just the single file to process
      const auto &theseFiles = shouldRetrieveAllClusters ? fFileNames : std::vector<std::string>({fFileNames[fileIdx]});
      // either all tree names or just the single tree to process
      const auto &theseTrees = shouldRetrieveAllClusters ? fTreeNames : std::vector<std::string>({fTreeNames[fileIdx]});
      // Evaluate clusters (with local entry numbers) and number of entries for this file, if needed
      const auto theseClustersAndEntries =
         shouldRetrieveAllClusters ? ClustersAndEntries{} : MakeClusters(theseTrees, theseFiles, maxTasksPerFile);

      // All clusters for the file to process, either with global or local entry numbers
      const auto &thisFileClusters = shouldRetrieveAllClusters ? clusters[fileIdx] : theseClustersAndEntries.first[0];

      // Either all number of entries or just the ones for this file
      const auto &theseEntries =
         shouldRetrieveAllClusters ? entries : std::vector<Long64_t>({theseClustersAndEntries.second[0]});

      auto processCluster = [&](const EntryCluster &c) {
         auto r = fTreeView->GetTreeReader(c.start, c.end, theseTrees, theseFiles, fFriendInfo, fEntryList,
                                           theseEntries, friendEntries);
         func(*r);
      };

      fPool.Foreach(processCluster, thisFileClusters);
   };

   std::vector<std::size_t> fileIdxs(fFileNames.size());
   std::iota(fileIdxs.begin(), fileIdxs.end(), 0u);

   fPool.Foreach(processFile, fileIdxs);
}

////////////////////////////////////////////////////////////////////////
/// \brief Retrieve the current value for the desired number of tasks per worker.
/// \return The desired number of tasks to be created per worker. TTreeProcessorMT uses this value as an hint.
unsigned int TTreeProcessorMT::GetTasksPerWorkerHint()
{
   return fgTasksPerWorkerHint;
}

////////////////////////////////////////////////////////////////////////
/// \brief Set the hint for the desired number of tasks created per worker.
/// \param[in] tasksPerWorkerHint Desired number of tasks per worker.
///
/// This allows to create a reasonable number of tasks even if any of the
/// processed files features a bad clustering, for example with a lot of
/// entries and just a few entries per cluster, or to limit the number of
/// tasks spawned when a very large number of files and workers is used.
void TTreeProcessorMT::SetTasksPerWorkerHint(unsigned int tasksPerWorkerHint)
{
   fgTasksPerWorkerHint = tasksPerWorkerHint;
}
