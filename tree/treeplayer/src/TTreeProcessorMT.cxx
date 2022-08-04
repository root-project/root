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

using EntryRange = std::pair<Long64_t, Long64_t>;

// note that this routine assumes global entry numbers
static bool ClustersAreSortedAndContiguous(const std::vector<std::vector<EntryRange>> &cls)
{
   Long64_t last_end = 0ll;
   for (const auto &fcl : cls) {
      for (const auto &c : fcl) {
         if (last_end != c.first)
            return false;
         last_end = c.second;
      }
   }
   return true;
}

/// Take a vector of vectors of EntryRanges (a vector per file), filter the entries according to entryList, and
/// and return a new vector of vectors of EntryRanges where cluster start/end entry numbers have been converted to
/// TEntryList-local entry numbers.
///
/// This routine assumes that entry numbers in the TEntryList (and, if present, in the sub-entrylists) are in
/// ascending order, i.e., for n > m:
///   elist.GetEntry(n) + tree_offset_for_entry_from_elist(n) > elist.GetEntry(m) + tree_offset_for_entry_from_elist(m)
static std::vector<std::vector<EntryRange>>
ConvertToElistClusters(std::vector<std::vector<EntryRange>> &&clusters, TEntryList &entryList,
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
      chain = ROOT::Internal::TreeUtils::MakeChainForMT();
      for (auto i = 0u; i < nFiles; ++i)
         chain->Add((fileNames[i] + "?#" + treeNames[i]).c_str(), entriesPerFile[i]);
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

   std::vector<std::vector<EntryRange>> elistClusters;

   for (auto fileN = 0u; fileN < nFiles; ++fileN) {
      std::vector<EntryRange> elistClustersForFile;
      for (const auto &c : clusters[fileN]) {
         if (entry >= c.second || entry == -1ll) // no entrylist entries in this cluster
            continue;
         R__ASSERT(entry >= c.first); // current entry should never come before the cluster we are looking at
         const Long64_t elistRangeStart = elistEntry;
         // advance entry list until the entrylist entry goes beyond the end of the cluster
         while (entry < c.second && entry != -1ll)
            entry = Next(elistEntry, entryList, chain.get());
         elistClustersForFile.emplace_back(EntryRange{elistRangeStart, elistEntry});
      }
      elistClusters.emplace_back(std::move(elistClustersForFile));
   }

   R__ASSERT(elistClusters.size() == clusters.size()); // same number of files
   R__ASSERT(ClustersAreSortedAndContiguous(elistClusters));

   entryList.GetEntry(0ll); // reset TEntryList internal state, lest we incur in ROOT-10807
   return elistClusters;
}

// EntryRanges and number of entries per file
using ClustersAndEntries = std::pair<std::vector<std::vector<EntryRange>>, std::vector<Long64_t>>;

////////////////////////////////////////////////////////////////////////
/// Return a vector of cluster boundaries for the given tree and files.
static ClustersAndEntries MakeClusters(const std::vector<std::string> &treeNames,
                                       const std::vector<std::string> &fileNames, const unsigned int maxTasksPerFile,
                                       const EntryRange &range = {0, std::numeric_limits<Long64_t>::max()})
{
   // Note that as a side-effect of opening all files that are going to be used in the
   // analysis once, all necessary streamers will be loaded into memory.
   TDirectory::TContext c;
   const auto nFileNames = fileNames.size();
   std::vector<std::vector<EntryRange>> clustersPerFile;
   std::vector<Long64_t> entriesPerFile;
   entriesPerFile.reserve(nFileNames);
   Long64_t offset = 0ll;
   bool rangeEndReached = false; // flag to break the outer loop
   for (auto i = 0u; i < nFileNames && !rangeEndReached; ++i) {
      const auto &fileName = fileNames[i];
      const auto &treeName = treeNames[i];

      std::unique_ptr<TFile> f(TFile::Open(
         fileName.c_str(), "READ_WITHOUT_GLOBALREGISTRATION")); // need TFile::Open to load plugins if need be
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

      // Avoid calling TROOT::RecursiveRemove for this tree, it takes the read lock and we don't need it.
      t->ResetBit(kMustCleanup);
      ROOT::Internal::TreeUtils::ClearMustCleanupBits(*t->GetListOfBranches());
      auto clusterIter = t->GetClusterIterator(0);
      Long64_t clusterStart = 0ll, clusterEnd = 0ll;
      const Long64_t entries = t->GetEntries();
      // Iterate over the clusters in the current file
      std::vector<EntryRange> entryRanges;
      while ((clusterStart = clusterIter()) < entries && !rangeEndReached) {
         clusterEnd = clusterIter.GetNextEntry();
         // Currently, if a user specified a range, the clusters will be only globally obtained
         // Assume that there are 3 files with entries: [0, 100], [0, 150], [0, 200] (in this order)
         // Since the cluster boundaries are obtained sequentially, applying the offsets, the boundaries
         // would be: 0, 100, 250, 450. Now assume that the user provided the range (150, 300)
         // Then, in the first iteration, nothing is going to be added to entryRanges since:
         // std::max(0, 150) < std::min(100, max). Then, by the same logic only a subset of the second
         // tree is added, i.e.: currentStart is now 200 and currentEnd is 250 (locally from 100 to 150).
         // Lastly, the last tree would take entries from 250 to 300 (or from 0 to 50 locally).
         // The current file's offset to start and end is added to make them (chain) global
         const auto currentStart = std::max(clusterStart + offset, range.first);
         const auto currentEnd = std::min(clusterEnd + offset, range.second);
         // This is not satified if the desired start is larger than the last entry of some cluster
         // In this case, this cluster is not going to be processes further
         if (currentStart < currentEnd)
            entryRanges.emplace_back(EntryRange{currentStart, currentEnd});
         if (currentEnd == range.second) // if the desired end is reached, stop reading further
            rangeEndReached = true;
      }
      offset += entries; // consistently keep track of the total number of entries
      clustersPerFile.emplace_back(std::move(entryRanges));
      // Keep track of the entries, even if their corresponding tree is out of the range, e.g. entryRanges is empty
      entriesPerFile.emplace_back(entries);
   }
   if (range.first >= offset && offset > 0) // do not error out on an empty tree
      throw std::logic_error(std::string("A range of entries was passed in the creation of the TTreeProcessorMT, ") +
                             "but the starting entry (" + range.first + ") is larger than the total number of " +
                             "entries (" + offset + ") in the dataset.");

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

   std::vector<std::vector<EntryRange>> eventRangesPerFile(clustersPerFile.size());
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
         const auto start = clustersInThisFile[i].first;
         // We lump together at least nFolds clusters, therefore
         // we need to jump ahead of nFolds-1.
         i += (nFolds - 1);
         // We now add a cluster if we have some reminder left
         if (nReminderClusters > 0) {
            i += 1U;
            nReminderClusters--;
         }
         const auto end = clustersInThisFile[i].second;
         eventRangesPerFileIt->emplace_back(EntryRange({start, end}));
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
            std::unique_ptr<TFile> curfile(
               TFile::Open(thisFriendFiles[fileidx].c_str(), "READ_WITHOUT_GLOBALREGISTRATION"));
            if (!curfile || curfile->IsZombie())
               throw std::runtime_error("TTreeProcessorMT::GetFriendEntries: Could not open file \"" +
                                        thisFriendFiles[fileidx] + "\"");
            // thisFriendChainSubNames[fileidx] stores the name of the current
            // subtree in the TChain stored in the current file.
            TTree *curtree = curfile->Get<TTree>(thisFriendChainSubNames[fileidx].c_str());
            if (!curtree)
               throw std::runtime_error("TTreeProcessorMT::GetFriendEntries: Could not retrieve TTree \"" +
                                        thisFriendChainSubNames[fileidx] + "\" from file \"" +
                                        thisFriendFiles[fileidx] + "\"");
            nEntries.emplace_back(curtree->GetEntries());
         }
         // Otherwise, if there are no sub names for the current friend, it means
         // it's a TTree. We can safely use `thisFriendName` as the name of the tree
         // to retrieve from the file in `thisFriendFiles`
      } else {
         for (const auto &fname : thisFriendFiles) {
            std::unique_ptr<TFile> f(TFile::Open(fname.c_str(), "READ_WITHOUT_GLOBALREGISTRATION"));
            if (!f || f->IsZombie())
               throw std::runtime_error("TTreeProcessorMT::GetFriendEntries: Could not open file \"" + fname + "\"");
            TTree *t = f->Get<TTree>(thisFriendName.c_str());
            if (!t)
               throw std::runtime_error("TTreeProcessorMT::GetFriendEntries: Could not retrieve TTree \"" +
                                        thisFriendName + "\" from file \"" + fname + "\"");
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

   fChain = ROOT::Internal::TreeUtils::MakeChainForMT();
   // Because of the range, we might have stopped reading entries earlier,
   // hence the size of nEntries can be smaller than the number of all files
   // TODO: pass "firstFileToProcess" index in case of a range,
   // and do not add files to the chain, which are before the desired start entry of the range
   const auto nFilesToProcess = nEntries.size();
   for (auto i = 0u; i < nFilesToProcess; ++i) {
      fChain->Add((fileNames[i] + "?#" + treeNames[i]).c_str(), nEntries[i]);
   }
   fNoCleanupNotifier.RegisterChain(*fChain.get());

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
      auto frChain = ROOT::Internal::TreeUtils::MakeChainForMT(thisFriendName);
      const auto nFileNames = friendFileNames[i].size();
      if (thisFriendChainSubNames.empty()) {
         // If there are no chain subnames, the friend was a TTree. It's safe
         // to add to the chain the filename directly.
         for (auto j = 0u; j < nFileNames; ++j) {
            frChain->Add(thisFriendFiles[j].c_str(), thisFriendEntries[j]);
         }
      } else {
         // Otherwise, the new friend chain needs to be built using the nomenclature
         // "filename/treename" as argument to `TChain::Add`
         for (auto j = 0u; j < nFileNames; ++j) {
            frChain->Add((thisFriendFiles[j] + "?#" + thisFriendChainSubNames[j]).c_str(), thisFriendEntries[j]);
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

////////////////////////////////////////////////////////////////////////
/// Clear the resources
void TTreeView::Reset()
{
   fChain.reset();
   fEntryList.reset();
   fFriends.clear();
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
/// \param[in] /// \param[in] globalRange Global entry range to process, {begin (inclusive), end (exclusive)}.
TTreeProcessorMT::TTreeProcessorMT(std::string_view filename, std::string_view treename, UInt_t nThreads,
                                   const EntryRange &globalRange)
   : fFileNames({std::string(filename)}),
     fTreeNames(treename.empty() ? FindTreeNames() : std::vector<std::string>{std::string(treename)}), fFriendInfo(),
     fPool(nThreads), fGlobalRange(globalRange)
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
/// \param[in] globalRange Global entry range to process, {begin (inclusive), end (exclusive)}.
///
/// If different files contain TTrees with different names and automatic TTree name detection is not an option
/// (for example, because some of the files contain multiple TTrees) please manually create a TChain and pass
/// it to the appropriate TTreeProcessorMT constructor.
TTreeProcessorMT::TTreeProcessorMT(const std::vector<std::string_view> &filenames, std::string_view treename,
                                   UInt_t nThreads, const EntryRange &globalRange)
   : fFileNames(CheckAndConvert(filenames)),
     fTreeNames(treename.empty() ? FindTreeNames()
                                 : std::vector<std::string>(fFileNames.size(), std::string(treename))),
     fFriendInfo(), fPool(nThreads), fGlobalRange(globalRange)
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
/// \param[in] globalRange Global entry range to process, {begin (inclusive), end (exclusive)}.
TTreeProcessorMT::TTreeProcessorMT(TTree &tree, UInt_t nThreads, const EntryRange &globalRange)
   : fFileNames(Internal::TreeUtils::GetFileNamesFromTree(tree)),
     fTreeNames(Internal::TreeUtils::GetTreeFullPaths(tree)), fFriendInfo(Internal::TreeUtils::GetFriendInfo(tree)),
     fPool(nThreads), fGlobalRange(globalRange)
{
}

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
   const bool shouldRetrieveAllClusters = hasFriends || hasEntryList || fGlobalRange.first > 0 ||
                                          fGlobalRange.second != std::numeric_limits<Long64_t>::max();
   ClustersAndEntries allClusterAndEntries{};
   auto &allClusters = allClusterAndEntries.first;
   const auto &allEntries = allClusterAndEntries.second;
   if (shouldRetrieveAllClusters) {
      allClusterAndEntries = MakeClusters(fTreeNames, fFileNames, maxTasksPerFile, fGlobalRange);
      if (hasEntryList)
         allClusters = ConvertToElistClusters(std::move(allClusters), fEntryList, fTreeNames, fFileNames, allEntries);
   }

   // Per-file processing in case we retrieved all cluster info upfront
   auto processFileUsingGlobalClusters = [&](std::size_t fileIdx) {
      auto processCluster = [&](const EntryRange &c) {
         auto r = fTreeView->GetTreeReader(c.first, c.second, fTreeNames, fFileNames, fFriendInfo, fEntryList,
                                           allEntries, GetFriendEntries(fFriendInfo));
         func(*r);
      };
      fPool.Foreach(processCluster, allClusters[fileIdx]);
   };

   // Per-file processing that also retrieves cluster info for a file
   auto processFileRetrievingClusters = [&](std::size_t fileIdx) {
      // Evaluate clusters (with local entry numbers) and number of entries for this file
      const auto &treeNames = std::vector<std::string>({fTreeNames[fileIdx]});
      const auto &fileNames = std::vector<std::string>({fFileNames[fileIdx]});
      const auto clustersAndEntries = MakeClusters(treeNames, fileNames, maxTasksPerFile);
      const auto &clusters = clustersAndEntries.first[0];
      const auto &entries = clustersAndEntries.second[0];
      auto processCluster = [&](const EntryRange &c) {
         auto r = fTreeView->GetTreeReader(c.first, c.second, treeNames, fileNames, fFriendInfo, fEntryList, {entries},
                                           std::vector<std::vector<Long64_t>>{});
         func(*r);
      };
      fPool.Foreach(processCluster, clusters);
   };

   const auto firstNonEmpty =
      fGlobalRange.first > 0u ? std::distance(allClusters.begin(), std::find_if(allClusters.begin(), allClusters.end(),
                                                                                [](auto &c) { return !c.empty(); }))
                              : 0u;

   std::vector<std::size_t> fileIdxs(allEntries.empty() ? fFileNames.size() : allEntries.size() - firstNonEmpty);
   std::iota(fileIdxs.begin(), fileIdxs.end(), firstNonEmpty);

   if (shouldRetrieveAllClusters)
      fPool.Foreach(processFileUsingGlobalClusters, fileIdxs);
   else
      fPool.Foreach(processFileRetrievingClusters, fileIdxs);

   // make sure TChains and TFiles are cleaned up since they are not globally tracked
   for (unsigned int islot = 0; islot < fTreeView.GetNSlots(); ++islot) {
      ROOT::Internal::TTreeView *view = fTreeView.GetAtSlotRaw(islot);
      if (view != nullptr) {
         view->Reset();
      }
   }
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
