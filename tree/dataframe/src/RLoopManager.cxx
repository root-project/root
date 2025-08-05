/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RConfigure.h" // R__USE_IMT
#include "ROOT/RDataSource.hxx"
#include "ROOT/RDF/GraphNode.hxx"
#include "ROOT/InternalTreeUtils.hxx" // GetTreeFullPaths
#include "ROOT/RDF/RActionBase.hxx"
#include "ROOT/RDF/RDefineBase.hxx"
#include "ROOT/RDF/RDefineReader.hxx" // RDefinesWithReaders
#include "ROOT/RDF/RFilterBase.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RNTupleProcessor.hxx"
#include "ROOT/RDF/RRangeBase.hxx"
#include "ROOT/RDF/RVariationBase.hxx"
#include "ROOT/RDF/RVariationReader.hxx" // RVariationsWithReaders
#include "ROOT/RLogger.hxx"
#include "ROOT/RNTuple.hxx"
#include "ROOT/RNTupleDS.hxx"
#include "RtypesCore.h" // Long64_t
#include "TStopwatch.h"
#include "TBranchElement.h"
#include "TBranchObject.h"
#include "TChain.h"
#include "TEntryList.h"
#include "TFile.h"
#include "TFriendElement.h"
#include "TROOT.h" // IsImplicitMTEnabled, gCoreMutex, R__*_LOCKGUARD
#include "TTreeReader.h"
#include "TTree.h" // For MaxTreeSizeRAII. Revert when #6640 will be solved.

#include "ROOT/RTTreeDS.hxx"

#ifdef R__USE_IMT
#include "ROOT/TThreadExecutor.hxx"
#include "ROOT/TTreeProcessorMT.hxx"
#include "ROOT/RSlotStack.hxx"
#endif

#ifdef R__UNIX
// Functions needed to perform EOS XRootD redirection in ChangeSpec
#include "TEnv.h"
#include "TSystem.h"
#ifndef R__FBSD
#include <sys/xattr.h>
#else
#include <sys/extattr.h>
#endif
#ifdef R__MACOSX
/* On macOS getxattr takes two extra arguments that should be set to 0 */
#define getxattr(path, name, value, size) getxattr(path, name, value, size, 0u, 0)
#endif
#ifdef R__FBSD
#define getxattr(path, name, value, size) extattr_get_file(path, EXTATTR_NAMESPACE_USER, name, value, size)
#endif
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>
#include <set>
#include <limits> // For MaxTreeSizeRAII. Revert when #6640 will be solved.

using namespace ROOT::Detail::RDF;
using namespace ROOT::Internal::RDF;

namespace {
/// A helper function that returns all RDF code that is currently scheduled for just-in-time compilation.
/// This allows different RLoopManager instances to share these data.
/// We want RLoopManagers to be able to add their code to a global "code to execute via cling",
/// so that, lazily, we can jit everything that's needed by all RDFs in one go, which is potentially
/// much faster than jitting each RLoopManager's code separately.
std::string &GetCodeToJit()
{
   static std::string code;
   return code;
}

void ThrowIfNSlotsChanged(unsigned int nSlots)
{
   const auto currentSlots = RDFInternal::GetNSlots();
   if (currentSlots != nSlots) {
      std::string msg = "RLoopManager::Run: when the RDataFrame was constructed the number of slots required was " +
                        std::to_string(nSlots) + ", but when starting the event loop it was " +
                        std::to_string(currentSlots) + ".";
      if (currentSlots > nSlots)
         msg += " Maybe EnableImplicitMT() was called after the RDataFrame was constructed?";
      else
         msg += " Maybe DisableImplicitMT() was called after the RDataFrame was constructed?";
      throw std::runtime_error(msg);
   }
}

/**
\struct MaxTreeSizeRAII
\brief Scope-bound change of `TTree::fgMaxTreeSize`.

This RAII object stores the current value result of `TTree::GetMaxTreeSize`,
changes it to maximum at construction time and restores it back at destruction
time. Needed for issue #6523 and should be reverted when #6640 will be solved.
*/
struct MaxTreeSizeRAII {
   Long64_t fOldMaxTreeSize;

   MaxTreeSizeRAII() : fOldMaxTreeSize(TTree::GetMaxTreeSize())
   {
      TTree::SetMaxTreeSize(std::numeric_limits<Long64_t>::max());
   }

   ~MaxTreeSizeRAII() { TTree::SetMaxTreeSize(fOldMaxTreeSize); }
};

struct DatasetLogInfo {
   std::string fDataSet;
   ULong64_t fRangeStart;
   ULong64_t fRangeEnd;
   unsigned int fSlot;
};

std::string LogRangeProcessing(const DatasetLogInfo &info)
{
   std::stringstream msg;
   msg << "Processing " << info.fDataSet << ": entry range [" << info.fRangeStart << "," << info.fRangeEnd - 1
       << "], using slot " << info.fSlot << " in thread " << std::this_thread::get_id() << '.';
   return msg.str();
}

auto MakeDatasetColReadersKey(std::string_view colName, const std::type_info &ti)
{
   // We use a combination of column name and column type name as the key because in some cases we might end up
   // with concrete readers that use different types for the same column, e.g. std::vector and RVec here:
   //    df.Sum<vector<int>>("stdVectorBranch");
   //    df.Sum<RVecI>("stdVectorBranch");
   return std::string(colName) + ':' + ti.name();
}
} // anonymous namespace

/**
 * \brief Helper function to open a file (or the first file from a glob).
 * This function is used at construction time of an RDataFrame, to check the
 * concrete type of the dataset stored inside the file.
 */
std::unique_ptr<TFile> OpenFileWithSanityChecks(std::string_view fileNameGlob)
{
   // Follow same logic in TChain::Add to find the correct string to look for globbing:
   // - If the extension ".root" is present in the file name, pass along the basename.
   // - If not, use the "?" token to delimit the part of the string which represents the basename.
   // - Otherwise, pass the full filename.
   auto &&baseNameAndQuery = [&fileNameGlob]() {
      constexpr std::string_view delim{".root"};
      if (auto &&it = std::find_end(fileNameGlob.begin(), fileNameGlob.end(), delim.begin(), delim.end());
          it != fileNameGlob.end()) {
         auto &&distanceToEndOfDelim = std::distance(fileNameGlob.begin(), it + delim.length());
         return std::make_pair(fileNameGlob.substr(0, distanceToEndOfDelim), fileNameGlob.substr(distanceToEndOfDelim));
      } else if (auto &&lastQuestionMark = fileNameGlob.find_last_of('?'); lastQuestionMark != std::string_view::npos)
         return std::make_pair(fileNameGlob.substr(0, lastQuestionMark), fileNameGlob.substr(lastQuestionMark));
      else
         return std::make_pair(fileNameGlob, std::string_view{});
   }();
   // Captured structured bindings variable are only valid since C++20
   auto &&baseName = baseNameAndQuery.first;
   auto &&query = baseNameAndQuery.second;

   std::string fileToOpen{fileNameGlob};
   if (baseName.find_first_of("[]*?") != std::string_view::npos) { // Wildcards accepted by TChain::Add
      const auto expanded = ROOT::Internal::TreeUtils::ExpandGlob(std::string{baseName});
      if (expanded.empty())
         throw std::invalid_argument{"RDataFrame: The glob expression '" + std::string{baseName} +
                                     "' did not match any files."};

      fileToOpen = expanded.front() + std::string{query};
   }

   ::TDirectory::TContext ctxt; // Avoid changing gDirectory;
   std::unique_ptr<TFile> inFile{TFile::Open(fileToOpen.c_str(), "READ_WITHOUT_GLOBALREGISTRATION")};
   if (!inFile || inFile->IsZombie())
      throw std::invalid_argument("RDataFrame: could not open file \"" + fileToOpen + "\".");

   return inFile;
}

namespace ROOT {
namespace Detail {
namespace RDF {

/// A RAII object that calls RLoopManager::CleanUpTask at destruction
struct RCallCleanUpTask {
   RLoopManager &fLoopManager;
   unsigned int fArg;
   TTreeReader *fReader;

   RCallCleanUpTask(RLoopManager &lm, unsigned int arg = 0u, TTreeReader *reader = nullptr)
      : fLoopManager(lm), fArg(arg), fReader(reader)
   {
   }
   ~RCallCleanUpTask() { fLoopManager.CleanUpTask(fReader, fArg); }
};

} // namespace RDF
} // namespace Detail
} // namespace ROOT

ROOT::Detail::RDF::RLoopManager::RLoopManager(const ROOT::Detail::RDF::ColumnNames_t &defaultColumns)
   : fDefaultColumns(defaultColumns),
     fNSlots(RDFInternal::GetNSlots()),
     fNewSampleNotifier(fNSlots),
     fSampleInfos(fNSlots),
     fDatasetColumnReaders(fNSlots)
{
}

RLoopManager::RLoopManager(TTree *tree, const ColumnNames_t &defaultBranches)
   : fDefaultColumns(defaultBranches),
     fNSlots(RDFInternal::GetNSlots()),
     fLoopType(ROOT::IsImplicitMTEnabled() ? ELoopType::kDataSourceMT : ELoopType::kDataSource),
     fDataSource(std::make_unique<ROOT::Internal::RDF::RTTreeDS>(ROOT::Internal::RDF::MakeAliasedSharedPtr(tree))),
     fNewSampleNotifier(fNSlots),
     fSampleInfos(fNSlots),
     fDatasetColumnReaders(fNSlots)
{
   fDataSource->SetNSlots(fNSlots);
}

RLoopManager::RLoopManager(ULong64_t nEmptyEntries)
   : fEmptyEntryRange(0, nEmptyEntries),
     fNSlots(RDFInternal::GetNSlots()),
     fLoopType(ROOT::IsImplicitMTEnabled() ? ELoopType::kNoFilesMT : ELoopType::kNoFiles),
     fNewSampleNotifier(fNSlots),
     fSampleInfos(fNSlots),
     fDatasetColumnReaders(fNSlots)
{
}

RLoopManager::RLoopManager(std::unique_ptr<RDataSource> ds, const ColumnNames_t &defaultBranches)
   : fDefaultColumns(defaultBranches),
     fNSlots(RDFInternal::GetNSlots()),
     fLoopType(ROOT::IsImplicitMTEnabled() ? ELoopType::kDataSourceMT : ELoopType::kDataSource),
     fDataSource(std::move(ds)),
     fNewSampleNotifier(fNSlots),
     fSampleInfos(fNSlots),
     fDatasetColumnReaders(fNSlots)
{
   fDataSource->SetNSlots(fNSlots);
}

RLoopManager::RLoopManager(ROOT::RDF::Experimental::RDatasetSpec &&spec)
   : fNSlots(RDFInternal::GetNSlots()),
     fLoopType(ROOT::IsImplicitMTEnabled() ? ELoopType::kDataSourceMT : ELoopType::kDataSource),
     fNewSampleNotifier(fNSlots),
     fSampleInfos(fNSlots),
     fDatasetColumnReaders(fNSlots)
{
   ChangeSpec(std::move(spec));
}

#ifdef R__UNIX
namespace {
std::optional<std::string> GetRedirectedSampleId(std::string_view path, std::string_view datasetName)
{
   // Mimick the redirection done in TFile::Open to see if the path points to a FUSE-mounted EOS path.
   // If so, we create a redirected sample ID with the full xroot URL.
   TString expandedUrl(path.data());
   gSystem->ExpandPathName(expandedUrl);
   if (gEnv->GetValue("TFile.CrossProtocolRedirects", 1) == 1) {
      TUrl fileurl(expandedUrl, /* default is file */ kTRUE);
      if (strcmp(fileurl.GetProtocol(), "file") == 0) {
         ssize_t len = getxattr(fileurl.GetFile(), "eos.url.xroot", nullptr, 0);
         if (len > 0) {
            std::string xurl(len, 0);
            std::string fileNameFromUrl{fileurl.GetFile()};
            if (getxattr(fileNameFromUrl.c_str(), "eos.url.xroot", &xurl[0], len) == len) {
               // Sometimes the `getxattr` call may return an invalid URL due
               // to the POSIX attribute not being yet completely filled by EOS.
               if (auto baseName = fileNameFromUrl.substr(fileNameFromUrl.find_last_of("/") + 1);
                   std::equal(baseName.crbegin(), baseName.crend(), xurl.crbegin())) {
                  return xurl + '/' + datasetName.data();
               }
            }
         }
      }
   }

   return std::nullopt;
}
} // namespace
#endif

/**
 * @brief Changes the internal TTree held by the RLoopManager.
 *
 * @warning This method may lead to potentially dangerous interactions if used
 *     after the construction of the RDataFrame. Changing the specification makes
 *     sense *if and only if* the schema of the dataset is *unchanged*, i.e. the
 *     new specification refers to exactly the same number of columns, with the
 *     same names and types. The actual use case of this method is moving the
 *     processing of the same RDataFrame to a different range of entries of the
 *     same dataset (which may be stored in a different set of files).
 *
 * @param spec The specification of the dataset to be adopted.
 */
void RLoopManager::ChangeSpec(ROOT::RDF::Experimental::RDatasetSpec &&spec)
{
   auto filesVec = spec.GetFileNameGlobs();
   auto inFile = OpenFileWithSanityChecks(
      filesVec[0]); // we only need the first file, we assume all files are either TTree or RNTuple
   auto datasetName = spec.GetTreeNames();

   // Change the range of entries to be processed
   fBeginEntry = spec.GetEntryRangeBegin();
   fEndEntry = spec.GetEntryRangeEnd();

   // Store the samples
   fSamples = spec.MoveOutSamples();
   fSampleMap.clear();

   const bool isTTree = inFile->Get<TTree>(datasetName[0].data());
   const bool isRNTuple = inFile->Get<ROOT::RNTuple>(datasetName[0].data());

   if (isTTree || isRNTuple) {

      if (isTTree) {
         // Create the internal main chain
         auto chain = ROOT::Internal::TreeUtils::MakeChainForMT();
         for (auto &sample : fSamples) {
            const auto &trees = sample.GetTreeNames();
            const auto &files = sample.GetFileNameGlobs();
            for (std::size_t i = 0ul; i < files.size(); ++i) {
               // We need to use `<filename>?#<treename>` as an argument to TChain::Add
               // (see https://github.com/root-project/root/pull/8820 for why)
               const auto fullpath = files[i] + "?#" + trees[i];
               chain->Add(fullpath.c_str());
               // ...but instead we use `<filename>/<treename>` as a sample ID (cannot
               // change this easily because of backward compatibility: the sample ID
               // is exposed to users via RSampleInfo and DefinePerSample).
               const auto sampleId = files[i] + '/' + trees[i];
               fSampleMap.insert({sampleId, &sample});
#ifdef R__UNIX
               // Also add redirected EOS xroot URL when available
               if (auto redirectedSampleId = GetRedirectedSampleId(files[i], trees[i]))
                  fSampleMap.insert({redirectedSampleId.value(), &sample});
#endif
            }
         }
         fDataSource = std::make_unique<ROOT::Internal::RDF::RTTreeDS>(std::move(chain), spec.GetFriendInfo());
      } else if (isRNTuple) {

         std::vector<std::string> fileNames;
         std::set<std::string> rntupleNames;

         for (auto &sample : fSamples) {
            const auto &trees = sample.GetTreeNames();
            const auto &files = sample.GetFileNameGlobs();
            for (std::size_t i = 0ul; i < files.size(); ++i) {
               const auto sampleId = files[i] + '/' + trees[i];
               fSampleMap.insert({sampleId, &sample});
               fileNames.push_back(files[i]);
               rntupleNames.insert(trees[i]);

#ifdef R__UNIX
               // Also add redirected EOS xroot URL when available
               if (auto redirectedSampleId = GetRedirectedSampleId(files[i], trees[i]))
                  fSampleMap.insert({redirectedSampleId.value(), &sample});
#endif
            }
         }

         if (rntupleNames.size() == 1) {
            fDataSource = std::make_unique<ROOT::RDF::RNTupleDS>(*rntupleNames.begin(), fileNames);

         } else {
            throw std::runtime_error(
               "More than one RNTuple name was found, please make sure to use RNTuples with the same name.");
         }
      }

      fDataSource->SetNSlots(fNSlots);

      for (unsigned int slot{}; slot < fNSlots; slot++) {
         for (auto &v : fDatasetColumnReaders[slot])
            v.second.reset();
      }
   } else {
      throw std::invalid_argument(
         "RDataFrame: unsupported data format for dataset. Make sure you use TTree or RNTuple.");
   }
}

/// Run event loop with no source files, in parallel.
void RLoopManager::RunEmptySourceMT()
{
#ifdef R__USE_IMT
   std::shared_ptr<ROOT::Internal::RSlotStack> slotStack = SlotStack();
   // Working with an empty tree.
   // Evenly partition the entries according to fNSlots. Produce around 2 tasks per slot.
   const auto nEmptyEntries = GetNEmptyEntries();
   const auto nEntriesPerSlot = nEmptyEntries / (fNSlots * 2);
   auto remainder = nEmptyEntries % (fNSlots * 2);
   std::vector<std::pair<ULong64_t, ULong64_t>> entryRanges;
   ULong64_t begin = fEmptyEntryRange.first;
   while (begin < fEmptyEntryRange.second) {
      ULong64_t end = begin + nEntriesPerSlot;
      if (remainder > 0) {
         ++end;
         --remainder;
      }
      entryRanges.emplace_back(begin, end);
      begin = end;
   }

   // Each task will generate a subrange of entries
   auto genFunction = [this, &slotStack](const std::pair<ULong64_t, ULong64_t> &range) {
      ROOT::Internal::RSlotStackRAII slotRAII(*slotStack);
      auto slot = slotRAII.fSlot;
      RCallCleanUpTask cleanup(*this, slot);
      InitNodeSlots(nullptr, slot);
      R__LOG_DEBUG(0, RDFLogChannel()) << LogRangeProcessing({"an empty source", range.first, range.second, slot});
      try {
         UpdateSampleInfo(slot, range);
         for (auto currEntry = range.first; currEntry < range.second; ++currEntry) {
            RunAndCheckFilters(slot, currEntry);
         }
      } catch (...) {
         // Error might throw in experiment frameworks like CMSSW
         std::cerr << "RDataFrame::Run: event loop was interrupted\n";
         throw;
      }
   };

   ROOT::TThreadExecutor pool;
   pool.Foreach(genFunction, entryRanges);

#endif // not implemented otherwise
}

/// Run event loop with no source files, in sequence.
void RLoopManager::RunEmptySource()
{
   InitNodeSlots(nullptr, 0);
   R__LOG_DEBUG(0, RDFLogChannel()) << LogRangeProcessing(
      {"an empty source", fEmptyEntryRange.first, fEmptyEntryRange.second, 0u});
   RCallCleanUpTask cleanup(*this);
   try {
      UpdateSampleInfo(/*slot*/ 0, fEmptyEntryRange);
      for (ULong64_t currEntry = fEmptyEntryRange.first;
           currEntry < fEmptyEntryRange.second && fNStopsReceived < fNChildren; ++currEntry) {
         RunAndCheckFilters(0, currEntry);
      }
   } catch (...) {
      std::cerr << "RDataFrame::Run: event loop was interrupted\n";
      throw;
   }
}

#ifdef R__USE_IMT
namespace {
/// Return true on succesful entry read.
///
/// TTreeReader encodes successful reads in the `kEntryValid` enum value, but
/// there can be other situations where the read is still valid. For now, these
/// are:
/// - If there was no match of the current entry in one or more friend trees
///   according to their respective indexes.
/// - If there was a missing branch at the start of a new tree in the dataset.
///
/// In such situations, although the entry is not complete, the processing
/// should not be aborted and nodes of the computation graph will take action
/// accordingly.
bool validTTreeReaderRead(TTreeReader &treeReader)
{
   treeReader.Next();
   switch (treeReader.GetEntryStatus()) {
   case TTreeReader::kEntryValid: return true;
   case TTreeReader::kIndexedFriendNoMatch: return true;
   case TTreeReader::kMissingBranchWhenSwitchingTree: return true;
   default: return false;
   }
}
} // namespace
#endif

namespace {
struct DSRunRAII {
   ROOT::RDF::RDataSource &fDS;
   DSRunRAII(ROOT::RDF::RDataSource &ds, const std::set<std::string> &suppressErrorsForMissingColumns) : fDS(ds)
   {
      ROOT::Internal::RDF::CallInitializeWithOpts(fDS, suppressErrorsForMissingColumns);
   }
   ~DSRunRAII() { fDS.Finalize(); }
};
} // namespace

struct ROOT::Internal::RDF::RDSRangeRAII {
   ROOT::Detail::RDF::RLoopManager &fLM;
   unsigned int fSlot;
   TTreeReader *fTreeReader;
   RDSRangeRAII(ROOT::Detail::RDF::RLoopManager &lm, unsigned int slot, ULong64_t firstEntry,
                TTreeReader *treeReader = nullptr)
      : fLM(lm), fSlot(slot), fTreeReader(treeReader)
   {
      fLM.InitNodeSlots(fTreeReader, fSlot);
      fLM.GetDataSource()->InitSlot(fSlot, firstEntry);
   }
   ~RDSRangeRAII() { fLM.GetDataSource()->FinalizeSlot(fSlot); }
};

/// Run event loop over data accessed through a DataSource, in sequence.
void RLoopManager::RunDataSource()
{
   assert(fDataSource != nullptr);
   // Shortcut if the entry range would result in not reading anything
   if (fBeginEntry == fEndEntry)
      return;
   // Apply global entry range if necessary
   if (fBeginEntry != 0 || fEndEntry != std::numeric_limits<Long64_t>::max())
      fDataSource->SetGlobalEntryRange(std::make_pair<std::uint64_t, std::uint64_t>(fBeginEntry, fEndEntry));
   // Initialize data source and book finalization
   DSRunRAII _{*fDataSource, fSuppressErrorsForMissingBranches};
   // Ensure cleanup task is always called at the end. Notably, this also resets the column readers for those data
   // sources that need it (currently only TTree).
   RCallCleanUpTask cleanup(*this);

   // Main event loop. We start with an empty vector of ranges because we need to initialize the nodes and the data
   // source before the first call to GetEntryRanges, since it could trigger reading (currently only happens with
   // TTree).
   std::uint64_t processedEntries{};
   std::vector<std::pair<ULong64_t, ULong64_t>> ranges{};
   do {

      ROOT::Internal::RDF::RDSRangeRAII __{*this, 0u, 0ull};

      ranges = fDataSource->GetEntryRanges();

      fSampleInfos[0] = ROOT::Internal::RDF::CreateSampleInfo(*fDataSource, /*slot*/ 0, fSampleMap);

      try {
         for (const auto &range : ranges) {
            const auto start = range.first;
            const auto end = range.second;
            R__LOG_DEBUG(0, RDFLogChannel()) << LogRangeProcessing({fDataSource->GetLabel(), start, end, 0u});
            for (auto entry = start; entry < end && fNStopsReceived < fNChildren; ++entry) {
               if (fDataSource->SetEntry(0u, entry)) {
                  RunAndCheckFilters(0u, entry);
               }
               processedEntries++;
            }
         }
      } catch (...) {
         std::cerr << "RDataFrame::Run: event loop was interrupted\n";
         throw;
      }

   } while (!ranges.empty() && fNStopsReceived < fNChildren);

   ROOT::Internal::RDF::RunFinalChecks(*fDataSource, (fNStopsReceived < fNChildren));

   if (fEndEntry != std::numeric_limits<Long64_t>::max() &&
       static_cast<std::uint64_t>(fEndEntry - fBeginEntry) > processedEntries) {
      std::ostringstream buf{};
      buf << "RDataFrame stopped processing after ";
      buf << processedEntries;
      buf << " entries, whereas an entry range (begin=";
      buf << fBeginEntry;
      buf << ",end=";
      buf << fEndEntry;
      buf << ") was requested. Consider adjusting the end value of the entry range to a maximum of ";
      buf << (fBeginEntry + processedEntries);
      buf << ".";
      Warning("RDataFrame::Run", "%s", buf.str().c_str());
   }
}

/// Run event loop over data accessed through a DataSource, in parallel.
void RLoopManager::RunDataSourceMT()
{
#ifdef R__USE_IMT
   assert(fDataSource != nullptr);
   // Shortcut if the entry range would result in not reading anything
   if (fBeginEntry == fEndEntry)
      return;
   // Apply global entry range if necessary
   if (fBeginEntry != 0 || fEndEntry != std::numeric_limits<Long64_t>::max())
      fDataSource->SetGlobalEntryRange(std::make_pair<std::uint64_t, std::uint64_t>(fBeginEntry, fEndEntry));

   DSRunRAII _{*fDataSource, fSuppressErrorsForMissingBranches};

   ROOT::Internal::RDF::ProcessMT(*fDataSource, *this);

#endif // not implemented otherwise (never called)
}

/// Execute actions and make sure named filters are called for each event.
/// Named filters must be called even if the analysis logic would not require it, lest they report confusing results.
void RLoopManager::RunAndCheckFilters(unsigned int slot, Long64_t entry)
{
   // data-block callbacks run before the rest of the graph
   if (fNewSampleNotifier.CheckFlag(slot)) {
      for (auto &callback : fSampleCallbacks)
         callback.second(slot, fSampleInfos[slot]);
      fNewSampleNotifier.UnsetFlag(slot);
   }

   for (auto *actionPtr : fBookedActions)
      actionPtr->Run(slot, entry);
   for (auto *namedFilterPtr : fBookedNamedFilters)
      namedFilterPtr->CheckFilters(slot, entry);
   for (auto &callback : fCallbacksEveryNEvents)
      callback(slot);
}

/// Build TTreeReaderValues for all nodes
/// This method loops over all filters, actions and other booked objects and
/// calls their `InitSlot` method, to get them ready for running a task.
void RLoopManager::InitNodeSlots(TTreeReader *r, unsigned int slot)
{
   SetupSampleCallbacks(r, slot);
   for (auto *ptr : fBookedActions)
      ptr->InitSlot(r, slot);
   for (auto *ptr : fBookedFilters)
      ptr->InitSlot(r, slot);
   for (auto *ptr : fBookedDefines)
      ptr->InitSlot(r, slot);
   for (auto *ptr : fBookedVariations)
      ptr->InitSlot(r, slot);

   for (auto &callback : fCallbacksOnce)
      callback(slot);
}

void RLoopManager::SetupSampleCallbacks(TTreeReader *r, unsigned int slot) {
   if (r != nullptr) {
      // we need to set a notifier so that we run the callbacks every time we switch to a new TTree
      // `PrependLink` inserts this notifier into the TTree/TChain's linked list of notifiers
      fNewSampleNotifier.GetChainNotifyLink(slot).PrependLink(*r->GetTree());
   }
   // Whatever the data source, initially set the "new data block" flag:
   // - for TChains, this ensures that we don't skip the first data block because
   //   the correct tree is already loaded
   // - for RDataSources and empty sources, which currently don't have data blocks, this
   //   ensures that we run once per task
   fNewSampleNotifier.SetFlag(slot);
}

void RLoopManager::UpdateSampleInfo(unsigned int slot, const std::pair<ULong64_t, ULong64_t> &range) {
   fSampleInfos[slot] = RSampleInfo(
      "Empty source, range: {" + std::to_string(range.first) + ", " + std::to_string(range.second) + "}", range);
}

void RLoopManager::UpdateSampleInfo(unsigned int slot, TTreeReader &r) {
   // one GetTree to retrieve the TChain, another to retrieve the underlying TTree
   auto *tree = r.GetTree()->GetTree();
   R__ASSERT(tree != nullptr);
   const std::string treename = ROOT::Internal::TreeUtils::GetTreeFullPaths(*tree)[0];
   auto *file = tree->GetCurrentFile();
   const std::string fname = file != nullptr ? file->GetName() : "#inmemorytree#";

   std::pair<Long64_t, Long64_t> range = r.GetEntriesRange();
   R__ASSERT(range.first >= 0);
   if (range.second == -1) {
      range.second = tree->GetEntries(); // convert '-1', i.e. 'until the end', to the actual entry number
   }
   // If the tree is stored in a subdirectory, treename will be the full path to it starting with the root directory '/'
   const std::string &id = fname + (treename.rfind('/', 0) == 0 ? "" : "/") + treename;
   if (fSampleMap.empty()) {
      fSampleInfos[slot] = RSampleInfo(id, range);
   } else {
      if (fSampleMap.find(id) == fSampleMap.end())
         throw std::runtime_error("Full sample identifier '" + id + "' cannot be found in the available samples.");
      fSampleInfos[slot] = RSampleInfo(id, range, fSampleMap[id]);
   }
}

/// Create a slot stack with the desired number of slots or reuse a shared instance.
/// When a LoopManager runs in isolation, it will create its own slot stack from the
/// number of slots. When it runs as part of RunGraphs(), each loop manager will be
/// assigned a shared slot stack, so dataframe helpers can be shared in a thread-safe
/// manner.
std::shared_ptr<ROOT::Internal::RSlotStack> RLoopManager::SlotStack() const
{
#ifdef R__USE_IMT
   if (auto shared = fSlotStack.lock(); shared) {
      return shared;
   } else {
      return std::make_shared<ROOT::Internal::RSlotStack>(fNSlots);
   }
#else
   return nullptr;
#endif
}

/// Initialize all nodes of the functional graph before running the event loop.
/// This method is called once per event-loop and performs generic initialization
/// operations that do not depend on the specific processing slot (i.e. operations
/// that are common for all threads).
void RLoopManager::InitNodes()
{
   EvalChildrenCounts();
   for (auto *filter : fBookedFilters)
      filter->InitNode();
   for (auto *range : fBookedRanges)
      range->InitNode();
   for (auto *ptr : fBookedActions)
      ptr->Initialize();
}

/// Perform clean-up operations. To be called at the end of each event loop.
void RLoopManager::CleanUpNodes()
{
   fMustRunNamedFilters = false;

   // forget RActions and detach TResultProxies
   for (auto *ptr : fBookedActions)
      ptr->Finalize();

   fRunActions.insert(fRunActions.begin(), fBookedActions.begin(), fBookedActions.end());
   fBookedActions.clear();

   // reset children counts
   fNChildren = 0;
   fNStopsReceived = 0;
   for (auto *ptr : fBookedFilters)
      ptr->ResetChildrenCount();
   for (auto *ptr : fBookedRanges)
      ptr->ResetChildrenCount();

   fCallbacksEveryNEvents.clear();
   fCallbacksOnce.clear();
}

/// Perform clean-up operations. To be called at the end of each task execution.
void RLoopManager::CleanUpTask(TTreeReader *r, unsigned int slot)
{
   if (r != nullptr)
      fNewSampleNotifier.GetChainNotifyLink(slot).RemoveLink(*r->GetTree());
   for (auto *ptr : fBookedActions)
      ptr->FinalizeSlot(slot);
   for (auto *ptr : fBookedFilters)
      ptr->FinalizeSlot(slot);
   for (auto *ptr : fBookedDefines)
      ptr->FinalizeSlot(slot);

   if (auto ds = GetDataSource(); ds && ds->GetLabel() == "TTreeDS") {
      // we are reading from a tree/chain and we need to re-create the RTreeColumnReaders at every task
      // because the TTreeReader object changes at every task
      for (auto &v : fDatasetColumnReaders[slot])
         v.second.reset();
   }
}

/// Add RDF nodes that require just-in-time compilation to the computation graph.
/// This method also clears the contents of GetCodeToJit().
void RLoopManager::Jit()
{
   {
      R__READ_LOCKGUARD(ROOT::gCoreMutex);
      if (GetCodeToJit().empty()) {
         R__LOG_INFO(RDFLogChannel()) << "Nothing to jit and execute.";
         return;
      }
   }

   const std::string code = []() {
      R__WRITE_LOCKGUARD(ROOT::gCoreMutex);
      return std::move(GetCodeToJit());
   }();

   TStopwatch s;
   s.Start();
   RDFInternal::InterpreterCalc(code, "RLoopManager::Run");
   s.Stop();
   R__LOG_INFO(RDFLogChannel()) << "Just-in-time compilation phase completed"
                                << (s.RealTime() > 1e-3 ? " in " + std::to_string(s.RealTime()) + " seconds."
                                                        : " in less than 1ms.");
}

/// Trigger counting of number of children nodes for each node of the functional graph.
/// This is done once before starting the event loop. Each action sends an `increase children count` signal
/// upstream, which is propagated until RLoopManager. Each time a node receives the signal, in increments its
/// children counter. Each node only propagates the signal once, even if it receives it multiple times.
/// Named filters also send an `increase children count` signal, just like actions, as they always execute during
/// the event loop so the graph branch they belong to must count as active even if it does not end in an action.
void RLoopManager::EvalChildrenCounts()
{
   for (auto *actionPtr : fBookedActions)
      actionPtr->TriggerChildrenCount();
   for (auto *namedFilterPtr : fBookedNamedFilters)
      namedFilterPtr->TriggerChildrenCount();
}

/// Start the event loop with a different mechanism depending on IMT/no IMT, data source/no data source.
/// Also perform a few setup and clean-up operations (jit actions if necessary, clear booked actions after the loop...).
/// The jitting phase is skipped if the `jit` parameter is `false` (unsafe, use with care).
void RLoopManager::Run(bool jit)
{
   // Change value of TTree::GetMaxTreeSize only for this scope. Revert when #6640 will be solved.
   MaxTreeSizeRAII ctxtmts;

   R__LOG_INFO(RDFLogChannel()) << "Starting event loop number " << fNRuns << '.';

   ThrowIfNSlotsChanged(GetNSlots());

   if (jit)
      Jit();

   InitNodes();

   // Exceptions can occur during the event loop. In order to ensure proper cleanup of nodes
   // we use RAII: even in case of an exception, the destructor of the object is invoked and
   // all the cleanup takes place.
   class NodesCleanerRAII {
      RLoopManager &fRLM;

   public:
      NodesCleanerRAII(RLoopManager &thisRLM) : fRLM(thisRLM) {}
      ~NodesCleanerRAII() { fRLM.CleanUpNodes(); }
   };

   NodesCleanerRAII runKeeper(*this);

   TStopwatch s;
   s.Start();

   switch (fLoopType) {
   case ELoopType::kInvalid:
      throw std::runtime_error("RDataFrame: executing the computation graph without a data source, aborting.");
      break;
   case ELoopType::kNoFilesMT: RunEmptySourceMT(); break;
   case ELoopType::kDataSourceMT: RunDataSourceMT(); break;
   case ELoopType::kNoFiles: RunEmptySource(); break;
   case ELoopType::kDataSource: RunDataSource(); break;
   }
   s.Stop();

   fNRuns++;

   R__LOG_INFO(RDFLogChannel()) << "Finished event loop number " << fNRuns - 1 << " (" << s.CpuTime() << "s CPU, "
                                << s.RealTime() << "s elapsed).";
}

/// Return the list of default columns -- empty if none was provided when constructing the RDataFrame
const ColumnNames_t &RLoopManager::GetDefaultColumnNames() const
{
   return fDefaultColumns;
}

void RLoopManager::Register(RDFInternal::RActionBase *actionPtr)
{
   fBookedActions.emplace_back(actionPtr);
   AddSampleCallback(actionPtr, actionPtr->GetSampleCallback());
}

void RLoopManager::Deregister(RDFInternal::RActionBase *actionPtr)
{
   RDFInternal::Erase(actionPtr, fRunActions);
   RDFInternal::Erase(actionPtr, fBookedActions);
   fSampleCallbacks.erase(actionPtr);
}

void RLoopManager::Register(RFilterBase *filterPtr)
{
   fBookedFilters.emplace_back(filterPtr);
   if (filterPtr->HasName()) {
      fBookedNamedFilters.emplace_back(filterPtr);
      fMustRunNamedFilters = true;
   }
}

void RLoopManager::Deregister(RFilterBase *filterPtr)
{
   RDFInternal::Erase(filterPtr, fBookedFilters);
   RDFInternal::Erase(filterPtr, fBookedNamedFilters);
}

void RLoopManager::Register(RRangeBase *rangePtr)
{
   fBookedRanges.emplace_back(rangePtr);
}

void RLoopManager::Deregister(RRangeBase *rangePtr)
{
   RDFInternal::Erase(rangePtr, fBookedRanges);
}

void RLoopManager::Register(RDefineBase *ptr)
{
   fBookedDefines.emplace_back(ptr);
}

void RLoopManager::Deregister(RDefineBase *ptr)
{
   RDFInternal::Erase(ptr, fBookedDefines);
   fSampleCallbacks.erase(ptr);
}

void RLoopManager::Register(RDFInternal::RVariationBase *v)
{
   fBookedVariations.emplace_back(v);
}

void RLoopManager::Deregister(RDFInternal::RVariationBase *v)
{
   RDFInternal::Erase(v, fBookedVariations);
}

// dummy call, end of recursive chain of calls
bool RLoopManager::CheckFilters(unsigned int, Long64_t)
{
   return true;
}

/// Call `FillReport` on all booked filters
void RLoopManager::Report(ROOT::RDF::RCutFlowReport &rep) const
{
   for (const auto *fPtr : fBookedNamedFilters)
      fPtr->FillReport(rep);
}

void RLoopManager::ToJitExec(const std::string &code) const
{
   R__WRITE_LOCKGUARD(ROOT::gCoreMutex);
   GetCodeToJit().append(code);
}

void RLoopManager::RegisterCallback(ULong64_t everyNEvents, std::function<void(unsigned int)> &&f)
{
   if (everyNEvents == 0ull)
      fCallbacksOnce.emplace_back(std::move(f), fNSlots);
   else
      fCallbacksEveryNEvents.emplace_back(everyNEvents, std::move(f), fNSlots);
}

std::vector<std::string> RLoopManager::GetFiltersNames()
{
   std::vector<std::string> filters;
   for (auto *filter : fBookedFilters) {
      auto name = (filter->HasName() ? filter->GetName() : "Unnamed Filter");
      filters.push_back(name);
   }
   return filters;
}

std::vector<RNodeBase *> RLoopManager::GetGraphEdges() const
{
   std::vector<RNodeBase *> nodes(fBookedFilters.size() + fBookedRanges.size());
   auto it = std::copy(fBookedFilters.begin(), fBookedFilters.end(), nodes.begin());
   std::copy(fBookedRanges.begin(), fBookedRanges.end(), it);
   return nodes;
}

std::vector<RDFInternal::RActionBase *> RLoopManager::GetAllActions() const
{
   std::vector<RDFInternal::RActionBase *> actions(fBookedActions.size() + fRunActions.size());
   auto it = std::copy(fBookedActions.begin(), fBookedActions.end(), actions.begin());
   std::copy(fRunActions.begin(), fRunActions.end(), it);
   return actions;
}

std::shared_ptr<ROOT::Internal::RDF::GraphDrawing::GraphNode> RLoopManager::GetGraph(
   std::unordered_map<void *, std::shared_ptr<ROOT::Internal::RDF::GraphDrawing::GraphNode>> &visitedMap)
{
   // If there is already a node for the RLoopManager return it. If there is not, return a new one.
   auto duplicateRLoopManagerIt = visitedMap.find((void *)this);
   if (duplicateRLoopManagerIt != visitedMap.end())
      return duplicateRLoopManagerIt->second;

   std::string name;
   if (fDataSource) {
      name = fDataSource->GetLabel();
   } else {
      name = "Empty source\\nEntries: " + std::to_string(GetNEmptyEntries());
   }
   auto thisNode = std::make_shared<ROOT::Internal::RDF::GraphDrawing::GraphNode>(
      name, visitedMap.size(), ROOT::Internal::RDF::GraphDrawing::ENodeType::kRoot);
   visitedMap[(void *)this] = thisNode;
   return thisNode;
}

/// Return true if AddDataSourceColumnReaders was called for column name col.
bool RLoopManager::HasDataSourceColumnReaders(std::string_view col, const std::type_info &ti) const
{
   const auto key = MakeDatasetColReadersKey(col, ti);
   assert(fDataSource != nullptr);
   // since data source column readers are always added for all slots at the same time,
   // if the reader is present for slot 0 we have it for all other slots as well.
   auto it = fDatasetColumnReaders[0].find(key);
   return (it != fDatasetColumnReaders[0].end() && it->second);
}

void RLoopManager::AddDataSourceColumnReaders(std::string_view col,
                                              std::vector<std::unique_ptr<RColumnReaderBase>> &&readers,
                                              const std::type_info &ti)
{
   const auto key = MakeDatasetColReadersKey(col, ti);
   assert(fDataSource != nullptr && !HasDataSourceColumnReaders(col, ti));
   assert(readers.size() == fNSlots);

   for (auto slot = 0u; slot < fNSlots; ++slot) {
      fDatasetColumnReaders[slot][key] = std::move(readers[slot]);
   }
}

// Differently from AddDataSourceColumnReaders, this can be called from multiple threads concurrently
/// \brief Register a new RTreeColumnReader with this RLoopManager.
/// \return A shared pointer to the inserted column reader.
RColumnReaderBase *RLoopManager::AddTreeColumnReader(unsigned int slot, std::string_view col,
                                                     std::unique_ptr<RColumnReaderBase> &&reader,
                                                     const std::type_info &ti)
{
   auto &readers = fDatasetColumnReaders[slot];
   const auto key = MakeDatasetColReadersKey(col, ti);
   // if a reader for this column and this slot was already there, we are doing something wrong
   assert(readers.find(key) == readers.end() || readers[key] == nullptr);
   auto *rptr = reader.get();
   readers[key] = std::move(reader);
   return rptr;
}

RColumnReaderBase *RLoopManager::AddDataSourceColumnReader(unsigned int slot, std::string_view col,
                                                           const std::type_info &ti, TTreeReader *treeReader)
{
   auto &readers = fDatasetColumnReaders[slot];
   const auto key = MakeDatasetColReadersKey(col, ti);
   // if a reader for this column and this slot was already there, we are doing something wrong
   assert(readers.find(key) == readers.end() || readers[key] == nullptr);
   assert(fDataSource && "Missing RDataSource to add column reader.");

   readers[key] = ROOT::Internal::RDF::CreateColumnReader(*fDataSource, slot, col, ti, treeReader);

   return readers[key].get();
}

RColumnReaderBase *
RLoopManager::GetDatasetColumnReader(unsigned int slot, std::string_view col, const std::type_info &ti) const
{
   const auto key = MakeDatasetColReadersKey(col, ti);
   if (auto it = fDatasetColumnReaders[slot].find(key); it != fDatasetColumnReaders[slot].end() && it->second)
      return it->second.get();
   else
      return nullptr;
}

void RLoopManager::AddSampleCallback(void *nodePtr, SampleCallback_t &&callback)
{
   if (callback)
      fSampleCallbacks.insert({nodePtr, std::move(callback)});
}

void RLoopManager::SetEmptyEntryRange(std::pair<ULong64_t, ULong64_t> &&newRange)
{
   fEmptyEntryRange = std::move(newRange);
}

void RLoopManager::ChangeBeginAndEndEntries(Long64_t begin, Long64_t end)
{
   fBeginEntry = begin;
   fEndEntry = end;
}

void ROOT::Detail::RDF::RLoopManager::SetTTreeLifeline(std::any lifeline)
{
   fTTreeLifeline = std::move(lifeline);
}

std::shared_ptr<ROOT::Detail::RDF::RLoopManager>
ROOT::Detail::RDF::CreateLMFromTTree(std::string_view datasetName, std::string_view fileNameGlob,
                                     const ROOT::RDF::ColumnNames_t &defaultColumns, bool checkFile)
{
   // Introduce the same behaviour as in CreateLMFromFile for consistency.
   // Creating an RDataFrame with a non-existing file will throw early rather
   // than wait for the start of the graph execution.
   if (checkFile) {
      OpenFileWithSanityChecks(fileNameGlob);
   }

   auto dataSource = std::make_unique<ROOT::Internal::RDF::RTTreeDS>(datasetName, fileNameGlob);
   auto lm = std::make_shared<ROOT::Detail::RDF::RLoopManager>(std::move(dataSource), defaultColumns);
   return lm;
}

std::shared_ptr<ROOT::Detail::RDF::RLoopManager>
ROOT::Detail::RDF::CreateLMFromTTree(std::string_view datasetName, const std::vector<std::string> &fileNameGlobs,
                                     const std::vector<std::string> &defaultColumns, bool checkFile)
{
   if (fileNameGlobs.size() == 0)
      throw std::invalid_argument("RDataFrame: empty list of input files.");
   // Introduce the same behaviour as in CreateLMFromFile for consistency.
   // Creating an RDataFrame with a non-existing file will throw early rather
   // than wait for the start of the graph execution.
   if (checkFile) {
      OpenFileWithSanityChecks(fileNameGlobs[0]);
   }
   auto dataSource = std::make_unique<ROOT::Internal::RDF::RTTreeDS>(datasetName, fileNameGlobs);
   auto lm = std::make_shared<ROOT::Detail::RDF::RLoopManager>(std::move(dataSource), defaultColumns);
   return lm;
}

std::shared_ptr<ROOT::Detail::RDF::RLoopManager>
ROOT::Detail::RDF::CreateLMFromRNTuple(std::string_view datasetName, std::string_view fileNameGlob,
                                       const ROOT::RDF::ColumnNames_t &defaultColumns)
{
   auto dataSource = std::make_unique<ROOT::RDF::RNTupleDS>(datasetName, fileNameGlob);
   auto lm = std::make_shared<ROOT::Detail::RDF::RLoopManager>(std::move(dataSource), defaultColumns);
   return lm;
}

std::shared_ptr<ROOT::Detail::RDF::RLoopManager>
ROOT::Detail::RDF::CreateLMFromRNTuple(std::string_view datasetName, const std::vector<std::string> &fileNameGlobs,
                                       const ROOT::RDF::ColumnNames_t &defaultColumns)
{
   auto dataSource = std::make_unique<ROOT::RDF::RNTupleDS>(datasetName, fileNameGlobs);
   auto lm = std::make_shared<ROOT::Detail::RDF::RLoopManager>(std::move(dataSource), defaultColumns);
   return lm;
}

std::shared_ptr<ROOT::Detail::RDF::RLoopManager>
ROOT::Detail::RDF::CreateLMFromFile(std::string_view datasetName, std::string_view fileNameGlob,
                                    const ROOT::RDF::ColumnNames_t &defaultColumns)
{

   auto inFile = OpenFileWithSanityChecks(fileNameGlob);

   if (inFile->Get<TTree>(datasetName.data())) {
      return CreateLMFromTTree(datasetName, fileNameGlob, defaultColumns, /*checkFile=*/false);
   } else if (inFile->Get<ROOT::RNTuple>(datasetName.data())) {
      return CreateLMFromRNTuple(datasetName, fileNameGlob, defaultColumns);
   }

   throw std::invalid_argument("RDataFrame: unsupported data format for dataset \"" + std::string(datasetName) +
                               "\" in file \"" + inFile->GetName() + "\".");
}

std::shared_ptr<ROOT::Detail::RDF::RLoopManager>
ROOT::Detail::RDF::CreateLMFromFile(std::string_view datasetName, const std::vector<std::string> &fileNameGlobs,
                                    const ROOT::RDF::ColumnNames_t &defaultColumns)
{

   if (fileNameGlobs.size() == 0)
      throw std::invalid_argument("RDataFrame: empty list of input files.");

   auto inFile = OpenFileWithSanityChecks(fileNameGlobs[0]);

   if (inFile->Get<TTree>(datasetName.data())) {
      return CreateLMFromTTree(datasetName, fileNameGlobs, defaultColumns, /*checkFile=*/false);
   } else if (inFile->Get<ROOT::RNTuple>(datasetName.data())) {
      return CreateLMFromRNTuple(datasetName, fileNameGlobs, defaultColumns);
   }

   throw std::invalid_argument("RDataFrame: unsupported data format for dataset \"" + std::string(datasetName) +
                               "\" in file \"" + inFile->GetName() + "\".");
}

// outlined to pin virtual table
ROOT::Detail::RDF::RLoopManager::~RLoopManager() = default;

void ROOT::Detail::RDF::RLoopManager::SetDataSource(std::unique_ptr<ROOT::RDF::RDataSource> dataSource)
{
   if (dataSource) {
      fDataSource = std::move(dataSource);
      fDataSource->SetNSlots(fNSlots);
      fLoopType = ROOT::IsImplicitMTEnabled() ? ELoopType::kDataSourceMT : ELoopType::kDataSource;
   }
}

void ROOT::Detail::RDF::RLoopManager::DataSourceThreadTask(const std::pair<ULong64_t, ULong64_t> &entryRange,
                                                           ROOT::Internal::RSlotStack &slotStack,
                                                           std::atomic<ULong64_t> &entryCount)
{
#ifdef R__USE_IMT
   ROOT::Internal::RSlotStackRAII slotRAII(slotStack);
   const auto &slot = slotRAII.fSlot;

   const auto &[start, end] = entryRange;
   const auto nEntries = end - start;
   entryCount.fetch_add(nEntries);

   RCallCleanUpTask cleanup(*this, slot);
   RDSRangeRAII _{*this, slot, start};

   fSampleInfos[slot] = ROOT::Internal::RDF::CreateSampleInfo(*fDataSource, slot, fSampleMap);

   R__LOG_DEBUG(0, RDFLogChannel()) << LogRangeProcessing({fDataSource->GetLabel(), start, end, slot});

   try {
      for (auto entry = start; entry < end; ++entry) {
         if (fDataSource->SetEntry(slot, entry)) {
            RunAndCheckFilters(slot, entry);
         }
      }
   } catch (...) {
      std::cerr << "RDataFrame::Run: event loop was interrupted\n";
      throw;
   }
   fDataSource->FinalizeSlot(slot);
#else
   (void)entryRange;
   (void)slotStack;
   (void)entryCount;
#endif
}

void ROOT::Detail::RDF::RLoopManager::TTreeThreadTask(TTreeReader &treeReader, ROOT::Internal::RSlotStack &slotStack,
                                                      std::atomic<ULong64_t> &entryCount)
{
#ifdef R__USE_IMT
   ROOT::Internal::RSlotStackRAII slotRAII(slotStack);
   const auto &slot = slotRAII.fSlot;

   const auto entryRange = treeReader.GetEntriesRange(); // we trust TTreeProcessorMT to call SetEntriesRange
   const auto &[start, end] = entryRange;
   const auto nEntries = end - start;
   auto count = entryCount.fetch_add(nEntries);

   RDSRangeRAII _{*this, slot, static_cast<ULong64_t>(start), &treeReader};
   RCallCleanUpTask cleanup(*this, slot, &treeReader);

   R__LOG_DEBUG(0, RDFLogChannel()) << LogRangeProcessing(
      {fDataSource->GetLabel(), static_cast<ULong64_t>(start), static_cast<ULong64_t>(end), slot});
   try {
      // recursive call to check filters and conditionally execute actions
      while (validTTreeReaderRead(treeReader)) {
         if (fNewSampleNotifier.CheckFlag(slot)) {
            UpdateSampleInfo(slot, treeReader);
         }
         RunAndCheckFilters(slot, count++);
      }
   } catch (...) {
      std::cerr << "RDataFrame::Run: event loop was interrupted\n";
      throw;
   }
   // fNStopsReceived < fNChildren is always true at the moment as we don't support event loop early quitting in
   // multi-thread runs, but it costs nothing to be safe and future-proof in case we add support for that later.
   if (treeReader.GetEntryStatus() != TTreeReader::kEntryBeyondEnd && fNStopsReceived < fNChildren) {
      // something went wrong in the TTreeReader event loop
      throw std::runtime_error("An error was encountered while processing the data. TTreeReader status code is: " +
                               std::to_string(treeReader.GetEntryStatus()));
   }
#else
   (void)treeReader;
   (void)slotStack;
   (void)entryCount;
#endif
}
