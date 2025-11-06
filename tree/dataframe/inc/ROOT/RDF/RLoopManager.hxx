// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RLOOPMANAGER
#define ROOT_RLOOPMANAGER

#include "ROOT/RDataSource.hxx"
#include "ROOT/RDF/RColumnReaderBase.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "ROOT/RDF/RNodeBase.hxx"
#include "ROOT/RDF/RNewSampleNotifier.hxx"
#include "ROOT/RDF/RSampleInfo.hxx"
#include "ROOT/RDF/Utils.hxx"

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <any>

// forward declarations
class TTree;
class TTreeReader;
class TDirectory;

namespace ROOT {
namespace RDF {
class RCutFlowReport;
class RDataSource;
} // ns RDF

namespace Internal {
class RSlotStack;
namespace RDF {
class GraphNode;
class RActionBase;
class RVariationBase;
class RDefinesWithReaders;
class RVariationsWithReaders;

namespace GraphDrawing {
class GraphCreatorHelper;
} // ns GraphDrawing

using Callback_t = std::function<void(unsigned int)>;

class RCallback {
   const Callback_t fFun;
   const ULong64_t fEveryN;
   std::vector<ULong64_t> fCounters;

public:
   RCallback(ULong64_t everyN, Callback_t &&f, unsigned int nSlots)
      : fFun(std::move(f)), fEveryN(everyN), fCounters(nSlots, 0ull)
   {
   }

   void operator()(unsigned int slot)
   {
      auto &c = fCounters[slot];
      ++c;
      if (c == fEveryN) {
         c = 0ull;
         fFun(slot);
      }
   }
};

class ROneTimeCallback {
   const Callback_t fFun;
   std::vector<int> fHasBeenCalled; // std::vector<bool> is thread-unsafe for our purposes (and generally evil)

public:
   ROneTimeCallback(Callback_t &&f, unsigned int nSlots) : fFun(std::move(f)), fHasBeenCalled(nSlots, 0) {}

   void operator()(unsigned int slot)
   {
      if (fHasBeenCalled[slot] == 1)
         return;
      fFun(slot);
      fHasBeenCalled[slot] = 1;
   }
};

struct RDSRangeRAII;

} // namespace RDF
} // namespace Internal
} // namespace ROOT

namespace ROOT {
namespace Detail {
namespace RDF {
namespace RDFInternal = ROOT::Internal::RDF;

class RFilterBase;
class RRangeBase;
class RDefineBase;
using ROOT::RDF::RDataSource;

/// The head node of a RDF computation graph.
/// This class is responsible of running the event loop.
class RLoopManager : public RNodeBase {
   using ColumnNames_t = std::vector<std::string>;
   enum class ELoopType {
      kInvalid,
      kNoFiles,
      kNoFilesMT,
      kDataSource,
      kDataSourceMT
   };

   friend struct RCallCleanUpTask;
   friend struct ROOT::Internal::RDF::RDSRangeRAII;

   /*
   A wrapped reference to a TTree dataset that can be shared by many computation graphs. Ensures lifetime management.
   It needs to be destroyed *after* the fDataSource data member, in case it is holding the only Python reference to the
   input dataset and the data source needs to access it before it is destroyed.
   */
   std::any fTTreeLifeline{};

   std::vector<RDFInternal::RActionBase *> fBookedActions; ///< Non-owning pointers to actions to be run
   std::vector<RDFInternal::RActionBase *> fRunActions;    ///< Non-owning pointers to actions already run
   std::vector<RFilterBase *> fBookedFilters;
   std::vector<RFilterBase *> fBookedNamedFilters; ///< Contains a subset of fBookedFilters, i.e. only the named filters
   std::vector<RRangeBase *> fBookedRanges;
   std::vector<RDefineBase *> fBookedDefines;
   std::vector<RDFInternal::RVariationBase *> fBookedVariations;

   Long64_t fBeginEntry{0};
   Long64_t fEndEntry{std::numeric_limits<Long64_t>::max()};

   /// Keys are `fname + "/" + treename` as RSampleInfo::fID; Values are pointers to the corresponding sample
   std::unordered_map<std::string, ROOT::RDF::Experimental::RSample *> fSampleMap;
   /// Samples need to survive throughout the whole event loop, hence stored as an attribute
   std::vector<ROOT::RDF::Experimental::RSample> fSamples;

   ColumnNames_t fDefaultColumns;
   /// Range of entries created when no data source is specified.
   std::pair<ULong64_t, ULong64_t> fEmptyEntryRange{};
   unsigned int fNSlots{1};
   bool fMustRunNamedFilters{true};
   /// The kind of event loop that is going to be run (e.g. on ROOT files, on no files)
   ELoopType fLoopType{ELoopType::kInvalid};
   std::unique_ptr<RDataSource> fDataSource{}; ///< Owning pointer to a data-source object. Null if no data-source
   /// Registered callbacks to be executed every N events.
   /// The registration happens via the RegisterCallback method.
   std::vector<RDFInternal::RCallback> fCallbacksEveryNEvents;
   /// Registered callbacks to invoke just once before running the loop.
   /// The registration happens via the RegisterCallback method.
   std::vector<RDFInternal::ROneTimeCallback> fCallbacksOnce;
   /// Registered callbacks to call at the beginning of each "data block".
   /// The key is the pointer of the corresponding node in the computation graph (a RDefinePerSample or a RAction).
   std::unordered_map<void *, ROOT::RDF::SampleCallback_t> fSampleCallbacks;
   RDFInternal::RNewSampleNotifier fNewSampleNotifier;
   std::vector<ROOT::RDF::RSampleInfo> fSampleInfos;
   unsigned int fNRuns{0}; ///< Number of event loops run

   /// Readers for TTree/RDataSource columns (one per slot), shared by all nodes in the computation graph.
   std::vector<std::unordered_map<std::string, std::unique_ptr<RColumnReaderBase>>> fDatasetColumnReaders;

   /// Pointer to a shared slot stack in case this instance runs concurrently with others:
   std::weak_ptr<ROOT::Internal::RSlotStack> fSlotStack;

   void RunEmptySourceMT();
   void RunEmptySource();
   void RunDataSourceMT();
   void RunDataSource();
   void RunAndCheckFilters(unsigned int slot, Long64_t entry);
   void InitNodeSlots(TTreeReader *r, unsigned int slot);
   void InitNodes();
   void CleanUpNodes();
   void CleanUpTask(TTreeReader *r, unsigned int slot);
   void EvalChildrenCounts();
   void SetupSampleCallbacks(TTreeReader *r, unsigned int slot);
   void UpdateSampleInfo(unsigned int slot, const std::pair<ULong64_t, ULong64_t> &range);
   void UpdateSampleInfo(unsigned int slot, TTreeReader &r);
   std::shared_ptr<ROOT::Internal::RSlotStack> SlotStack() const;

   // List of branches for which we want to suppress the printed error about
   // missing branch when switching to a new tree. This is modified by readers,
   // so must be declared before them in this class.
   std::set<std::string> fSuppressErrorsForMissingBranches{};
   ROOT::Internal::RDF::RStringCache fCachedColNames;
   std::set<std::pair<std::string_view, std::unique_ptr<ROOT::Internal::RDF::RDefinesWithReaders>>>
      fUniqueDefinesWithReaders;
   std::set<std::pair<std::string_view, std::unique_ptr<ROOT::Internal::RDF::RVariationsWithReaders>>>
      fUniqueVariationsWithReaders;

public:
   RLoopManager(const ColumnNames_t &defaultColumns = {});
   RLoopManager(TTree *tree, const ColumnNames_t &defaultBranches);
   RLoopManager(ULong64_t nEmptyEntries);
   RLoopManager(std::unique_ptr<RDataSource> ds, const ColumnNames_t &defaultBranches);
   RLoopManager(ROOT::RDF::Experimental::RDatasetSpec &&spec);

   // Rule of five

   RLoopManager(const RLoopManager &) = delete;
   RLoopManager &operator=(const RLoopManager &) = delete;
   RLoopManager(RLoopManager &&) = delete;
   RLoopManager &operator=(RLoopManager &&) = delete;
   ~RLoopManager() override;

   void Jit();
   RLoopManager *GetLoopManagerUnchecked() final { return this; }
   void Run(bool jit = true);
   const ColumnNames_t &GetDefaultColumnNames() const;
   ULong64_t GetNEmptyEntries() const { return fEmptyEntryRange.second - fEmptyEntryRange.first; }
   RDataSource *GetDataSource() const { return fDataSource.get(); }
   void Register(RDFInternal::RActionBase *actionPtr);
   void Deregister(RDFInternal::RActionBase *actionPtr);
   void Register(RFilterBase *filterPtr);
   void Deregister(RFilterBase *filterPtr);
   void Register(RRangeBase *rangePtr);
   void Deregister(RRangeBase *rangePtr);
   void Register(RDefineBase *definePtr);
   void Deregister(RDefineBase *definePtr);
   void Register(RDFInternal::RVariationBase *varPtr);
   void Deregister(RDFInternal::RVariationBase *varPtr);
   bool CheckFilters(unsigned int, Long64_t) final;
   unsigned int GetNSlots() const { return fNSlots; }
   void Report(ROOT::RDF::RCutFlowReport &rep) const final;
   /// End of recursive chain of calls, does nothing
   void PartialReport(ROOT::RDF::RCutFlowReport &) const final {}
   void IncrChildrenCount() final { ++fNChildren; }
   void StopProcessing() final { ++fNStopsReceived; }
   void ToJitExec(const std::string &) const;
   void RegisterCallback(ULong64_t everyNEvents, std::function<void(unsigned int)> &&f);
   unsigned int GetNRuns() const { return fNRuns; }
   bool HasDataSourceColumnReaders(std::string_view col, const std::type_info &ti) const;
   void AddDataSourceColumnReaders(std::string_view col, std::vector<std::unique_ptr<RColumnReaderBase>> &&readers,
                                   const std::type_info &ti);
   RColumnReaderBase *GetDatasetColumnReader(unsigned int slot, std::string_view col, const std::type_info &ti) const;
   RColumnReaderBase *AddDataSourceColumnReader(unsigned int slot, std::string_view col, const std::type_info &ti,
                                                TTreeReader *treeReader);

   /// End of recursive chain of calls, does nothing
   void AddFilterName(std::vector<std::string> &) final {}
   /// For each booked filter, returns either the name or "Unnamed Filter"
   std::vector<std::string> GetFiltersNames();

   /// Return all graph edges known to RLoopManager
   /// This includes Filters and Ranges but not Defines.
   std::vector<RNodeBase *> GetGraphEdges() const;

   /// Return all actions, either booked or already run
   std::vector<RDFInternal::RActionBase *> GetAllActions() const;

   std::shared_ptr<ROOT::Internal::RDF::GraphDrawing::GraphNode>
   GetGraph(std::unordered_map<void *, std::shared_ptr<ROOT::Internal::RDF::GraphDrawing::GraphNode>> &visitedMap) final;

   void AddSampleCallback(void *nodePtr, ROOT::RDF::SampleCallback_t &&callback);

   void SetEmptyEntryRange(std::pair<ULong64_t, ULong64_t> &&newRange);
   void ChangeBeginAndEndEntries(Long64_t begin, Long64_t end);
   void ChangeSpec(ROOT::RDF::Experimental::RDatasetSpec &&spec);

   ROOT::Internal::RDF::RStringCache &GetColumnNamesCache() { return fCachedColNames; }
   std::set<std::pair<std::string_view, std::unique_ptr<ROOT::Internal::RDF::RDefinesWithReaders>>> &
   GetUniqueDefinesWithReaders()
   {
      return fUniqueDefinesWithReaders;
   }
   std::set<std::pair<std::string_view, std::unique_ptr<ROOT::Internal::RDF::RVariationsWithReaders>>> &
   GetUniqueVariationsWithReaders()
   {
      return fUniqueVariationsWithReaders;
   }

   void SetTTreeLifeline(std::any lifeline);
   /// Register a slot stack to be used by this RLoopManager. This allows for sharing RDataFrame helpers safely in the
   /// context of RunGraphs(). Note that the loop manager only stores a weak_ptr, in between runs.
   void SetSlotStack(const std::shared_ptr<ROOT::Internal::RSlotStack> &slotStack) { fSlotStack = slotStack; }

   void SetDataSource(std::unique_ptr<ROOT::RDF::RDataSource> dataSource);

   void InsertSuppressErrorsForMissingBranch(const std::string &branchName)
   {
      fSuppressErrorsForMissingBranches.insert(branchName);
   }
   void EraseSuppressErrorsForMissingBranch(const std::string &branchName)
   {
      fSuppressErrorsForMissingBranches.erase(branchName);
   }
   const std::set<std::string> &GetSuppressErrorsForMissingBranches() const
   {
      return fSuppressErrorsForMissingBranches;
   }

   /// The task run by every thread on the input entry range, for the generic RDataSource.
   void DataSourceThreadTask(const std::pair<ULong64_t, ULong64_t> &entryRange, ROOT::Internal::RSlotStack &slotStack,
                             std::atomic<ULong64_t> &entryCount);
   /// The task run by every thread on an entry range (known by the input TTreeReader), for the TTree data source.
   void
   TTreeThreadTask(TTreeReader &treeReader, ROOT::Internal::RSlotStack &slotStack, std::atomic<ULong64_t> &entryCount);
};

/// \brief Create an RLoopManager that reads a TChain.
/// \param[in] datasetName Name of the TChain
/// \param[in] fileNameGlob File name (or glob) in which the TChain is stored.
/// \param[in] defaultColumns List of default columns, see
/// \param[in] checkFile file validator boolean
/// \ref https://root.cern/doc/master/classROOT_1_1RDataFrame.html#default-branches "Default column lists"
/// \return the RLoopManager instance.
std::shared_ptr<ROOT::Detail::RDF::RLoopManager>
CreateLMFromTTree(std::string_view datasetName, std::string_view fileNameGlob,
                  const std::vector<std::string> &defaultColumns, bool checkFile = true);

/// \brief Create an RLoopManager that reads a TChain.
/// \param[in] datasetName Name of the TChain
/// \param[in] fileNameGlobs List of file names (potentially globs).
/// \param[in] defaultColumns List of default columns, see
/// \param[in] checkFile file validator boolean
/// \ref https://root.cern/doc/master/classROOT_1_1RDataFrame.html#default-branches "Default column lists"
/// \return the RLoopManager instance.
std::shared_ptr<ROOT::Detail::RDF::RLoopManager>
CreateLMFromTTree(std::string_view datasetName, const std::vector<std::string> &fileNameGlobs,
                  const std::vector<std::string> &defaultColumns, bool checkFile = true);

/// \brief Create an RLoopManager that reads an RNTuple.
/// \param[in] datasetName Name of the RNTuple
/// \param[in] fileNameGlob File name (or glob) in which the RNTuple is stored.
/// \param[in] defaultColumns List of default columns, see
/// \ref https://root.cern/doc/master/classROOT_1_1RDataFrame.html#default-branches "Default column lists"
/// \return the RLoopManager instance.
std::shared_ptr<ROOT::Detail::RDF::RLoopManager> CreateLMFromRNTuple(std::string_view datasetName,
                                                                     std::string_view fileNameGlob,
                                                                     const std::vector<std::string> &defaultColumns);

/// \brief Create an RLoopManager that reads multiple RNTuples chained vertically.
/// \param[in] datasetName Name of the RNTuple
/// \param[in] fileNameGlobs List of file names (potentially globs).
/// \param[in] defaultColumns List of default columns, see
/// \ref https://root.cern/doc/master/classROOT_1_1RDataFrame.html#default-branches "Default column lists"
/// \return the RLoopManager instance.
std::shared_ptr<ROOT::Detail::RDF::RLoopManager> CreateLMFromRNTuple(std::string_view datasetName,
                                                                     const std::vector<std::string> &fileNameGlobs,
                                                                     const std::vector<std::string> &defaultColumns);

/// \brief Create an RLoopManager opening a file and checking the data format of the dataset.
/// \param[in] datasetName Name of the dataset in the file.
/// \param[in] fileNameGlob File name (or glob) in which the dataset is stored.
/// \param[in] defaultColumns List of default columns, see
/// \ref https://root.cern/doc/master/classROOT_1_1RDataFrame.html#default-branches "Default column lists"
/// \throws std::invalid_argument if the file could not be opened.
/// \return an RLoopManager of the appropriate data source.
std::shared_ptr<ROOT::Detail::RDF::RLoopManager> CreateLMFromFile(std::string_view datasetName,
                                                                  std::string_view fileNameGlob,
                                                                  const std::vector<std::string> &defaultColumns);

/// \brief Create an RLoopManager that reads many files. The first is opened to infer the data source type.
/// \param[in] datasetName Name of the dataset.
/// \param[in] fileNameGlobs List of file names (potentially globs).
/// \param[in] defaultColumns List of default columns, see
/// \ref https://root.cern/doc/master/classROOT_1_1RDataFrame.html#default-branches "Default column lists"
/// \throws std::invalid_argument if the file could not be opened.
/// \return an RLoopManager of the appropriate data source.
std::shared_ptr<ROOT::Detail::RDF::RLoopManager> CreateLMFromFile(std::string_view datasetName,
                                                                  const std::vector<std::string> &fileNameGlobs,
                                                                  const std::vector<std::string> &defaultColumns);

} // namespace RDF
} // namespace Detail
} // namespace ROOT

#endif
