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

#include "ROOT/InternalTreeUtils.hxx" // RNoCleanupNotifier
#include "ROOT/RDF/RColumnReaderBase.hxx"
#include "ROOT/RDF/RNodeBase.hxx"
#include "ROOT/RDF/RNewSampleNotifier.hxx"
#include "ROOT/RDF/RSampleInfo.hxx"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

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
namespace RDF {
std::vector<std::string> GetBranchNames(TTree &t, bool allowDuplicates = true);

class GraphNode;
class RActionBase;
class RVariationBase;

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

} // ns RDF
} // ns Internal

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
   enum class ELoopType { kROOTFiles, kROOTFilesMT, kNoFiles, kNoFilesMT, kDataSource, kDataSourceMT };

   friend struct RCallCleanUpTask;

   std::vector<RDFInternal::RActionBase *> fBookedActions; ///< Non-owning pointers to actions to be run
   std::vector<RDFInternal::RActionBase *> fRunActions;    ///< Non-owning pointers to actions already run
   std::vector<RFilterBase *> fBookedFilters;
   std::vector<RFilterBase *> fBookedNamedFilters; ///< Contains a subset of fBookedFilters, i.e. only the named filters
   std::vector<RRangeBase *> fBookedRanges;
   std::vector<RDefineBase *> fBookedDefines;
   std::vector<RDFInternal::RVariationBase *> fBookedVariations;

   /// Shared pointer to the input TTree. It does not delete the pointee if the TTree/TChain was passed directly as an
   /// argument to RDataFrame's ctor (in which case we let users retain ownership).
   std::shared_ptr<TTree> fTree{nullptr};
   const ColumnNames_t fDefaultColumns;
   const ULong64_t fNEmptyEntries{0};
   const unsigned int fNSlots{1};
   bool fMustRunNamedFilters{true};
   const ELoopType fLoopType; ///< The kind of event loop that is going to be run (e.g. on ROOT files, on no files)
   const std::unique_ptr<RDataSource> fDataSource; ///< Owning pointer to a data-source object. Null if no data-source
   std::vector<RDFInternal::RCallback> fCallbacks;         ///< Registered callbacks
   /// Registered callbacks to invoke just once before running the loop
   std::vector<RDFInternal::ROneTimeCallback> fCallbacksOnce;
   /// Registered callbacks to call at the beginning of each "data block"
   std::vector<ROOT::RDF::SampleCallback_t> fSampleCallbacks;
   RDFInternal::RNewSampleNotifier fNewSampleNotifier;
   std::vector<ROOT::RDF::RSampleInfo> fSampleInfos;
   unsigned int fNRuns{0}; ///< Number of event loops run

   /// Readers for TTree/RDataSource columns (one per slot), shared by all nodes in the computation graph.
   std::vector<std::unordered_map<std::string, std::shared_ptr<RColumnReaderBase>>> fDatasetColumnReaders;

   /// Cache of the tree/chain branch names. Never access directy, always use GetBranchNames().
   ColumnNames_t fValidBranchNames;

   ROOT::Internal::TreeUtils::RNoCleanupNotifier fNoCleanupNotifier;

   void RunEmptySourceMT();
   void RunEmptySource();
   void RunTreeProcessorMT();
   void RunTreeReader();
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

public:
   RLoopManager(TTree *tree, const ColumnNames_t &defaultBranches);
   RLoopManager(ULong64_t nEmptyEntries);
   RLoopManager(std::unique_ptr<RDataSource> ds, const ColumnNames_t &defaultBranches);
   RLoopManager(const RLoopManager &) = delete;
   RLoopManager &operator=(const RLoopManager &) = delete;

   void JitDeclarations();
   void Jit();
   RLoopManager *GetLoopManagerUnchecked() final { return this; }
   void Run();
   const ColumnNames_t &GetDefaultColumnNames() const;
   TTree *GetTree() const;
   ::TDirectory *GetDirectory() const;
   ULong64_t GetNEmptyEntries() const { return fNEmptyEntries; }
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
   void SetTree(std::shared_ptr<TTree> tree);
   void IncrChildrenCount() final { ++fNChildren; }
   void StopProcessing() final { ++fNStopsReceived; }
   void ToJitExec(const std::string &) const;
   void RegisterCallback(ULong64_t everyNEvents, std::function<void(unsigned int)> &&f);
   unsigned int GetNRuns() const { return fNRuns; }
   bool HasDataSourceColumnReaders(const std::string &col) const;
   void AddDataSourceColumnReaders(const std::string &col, std::vector<std::unique_ptr<RColumnReaderBase>> &&readers);
   std::shared_ptr<RColumnReaderBase> GetDatasetColumnReader(unsigned int slot, const std::string &col) const;

   /// End of recursive chain of calls, does nothing
   void AddFilterName(std::vector<std::string> &) {}
   /// For each booked filter, returns either the name or "Unnamed Filter"
   std::vector<std::string> GetFiltersNames();

   /// Return all graph edges known to RLoopManager
   /// This includes Filters and Ranges but not Defines.
   std::vector<RNodeBase *> GetGraphEdges() const;

   /// Return all actions, either booked or already run
   std::vector<RDFInternal::RActionBase *> GetAllActions() const;

   std::shared_ptr<ROOT::Internal::RDF::GraphDrawing::GraphNode>
   GetGraph(std::unordered_map<void *, std::shared_ptr<ROOT::Internal::RDF::GraphDrawing::GraphNode>> &visitedMap);

   const ColumnNames_t &GetBranchNames();

   void AddSampleCallback(ROOT::RDF::SampleCallback_t &&callback);
};

} // ns RDF
} // ns Detail
} // ns ROOT

#endif
