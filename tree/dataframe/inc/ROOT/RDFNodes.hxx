// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFNODES
#define ROOT_RDFNODES

#include "ROOT/GraphNode.hxx"
#include "ROOT/RActionBase.hxx"
#include "ROOT/RDFColumnValue.hxx"
#include "ROOT/RCustomColumnBase.hxx"
#include "ROOT/RDataSource.hxx"
#include "ROOT/RDFBookedCustomColumns.hxx"
#include "ROOT/RDFNodesUtils.hxx"
#include "ROOT/RDFUtils.hxx"
#include "ROOT/RIntegerSequence.hxx"
#include "ROOT/RLoopManager.hxx"
#include "ROOT/RMakeUnique.hxx"
#include "ROOT/RNodeBase.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/TypeTraits.hxx"
#include "TError.h"
#include "TTreeReaderArray.h"
#include "TTreeReaderValue.h"

#include <deque> // std::vector substitute in case of vector<bool>
#include <limits>
#include <memory>
#include <stack>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace ROOT {
namespace Internal {
namespace RDF {

namespace RDFDetail = ROOT::Detail::RDF;
namespace RDFGraphDrawing = ROOT::Internal::RDF::GraphDrawing;

// fwd decl for RAction, RFilter
namespace GraphDrawing {
std::shared_ptr<GraphNode>
CreateDefineNode(const std::string &columnName, const RDFDetail::RCustomColumnBase *columnPtr);
} // namespace GraphDrawing

// fwd decl for RAction
namespace GraphDrawing {
bool CheckIfDefaultOrDSColumn(const std::string &name, const std::shared_ptr<RDFDetail::RCustomColumnBase> &column);
} // ns GraphDrawing

class RJittedAction : public RActionBase {
private:
   std::unique_ptr<RActionBase> fConcreteAction;

public:
   RJittedAction(RLoopManager &lm) : RActionBase(&lm, lm.GetNSlots(), {}, RDFInternal::RBookedCustomColumns{}) {}

   void SetAction(std::unique_ptr<RActionBase> a) { fConcreteAction = std::move(a); }

   void Run(unsigned int slot, Long64_t entry) final;
   void Initialize() final;
   void InitSlot(TTreeReader *r, unsigned int slot) final;
   void TriggerChildrenCount() final;
   void FinalizeSlot(unsigned int) final;
   void Finalize() final;
   void *PartialUpdate(unsigned int slot) final;
   bool HasRun() const final;
   void ClearValueReaders(unsigned int slot) final;

   std::shared_ptr< ROOT::Internal::RDF::GraphDrawing::GraphNode> GetGraph();
};

template <typename Helper, typename PrevDataFrame, typename ColumnTypes_t = typename Helper::ColumnTypes_t>
class RAction final : public RActionBase {
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;

   Helper fHelper;
   const std::shared_ptr<PrevDataFrame> fPrevDataPtr;
   PrevDataFrame &fPrevData;
   std::vector<RDFValueTuple_t<ColumnTypes_t>> fValues;

public:
   RAction(Helper &&h, const ColumnNames_t &bl, std::shared_ptr<PrevDataFrame> pd,
           const RBookedCustomColumns &customColumns)
      : RActionBase(pd->GetLoopManagerUnchecked(), pd->GetLoopManagerUnchecked()->GetNSlots(), bl, customColumns),
        fHelper(std::forward<Helper>(h)), fPrevDataPtr(std::move(pd)), fPrevData(*fPrevDataPtr), fValues(fNSlots)
   {
   }

   RAction(const RAction &) = delete;
   RAction &operator=(const RAction &) = delete;

   void Initialize() final {
      fHelper.Initialize();
   }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      for (auto &bookedBranch : fCustomColumns.GetColumns())
         bookedBranch.second->InitSlot(r, slot);

      InitRDFValues(slot, fValues[slot], r, fColumnNames, fCustomColumns, TypeInd_t());
      fHelper.InitTask(r, slot);
   }

   void Run(unsigned int slot, Long64_t entry) final
   {
      // check if entry passes all filters
      if (fPrevData.CheckFilters(slot, entry))
         Exec(slot, entry, TypeInd_t());
   }

   template <std::size_t... S>
   void Exec(unsigned int slot, Long64_t entry, std::index_sequence<S...>)
   {
      (void)entry; // avoid bogus 'unused parameter' warning in gcc4.9
      fHelper.Exec(slot, std::get<S>(fValues[slot]).Get(entry)...);
   }

   void TriggerChildrenCount() final { fPrevData.IncrChildrenCount(); }

   void FinalizeSlot(unsigned int slot) final
   {
      ClearValueReaders(slot);
      for (auto &column : fCustomColumns.GetColumns()) {
         column.second->ClearValueReaders(slot);
      }
      fHelper.CallFinalizeTask(slot);
   }

   void ClearValueReaders(unsigned int slot) { ResetRDFValueTuple(fValues[slot], TypeInd_t()); }

   void Finalize() final
   {
      fHelper.Finalize();
      fHasRun = true;
   }

   std::shared_ptr<RDFGraphDrawing::GraphNode> GetGraph()
   {
      auto prevNode = fPrevData.GetGraph();
      auto prevColumns = prevNode->GetDefinedColumns();

      // Action nodes do not need to ask an helper to create the graph nodes. They are never common nodes between
      // multiple branches
      auto thisNode = std::make_shared< RDFGraphDrawing::GraphNode>(fHelper.GetActionName());
      auto evaluatedNode = thisNode;
      for (auto &column : fCustomColumns.GetColumns()) {
         /* Each column that this node has but the previous hadn't has been defined in between,
          * so it has to be built and appended. */
         if (RDFGraphDrawing::CheckIfDefaultOrDSColumn(column.first, column.second))
            continue;
         if (std::find(prevColumns.begin(), prevColumns.end(), column.first) == prevColumns.end()) {
            auto defineNode = RDFGraphDrawing::CreateDefineNode(column.first, column.second.get());
            evaluatedNode->SetPrevNode(defineNode);
            evaluatedNode = defineNode;
         }
      }

      thisNode->AddDefinedColumns(fCustomColumns.GetNames());
      thisNode->SetAction(HasRun());
      evaluatedNode->SetPrevNode(prevNode);
      return thisNode;
   }

   /// This method is invoked to update a partial result during the event loop, right before passing the result to a
   /// user-defined callback registered via RResultPtr::RegisterCallback
   void *PartialUpdate(unsigned int slot) final { return PartialUpdateImpl(slot); }

private:
   // this overload is SFINAE'd out if Helper does not implement `PartialUpdate`
   // the template parameter is required to defer instantiation of the method to SFINAE time
   template <typename H = Helper>
   auto PartialUpdateImpl(unsigned int slot) -> decltype(std::declval<H>().PartialUpdate(slot), (void *)(nullptr))
   {
      return &fHelper.PartialUpdate(slot);
   }
   // this one is always available but has lower precedence thanks to `...`
   void *PartialUpdateImpl(...) { throw std::runtime_error("This action does not support callbacks yet!"); }
};

} // namespace RDF
} // namespace Internal

namespace Detail {
namespace RDF {
namespace RDFGraphDrawing = ROOT::Internal::RDF::GraphDrawing;

// clang-format off
namespace CustomColExtraArgs {
struct None{};
struct Slot{};
struct SlotAndEntry{};
}
// clang-format on

template <typename F, typename ExtraArgsTag = CustomColExtraArgs::None>
class RCustomColumn final : public RCustomColumnBase {
   // shortcuts
   using NoneTag = CustomColExtraArgs::None;
   using SlotTag = CustomColExtraArgs::Slot;
   using SlotAndEntryTag = CustomColExtraArgs::SlotAndEntry;
   // other types
   using FunParamTypes_t = typename CallableTraits<F>::arg_types;
   using ColumnTypesTmp_t =
      RDFInternal::RemoveFirstParameterIf_t<std::is_same<ExtraArgsTag, SlotTag>::value, FunParamTypes_t>;
   using ColumnTypes_t =
      RDFInternal::RemoveFirstTwoParametersIf_t<std::is_same<ExtraArgsTag, SlotAndEntryTag>::value, ColumnTypesTmp_t>;
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;
   using ret_type = typename CallableTraits<F>::ret_type;
   // Avoid instantiating vector<bool> as `operator[]` returns temporaries in that case. Use std::deque instead.
   using ValuesPerSlot_t =
      typename std::conditional<std::is_same<ret_type, bool>::value, std::deque<ret_type>, std::vector<ret_type>>::type;

   F fExpression;
   const ColumnNames_t fBranches;
   ValuesPerSlot_t fLastResults;

   std::vector<RDFInternal::RDFValueTuple_t<ColumnTypes_t>> fValues;

public:
   RCustomColumn(RLoopManager *lm, std::string_view name, F &&expression, const ColumnNames_t &bl, unsigned int nSlots,
                 const RDFInternal::RBookedCustomColumns &customColumns, bool isDSColumn = false)
      : RCustomColumnBase(lm, name, nSlots, isDSColumn, customColumns), fExpression(std::forward<F>(expression)),
        fBranches(bl), fLastResults(fNSlots), fValues(fNSlots)
   {
   }

   RCustomColumn(const RCustomColumn &) = delete;
   RCustomColumn &operator=(const RCustomColumn &) = delete;


   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
       //TODO: Each node calls this method for each column it uses. Multiple nodes may share the same columns, and this would lead to this method being called multiple times.
      RDFInternal::InitRDFValues(slot, fValues[slot], r, fBranches, fCustomColumns, TypeInd_t());
   }

   void *GetValuePtr(unsigned int slot) final { return static_cast<void *>(&fLastResults[slot]); }

   void Update(unsigned int slot, Long64_t entry) final
   {
      if (entry != fLastCheckedEntry[slot]) {
         // evaluate this filter, cache the result
         UpdateHelper(slot, entry, TypeInd_t(), ColumnTypes_t(), ExtraArgsTag{});
         fLastCheckedEntry[slot] = entry;
      }
   }

   const std::type_info &GetTypeId() const
   {
      return fIsDataSourceColumn ? typeid(typename std::remove_pointer<ret_type>::type) : typeid(ret_type);
   }

   template <std::size_t... S, typename... BranchTypes>
   void UpdateHelper(unsigned int slot, Long64_t entry, std::index_sequence<S...>, TypeList<BranchTypes...>, NoneTag)
   {
      fLastResults[slot] = fExpression(std::get<S>(fValues[slot]).Get(entry)...);
      // silence "unused parameter" warnings in gcc
      (void)slot;
      (void)entry;
   }

   template <std::size_t... S, typename... BranchTypes>
   void UpdateHelper(unsigned int slot, Long64_t entry, std::index_sequence<S...>, TypeList<BranchTypes...>, SlotTag)
   {
      fLastResults[slot] = fExpression(slot, std::get<S>(fValues[slot]).Get(entry)...);
      // silence "unused parameter" warnings in gcc
      (void)slot;
      (void)entry;
   }

   template <std::size_t... S, typename... BranchTypes>
   void
   UpdateHelper(unsigned int slot, Long64_t entry, std::index_sequence<S...>, TypeList<BranchTypes...>, SlotAndEntryTag)
   {
      fLastResults[slot] = fExpression(slot, entry, std::get<S>(fValues[slot]).Get(entry)...);
      // silence "unused parameter" warnings in gcc
      (void)slot;
      (void)entry;
   }

   void ClearValueReaders(unsigned int slot) final
   {
      //TODO: Each node calls this method for each column it uses. Multiple nodes may share the same columns, and this would lead to this method being called multiple times.
      RDFInternal::ResetRDFValueTuple(fValues[slot], TypeInd_t());
   }
};

// fwd decl for RFilterBase
namespace RDF {
class RCutFlowReport;
} // ns RDF

} // ns RDF
} // ns Detail

namespace Internal {
namespace RDF {
namespace GraphDrawing {
std::shared_ptr<GraphNode> CreateFilterNode(const ROOT::Detail::RDF::RFilterBase *filterPtr);

bool CheckIfDefaultOrDSColumn(const std::string &name,
                              const std::shared_ptr<ROOT::Detail::RDF::RCustomColumnBase> &column);
} // ns GraphDrawing
} // ns RDF
} // ns Internal

namespace Detail {
namespace RDF {

class RFilterBase : public RNodeBase {
protected:
   std::vector<Long64_t> fLastCheckedEntry;
   std::vector<int> fLastResult = {true}; // std::vector<bool> cannot be used in a MT context safely
   std::vector<ULong64_t> fAccepted = {0};
   std::vector<ULong64_t> fRejected = {0};
   const std::string fName;
   const unsigned int fNSlots;      ///< Number of thread slots used by this node, inherited from parent node.

   RDFInternal::RBookedCustomColumns fCustomColumns;

public:
   RFilterBase(RLoopManager *df, std::string_view name, const unsigned int nSlots,
               const RDFInternal::RBookedCustomColumns &customColumns);
   RFilterBase &operator=(const RFilterBase &) = delete;
   virtual ~RFilterBase() { fLoopManager->Deregister(this); }

   virtual void InitSlot(TTreeReader *r, unsigned int slot) = 0;
   bool HasName() const;
   std::string GetName() const;
   virtual void FillReport(ROOT::RDF::RCutFlowReport &) const;
   virtual void TriggerChildrenCount() = 0;
   virtual void ResetReportCount()
   {
      R__ASSERT(!fName.empty()); // this method is to only be called on named filters
      // fAccepted and fRejected could be different than 0 if this is not the first event-loop run using this filter
      std::fill(fAccepted.begin(), fAccepted.end(), 0);
      std::fill(fRejected.begin(), fRejected.end(), 0);
   }
   virtual void ClearValueReaders(unsigned int slot) = 0;
   virtual void ClearTask(unsigned int slot) = 0;
   virtual void InitNode();
   virtual void AddFilterName(std::vector<std::string> &filters) = 0;
};

/// A wrapper around a concrete RFilter, which forwards all calls to it
/// RJittedFilter is the type of the node returned by jitted Filter calls: the concrete filter can be created and set
/// at a later time, from jitted code.
class RJittedFilter final : public RFilterBase {
   std::unique_ptr<RFilterBase> fConcreteFilter = nullptr;

public:
   RJittedFilter(RLoopManager *lm, std::string_view name)
      : RFilterBase(lm, name, lm->GetNSlots(), RDFInternal::RBookedCustomColumns())
   {
   }

   void SetFilter(std::unique_ptr<RFilterBase> f);

   void InitSlot(TTreeReader *r, unsigned int slot) final;
   bool CheckFilters(unsigned int slot, Long64_t entry) final;
   void Report(ROOT::RDF::RCutFlowReport &) const final;
   void PartialReport(ROOT::RDF::RCutFlowReport &) const final;
   void FillReport(ROOT::RDF::RCutFlowReport &) const final;
   void IncrChildrenCount() final;
   void StopProcessing() final;
   void ResetChildrenCount() final;
   void TriggerChildrenCount() final;
   void ResetReportCount() final;
   void ClearValueReaders(unsigned int slot) final;
   void InitNode() final;
   void AddFilterName(std::vector<std::string> &filters) final;
   void ClearTask(unsigned int slot) final;

   std::shared_ptr<RDFGraphDrawing::GraphNode> GetGraph(){
      if(fConcreteFilter != nullptr ){
         //Here the filter exists, so it can be served
         return fConcreteFilter->GetGraph();
      }
      throw std::runtime_error("The Jitting should have been invoked before this method.");
   }

};

template <typename FilterF, typename PrevDataFrame>
class RFilter final : public RFilterBase {
   using ColumnTypes_t = typename CallableTraits<FilterF>::arg_types;
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;

   FilterF fFilter;
   const ColumnNames_t fBranches;
   const std::shared_ptr<PrevDataFrame> fPrevDataPtr;
   PrevDataFrame &fPrevData;
   std::vector<RDFInternal::RDFValueTuple_t<ColumnTypes_t>> fValues;

public:
   RFilter(FilterF &&f, const ColumnNames_t &bl, std::shared_ptr<PrevDataFrame> pd,
           const RDFInternal::RBookedCustomColumns &customColumns, std::string_view name = "")
      : RFilterBase(pd->GetLoopManagerUnchecked(), name, pd->GetLoopManagerUnchecked()->GetNSlots(), customColumns),
        fFilter(std::forward<FilterF>(f)), fBranches(bl), fPrevDataPtr(std::move(pd)), fPrevData(*fPrevDataPtr),
        fValues(fNSlots)
   {
   }

   RFilter(const RFilter &) = delete;
   RFilter &operator=(const RFilter &) = delete;

   bool CheckFilters(unsigned int slot, Long64_t entry) final
   {
      if (entry != fLastCheckedEntry[slot]) {
         if (!fPrevData.CheckFilters(slot, entry)) {
            // a filter upstream returned false, cache the result
            fLastResult[slot] = false;
         } else {
            // evaluate this filter, cache the result
            auto passed = CheckFilterHelper(slot, entry, TypeInd_t());
            passed ? ++fAccepted[slot] : ++fRejected[slot];
            fLastResult[slot] = passed;
         }
         fLastCheckedEntry[slot] = entry;
      }
      return fLastResult[slot];
   }

   template <std::size_t... S>
   bool CheckFilterHelper(unsigned int slot, Long64_t entry, std::index_sequence<S...>)
   {
      // silence "unused parameter" warnings in gcc
      (void)slot;
      (void)entry;
      return fFilter(std::get<S>(fValues[slot]).Get(entry)...);
   }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      for (auto &bookedBranch : fCustomColumns.GetColumns())
         bookedBranch.second->InitSlot(r, slot);
      RDFInternal::InitRDFValues(slot, fValues[slot], r, fBranches, fCustomColumns, TypeInd_t());
   }

   // recursive chain of `Report`s
   void Report(ROOT::RDF::RCutFlowReport &rep) const final { PartialReport(rep); }

   void PartialReport(ROOT::RDF::RCutFlowReport &rep) const final
   {
      fPrevData.PartialReport(rep);
      FillReport(rep);
   }

   void StopProcessing() final
   {
      ++fNStopsReceived;
      if (fNStopsReceived == fNChildren)
         fPrevData.StopProcessing();
   }

   void IncrChildrenCount() final
   {
      ++fNChildren;
      // propagate "children activation" upstream. named filters do the propagation via `TriggerChildrenCount`.
      if (fNChildren == 1 && fName.empty())
         fPrevData.IncrChildrenCount();
   }

   void TriggerChildrenCount() final
   {
      R__ASSERT(!fName.empty()); // this method is to only be called on named filters
      fPrevData.IncrChildrenCount();
   }

   virtual void ClearValueReaders(unsigned int slot) final
   {
      RDFInternal::ResetRDFValueTuple(fValues[slot], TypeInd_t());
   }

   void AddFilterName(std::vector<std::string> &filters)
   {
      fPrevData.AddFilterName(filters);
      auto name = (HasName() ? fName : "Unnamed Filter");
      filters.push_back(name);
   }

   virtual void ClearTask(unsigned int slot) final
   {
      for (auto &column : fCustomColumns.GetColumns()) {
         column.second->ClearValueReaders(slot);
      }

      ClearValueReaders(slot);
   }

   std::shared_ptr<RDFGraphDrawing::GraphNode> GetGraph(){
      // Recursively call for the previous node.
      auto prevNode = fPrevData.GetGraph();
      auto prevColumns = prevNode->GetDefinedColumns();

      auto thisNode = RDFGraphDrawing::CreateFilterNode(this);

      /* If the returned node is not new, there is no need to perform any other operation.
       * This is a likely scenario when building the entire graph in which branches share
       * some nodes. */
      if(!thisNode->GetIsNew()){
         return thisNode;
      }

      auto evaluatedNode = thisNode;
      /* Each column that this node has but the previous hadn't has been defined in between,
       * so it has to be built and appended. */

      for (auto &column: fCustomColumns.GetColumns()){
         // Even if treated as custom columns by the Dataframe, datasource columns must not be in the graph.
         if(RDFGraphDrawing::CheckIfDefaultOrDSColumn(column.first, column.second))
            continue;
         if(std::find(prevColumns.begin(), prevColumns.end(), column.first) == prevColumns.end()){
            auto defineNode = RDFGraphDrawing::CreateDefineNode(column.first, column.second.get());
            evaluatedNode->SetPrevNode(defineNode);
            evaluatedNode = defineNode;
         }
      }

      // Keep track of the columns defined up to this point.
      thisNode->AddDefinedColumns(fCustomColumns.GetNames());

      evaluatedNode->SetPrevNode(prevNode);
      return thisNode;
   }
};

} // ns RDF
} // ns Detail

// fwd decl for RRangeBase
namespace Internal {
namespace RDF {
namespace GraphDrawing {
std::shared_ptr<GraphNode> CreateRangeNode(const ROOT::Detail::RDF::RRangeBase *rangePtr);
} // namespace GraphDrawing
} // namespace RDF
} // namespace Internal

namespace Detail {
namespace RDF {

class RRangeBase : public RNodeBase {
protected:
   unsigned int fStart;
   unsigned int fStop;
   unsigned int fStride;
   Long64_t fLastCheckedEntry{-1};
   bool fLastResult{true};
   ULong64_t fNProcessedEntries{0};
   bool fHasStopped{false};         ///< True if the end of the range has been reached
   const unsigned int fNSlots;      ///< Number of thread slots used by this node, inherited from parent node.

   void ResetCounters();

public:
   RRangeBase(RLoopManager *implPtr, unsigned int start, unsigned int stop, unsigned int stride,
              const unsigned int nSlots);

   RRangeBase &operator=(const RRangeBase &) = delete;
   virtual ~RRangeBase() { fLoopManager->Deregister(this); }

   void InitNode() { ResetCounters(); }
   virtual std::shared_ptr<RDFGraphDrawing::GraphNode> GetGraph() = 0;
};

template <typename PrevData>
class RRange final : public RRangeBase {
   const std::shared_ptr<PrevData> fPrevDataPtr;
   PrevData &fPrevData;

public:
   RRange(unsigned int start, unsigned int stop, unsigned int stride, std::shared_ptr<PrevData> pd)
      : RRangeBase(pd->GetLoopManagerUnchecked(), start, stop, stride, pd->GetLoopManagerUnchecked()->GetNSlots()),
        fPrevDataPtr(std::move(pd)), fPrevData(*fPrevDataPtr)
   {
   }

   RRange(const RRange &) = delete;
   RRange &operator=(const RRange &) = delete;

   /// Ranges act as filters when it comes to selecting entries that downstream nodes should process
   bool CheckFilters(unsigned int slot, Long64_t entry) final
   {
      if (entry != fLastCheckedEntry) {
         if (fHasStopped)
            return false;
         if (!fPrevData.CheckFilters(slot, entry)) {
            // a filter upstream returned false, cache the result
            fLastResult = false;
         } else {
            // apply range filter logic, cache the result
            ++fNProcessedEntries;
            if (fNProcessedEntries <= fStart || (fStop > 0 && fNProcessedEntries > fStop) ||
                (fStride != 1 && fNProcessedEntries % fStride != 0))
               fLastResult = false;
            else
               fLastResult = true;
            if (fNProcessedEntries == fStop) {
               fHasStopped = true;
               fPrevData.StopProcessing();
            }
         }
         fLastCheckedEntry = entry;
      }
      return fLastResult;
   }

   // recursive chain of `Report`s
   // RRange simply forwards these calls to the previous node
   void Report(ROOT::RDF::RCutFlowReport &rep) const final { fPrevData.PartialReport(rep); }

   void PartialReport(ROOT::RDF::RCutFlowReport &rep) const final { fPrevData.PartialReport(rep); }

   void StopProcessing() final
   {
      ++fNStopsReceived;
      if (fNStopsReceived == fNChildren && !fHasStopped)
         fPrevData.StopProcessing();
   }

   void IncrChildrenCount() final
   {
      ++fNChildren;
      // propagate "children activation" upstream
      if (fNChildren == 1)
         fPrevData.IncrChildrenCount();
   }

   /// This function must be defined by all nodes, but only the filters will add their name
   void AddFilterName(std::vector<std::string> &filters) { fPrevData.AddFilterName(filters); }
   std::shared_ptr<RDFGraphDrawing::GraphNode> GetGraph()
   {
      // TODO: Ranges node have no information about custom columns, hence it is not possible now
      // if defines have been used before.
      auto prevNode = fPrevData.GetGraph();
      auto prevColumns = prevNode->GetDefinedColumns();

      auto thisNode = RDFGraphDrawing::CreateRangeNode(this);

      /* If the returned node is not new, there is no need to perform any other operation.
       * This is a likely scenario when building the entire graph in which branches share
       * some nodes. */
      if (!thisNode->GetIsNew()) {
         return thisNode;
      }
      thisNode->SetPrevNode(prevNode);

      // If there have been some defines before it, this node won't detect them.
      thisNode->AddDefinedColumns(prevColumns);

      return thisNode;
   }
};

} // namespace RDF
} // namespace Detail
} // namespace ROOT
#endif // ROOT_RDFNODES
