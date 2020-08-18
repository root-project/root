// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RACTION
#define ROOT_RACTION

#include "ROOT/RDF/ColumnReaders.hxx"
#include "ROOT/RDF/GraphNode.hxx"
#include "ROOT/RDF/RActionBase.hxx"
#include "ROOT/RDF/Utils.hxx" // ColumnNames_t, IsInternalColumn
#include "ROOT/RDF/RLoopManager.hxx"

#include <cstddef> // std::size_t
#include <memory>
#include <string>
#include <vector>

namespace ROOT {
namespace Internal {
namespace RDF {

namespace RDFDetail = ROOT::Detail::RDF;
namespace RDFGraphDrawing = ROOT::Internal::RDF::GraphDrawing;

// fwd declarations for RActionCRTP
namespace GraphDrawing {
std::shared_ptr<GraphNode> CreateDefineNode(const std::string &colName, const RDFDetail::RCustomColumnBase *columnPtr);
} // ns GraphDrawing

/// Unused, not instantiatable. Only the partial specialization RActionCRTP<RAction<...>> can be used.
template <typename Dummy>
class RActionCRTP {
   static_assert(sizeof(Dummy) < 0, "The unspecialized version of RActionCRTP should never be instantiated");
};

/// A type-erasing wrapper around RColumnValue.
/// Used to reduce compile time by avoiding instantiation of very large tuples and/or (std::get<N>...) fold expressions.
class R__CLING_PTRCHECK(off) RTypeErasedColumnValue {
   std::shared_ptr<void> fPtr;  // shared_ptr to take advantage of the type-erased custom deleter

public:
   template <typename T>
   RTypeErasedColumnValue(std::unique_ptr<RColumnReaderBase<T>> v) : fPtr(std::move(v))
   {
   }

   template <typename T>
   T &Get(ULong64_t e)
   {
      return static_cast<RColumnReaderBase<T> *>(fPtr.get())->Get(e);
   }

   template <typename T>
   RColumnReaderBase<T> *Cast()
   {
      return static_cast<RColumnReaderBase<T> *>(fPtr.get());
   }
};

inline const std::vector<void *> *
GetValuePtrsPtr(const std::string &colName, const std::map<std::string, std::vector<void *>> &DSValuePtrsMap)
{
   const auto DSValuePtrsIt = DSValuePtrsMap.find(colName);
   const std::vector<void *> *DSValuePtrsPtr = DSValuePtrsIt != DSValuePtrsMap.end() ? &DSValuePtrsIt->second : nullptr;
   return DSValuePtrsPtr;
}

/// This overload is specialized to act on RTypeErasedColumnValues instead of RColumnValues.
template <typename... ColTypes>
void InitColumnReaders(unsigned int slot, std::vector<RTypeErasedColumnValue> &values, TTreeReader *r,
                       const ColumnNames_t &bn, const RBookedCustomColumns &customCols,
                       ROOT::TypeTraits::TypeList<ColTypes...>,
                       const std::array<bool, sizeof...(ColTypes)> &isTmpColumn,
                       const std::map<std::string, std::vector<void *>> &DSValuePtrsMap)
{
   const auto &customColMap = customCols.GetColumns();

   // Construct the column readers
   using expander = int[];
   int i = 0;
   (void)expander{
      (values.emplace_back(MakeColumnReader<ColTypes>(slot, isTmpColumn[i] ? customColMap.at(bn[i]).get() : nullptr, r,
                                                      GetValuePtrsPtr(bn[i], DSValuePtrsMap), bn[i])),
       ++i)...,
      0};

   (void)slot; // avoid bogus 'unused parameter' warning
   (void)r;    // avoid bogus 'unused parameter' warning
}

/// This overload is specialized to act on RTypeErasedColumnValues instead of RColumnValues.
template <std::size_t... S, typename... ColTypes>
void ResetColumnReaders(std::vector<RTypeErasedColumnValue> &values, std::index_sequence<S...>,
                        ROOT::TypeTraits::TypeList<ColTypes...>)
{
   using expander = int[];
   (void)expander{(values[S].Cast<ColTypes>()->Reset(), 0)...};
   values.clear();
}

// fwd decl for RActionCRTP
template <typename Helper, typename PrevDataFrame, typename ColumnTypes_t>
class RAction;

/// A common template base class for all RActions. Avoids code repetition for specializations of RActions
/// for different helpers, implementing all of the common logic.
template <typename Helper, typename PrevDataFrame, typename ColumnTypes_t>
class RActionCRTP<RAction<Helper, PrevDataFrame, ColumnTypes_t>> : public RActionBase {
   using Action_t = RAction<Helper, PrevDataFrame, ColumnTypes_t>;

   Helper fHelper;
   const std::shared_ptr<PrevDataFrame> fPrevDataPtr;
   PrevDataFrame &fPrevData;

protected:
   /// The nth flag signals whether the nth input column is a custom column or not.
   std::array<bool, ColumnTypes_t::list_size> fIsCustomColumn;

public:
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;

   RActionCRTP(Helper &&h, const ColumnNames_t &columns, std::shared_ptr<PrevDataFrame> pd,
               const RBookedCustomColumns &customColumns)
      : RActionBase(pd->GetLoopManagerUnchecked(), columns, customColumns), fHelper(std::forward<Helper>(h)),
        fPrevDataPtr(std::move(pd)), fPrevData(*fPrevDataPtr), fIsCustomColumn()
   {
      const auto nColumns = columns.size();
      const auto &customCols = GetCustomColumns();
      for (auto i = 0u; i < nColumns; ++i)
         fIsCustomColumn[i] = customCols.HasName(columns[i]);
   }

   RActionCRTP(const RActionCRTP &) = delete;
   RActionCRTP &operator=(const RActionCRTP &) = delete;
   // must call Deregister here, before fPrevDataFrame is destroyed,
   // otherwise if fPrevDataFrame is fLoopManager we get a use after delete
   ~RActionCRTP() { fLoopManager->Deregister(this); }

   Helper &GetHelper() { return fHelper; }

   /**
      Retrieve a wrapper to the result of the action that knows how to merge
      with others of the same type.
   */
   std::unique_ptr<RDFDetail::RMergeableValueBase> GetMergeableValue() const final
   {
      return fHelper.GetMergeableValue();
   }

   void Initialize() final { fHelper.Initialize(); }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      for (auto &bookedBranch : GetCustomColumns().GetColumns())
         bookedBranch.second->InitSlot(r, slot);
      static_cast<Action_t *>(this)->InitColumnValues(r, slot);
      fHelper.InitTask(r, slot);
   }

   void Run(unsigned int slot, Long64_t entry) final
   {
      // check if entry passes all filters
      if (fPrevData.CheckFilters(slot, entry))
         static_cast<Action_t *>(this)->Exec(slot, entry, TypeInd_t());
   }

   void TriggerChildrenCount() final { fPrevData.IncrChildrenCount(); }

   void FinalizeSlot(unsigned int slot) final
   {
      ClearValueReaders(slot);
      for (auto &column : GetCustomColumns().GetColumns()) {
         column.second->ClearValueReaders(slot);
      }
      fHelper.CallFinalizeTask(slot);
   }

   void ClearValueReaders(unsigned int slot) { static_cast<Action_t *>(this)->ResetColumnValues(slot, TypeInd_t()); }

   void Finalize() final
   {
      fHelper.Finalize();
      SetHasRun();
   }

   std::shared_ptr<RDFGraphDrawing::GraphNode> GetGraph()
   {
      auto prevNode = fPrevData.GetGraph();
      auto prevColumns = prevNode->GetDefinedColumns();

      // Action nodes do not need to ask an helper to create the graph nodes. They are never common nodes between
      // multiple branches
      auto thisNode = std::make_shared<RDFGraphDrawing::GraphNode>(fHelper.GetActionName());
      auto evaluatedNode = thisNode;
      for (auto &column : GetCustomColumns().GetColumns()) {
         /* Each column that this node has but the previous hadn't has been defined in between,
          * so it has to be built and appended. */
         if (RDFInternal::IsInternalColumn(column.first))
            continue;
         if (std::find(prevColumns.begin(), prevColumns.end(), column.first) == prevColumns.end()) {
            auto defineNode = RDFGraphDrawing::CreateDefineNode(column.first, column.second.get());
            evaluatedNode->SetPrevNode(defineNode);
            evaluatedNode = defineNode;
         }
      }

      thisNode->AddDefinedColumns(GetCustomColumns().GetNames());
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
   void *PartialUpdateImpl(...) { throw std::runtime_error("This action does not support callbacks!"); }
};

/// An action node in a RDF computation graph.
template <typename Helper, typename PrevDataFrame, typename ColumnTypes_t = typename Helper::ColumnTypes_t>
class RAction final : public RActionCRTP<RAction<Helper, PrevDataFrame, ColumnTypes_t>> {
   std::vector<RDFValueTuple_t<ColumnTypes_t>> fValues;

public:
   using ActionCRTP_t = RActionCRTP<RAction<Helper, PrevDataFrame, ColumnTypes_t>>;

   RAction(Helper &&h, const ColumnNames_t &bl, std::shared_ptr<PrevDataFrame> pd,
           const RBookedCustomColumns &customColumns)
      : ActionCRTP_t(std::forward<Helper>(h), bl, std::move(pd), customColumns), fValues(GetNSlots())
   {
   }

   void InitColumnValues(TTreeReader *r, unsigned int slot)
   {
      InitColumnReaders(slot, fValues[slot], r, RActionBase::GetColumnNames(), RActionBase::GetCustomColumns(),
                        typename ActionCRTP_t::TypeInd_t{}, ActionCRTP_t::fIsCustomColumn,
                        ActionCRTP_t::fLoopManager->GetDSValuePtrs());
   }

   template <std::size_t... S>
   void Exec(unsigned int slot, Long64_t entry, std::index_sequence<S...>)
   {
      (void)entry; // avoid bogus 'unused parameter' warning in gcc4.9
      ActionCRTP_t::GetHelper().Exec(slot, std::get<S>(fValues[slot])->Get(entry)...);
   }

   template <std::size_t... S>
   void ResetColumnValues(unsigned int slot, std::index_sequence<S...> s)
   {
      ResetColumnReaders(fValues[slot], s);
   }
};

// These specializations let RAction<SnapshotHelper[MT]> type-erase their column values, for (presumably) a small hit in
// performance (which hopefully be completely swallowed by the cost of I/O during the event loop) and a large,
// measurable gain in compile time and therefore jitting time.
// Snapshot is the action that most suffers from long compilation times because it happens to be called with dozens
// if not with a few hundred template parameters, which pretty much never happens for other actions.

// fwd decl
template <typename... BranchTypes>
class SnapshotHelper;

template <typename... BranchTypes>
class SnapshotHelperMT;

template <typename PrevDataFrame, typename... ColTypes>
class RAction<SnapshotHelper<ColTypes...>, PrevDataFrame, ROOT::TypeTraits::TypeList<ColTypes...>> final
   : public RActionCRTP<RAction<SnapshotHelper<ColTypes...>, PrevDataFrame, ROOT::TypeTraits::TypeList<ColTypes...>>> {

   using ActionCRTP_t =
      RActionCRTP<RAction<SnapshotHelper<ColTypes...>, PrevDataFrame, ROOT::TypeTraits::TypeList<ColTypes...>>>;
   using ColumnTypes_t = typename SnapshotHelper<ColTypes...>::ColumnTypes_t;

   std::vector<std::vector<RTypeErasedColumnValue>> fValues;

public:
   RAction(SnapshotHelper<ColTypes...> &&h, const ColumnNames_t &bl, std::shared_ptr<PrevDataFrame> pd,
           const RBookedCustomColumns &customColumns)
      : ActionCRTP_t(std::forward<SnapshotHelper<ColTypes...>>(h), bl, std::move(pd), std::move(customColumns)),
        fValues(GetNSlots())
   {
      for (auto &v : fValues)
         v.reserve(sizeof...(ColTypes));
   }

   void InitColumnValues(TTreeReader *r, unsigned int slot)
   {
      InitColumnReaders(slot, fValues[slot], r, RActionBase::GetColumnNames(), RActionBase::GetCustomColumns(),
                        ColumnTypes_t{}, ActionCRTP_t::fIsCustomColumn, ActionCRTP_t::fLoopManager->GetDSValuePtrs());
   }

   template <std::size_t... S>
   void Exec(unsigned int slot, Long64_t entry, std::index_sequence<S...>)
   {
      (void)entry; // avoid bogus 'unused parameter' warning in gcc4.9
      ActionCRTP_t::GetHelper().Exec(slot, fValues[slot][S].template Get<ColTypes>(entry)...);
   }

   template <std::size_t... S>
   void ResetColumnValues(unsigned int slot, std::index_sequence<S...> s)
   {
      ResetColumnReaders(fValues[slot], s, ColumnTypes_t{});
   }
};

// Same exact code as above, but for SnapshotHelperMT. I don't know how to avoid repeating this code
template <typename PrevDataFrame, typename... ColTypes>
class RAction<SnapshotHelperMT<ColTypes...>, PrevDataFrame, ROOT::TypeTraits::TypeList<ColTypes...>> final
   : public RActionCRTP<
        RAction<SnapshotHelperMT<ColTypes...>, PrevDataFrame, ROOT::TypeTraits::TypeList<ColTypes...>>> {

   using ActionCRTP_t =
      RActionCRTP<RAction<SnapshotHelperMT<ColTypes...>, PrevDataFrame, ROOT::TypeTraits::TypeList<ColTypes...>>>;
   using ColumnTypes_t = typename SnapshotHelperMT<ColTypes...>::ColumnTypes_t;

   std::vector<std::vector<RTypeErasedColumnValue>> fValues;

public:
   RAction(SnapshotHelperMT<ColTypes...> &&h, const ColumnNames_t &bl, std::shared_ptr<PrevDataFrame> pd,
           const RBookedCustomColumns &customColumns)
      : ActionCRTP_t(std::forward<SnapshotHelperMT<ColTypes...>>(h), bl, std::move(pd), std::move(customColumns)),
        fValues(GetNSlots())
   {
      for (auto &v : fValues)
         v.reserve(sizeof...(ColTypes));
   }

   void InitColumnValues(TTreeReader *r, unsigned int slot)
   {
      InitColumnReaders(slot, fValues[slot], r, RActionBase::GetColumnNames(), RActionBase::GetCustomColumns(),
                        ColumnTypes_t{}, ActionCRTP_t::fIsCustomColumn, ActionCRTP_t::fLoopManager->GetDSValuePtrs());
   }

   template <std::size_t... S>
   void Exec(unsigned int slot, Long64_t entry, std::index_sequence<S...>)
   {
      (void)entry; // avoid bogus 'unused parameter' warning in gcc4.9
      ActionCRTP_t::GetHelper().Exec(slot, fValues[slot][S].template Get<ColTypes>(entry)...);
   }

   template <std::size_t... S>
   void ResetColumnValues(unsigned int slot, std::index_sequence<S...> s)
   {
      ResetColumnReaders(fValues[slot], s, ColumnTypes_t{});
   }
};

} // ns RDF
} // ns Internal
} // ns ROOT

#endif // ROOT_RACTION
