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

/// A type-erasing wrapper around RColumnReaderBase
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
                       ROOT::TypeTraits::TypeList<ColTypes...>, const RColumnReadersInfo &colInfo)
{
   // see RColumnReadersInfo for why we pass these arguments like this rather than directly as function arguments
   const auto &colNames = colInfo.fColNames;
   const auto &customCols = colInfo.fCustomCols;
   const bool *isCustomColumn = colInfo.fIsCustomColumn;
   const auto &DSValuePtrsMap = colInfo.fDSValuePtrsMap;

   const auto &customColMap = customCols.GetColumns();

   // Construct the column readers
   using expander = int[];
   int i = 0;
   (void)expander{
      (values.emplace_back(MakeColumnReader<ColTypes>(slot, isCustomColumn[i] ? customColMap.at(colNames[i]).get() : nullptr, r,
                                                      GetValuePtrsPtr(colNames[i], DSValuePtrsMap), colNames[i])),
       ++i)...,
      0};

   (void)slot; // avoid bogus 'unused parameter' warning
   (void)r;    // avoid bogus 'unused parameter' warning
}

/// This overload is specialized to act on RTypeErasedColumnValues instead of RColumnValues.
template <typename... ColTypes>
void ResetColumnReaders(std::vector<RTypeErasedColumnValue> &values, ROOT::TypeTraits::TypeList<ColTypes...>)
{
   using expander = int[];
   int i = 0;
   (void)expander{(values[i].Cast<ColTypes>()->Reset(), ++i)...};
   values.clear();
}

// Forward declarations
template <typename... ColTypes>
class SnapshotHelper;

template <typename... ColTypes>
class SnapshotHelperMT;

namespace RDFDetail = ROOT::Detail::RDF;
namespace RDFGraphDrawing = ROOT::Internal::RDF::GraphDrawing;

namespace GraphDrawing {
std::shared_ptr<GraphNode> CreateDefineNode(const std::string &colName, const RDFDetail::RCustomColumnBase *columnPtr);
} // namespace GraphDrawing

// helper functions and types
template <typename... ColTypes>
inline constexpr bool IsSnapshotHelper(SnapshotHelper<ColTypes...> *) { return true; }

template <typename... ColTypes>
inline constexpr bool IsSnapshotHelper(SnapshotHelperMT<ColTypes...> *) { return true; }

template <typename T>
inline constexpr bool IsSnapshotHelper(T *) { return false; }

template <typename Helper, typename ColumnTypes, bool IsSnapshot = IsSnapshotHelper(static_cast<Helper *>(nullptr))>
struct ActionImpl {
   using TypeInd_t = std::make_index_sequence<ColumnTypes::list_size>;
   using Values_t = RDFValueTuple_t<ColumnTypes>;

   static void InitColumnReaders(unsigned int slot, Values_t &values, TTreeReader *r, const ColumnNames_t &colNames,
                                 const RBookedCustomColumns &customCols,
                                 const std::array<bool, ColumnTypes::list_size> &isCustomColumn,
                                 const std::map<std::string, std::vector<void *>> &DSValuePtrs)
   {
      RDFInternal::RColumnReadersInfo info{colNames, customCols, isCustomColumn.data(), DSValuePtrs};
      RDFInternal::InitColumnReaders(slot, values, r, TypeInd_t{}, info);
   }

   template <std::size_t... S>
   static void CallExec(unsigned int slot, Long64_t entry, Helper &helper, Values_t &values, std::index_sequence<S...>)
   {
      helper.Exec(slot, std::get<S>(values)->Get(entry)...);
      (void)entry; // avoid bogus unused parameter warnings
   }

   static void Exec(unsigned int slot, Long64_t entry, Helper &helper, Values_t &values)
   {
      CallExec(slot, entry, helper, values, TypeInd_t{});
   }

   static void ResetColumnReaders(Values_t &values) { RDFInternal::ResetColumnReaders(values, TypeInd_t{}); }
};

template <typename Helper, typename ColumnTypes>
struct ActionImpl<Helper, ColumnTypes, true> {
   using TypeInd_t = std::make_index_sequence<ColumnTypes::list_size>;
   using Values_t = std::vector<RTypeErasedColumnValue>;

   static void InitColumnReaders(unsigned int slot, Values_t &values, TTreeReader *r, const ColumnNames_t &colNames,
                                 const RBookedCustomColumns &customCols,
                                 const std::array<bool, ColumnTypes::list_size> &isCustomColumn,
                                 const std::map<std::string, std::vector<void *>> &DSValuePtrs)
   {
      RDFInternal::RColumnReadersInfo info{colNames, customCols, isCustomColumn.data(), DSValuePtrs};
      RDFInternal::InitColumnReaders(slot, values, r, ColumnTypes{}, info);
   }

   template <std::size_t... S, typename... ColTypes>
   static void CallExec(unsigned int slot, Long64_t entry, Helper &helper, Values_t &values, std::index_sequence<S...>,
                        ROOT::TypeTraits::TypeList<ColTypes...>)
   {
      helper.Exec(slot, values[S].template Get<ColTypes>(entry)...);
      (void)entry; // avoid bogus unused parameter warnings
   }

   static void Exec(unsigned int slot, Long64_t entry, Helper &helper, Values_t &values)
   {
      CallExec(slot, entry, helper, values, TypeInd_t{}, ColumnTypes{});
   }

   static void ResetColumnReaders(Values_t &values) { RDFInternal::ResetColumnReaders(values, ColumnTypes{}); }
};

// clang-format off
/**
 * \class ROOT::Internal::RDF::RAction
 * \ingroup dataframe
 * \brief A RDataFrame node that produces a result
 * \tparam Helper The action helper type, which implements the concrete action logic (e.g. FillHelper, SnapshotHelper)
 * \tparam PrevDataFrame The type of the parent node in the computation graph
 * \tparam ColumnTypes_t A TypeList with the types of the input columns
 *
 */
// clang-format on
template <typename Helper, typename PrevDataFrame, typename ColumnTypes_t = typename Helper::ColumnTypes_t>
class RAction : public RActionBase {
   using ActionImpl_t = ActionImpl<Helper, ColumnTypes_t>;

   Helper fHelper;
   const std::shared_ptr<PrevDataFrame> fPrevDataPtr;
   PrevDataFrame &fPrevData;
   std::vector<typename ActionImpl_t::Values_t> fValues;

   /// The nth flag signals whether the nth input column is a custom column or not.
   std::array<bool, ColumnTypes_t::list_size> fIsCustomColumn;

public:
   RAction(Helper &&h, const ColumnNames_t &columns, std::shared_ptr<PrevDataFrame> pd,
           const RBookedCustomColumns &customColumns)
      : RActionBase(pd->GetLoopManagerUnchecked(), columns, customColumns), fHelper(std::forward<Helper>(h)),
        fPrevDataPtr(std::move(pd)), fPrevData(*fPrevDataPtr), fValues(GetNSlots()), fIsCustomColumn()
   {
      const auto nColumns = columns.size();
      const auto &customCols = GetCustomColumns();
      for (auto i = 0u; i < nColumns; ++i)
         fIsCustomColumn[i] = customCols.HasName(columns[i]);
   }

   RAction(const RAction &) = delete;
   RAction &operator=(const RAction &) = delete;
   // must call Deregister here, before fPrevDataFrame is destroyed,
   // otherwise if fPrevDataFrame is fLoopManager we get a use after delete
   ~RAction() { fLoopManager->Deregister(this); }

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
      ActionImpl_t::InitColumnReaders(slot, fValues[slot], r, RActionBase::GetColumnNames(),
                                      RActionBase::GetCustomColumns(), fIsCustomColumn, fLoopManager->GetDSValuePtrs());
      fHelper.InitTask(r, slot);
   }

   void Run(unsigned int slot, Long64_t entry) final
   {
      // check if entry passes all filters
      if (fPrevData.CheckFilters(slot, entry))
         ActionImpl_t::Exec(slot, entry, fHelper, fValues[slot]);
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

   void ClearValueReaders(unsigned int slot) { ActionImpl_t::ResetColumnReaders(fValues[slot]); }

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

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif // ROOT_RACTION
