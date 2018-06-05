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

#ifndef NDEBUG
#include "TError.h"
#endif
#include "ROOT/RIntegerSequence.hxx"
#include "ROOT/TypeTraits.hxx"
#include "ROOT/RCutFlowReport.hxx"
#include "ROOT/RDataSource.hxx"
#include "ROOT/RDFNodesUtils.hxx"
#include "ROOT/RDFUtils.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/TSpinMutex.hxx"
#include "TTreeReaderArray.h"
#include "TTreeReaderValue.h"
#include "TError.h"

#include <map>
#include <numeric> // std::accumulate (FillReport), std::iota (TSlotStack)
#include <string>
#include <tuple>
#include <cassert>
#include <climits>
#include <deque> // std::vector substitute in case of vector<bool>
#include <functional>

namespace ROOT {
namespace Internal {
namespace RDF {
class RActionBase;

// This is an helper class to allow to pick a slot without resorting to a map
// indexed by thread ids.
// WARNING: this class does not work as a regular stack. The size is
// fixed at construction time and no blocking is foreseen.
class TSlotStack {
private:
   unsigned int &GetCount()
   {
      TTHREAD_TLS(unsigned int) count = 0U;
      return count;
   }
   unsigned int &GetIndex()
   {
      TTHREAD_TLS(unsigned int) index = UINT_MAX;
      return index;
   }
   unsigned int fCursor;
   std::vector<unsigned int> fBuf;
   ROOT::TSpinMutex fMutex;

public:
   TSlotStack() = delete;
   TSlotStack(unsigned int size) : fCursor(size), fBuf(size) { std::iota(fBuf.begin(), fBuf.end(), 0U); }
   void ReturnSlot(unsigned int slotNumber);
   unsigned int GetSlot();
};
} // ns RDF
} // ns Internal

namespace Detail {
namespace RDF {
using namespace ROOT::TypeTraits;
namespace RDFInternal = ROOT::Internal::RDF;

// forward declarations for RLoopManager
using ActionBasePtr_t = std::shared_ptr<RDFInternal::RActionBase>;
using ActionBaseVec_t = std::vector<ActionBasePtr_t>;
class RCustomColumnBase;
using RCustomColumnBasePtr_t = std::shared_ptr<RCustomColumnBase>;
class RFilterBase;
using FilterBasePtr_t = std::shared_ptr<RFilterBase>;
using FilterBaseVec_t = std::vector<FilterBasePtr_t>;
class RRangeBase;
using RangeBasePtr_t = std::shared_ptr<RRangeBase>;
using RangeBaseVec_t = std::vector<RangeBasePtr_t>;

class RLoopManager {
   using RDataSource = ROOT::RDF::RDataSource;
   enum class ELoopType { kROOTFiles, kROOTFilesMT, kNoFiles, kNoFilesMT, kDataSource, kDataSourceMT };

   using Callback_t = std::function<void(unsigned int)>;
   class TCallback {
      const Callback_t fFun;
      const ULong64_t fEveryN;
      std::vector<ULong64_t> fCounters;

   public:
      TCallback(ULong64_t everyN, Callback_t &&f, unsigned int nSlots)
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

   class TOneTimeCallback {
      const Callback_t fFun;
      std::vector<int> fHasBeenCalled; // std::vector<bool> is thread-unsafe for our purposes (and generally evil)

   public:
      TOneTimeCallback(Callback_t &&f, unsigned int nSlots) : fFun(std::move(f)), fHasBeenCalled(nSlots, 0) {}

      void operator()(unsigned int slot)
      {
         if (fHasBeenCalled[slot] == 1)
            return;
         fFun(slot);
         fHasBeenCalled[slot] = 1;
      }
   };

   ActionBaseVec_t fBookedActions;
   FilterBaseVec_t fBookedFilters;
   FilterBaseVec_t fBookedNamedFilters; ///< Contains a subset of fBookedFilters, i.e. only the named filters
   std::map<std::string, RCustomColumnBasePtr_t> fBookedCustomColumns;
   ColumnNames_t fCustomColumnNames; ///< Contains names of all custom columns defined in the functional graph.
   RangeBaseVec_t fBookedRanges;
   std::vector<std::shared_ptr<bool>> fResProxyReadiness;
   ::TDirectory *const fDirPtr{nullptr};
   std::shared_ptr<TTree> fTree{nullptr}; //< Shared pointer to the input TTree/TChain. It does not own the pointee if
   // the TTree/TChain was passed directly as an argument to RDataFrame's ctor (in
   // which case we let users retain ownership).
   const ColumnNames_t fDefaultColumns;
   const ULong64_t fNEmptyEntries{0};
   const unsigned int fNSlots{1};
   bool fMustRunNamedFilters{true};
   unsigned int fNChildren{0};      ///< Number of nodes of the functional graph hanging from this object
   unsigned int fNStopsReceived{0}; ///< Number of times that a children node signaled to stop processing entries.
   const ELoopType fLoopType; ///< The kind of event loop that is going to be run (e.g. on ROOT files, on no files)
   std::string fToJit;        ///< string containing all `BuildAndBook` actions that should be jitted before running
   const std::unique_ptr<RDataSource> fDataSource; ///< Owning pointer to a data-source object. Null if no data-source
   ColumnNames_t fDefinedDataSourceColumns;        ///< List of data-source columns that have been `Define`d so far
   std::map<std::string, std::string> fAliasColumnNameMap; ///< ColumnNameAlias-columnName pairs
   std::vector<TCallback> fCallbacks;                      ///< Registered callbacks
   std::vector<TOneTimeCallback> fCallbacksOnce; ///< Registered callbacks to invoke just once before running the loop
   /// A unique ID that identifies the computation graph that starts with this RLoopManager.
   /// Used, for example, to jit objects in a namespace reserved for this computation graph
   const unsigned int fID = GetNextID();

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
   void CleanUpTask(unsigned int slot);
   void JitActions();
   void EvalChildrenCounts();
   unsigned int GetNextID() const;

public:
   RLoopManager(TTree *tree, const ColumnNames_t &defaultBranches);
   RLoopManager(ULong64_t nEmptyEntries);
   RLoopManager(std::unique_ptr<RDataSource> ds, const ColumnNames_t &defaultBranches);
   RLoopManager(const RLoopManager &) = delete;
   RLoopManager &operator=(const RLoopManager &) = delete;

   void Run();
   RLoopManager *GetLoopManagerUnchecked();
   const ColumnNames_t &GetDefaultColumnNames() const;
   const ColumnNames_t &GetCustomColumnNames() const { return fCustomColumnNames; };
   TTree *GetTree() const;
   const std::map<std::string, RCustomColumnBasePtr_t> &GetBookedColumns() const { return fBookedCustomColumns; }
   ::TDirectory *GetDirectory() const;
   ULong64_t GetNEmptyEntries() const { return fNEmptyEntries; }
   RDataSource *GetDataSource() const { return fDataSource.get(); }
   void Book(const ActionBasePtr_t &actionPtr);
   void Book(const FilterBasePtr_t &filterPtr);
   void Book(const RCustomColumnBasePtr_t &branchPtr);
   void Book(const std::shared_ptr<bool> &branchPtr);
   void Book(const RangeBasePtr_t &rangePtr);
   bool CheckFilters(int, unsigned int);
   unsigned int GetNSlots() const { return fNSlots; }
   bool MustRunNamedFilters() const { return fMustRunNamedFilters; }
   void Report(ROOT::RDF::RCutFlowReport &rep) const;
   /// End of recursive chain of calls, does nothing
   void PartialReport(ROOT::RDF::RCutFlowReport &) const {}
   void SetTree(const std::shared_ptr<TTree> &tree) { fTree = tree; }
   void IncrChildrenCount() { ++fNChildren; }
   void StopProcessing() { ++fNStopsReceived; }
   void ToJit(const std::string &s) { fToJit.append(s); }
   const ColumnNames_t &GetDefinedDataSourceColumns() const { return fDefinedDataSourceColumns; }
   void AddDataSourceColumn(std::string_view name) { fDefinedDataSourceColumns.emplace_back(name); }
   void AddColumnAlias(const std::string &alias, const std::string &colName) { fAliasColumnNameMap[alias] = colName; }
   void AddCustomColumnName(std::string_view name) { fCustomColumnNames.emplace_back(name); }
   const std::map<std::string, std::string> &GetAliasMap() const { return fAliasColumnNameMap; }
   void RegisterCallback(ULong64_t everyNEvents, std::function<void(unsigned int)> &&f);
   unsigned int GetID() const { return fID; }
};
} // end ns RDF
} // end ns Detail

namespace Internal {
namespace RDF {
using namespace ROOT::Detail::RDF;

/**
\class ROOT::Internal::RDF::TColumnValue
\ingroup dataframe
\brief Helper class that updates and returns TTree branches as well as RDataFrame temporary columns
\tparam T The type of the column

RDataFrame nodes must access two different types of values during the event loop:
values of real branches, for which TTreeReader{Values,Arrays} act as proxies, or
temporary columns whose values are generated on the fly. While the type of the
value is known at compile time (or just-in-time), it is only at runtime that nodes
can check whether a certain value is generated on the fly or not.

TColumnValue abstracts this difference by providing the same interface for
both cases and handling the reading or generation of new values transparently.
Only one of the two data members fReaderProxy or fValuePtr will be non-null
for a given TColumnValue, depending on whether the value comes from a real
TTree branch or from a temporary column respectively.

RDataFrame nodes can store tuples of TColumnValues and retrieve an updated
value for the column via the `Get` method.
**/
template <typename T, bool MustUseRVec = IsRVec_t<T>::value>
class TColumnValue {
   // ColumnValue_t is the type of the column or the type of the elements of an array column
   using ColumnValue_t = typename std::conditional<MustUseRVec, TakeFirstParameter_t<T>, T>::type;
   using TreeReader_t =
      typename std::conditional<MustUseRVec, TTreeReaderArray<ColumnValue_t>, TTreeReaderValue<ColumnValue_t>>::type;

   /// TColumnValue has a slightly different behaviour whether the column comes from a TTreeReader, a RDataFrame Define
   /// or a RDataSource. It stores which it is as an enum.
   enum class EColumnKind { kTree, kCustomColumn, kDataSource, kInvalid };
   // Set to the correct value by MakeProxy or SetTmpColumn
   EColumnKind fColumnKind = EColumnKind::kInvalid;
   /// The slot this value belongs to. Only needed when querying custom column values, it is set in `SetTmpColumn`.
   unsigned int fSlot = std::numeric_limits<unsigned int>::max();

   // Each element of the following data members will be in use by a _single task_.
   // The vectors are used as very small stacks (1-2 elements typically) that fill in case of interleaved task execution
   // i.e. when more than one task needs readers in this worker thread.

   /// Owning ptrs to a TTreeReaderValue or TTreeReaderArray. Only used for Tree columns.
   std::vector<std::unique_ptr<TreeReader_t>> fTreeReaders;
   /// Non-owning ptrs to the value of a custom column.
   std::vector<T *> fCustomValuePtrs;
   /// Non-owning ptrs to the value of a data-source column.
   std::vector<T **> fDSValuePtrs;
   /// Non-owning ptrs to the node responsible for the custom column. Needed when querying custom values.
   std::vector<RCustomColumnBase *> fCustomColumns;
   /// Enumerator for the different properties of the branch storage in memory
   enum class EStorageType : char { kContiguous, kUnknown, kSparse};
   /// Signal whether we ever checked that the branch we are reading with a TTreeReaderArray stores array elements
   /// in contiguous memory. Only used when T == RVec<U>.
   EStorageType fStorageType = EStorageType::kUnknown;
   /// If MustUseRVec, i.e. we are reading an array, we return a reference to this RVec to clients
   RVec<ColumnValue_t> fRVec;
   bool fCopyWarningPrinted = false;

public:
   static constexpr bool fgMustUseRVec = MustUseRVec;

   TColumnValue() = default;

   void SetTmpColumn(unsigned int slot, RCustomColumnBase *tmpColumn);

   void MakeProxy(TTreeReader *r, const std::string &bn)
   {
      fColumnKind = EColumnKind::kTree;
      fTreeReaders.emplace_back(new TreeReader_t(*r, bn.c_str()));
   }

   /// This overload is used to return scalar quantities (i.e. types that are not read into a RVec)
   template <typename U = T, typename std::enable_if<!TColumnValue<U>::fgMustUseRVec, int>::type = 0>
   T &Get(Long64_t entry);

   /// This overload is used to return arrays (i.e. types that are read into a RVec).
   /// In this case the returned T is always a RVec<ColumnValue_t>.
   template <typename U = T, typename std::enable_if<TColumnValue<U>::fgMustUseRVec, int>::type = 0>
   T &Get(Long64_t entry);

   void Reset()
   {
      switch (fColumnKind) {
      case EColumnKind::kTree: fTreeReaders.pop_back(); break;
      case EColumnKind::kCustomColumn:
         fCustomColumns.pop_back();
         fCustomValuePtrs.pop_back();
         break;
      case EColumnKind::kDataSource:
         fCustomColumns.pop_back();
         fDSValuePtrs.pop_back();
         break;
      case EColumnKind::kInvalid: throw std::runtime_error("ColumnKind not set for this TColumnValue");
      }
   }
};

template <typename T>
struct TRDFValueTuple {
};

template <typename... BranchTypes>
struct TRDFValueTuple<TypeList<BranchTypes...>> {
   using type = std::tuple<TColumnValue<BranchTypes>...>;
};

template <typename BranchType>
using RDFValueTuple_t = typename TRDFValueTuple<BranchType>::type;

/// Clear the proxies of a tuple of TColumnValues
template <typename ValueTuple, std::size_t... S>
void ResetRDFValueTuple(ValueTuple &values, std::index_sequence<S...>)
{
   // hack to expand a parameter pack without c++17 fold expressions.
   std::initializer_list<int> expander{(std::get<S>(values).Reset(), 0)...};
   (void)expander; // avoid "unused variable" warnings
}

class RActionBase {
protected:
   RLoopManager *fLoopManager; ///< A raw pointer to the RLoopManager at the root of this functional
                               /// graph. It is only guaranteed to contain a valid address during an
                               /// event loop.
   const unsigned int fNSlots; ///< Number of thread slots used by this node.

public:
   RActionBase(RLoopManager *implPtr, const unsigned int nSlots);
   RActionBase(const RActionBase &) = delete;
   RActionBase &operator=(const RActionBase &) = delete;
   virtual ~RActionBase() = default;

   virtual void Run(unsigned int slot, Long64_t entry) = 0;
   virtual void Initialize() = 0;
   virtual void InitSlot(TTreeReader *r, unsigned int slot) = 0;
   virtual void TriggerChildrenCount() = 0;
   virtual void ClearValueReaders(unsigned int slot) = 0;
   /// This method is invoked to update a partial result during the event loop, right before passing the result to a
   /// user-defined callback registered via RResultPtr::RegisterCallback
   virtual void *PartialUpdate(unsigned int slot) = 0;
};

template <typename Helper, typename PrevDataFrame, typename ColumnTypes_t = typename Helper::ColumnTypes_t>
class RAction final : public RActionBase {
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;

   Helper fHelper;
   const ColumnNames_t fBranches;
   PrevDataFrame &fPrevData;
   std::vector<RDFValueTuple_t<ColumnTypes_t>> fValues;

public:
   RAction(Helper &&h, const ColumnNames_t &bl, PrevDataFrame &pd)
      : RActionBase(pd.GetLoopManagerUnchecked(), pd.GetLoopManagerUnchecked()->GetNSlots()), fHelper(std::move(h)),
        fBranches(bl), fPrevData(pd), fValues(fNSlots)
   {
   }

   RAction(const RAction &) = delete;
   RAction &operator=(const RAction &) = delete;
   ~RAction() { fHelper.Finalize(); }

   void Initialize() final { fHelper.Initialize(); }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      InitRDFValues(slot, fValues[slot], r, fBranches, fLoopManager->GetCustomColumnNames(),
                    fLoopManager->GetBookedColumns(), TypeInd_t());
      fHelper.InitSlot(r, slot);
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

   virtual void ClearValueReaders(unsigned int slot) final { ResetRDFValueTuple(fValues[slot], TypeInd_t()); }

   /// This method is invoked to update a partial result during the event loop, right before passing the result to a
   /// user-defined callback registered via RResultPtr::RegisterCallback
   /// TODO the PartialUpdateImpl trick can go away once all action helpers will implement PartialUpdate
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

} // end NS RDF
} // end NS Internal

namespace Detail {
namespace RDF {

class RCustomColumnBase {
protected:
   RLoopManager *fLoopManager; ///< A raw pointer to the RLoopManager at the root of this functional graph. It is only
                               /// guaranteed to contain a valid address during an event loop.
   const std::string fName;
   unsigned int fNChildren{0};      ///< number of nodes of the functional graph hanging from this object
   unsigned int fNStopsReceived{0}; ///< number of times that a children node signaled to stop processing entries.
   const unsigned int fNSlots;      ///< number of thread slots used by this node, inherited from parent node.
   const bool fIsDataSourceColumn; ///< does the custom column refer to a data-source column? (or a user-define column?)
   std::vector<Long64_t> fLastCheckedEntry;

public:
   RCustomColumnBase(RLoopManager *df, std::string_view name, const unsigned int nSlots, const bool isDSColumn);
   RCustomColumnBase &operator=(const RCustomColumnBase &) = delete;
   virtual ~RCustomColumnBase(); // outlined defaulted.
   virtual void InitSlot(TTreeReader *r, unsigned int slot) = 0;
   virtual void *GetValuePtr(unsigned int slot) = 0;
   virtual const std::type_info &GetTypeId() const = 0;
   RLoopManager *GetLoopManagerUnchecked() const;
   std::string GetName() const;
   virtual void Update(unsigned int slot, Long64_t entry) = 0;
   virtual void ClearValueReaders(unsigned int slot) = 0;
   bool IsDataSourceColumn() const { return fIsDataSourceColumn; }
   void InitNode();
};

// clang-format off
namespace TCCHelperTypes {
struct TNothing;
struct TSlot;
struct TSlotAndEntry;
}
// clang-format on

template <typename F, typename UPDATE_HELPER_TYPE = TCCHelperTypes::TNothing>
class RCustomColumn final : public RCustomColumnBase {
   // shortcuts
   using TNothing = TCCHelperTypes::TNothing;
   using TSlot = TCCHelperTypes::TSlot;
   using TSlotAndEntry = TCCHelperTypes::TSlotAndEntry;
   using UHT_t = UPDATE_HELPER_TYPE;
   // other types
   using FunParamTypes_t = typename CallableTraits<F>::arg_types;
   using BranchTypesTmp_t =
      typename RDFInternal::RemoveFirstParameterIf<std::is_same<TSlot, UHT_t>::value, FunParamTypes_t>::type;
   using ColumnTypes_t = typename RDFInternal::RemoveFirstTwoParametersIf<std::is_same<TSlotAndEntry, UHT_t>::value,
                                                                          BranchTypesTmp_t>::type;
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
   RCustomColumn(std::string_view name, F &&expression, const ColumnNames_t &bl, RLoopManager *lm,
                 bool isDSColumn = false)
      : RCustomColumnBase(lm, name, lm->GetNSlots(), isDSColumn), fExpression(std::move(expression)), fBranches(bl),
        fLastResults(fNSlots), fValues(fNSlots)
   {
   }

   RCustomColumn(const RCustomColumn &) = delete;
   RCustomColumn &operator=(const RCustomColumn &) = delete;

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      RDFInternal::InitRDFValues(slot, fValues[slot], r, fBranches, fLoopManager->GetCustomColumnNames(),
                                 fLoopManager->GetBookedColumns(), TypeInd_t());
   }

   void *GetValuePtr(unsigned int slot) final { return static_cast<void *>(&fLastResults[slot]); }

   void Update(unsigned int slot, Long64_t entry) final
   {
      if (entry != fLastCheckedEntry[slot]) {
         // evaluate this filter, cache the result
         UpdateHelper(slot, entry, TypeInd_t(), ColumnTypes_t(), (UPDATE_HELPER_TYPE *)nullptr);
         fLastCheckedEntry[slot] = entry;
      }
   }

   const std::type_info &GetTypeId() const
   {
      return fIsDataSourceColumn ? typeid(typename std::remove_pointer<ret_type>::type) : typeid(ret_type);
   }

   template <std::size_t... S, typename... BranchTypes>
   void UpdateHelper(unsigned int slot, Long64_t entry, std::index_sequence<S...>, TypeList<BranchTypes...>,
                     TCCHelperTypes::TNothing *)
   {
      fLastResults[slot] = fExpression(std::get<S>(fValues[slot]).Get(entry)...);
      // silence "unused parameter" warnings in gcc
      (void)slot;
      (void)entry;
   }

   template <std::size_t... S, typename... BranchTypes>
   void UpdateHelper(unsigned int slot, Long64_t entry, std::index_sequence<S...>, TypeList<BranchTypes...>,
                     TCCHelperTypes::TSlot *)
   {
      fLastResults[slot] = fExpression(slot, std::get<S>(fValues[slot]).Get(entry)...);
      // silence "unused parameter" warnings in gcc
      (void)slot;
      (void)entry;
   }

   template <std::size_t... S, typename... BranchTypes>
   void UpdateHelper(unsigned int slot, Long64_t entry, std::index_sequence<S...>, TypeList<BranchTypes...>,
                     TCCHelperTypes::TSlotAndEntry *)
   {
      fLastResults[slot] = fExpression(slot, entry, std::get<S>(fValues[slot]).Get(entry)...);
      // silence "unused parameter" warnings in gcc
      (void)slot;
      (void)entry;
   }

   void ClearValueReaders(unsigned int slot) final { RDFInternal::ResetRDFValueTuple(fValues[slot], TypeInd_t()); }
};

class RFilterBase {
protected:
   RLoopManager *fLoopManager; ///< A raw pointer to the RLoopManager at the root of this functional graph. It is only
                               /// guaranteed to contain a valid address during an event loop.
   std::vector<Long64_t> fLastCheckedEntry;
   std::vector<int> fLastResult = {true}; // std::vector<bool> cannot be used in a MT context safely
   std::vector<ULong64_t> fAccepted = {0};
   std::vector<ULong64_t> fRejected = {0};
   const std::string fName;
   unsigned int fNChildren{0};      ///< Number of nodes of the functional graph hanging from this object
   unsigned int fNStopsReceived{0}; ///< Number of times that a children node signaled to stop processing entries.
   const unsigned int fNSlots;      ///< Number of thread slots used by this node, inherited from parent node.

public:
   RFilterBase(RLoopManager *df, std::string_view name, const unsigned int nSlots);
   RFilterBase &operator=(const RFilterBase &) = delete;
   virtual ~RFilterBase() = default;

   virtual void InitSlot(TTreeReader *r, unsigned int slot) = 0;
   virtual bool CheckFilters(unsigned int slot, Long64_t entry) = 0;
   virtual void Report(ROOT::RDF::RCutFlowReport &) const = 0;
   virtual void PartialReport(ROOT::RDF::RCutFlowReport &) const = 0;
   RLoopManager *GetLoopManagerUnchecked() const;
   bool HasName() const;
   virtual void FillReport(ROOT::RDF::RCutFlowReport &) const;
   virtual void IncrChildrenCount() = 0;
   virtual void StopProcessing() = 0;
   virtual void ResetChildrenCount()
   {
      fNChildren = 0;
      fNStopsReceived = 0;
   }
   virtual void TriggerChildrenCount() = 0;
   virtual void ResetReportCount()
   {
      assert(!fName.empty()); // this method is to only be called on named filters
      // fAccepted and fRejected could be different than 0 if this is not the first event-loop run using this filter
      std::fill(fAccepted.begin(), fAccepted.end(), 0);
      std::fill(fRejected.begin(), fRejected.end(), 0);
   }
   virtual void ClearValueReaders(unsigned int slot) = 0;
   virtual void InitNode();
};

/// A wrapper around a concrete RFilter, which forwards all calls to it
/// RJittedFilter is the type of the node returned by jitted Filter calls: the concrete filter can be created and set
/// at a later time, from jitted code.
// FIXME after switching to the new ownership model, RJittedFilter should avoid inheriting from RFilterBase and
// overriding all of its methods: it can just implement them, and RFilterBase's can go back to have non-virtual methods
class RJittedFilter final : public RFilterBase {
   std::unique_ptr<RFilterBase> fConcreteFilter = nullptr;

public:
   RJittedFilter(RLoopManager *lm, std::string_view name) : RFilterBase(lm, name, lm->GetNSlots()) {}

   void SetFilter(std::unique_ptr<RFilterBase> f);

   void InitSlot(TTreeReader *r, unsigned int slot) override final;
   bool CheckFilters(unsigned int slot, Long64_t entry) override final;
   void Report(ROOT::RDF::RCutFlowReport &) const override final;
   void PartialReport(ROOT::RDF::RCutFlowReport &) const override final;
   void FillReport(ROOT::RDF::RCutFlowReport &) const override final;
   void IncrChildrenCount() override final;
   void StopProcessing() override final;
   void ResetChildrenCount() override final;
   void TriggerChildrenCount() override final;
   void ResetReportCount() override final;
   void ClearValueReaders(unsigned int slot) override final;
   void InitNode() override final;
};

template <typename FilterF, typename PrevDataFrame>
class RFilter final : public RFilterBase {
   using ColumnTypes_t = typename CallableTraits<FilterF>::arg_types;
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;

   FilterF fFilter;
   const ColumnNames_t fBranches;
   PrevDataFrame &fPrevData;
   std::vector<RDFInternal::RDFValueTuple_t<ColumnTypes_t>> fValues;

public:
   RFilter(FilterF &&f, const ColumnNames_t &bl, PrevDataFrame &pd, std::string_view name = "")
      : RFilterBase(pd.GetLoopManagerUnchecked(), name, pd.GetLoopManagerUnchecked()->GetNSlots()),
        fFilter(std::move(f)), fBranches(bl), fPrevData(pd), fValues(fNSlots)
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
      return fFilter(std::get<S>(fValues[slot]).Get(entry)...);
      // silence "unused parameter" warnings in gcc
      (void)slot;
      (void)entry;
   }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      RDFInternal::InitRDFValues(slot, fValues[slot], r, fBranches, fLoopManager->GetCustomColumnNames(),
                                 fLoopManager->GetBookedColumns(), TypeInd_t());
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
      assert(!fName.empty()); // this method is to only be called on named filters
      fPrevData.IncrChildrenCount();
   }

   virtual void ClearValueReaders(unsigned int slot) final
   {
      RDFInternal::ResetRDFValueTuple(fValues[slot], TypeInd_t());
   }
};

class RRangeBase {
protected:
   RLoopManager *fLoopManager; ///< A raw pointer to the RLoopManager at the root of this functional graph. It is only
                               /// guaranteed to contain a valid address during an event loop.
   unsigned int fStart;
   unsigned int fStop;
   unsigned int fStride;
   Long64_t fLastCheckedEntry{-1};
   bool fLastResult{true};
   ULong64_t fNProcessedEntries{0};
   unsigned int fNChildren{0};      ///< Number of nodes of the functional graph hanging from this object
   unsigned int fNStopsReceived{0}; ///< Number of times that a children node signaled to stop processing entries.
   bool fHasStopped{false};         ///< True if the end of the range has been reached
   const unsigned int fNSlots;      ///< Number of thread slots used by this node, inherited from parent node.

   void ResetCounters();

public:
   RRangeBase(RLoopManager *implPtr, unsigned int start, unsigned int stop, unsigned int stride,
              const unsigned int nSlots);
   RRangeBase &operator=(const RRangeBase &) = delete;
   virtual ~RRangeBase() = default;

   RLoopManager *GetLoopManagerUnchecked() const;
   virtual bool CheckFilters(unsigned int slot, Long64_t entry) = 0;
   virtual void Report(ROOT::RDF::RCutFlowReport &) const = 0;
   virtual void PartialReport(ROOT::RDF::RCutFlowReport &) const = 0;
   virtual void IncrChildrenCount() = 0;
   virtual void StopProcessing() = 0;
   void ResetChildrenCount()
   {
      fNChildren = 0;
      fNStopsReceived = 0;
   }
   void InitNode() { ResetCounters(); }
};

template <typename PrevData>
class RRange final : public RRangeBase {
   PrevData &fPrevData;

public:
   RRange(unsigned int start, unsigned int stop, unsigned int stride, PrevData &pd)
      : RRangeBase(pd.GetLoopManagerUnchecked(), start, stop, stride, pd.GetLoopManagerUnchecked()->GetNSlots()),
        fPrevData(pd)
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
};

} // namespace RDF
} // namespace Detail

// method implementations
namespace Internal {
namespace RDF {

template <typename T, bool B>
void TColumnValue<T, B>::SetTmpColumn(unsigned int slot, ROOT::Detail::RDF::RCustomColumnBase *customColumn)
{
   fCustomColumns.emplace_back(customColumn);
   if (customColumn->GetTypeId() != typeid(T))
      throw std::runtime_error(
         std::string("TColumnValue: type specified for column \"" + customColumn->GetName() + "\" is ") +
         TypeID2TypeName(typeid(T)) + " but temporary column has type " + TypeID2TypeName(customColumn->GetTypeId()));

   if (customColumn->IsDataSourceColumn()) {
      fColumnKind = EColumnKind::kDataSource;
      fDSValuePtrs.emplace_back(static_cast<T **>(customColumn->GetValuePtr(slot)));
   } else {
      fColumnKind = EColumnKind::kCustomColumn;
      fCustomValuePtrs.emplace_back(static_cast<T *>(customColumn->GetValuePtr(slot)));
   }
   fSlot = slot;
}

// This method is executed inside the event-loop, many times per entry
// If need be, the if statement can be avoided using thunks
// (have both branches inside functions and have a pointer to the branch to be executed)
template <typename T, bool B>
template <typename U, typename std::enable_if<!TColumnValue<U>::fgMustUseRVec, int>::type>
T &TColumnValue<T, B>::Get(Long64_t entry)
{
   if (fColumnKind == EColumnKind::kTree) {
      return *(fTreeReaders.back()->Get());
   } else {
      fCustomColumns.back()->Update(fSlot, entry);
      return fColumnKind == EColumnKind::kCustomColumn ? *fCustomValuePtrs.back() : **fDSValuePtrs.back();
   }
}

/// This overload is used to return arrays (i.e. types that are read into a RVec)
template <typename T, bool B>
template <typename U, typename std::enable_if<TColumnValue<U>::fgMustUseRVec, int>::type>
T &TColumnValue<T, B>::Get(Long64_t entry)
{
   if (fColumnKind == EColumnKind::kTree) {
      auto &readerArray = *fTreeReaders.back();
      // We only use TTreeReaderArrays to read columns that users flagged as type `RVec`, so we need to check
      // that the branch stores the array as contiguous memory that we can actually wrap in an `RVec`.
      // Currently we need the first entry to have been loaded to perform the check
      // TODO Move check to `MakeProxy` once Axel implements this kind of check in TTreeReaderArray using
      // TBranchProxy

      if (EStorageType::kUnknown == fStorageType && readerArray.GetSize() > 1) {
         // We can decide since the array is long enough
         fStorageType = (1 == (&readerArray[1] - &readerArray[0])) ? EStorageType::kContiguous : EStorageType::kSparse;
      }

      const auto readerArraySize = readerArray.GetSize();
      if (EStorageType::kContiguous == fStorageType ||
          (EStorageType::kUnknown == fStorageType && readerArray.GetSize() < 2)) {
         if (readerArraySize > 0) {
            // trigger loading of the contens of the TTreeReaderArray
            // the address of the first element in the reader array is not necessarily equal to
            // the address returned by the GetAddress method
            auto readerArrayAddr = &readerArray.At(0);
            T tvec(readerArrayAddr, readerArraySize);
            swap(fRVec, tvec);
         } else {
            T emptyVec{};
            swap(fRVec, emptyVec);
         }
      } else {
         // The storage is not contiguous or we don't know yet: we cannot but copy into the tvec
#ifndef NDEBUG
         if (!fCopyWarningPrinted) {
            Warning("TColumnValue::Get",
                  "Branch %s hangs from a non-split branch. For this reason, it cannot be accessed via a RVec. A copy is being performed in order to properly read the content.",
                  readerArray.GetBranchName());
            fCopyWarningPrinted = true;
         }
#else
         (void)fCopyWarningPrinted;
#endif
         if (readerArraySize > 0) {
            (void)readerArray.At(0); // trigger deserialisation
            T tvec(readerArray.begin(), readerArray.end());
            swap(fRVec, tvec);
         } else {
            T emptyVec{};
            swap(fRVec, emptyVec);
         }
      }
      return fRVec;

   } else {
      fCustomColumns.back()->Update(fSlot, entry);
      return fColumnKind == EColumnKind::kCustomColumn ? *fCustomValuePtrs.back() : **fDSValuePtrs.back();
   }
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT
#endif // ROOT_RDFNODES
