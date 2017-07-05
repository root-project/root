// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDFNODES
#define ROOT_TDFNODES

#include "ROOT/TypeTraits.hxx"
#include "ROOT/TDFUtils.hxx"
#include "ROOT/RArrayView.hxx"
#include "ROOT/TSpinMutex.hxx"
#include "TTreeReaderArray.h"
#include "TTreeReaderValue.h"

#include <map>
#include <numeric> // std::iota for TSlotStack
#include <string>
#include <tuple>
#include <cassert>

namespace ROOT {

namespace Internal {
namespace TDF {
class TActionBase;
}
}

namespace Detail {
namespace TDF {
using namespace ROOT::TypeTraits;
namespace TDFInternal = ROOT::Internal::TDF;

// forward declarations for TLoopManager
using ActionBasePtr_t = std::shared_ptr<TDFInternal::TActionBase>;
using ActionBaseVec_t = std::vector<ActionBasePtr_t>;
class TCustomColumnBase;
using TmpBranchBasePtr_t = std::shared_ptr<TCustomColumnBase>;
class TFilterBase;
using FilterBasePtr_t = std::shared_ptr<TFilterBase>;
using FilterBaseVec_t = std::vector<FilterBasePtr_t>;
class TRangeBase;
using RangeBasePtr_t = std::shared_ptr<TRangeBase>;
using RangeBaseVec_t = std::vector<RangeBasePtr_t>;

class TLoopManager : public std::enable_shared_from_this<TLoopManager> {

   enum class ELoopType { kROOTFiles, kNoFiles };

   ActionBaseVec_t fBookedActions;
   FilterBaseVec_t fBookedFilters;
   FilterBaseVec_t fBookedNamedFilters;
   std::map<std::string, TmpBranchBasePtr_t> fBookedBranches;
   RangeBaseVec_t fBookedRanges;
   std::vector<std::shared_ptr<bool>> fResProxyReadiness;
   ::TDirectory * const fDirPtr{nullptr};
   std::shared_ptr<TTree> fTree{nullptr};
   const ColumnNames_t fDefaultColumns;
   const ULong64_t fNEmptyEntries{0};
   const unsigned int fNSlots{1};
   bool fHasRunAtLeastOnce{false};
   unsigned int fNChildren{0};      ///< Number of nodes of the functional graph hanging from this object
   unsigned int fNStopsReceived{0}; ///< Number of times that a children node signaled to stop processing entries.
   const ELoopType fLoopType; ///< The kind of event loop that is going to be run (e.g. on ROOT files, on no files)
   std::string fToJit; ///< string containing all `BuildAndBook` actions that should be jitted before running

   void RunEmptySourceMT();
   void RunEmptySource();
   void RunTreeProcessorMT();
   void RunTreeReader();
   void RunAndCheckFilters(unsigned int slot, Long64_t entry);
   void InitNodeSlots(TTreeReader *r, unsigned int slot);
   void InitNodes();
   void CleanUp();
   void JitActions();
   void EvalChildrenCounts();

public:
   TLoopManager(TTree *tree, const ColumnNames_t &defaultBranches);
   TLoopManager(ULong64_t nEmptyEntries);
   TLoopManager(const TLoopManager &) = delete;
   ~TLoopManager(){};
   void Run();
   TLoopManager *GetImplPtr();
   std::shared_ptr<TLoopManager> GetSharedPtr() { return shared_from_this(); }
   const ColumnNames_t &GetDefaultColumnNames() const;
   const ColumnNames_t GetTmpBranches() const { return {}; };
   TTree *GetTree() const;
   TCustomColumnBase *GetBookedBranch(const std::string &name) const;
   const std::map<std::string, TmpBranchBasePtr_t> &GetBookedBranches() const { return fBookedBranches; }
   ::TDirectory *GetDirectory() const;
   ULong64_t GetNEmptyEntries() const { return fNEmptyEntries; }
   void Book(const ActionBasePtr_t &actionPtr);
   void Book(const FilterBasePtr_t &filterPtr);
   void Book(const TmpBranchBasePtr_t &branchPtr);
   void Book(const std::shared_ptr<bool> &branchPtr);
   void Book(const RangeBasePtr_t &rangePtr);
   bool CheckFilters(int, unsigned int);
   unsigned int GetNSlots() const { return fNSlots; }
   bool HasRunAtLeastOnce() const { return fHasRunAtLeastOnce; }
   void Report() const;
   /// End of recursive chain of calls, does nothing
   void PartialReport() const {}
   void SetTree(std::shared_ptr<TTree> tree) { fTree = tree; }
   void IncrChildrenCount() { ++fNChildren; }
   void StopProcessing() { ++fNStopsReceived; }
   void Jit(const std::string& s) { fToJit.append(s); }
};
} // end ns TDF
} // end ns Detail

namespace Internal {
namespace TDF {
using namespace ROOT::Detail::TDF;

/**
\class ROOT::Internal::TDF::TColumnValue
\ingroup dataframe
\brief Helper class that updates and returns TTree branches as well as TDataFrame temporary columns
\tparam T The type of the column

TDataFrame nodes must access two different types of values during the event loop:
values of real branches, for which TTreeReader{Values,Arrays} act as proxies, or
temporary columns whose values are generated on the fly. While the type of the
value is known at compile time (or just-in-time), it is only at runtime that nodes
can check whether a certain value is generated on the fly or not.

TColumnValuePtr abstracts this difference by providing the same interface for
both cases and handling the reading or generation of new values transparently.
Only one of the two data members fReaderProxy or fValuePtr will be non-null
for a given TColumnValue, depending on whether the value comes from a real
TTree branch or from a temporary column respectively.

TDataFrame nodes can store tuples of TColumnValues and retrieve an updated
value for the column via the `Get` method.
**/
template <typename T>
class TColumnValue {
   // following line is equivalent to pseudo-code: ProxyParam_t == array_view<U> ? U : T
   // ReaderValueOrArray_t is a TTreeReaderValue<T> unless T is array_view<U>
   using ProxyParam_t = typename std::conditional<std::is_same<ReaderValueOrArray_t<T>, TTreeReaderValue<T>>::value, T,
                                                  TakeFirstParameter_t<T>>::type;
   std::unique_ptr<TTreeReaderValue<T>> fReaderValue{nullptr}; //< Owning ptr to a TTreeReaderValue. Used for
                                                               /// non-temporary columns and T != std::array_view<U>
   std::unique_ptr<TTreeReaderArray<ProxyParam_t>> fReaderArray{nullptr}; //< Owning ptr to a TTreeReaderArray. Used for
                                                                          /// non-temporary columsn and
                                                                          /// T == std::array_view<U>.
   T *fValuePtr{nullptr};                  //< Non-owning ptr to the value of a temporary column.
   TCustomColumnBase *fTmpColumn{nullptr}; //< Non-owning ptr to the node responsible for the temporary column.
   unsigned int fSlot{0}; //< The slot this value belongs to. Only used for temporary columns, not for real branches.

public:
   TColumnValue() = default;

   void SetTmpColumn(unsigned int slot, TCustomColumnBase *tmpColumn);

   void MakeProxy(TTreeReader *r, const std::string &bn)
   {
      Reset();
      bool useReaderValue = std::is_same<ProxyParam_t, T>::value;
      if (useReaderValue)
         fReaderValue.reset(new TTreeReaderValue<T>(*r, bn.c_str()));
      else
         fReaderArray.reset(new TTreeReaderArray<ProxyParam_t>(*r, bn.c_str()));
   }

   template <typename U = T,
             typename std::enable_if<std::is_same<typename TColumnValue<U>::ProxyParam_t, U>::value, int>::type = 0>
   T &Get(Long64_t entry);

   template <typename U = T, typename std::enable_if<!std::is_same<ProxyParam_t, U>::value, int>::type = 0>
   std::array_view<ProxyParam_t> Get(Long64_t)
   {
      auto &readerArray = *fReaderArray;
      if (readerArray.GetSize() > 1 && 1 != (&readerArray[1] - &readerArray[0])) {
         std::string exceptionText = "Branch ";
         exceptionText += fReaderArray->GetBranchName();
         exceptionText += " hangs from a non-split branch. For this reason, it cannot be accessed via an array_view."
                          " Please read the top level branch instead.";
         throw std::runtime_error(exceptionText.c_str());
      }

      return std::array_view<ProxyParam_t>(fReaderArray->begin(), fReaderArray->end());
   }

   void Reset()
   {
      fReaderValue = nullptr;
      fReaderArray = nullptr;
      fValuePtr = nullptr;
      fTmpColumn = nullptr;
      fSlot = 0;
   }
};

template <typename T>
struct TTDFValueTuple {
};

template <typename... BranchTypes>
struct TTDFValueTuple<TypeList<BranchTypes...>> {
   using type = std::tuple<TColumnValue<BranchTypes>...>;
};

template <typename BranchType>
using TDFValueTuple_t = typename TTDFValueTuple<BranchType>::type;

class TActionBase {
protected:
   TLoopManager *fImplPtr; ///< A raw pointer to the TLoopManager at the root of this functional
                           /// graph. It is only guaranteed to contain a valid address during an
                           /// event loop.
   const ColumnNames_t fTmpBranches;
   const unsigned int fNSlots; ///< Number of thread slots used by this node.

public:
   TActionBase(TLoopManager *implPtr, const ColumnNames_t &tmpBranches, unsigned int nSlots);
   virtual ~TActionBase() {}
   virtual void Run(unsigned int slot, Long64_t entry) = 0;
   virtual void InitSlot(TTreeReader *r, unsigned int slot) = 0;
   virtual void TriggerChildrenCount() = 0;
   unsigned int GetNSlots() const { return fNSlots; }
};

template <typename Helper, typename PrevDataFrame, typename BranchTypes_t = typename Helper::BranchTypes_t>
class TAction final : public TActionBase {
   using TypeInd_t = TDFInternal::GenStaticSeq_t<BranchTypes_t::list_size>;

   Helper fHelper;
   const ColumnNames_t fBranches;
   PrevDataFrame &fPrevData;
   std::vector<TDFValueTuple_t<BranchTypes_t>> fValues;

public:
   TAction(Helper &&h, const ColumnNames_t &bl, PrevDataFrame &pd)
      : TActionBase(pd.GetImplPtr(), pd.GetTmpBranches(), pd.GetNSlots()), fHelper(std::move(h)), fBranches(bl),
        fPrevData(pd), fValues(fNSlots)
   {
   }

   TAction(const TAction &) = delete;

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      InitTDFValues(slot, fValues[slot], r, fBranches, fTmpBranches, fImplPtr->GetBookedBranches(), TypeInd_t());
      fHelper.InitSlot(r, slot);
   }

   void Run(unsigned int slot, Long64_t entry) final
   {
      // check if entry passes all filters
      if (fPrevData.CheckFilters(slot, entry)) Exec(slot, entry, TypeInd_t());
   }

   template <int... S>
   void Exec(unsigned int slot, Long64_t entry, TDFInternal::StaticSeq<S...>)
   {
      (void)entry; // avoid bogus 'unused parameter' warning in gcc4.9
      fHelper.Exec(slot, std::get<S>(fValues[slot]).Get(entry)...);
   }

   void TriggerChildrenCount() final { fPrevData.IncrChildrenCount(); }

   ~TAction() { fHelper.Finalize(); }
};

} // end NS TDF
} // end NS Internal

namespace Detail {
namespace TDF {

class TCustomColumnBase {
protected:
   TLoopManager *fImplPtr; ///< A raw pointer to the TLoopManager at the root of this functional graph. It is only
                           /// guaranteed to contain a valid address during an event loop.
   ColumnNames_t fTmpBranches;
   const std::string fName;
   unsigned int fNChildren{0};      ///< Number of nodes of the functional graph hanging from this object
   unsigned int fNStopsReceived{0}; ///< Number of times that a children node signaled to stop processing entries.
   const unsigned int fNSlots; ///< Number of thread slots used by this node, inherited from parent node.

public:
   TCustomColumnBase(TLoopManager *df, const ColumnNames_t &tmpBranches, std::string_view name, unsigned int nSlots);
   virtual ~TCustomColumnBase() {}
   virtual void InitSlot(TTreeReader *r, unsigned int slot) = 0;
   virtual void *GetValuePtr(unsigned int slot) = 0;
   virtual const std::type_info &GetTypeId() const = 0;
   virtual bool CheckFilters(unsigned int slot, Long64_t entry) = 0;
   TLoopManager *GetImplPtr() const;
   virtual void Report() const = 0;
   virtual void PartialReport() const = 0;
   std::string GetName() const;
   ColumnNames_t GetTmpBranches() const;
   virtual void Update(unsigned int slot, Long64_t entry) = 0;
   virtual void IncrChildrenCount() = 0;
   virtual void StopProcessing() = 0;
   void ResetChildrenCount() { fNChildren = 0; fNStopsReceived = 0; }
   unsigned int GetNSlots() const { return fNSlots; }
};

template <typename F, typename PrevData>
class TCustomColumn final : public TCustomColumnBase {
   using BranchTypes_t = typename CallableTraits<F>::arg_types;
   using TypeInd_t = TDFInternal::GenStaticSeq_t<BranchTypes_t::list_size>;
   using ret_type = typename CallableTraits<F>::ret_type;

   F fExpression;
   const ColumnNames_t fBranches;
   std::vector<std::unique_ptr<ret_type>> fLastResultPtr;
   PrevData &fPrevData;
   std::vector<Long64_t> fLastCheckedEntry = {-1};

   std::vector<TDFInternal::TDFValueTuple_t<BranchTypes_t>> fValues;

public:
   TCustomColumn(std::string_view name, F &&expression, const ColumnNames_t &bl, PrevData &pd)
      : TCustomColumnBase(pd.GetImplPtr(), pd.GetTmpBranches(), name, pd.GetNSlots()),
        fExpression(std::move(expression)), fBranches(bl), fLastResultPtr(fNSlots), fPrevData(pd),
        fLastCheckedEntry(fNSlots, -1), fValues(fNSlots)
   {
      std::generate(fLastResultPtr.begin(), fLastResultPtr.end(),
                    []() { return std::unique_ptr<ret_type>(new ret_type()); });
      fTmpBranches.emplace_back(name);
   }

   TCustomColumn(const TCustomColumn &) = delete;

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      TDFInternal::InitTDFValues(slot, fValues[slot], r, fBranches, fTmpBranches, fImplPtr->GetBookedBranches(),
                                 TypeInd_t());
   }

   void *GetValuePtr(unsigned int slot) final { return static_cast<void *>(fLastResultPtr[slot].get()); }

   void Update(unsigned int slot, Long64_t entry) final
   {
      if (entry != fLastCheckedEntry[slot]) {
         // evaluate this filter, cache the result
         UpdateHelper(slot, entry, TypeInd_t(), BranchTypes_t());
         fLastCheckedEntry[slot] = entry;
      }
   }

   const std::type_info &GetTypeId() const { return typeid(ret_type); }

   bool CheckFilters(unsigned int slot, Long64_t entry) final
   {
      // dummy call: it just forwards to the previous object in the chain
      return fPrevData.CheckFilters(slot, entry);
   }

   template <int... S, typename... BranchTypes>
   void UpdateHelper(unsigned int slot, Long64_t entry, TDFInternal::StaticSeq<S...>, TypeList<BranchTypes...>)
   {
      *fLastResultPtr[slot] = fExpression(std::get<S>(fValues[slot]).Get(entry)...);
      // silence "unused parameter" warnings in gcc
      (void)slot;
      (void)entry;
   }

   // recursive chain of `Report`s
   // TCustomColumn simply forwards the call to the previous node
   void Report() const final { fPrevData.PartialReport(); }

   void PartialReport() const final { fPrevData.PartialReport(); }

   void StopProcessing()
   {
      ++fNStopsReceived;
      if (fNStopsReceived == fNChildren) fPrevData.StopProcessing();
   }

   void IncrChildrenCount()
   {
      ++fNChildren;
      // propagate "children activation" upstream
      if (fNChildren == 1) fPrevData.IncrChildrenCount();
   }
};

class TFilterBase {
protected:
   TLoopManager *fImplPtr; ///< A raw pointer to the TLoopManager at the root of this functional graph. It is only
                           /// guaranteed to contain a valid address during an event loop.
   const ColumnNames_t fTmpBranches;
   std::vector<Long64_t> fLastCheckedEntry = {-1};
   std::vector<int> fLastResult = {true}; // std::vector<bool> cannot be used in a MT context safely
   std::vector<ULong64_t> fAccepted = {0};
   std::vector<ULong64_t> fRejected = {0};
   const std::string fName;
   unsigned int fNChildren{0};      ///< Number of nodes of the functional graph hanging from this object
   unsigned int fNStopsReceived{0}; ///< Number of times that a children node signaled to stop processing entries.
   const unsigned int fNSlots; ///< Number of thread slots used by this node, inherited from parent node.

public:
   TFilterBase(TLoopManager *df, const ColumnNames_t &tmpBranches, std::string_view name, unsigned int nSlots);
   virtual ~TFilterBase() {}
   virtual void InitSlot(TTreeReader *r, unsigned int slot) = 0;
   virtual bool CheckFilters(unsigned int slot, Long64_t entry) = 0;
   virtual void Report() const = 0;
   virtual void PartialReport() const = 0;
   TLoopManager *GetImplPtr() const;
   ColumnNames_t GetTmpBranches() const;
   bool HasName() const;
   void PrintReport() const;
   virtual void IncrChildrenCount() = 0;
   virtual void StopProcessing() = 0;
   void ResetChildrenCount() { fNChildren = 0; fNStopsReceived = 0; }
   virtual void TriggerChildrenCount() = 0;
   unsigned int GetNSlots() const { return fNSlots; }
   virtual void ResetReportCount() = 0;
};

template <typename FilterF, typename PrevDataFrame>
class TFilter final : public TFilterBase {
   using BranchTypes_t = typename CallableTraits<FilterF>::arg_types;
   using TypeInd_t = TDFInternal::GenStaticSeq_t<BranchTypes_t::list_size>;

   FilterF fFilter;
   const ColumnNames_t fBranches;
   PrevDataFrame &fPrevData;
   std::vector<TDFInternal::TDFValueTuple_t<BranchTypes_t>> fValues;

public:
   TFilter(FilterF &&f, const ColumnNames_t &bl, PrevDataFrame &pd, std::string_view name = "")
      : TFilterBase(pd.GetImplPtr(), pd.GetTmpBranches(), name, pd.GetNSlots()), fFilter(std::move(f)), fBranches(bl),
        fPrevData(pd), fValues(fNSlots)
   {
   }

   TFilter(const TFilter &) = delete;

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

   template <int... S>
   bool CheckFilterHelper(unsigned int slot, Long64_t entry, TDFInternal::StaticSeq<S...>)
   {
      return fFilter(std::get<S>(fValues[slot]).Get(entry)...);
      // silence "unused parameter" warnings in gcc
      (void)slot;
      (void)entry;
   }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      TDFInternal::InitTDFValues(slot, fValues[slot], r, fBranches, fTmpBranches, fImplPtr->GetBookedBranches(),
                                 TypeInd_t());
   }

   // recursive chain of `Report`s
   void Report() const final { PartialReport(); }

   void PartialReport() const final
   {
      fPrevData.PartialReport();
      PrintReport();
   }

   void StopProcessing()
   {
      ++fNStopsReceived;
      if (fNStopsReceived == fNChildren) fPrevData.StopProcessing();
   }

   void IncrChildrenCount()
   {
      ++fNChildren;
      // propagate "children activation" upstream. named filters do the propagation via `TriggerChildrenCount`.
      if (fNChildren == 1 && fName.empty()) fPrevData.IncrChildrenCount();
   }

   void TriggerChildrenCount() final
   {
      assert(!fName.empty()); // this method is to only be called on named filters
      fPrevData.IncrChildrenCount();
   }

   void ResetReportCount()
   {
      assert(!fName.empty()); // this method is to only be called on named filters
      // fAccepted and fRejected could be different than 0 if this is not the first event-loop run using this filter
      std::fill(fAccepted.begin(), fAccepted.end(), 0);
      std::fill(fRejected.begin(), fRejected.end(), 0);
   }
};

class TRangeBase {
protected:
   TLoopManager *fImplPtr; ///< A raw pointer to the TLoopManager at the root of this functional graph. It is only
                           /// guaranteed to contain a valid address during an event loop.
   ColumnNames_t fTmpBranches;
   unsigned int fStart;
   unsigned int fStop;
   unsigned int fStride;
   Long64_t fLastCheckedEntry{-1};
   bool fLastResult{true};
   ULong64_t fNProcessedEntries{0};
   unsigned int fNChildren{0};      ///< Number of nodes of the functional graph hanging from this object
   unsigned int fNStopsReceived{0}; ///< Number of times that a children node signaled to stop processing entries.
   bool fHasStopped{false}; ///< True if the end of the range has been reached
   const unsigned int fNSlots; ///< Number of thread slots used by this node, inherited from parent node.

public:
   TRangeBase(TLoopManager *implPtr, const ColumnNames_t &tmpBranches, unsigned int start, unsigned int stop,
              unsigned int stride, unsigned int nSlots);
   virtual ~TRangeBase() {}
   TLoopManager *GetImplPtr() const;
   ColumnNames_t GetTmpBranches() const;
   virtual bool CheckFilters(unsigned int slot, Long64_t entry) = 0;
   virtual void Report() const = 0;
   virtual void PartialReport() const = 0;
   virtual void IncrChildrenCount() = 0;
   virtual void StopProcessing() = 0;
   void ResetChildrenCount() { fNChildren = 0; fNStopsReceived = 0; }
   unsigned int GetNSlots() const { return fNSlots; }
};

template <typename PrevData>
class TRange final : public TRangeBase {
   PrevData &fPrevData;

public:
   TRange(unsigned int start, unsigned int stop, unsigned int stride, PrevData &pd)
      : TRangeBase(pd.GetImplPtr(), pd.GetTmpBranches(), start, stop, stride, pd.GetNSlots()), fPrevData(pd)
   {
   }

   TRange(const TRange &) = delete;

   /// Ranges act as filters when it comes to selecting entries that downstream nodes should process
   bool CheckFilters(unsigned int slot, Long64_t entry) final
   {
      if (fHasStopped) {
         return false;
      } else if (entry != fLastCheckedEntry) {
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
   // TRange simply forwards these calls to the previous node
   void Report() const final { fPrevData.PartialReport(); }

   void PartialReport() const final { fPrevData.PartialReport(); }

   void StopProcessing()
   {
      ++fNStopsReceived;
      if (fNStopsReceived == fNChildren && !fHasStopped) fPrevData.StopProcessing();
   }

   void IncrChildrenCount()
   {
      ++fNChildren;
      // propagate "children activation" upstream
      if (fNChildren == 1) fPrevData.IncrChildrenCount();
   }
};

} // namespace TDF
} // namespace Detail
} // namespace ROOT

// method implementations
template <typename T>
void ROOT::Internal::TDF::TColumnValue<T>::SetTmpColumn(unsigned int slot,
                                                        ROOT::Detail::TDF::TCustomColumnBase *tmpColumn)
{
   Reset();
   fTmpColumn = tmpColumn;
   if (tmpColumn->GetTypeId() != typeid(T))
      throw std::runtime_error(std::string("TColumnValue: type specified is ") + typeid(T).name() +
                               " but temporary column has type " + tmpColumn->GetTypeId().name());
   fValuePtr = static_cast<T *>(tmpColumn->GetValuePtr(slot));
   fSlot = slot;
}

// This method is executed inside the event-loop, many times per entry
// If need be, the if statement can be avoided using thunks
// (have both branches inside functions and have a pointer to
// the branch to be executed)
template <typename T>
template <typename U,
          typename std::enable_if<std::is_same<typename ROOT::Internal::TDF::TColumnValue<U>::ProxyParam_t, U>::value,
                                  int>::type>
T &ROOT::Internal::TDF::TColumnValue<T>::Get(Long64_t entry)
{
   if (fReaderValue) {
      return *(fReaderValue->Get());
   } else {
      fTmpColumn->Update(fSlot, entry);
      return *fValuePtr;
   }
}

#endif // ROOT_TDFNODES
