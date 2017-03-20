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

#include "ROOT/TDFUtils.hxx"
#include "ROOT/RArrayView.hxx"
#include "ROOT/TSpinMutex.hxx"
#include "TTreeReaderArray.h"
#include "TTreeReaderValue.h"

#include <map>
#include <numeric> // std::iota for TSlotStack
#include <string>
#include <tuple>

namespace ROOT {

namespace Internal {
class TDataFrameActionBase;
}

namespace Detail {

// forward declarations for TDataFrameImpl
using ActionBasePtr_t = std::shared_ptr<ROOT::Internal::TDataFrameActionBase>;
using ActionBaseVec_t = std::vector<ActionBasePtr_t>;
class TDataFrameBranchBase;
using TmpBranchBasePtr_t = std::shared_ptr<ROOT::Detail::TDataFrameBranchBase>;
class TDataFrameFilterBase;
using FilterBasePtr_t = std::shared_ptr<ROOT::Detail::TDataFrameFilterBase>;
using FilterBaseVec_t = std::vector<FilterBasePtr_t>;
class TDataFrameRangeBase;
using RangeBasePtr_t = std::shared_ptr<ROOT::Detail::TDataFrameRangeBase>;
using RangeBaseVec_t = std::vector<RangeBasePtr_t>;

class TDataFrameImpl : public std::enable_shared_from_this<TDataFrameImpl> {

   // This is an helper class to allow to pick a slot without resorting to a map
   // indexed by thread ids.
   // WARNING: this class does not work as a regular stack. The size is
   // fixed at construction time and no blocking is foreseen.
   class TSlotStack {
   private:
      unsigned int              fCursor;
      std::vector<unsigned int> fBuf;
      ROOT::TSpinMutex          fMutex;

   public:
      TSlotStack() = delete;
      TSlotStack(unsigned int size) : fCursor(size), fBuf(size) { std::iota(fBuf.begin(), fBuf.end(), 0U); }
      void Push(unsigned int slotNumber);
      unsigned int Pop();
   };

   ROOT::Detail::ActionBaseVec_t fBookedActions;
   ROOT::Detail::FilterBaseVec_t fBookedFilters;
   ROOT::Detail::FilterBaseVec_t fBookedNamedFilters;
   std::map<std::string, TmpBranchBasePtr_t> fBookedBranches;
   ROOT::Detail::RangeBaseVec_t       fBookedRanges;
   std::vector<std::shared_ptr<bool>> fResProxyReadiness;
   ::TDirectory *                     fDirPtr{nullptr};
   TTree *                            fTree{nullptr};
   const BranchNames_t                fDefaultBranches;
   const unsigned int                 fNSlots{0};
   bool                               fHasRunAtLeastOnce{false};
   unsigned int fNChildren{0};      ///< Number of nodes of the functional graph hanging from this object
   unsigned int fNStopsReceived{0}; ///< Number of times that a children node signaled to stop processing entries.

public:
   TDataFrameImpl(TTree *tree, const BranchNames_t &defaultBranches);
   TDataFrameImpl(const TDataFrameImpl &) = delete;
   ~TDataFrameImpl(){};
   void Run();
   void BuildAllReaderValues(TTreeReader &r, unsigned int slot);
   void CreateSlots(unsigned int nSlots);
   TDataFrameImpl *                GetImplPtr();
   std::shared_ptr<TDataFrameImpl> GetSharedPtr() { return shared_from_this(); }
   const BranchNames_t &           GetDefaultBranches() const;
   const BranchNames_t             GetTmpBranches() const { return {}; };
   TTree *                         GetTree() const;
   TDataFrameBranchBase *GetBookedBranch(const std::string &name) const;
   const std::map<std::string, TmpBranchBasePtr_t> &GetBookedBranches() const { return fBookedBranches; }
   ::TDirectory *GetDirectory() const;
   std::string   GetTreeName() const;
   void Book(const ActionBasePtr_t &actionPtr);
   void Book(const ROOT::Detail::FilterBasePtr_t &filterPtr);
   void Book(const ROOT::Detail::TmpBranchBasePtr_t &branchPtr);
   void Book(const std::shared_ptr<bool> &branchPtr);
   void Book(const ROOT::Detail::RangeBasePtr_t &rangePtr);
   bool         CheckFilters(int, unsigned int);
   unsigned int GetNSlots() const;
   bool         HasRunAtLeastOnce() const { return fHasRunAtLeastOnce; }
   void         Report() const;
   /// End of recursive chain of calls, does nothing
   void PartialReport() const {}
   void SetTree(TTree *tree) { fTree = tree; }
   void                IncrChildrenCount() { ++fNChildren; }
   void                StopProcessing() { ++fNStopsReceived; }
};
}

namespace Internal {

/**
\class ROOT::Experimental::TDataFrameValue
\ingroup dataframe
\brief Helper class that updates and returns TTree branches as well as TDataFrame temporary columns
\tparam T The type of the column

TDataFrame nodes must access two different types of values during the event loop:
values of real branches, for which TTreeReader{Values,Arrays} act as proxies, or
temporary columns whose values are generated on the fly. While the type of the
value is known at compile time (or just-in-time), it is only at runtime that nodes
can check whether a certain value is generated on the fly or not.

TDataFrameValuePtr abstracts this difference by providing the same interface for
both cases and handling the reading or generation of new values transparently.
Only one of the two data members fReaderProxy or fValuePtr will be non-null
for a given TDataFrameValue, depending on whether the value comes from a real
TTree branch or from a temporary column respectively.

TDataFrame nodes can store tuples of TDataFrameValues and retrieve an updated
value for the column via the `Get` method.
**/
template <typename T>
class TDataFrameValue {
   // following line is equivalent to pseudo-code: ProxyParam_t == array_view<U> ? U : T
   // ReaderValueOrArray_t is a TTreeReaderValue<T> unless T is array_view<U>
   using ProxyParam_t = typename std::conditional<std::is_same<ReaderValueOrArray_t<T>, TTreeReaderValue<T>>::value, T,
                                                  TDFTraitsUtils::ExtractType_t<T>>::type;
   std::unique_ptr<TTreeReaderValue<T>> fReaderValue{nullptr}; //< Owning ptr to a TTreeReaderValue. Used for
                                                               /// non-temporary columns and T != std::array_view<U>
   std::unique_ptr<TTreeReaderArray<ProxyParam_t>> fReaderArray{nullptr}; //< Owning ptr to a TTreeReaderArray. Used for
                                                                          /// non-temporary columsn and
                                                                          /// T == std::array_view<U>.
   T *                                 fValuePtr{nullptr}; //< Non-owning ptr to the value of a temporary column.
   ROOT::Detail::TDataFrameBranchBase *fTmpColumn{
      nullptr};           //< Non-owning ptr to the node responsible for the temporary column.
   unsigned int fSlot{0}; //< The slot this value belongs to. Only used for temporary columns, not for real branches.

public:
   TDataFrameValue() = default;

   void SetTmpColumn(unsigned int slot, ROOT::Detail::TDataFrameBranchBase *tmpColumn);

   void MakeProxy(TTreeReader &r, const std::string &bn)
   {
      bool useReaderValue = std::is_same<ProxyParam_t, T>::value;
      if (useReaderValue)
         fReaderValue.reset(new TTreeReaderValue<T>(r, bn.c_str()));
      else
         fReaderArray.reset(new TTreeReaderArray<ProxyParam_t>(r, bn.c_str()));
   }

   template <typename U = T,
             typename std::enable_if<std::is_same<typename ROOT::Internal::TDataFrameValue<U>::ProxyParam_t, U>::value,
                                     int>::type = 0>
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
};

template <typename T>
struct TTDFValueTuple {
};

template <typename... BranchTypes>
struct TTDFValueTuple<ROOT::Internal::TDFTraitsUtils::TTypeList<BranchTypes...>> {
   using type = std::tuple<ROOT::Internal::TDataFrameValue<BranchTypes>...>;
};

template <typename BranchType>
using TDFValueTuple_t = typename TTDFValueTuple<BranchType>::type;

class TDataFrameActionBase {
protected:
   ROOT::Detail::TDataFrameImpl *fImplPtr; ///< A raw pointer to the TDataFrameImpl at the root of this functional
                                           /// graph. It is only guaranteed to contain a valid address during an event
                                           /// loop.
   const BranchNames_t fTmpBranches;

public:
   TDataFrameActionBase(ROOT::Detail::TDataFrameImpl *implPtr, const BranchNames_t &tmpBranches);
   virtual ~TDataFrameActionBase() {}
   virtual void Run(unsigned int slot, Long64_t entry)               = 0;
   virtual void BuildReaderValues(TTreeReader &r, unsigned int slot) = 0;
   virtual void CreateSlots(unsigned int nSlots) = 0;
};

template <typename Helper, typename PrevDataFrame, typename BranchTypes_t = typename Helper::BranchTypes_t>
class TDataFrameAction final : public TDataFrameActionBase {
   using TypeInd_t = typename TDFTraitsUtils::TGenStaticSeq<BranchTypes_t::fgSize>::Type_t;

   Helper                                                      fHelper;
   const BranchNames_t                                         fBranches;
   PrevDataFrame &                                             fPrevData;
   std::vector<ROOT::Internal::TDFValueTuple_t<BranchTypes_t>> fValues;

public:
   TDataFrameAction(Helper &&h, const BranchNames_t &bl, PrevDataFrame &pd)
      : TDataFrameActionBase(pd.GetImplPtr(), pd.GetTmpBranches()), fHelper(std::move(h)), fBranches(bl), fPrevData(pd)
   {
   }

   TDataFrameAction(const TDataFrameAction &) = delete;

   void CreateSlots(unsigned int nSlots) final { fValues.resize(nSlots); }

   void BuildReaderValues(TTreeReader &r, unsigned int slot) final
   {
      ROOT::Internal::InitTDFValues(slot, fValues[slot], r, fBranches, fTmpBranches, fImplPtr->GetBookedBranches(),
                                    BranchTypes_t(), TypeInd_t());
   }

   void Run(unsigned int slot, Long64_t entry) final
   {
      // check if entry passes all filters
      if (fPrevData.CheckFilters(slot, entry)) Exec(slot, entry, TypeInd_t());
   }

   template <int... S>
   void Exec(unsigned int slot, Long64_t entry, TDFTraitsUtils::TStaticSeq<S...>)
   {
      (void)entry; // avoid bogus 'unused parameter' warning in gcc4.9
      fHelper.Exec(slot, std::get<S>(fValues[slot]).Get(entry)...);
   }

   ~TDataFrameAction() { fHelper.Finalize(); }
};

} // end NS Internal

namespace Detail {

class TDataFrameBranchBase {
protected:
   TDataFrameImpl *fImplPtr; ///< A raw pointer to the TDataFrameImpl at the root of this functional graph. It is only
                             /// guaranteed to contain a valid address during an event loop.
   BranchNames_t     fTmpBranches;
   const std::string fName;
   unsigned int      fNChildren{0};      ///< Number of nodes of the functional graph hanging from this object
   unsigned int      fNStopsReceived{0}; ///< Number of times that a children node signaled to stop processing entries.

public:
   TDataFrameBranchBase(TDataFrameImpl *df, const BranchNames_t &tmpBranches, const std::string &name);
   virtual ~TDataFrameBranchBase() {}
   virtual void BuildReaderValues(TTreeReader &r, unsigned int slot) = 0;
   virtual void CreateSlots(unsigned int nSlots)   = 0;
   virtual void *GetValuePtr(unsigned int slot)    = 0;
   virtual const std::type_info &GetTypeId() const = 0;
   virtual bool CheckFilters(unsigned int slot, Long64_t entry) = 0;
   TDataFrameImpl *GetImplPtr() const;
   virtual void    Report() const        = 0;
   virtual void    PartialReport() const = 0;
   std::string     GetName() const;
   BranchNames_t   GetTmpBranches() const;
   virtual void Update(unsigned int slot, Long64_t entry) = 0;
   void         IncrChildrenCount() { ++fNChildren; }
   virtual void StopProcessing() = 0;
};

template <typename F, typename PrevData>
class TDataFrameBranch final : public TDataFrameBranchBase {
   using BranchTypes_t = typename ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::Args_t;
   using TypeInd_t     = typename ROOT::Internal::TDFTraitsUtils::TGenStaticSeq<BranchTypes_t::fgSize>::Type_t;
   using Ret_t         = typename ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::Ret_t;

   F                                   fExpression;
   const BranchNames_t                 fBranches;
   std::vector<std::unique_ptr<Ret_t>> fLastResultPtr;
   PrevData &                          fPrevData;
   std::vector<Long64_t>               fLastCheckedEntry = {-1};

   std::vector<ROOT::Internal::TDFValueTuple_t<BranchTypes_t>> fValues;

public:
   TDataFrameBranch(const std::string &name, F &&expression, const BranchNames_t &bl, PrevData &pd)
      : TDataFrameBranchBase(pd.GetImplPtr(), pd.GetTmpBranches(), name), fExpression(std::move(expression)),
        fBranches(bl), fPrevData(pd)
   {
      fTmpBranches.emplace_back(name);
   }

   TDataFrameBranch(const TDataFrameBranch &) = delete;

   void BuildReaderValues(TTreeReader &r, unsigned int slot) final
   {
      ROOT::Internal::InitTDFValues(slot, fValues[slot], r, fBranches, fTmpBranches, fImplPtr->GetBookedBranches(),
                                    BranchTypes_t(), TypeInd_t());
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

   const std::type_info &GetTypeId() const { return typeid(Ret_t); }

   void CreateSlots(unsigned int nSlots) final
   {
      fValues.resize(nSlots);
      fLastCheckedEntry.resize(nSlots, -1);
      fLastResultPtr.resize(nSlots);
      std::generate(fLastResultPtr.begin(), fLastResultPtr.end(), []() { return std::unique_ptr<Ret_t>(new Ret_t()); });
   }

   bool CheckFilters(unsigned int slot, Long64_t entry) final
   {
      // dummy call: it just forwards to the previous object in the chain
      return fPrevData.CheckFilters(slot, entry);
   }

   template <int... S, typename... BranchTypes>
   void UpdateHelper(unsigned int slot, Long64_t entry, ROOT::Internal::TDFTraitsUtils::TStaticSeq<S...>,
                     ROOT::Internal::TDFTraitsUtils::TTypeList<BranchTypes...>)
   {
      *fLastResultPtr[slot] = fExpression(std::get<S>(fValues[slot]).Get(entry)...);
   }

   // recursive chain of `Report`s
   // TDataFrameBranch simply forwards the call to the previous node
   void Report() const final { fPrevData.PartialReport(); }

   void PartialReport() const final { fPrevData.PartialReport(); }

   void StopProcessing()
   {
      ++fNStopsReceived;
      if (fNStopsReceived == fNChildren) fPrevData.StopProcessing();
   }
};

class TDataFrameFilterBase {
protected:
   TDataFrameImpl *fImplPtr; ///< A raw pointer to the TDataFrameImpl at the root of this functional graph. It is only
                             /// guaranteed to contain a valid address during an event loop.
   const BranchNames_t    fTmpBranches;
   std::vector<Long64_t>  fLastCheckedEntry = {-1};
   std::vector<int>       fLastResult       = {true}; // std::vector<bool> cannot be used in a MT context safely
   std::vector<ULong64_t> fAccepted         = {0};
   std::vector<ULong64_t> fRejected         = {0};
   const std::string      fName;
   unsigned int           fNChildren{0}; ///< Number of nodes of the functional graph hanging from this object
   unsigned int fNStopsReceived{0};      ///< Number of times that a children node signaled to stop processing entries.

public:
   TDataFrameFilterBase(TDataFrameImpl *df, const BranchNames_t &tmpBranches, const std::string &name);
   virtual ~TDataFrameFilterBase() {}
   virtual void BuildReaderValues(TTreeReader &r, unsigned int slot) = 0;
   virtual bool CheckFilters(unsigned int slot, Long64_t entry)      = 0;
   virtual void    Report() const        = 0;
   virtual void    PartialReport() const = 0;
   TDataFrameImpl *GetImplPtr() const;
   BranchNames_t   GetTmpBranches() const;
   bool            HasName() const;
   virtual void CreateSlots(unsigned int nSlots) = 0;
   void         PrintReport() const;
   void         IncrChildrenCount() { ++fNChildren; }
   virtual void StopProcessing() = 0;
};

template <typename FilterF, typename PrevDataFrame>
class TDataFrameFilter final : public TDataFrameFilterBase {
   using BranchTypes_t = typename ROOT::Internal::TDFTraitsUtils::TFunctionTraits<FilterF>::Args_t;
   using TypeInd_t     = typename ROOT::Internal::TDFTraitsUtils::TGenStaticSeq<BranchTypes_t::fgSize>::Type_t;

   FilterF                                                     fFilter;
   const BranchNames_t                                         fBranches;
   PrevDataFrame &                                             fPrevData;
   std::vector<ROOT::Internal::TDFValueTuple_t<BranchTypes_t>> fValues;

public:
   TDataFrameFilter(FilterF &&f, const BranchNames_t &bl, PrevDataFrame &pd, const std::string &name = "")
      : TDataFrameFilterBase(pd.GetImplPtr(), pd.GetTmpBranches(), name), fFilter(std::move(f)), fBranches(bl),
        fPrevData(pd)
   {
   }

   TDataFrameFilter(const TDataFrameFilter &) = delete;

   void CreateSlots(unsigned int nSlots)
   {
      fValues.resize(nSlots);
      fLastCheckedEntry.resize(nSlots, -1);
      fLastResult.resize(nSlots);
      fAccepted.resize(nSlots);
      fRejected.resize(nSlots);
      // fAccepted and fRejected could be different than 0 if this is not the
      // first event-loop run using this filter
      std::fill(fAccepted.begin(), fAccepted.end(), 0);
      std::fill(fRejected.begin(), fRejected.end(), 0);
   }

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
   bool CheckFilterHelper(unsigned int slot, Long64_t entry, ROOT::Internal::TDFTraitsUtils::TStaticSeq<S...>)
   {
      return fFilter(std::get<S>(fValues[slot]).Get(entry)...);
   }

   void BuildReaderValues(TTreeReader &r, unsigned int slot) final
   {
      ROOT::Internal::InitTDFValues(slot, fValues[slot], r, fBranches, fTmpBranches, fImplPtr->GetBookedBranches(),
                                    BranchTypes_t(), TypeInd_t());
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
};

class TDataFrameRangeBase {
protected:
   TDataFrameImpl *fImplPtr; ///< A raw pointer to the TDataFrameImpl at the root of this functional graph. It is only
                             /// guaranteed to contain a valid address during an event loop.
   BranchNames_t fTmpBranches;
   unsigned int  fStart;
   unsigned int  fStop;
   unsigned int  fStride;
   Long64_t      fLastCheckedEntry{-1};
   bool          fLastResult{true};
   ULong64_t     fNProcessedEntries{0};
   unsigned int  fNChildren{0};      ///< Number of nodes of the functional graph hanging from this object
   unsigned int  fNStopsReceived{0}; ///< Number of times that a children node signaled to stop processing entries.

public:
   TDataFrameRangeBase(TDataFrameImpl *implPtr, const BranchNames_t &tmpBranches, unsigned int start, unsigned int stop,
                       unsigned int stride);
   virtual ~TDataFrameRangeBase() {}
   TDataFrameImpl *GetImplPtr() const;
   BranchNames_t   GetTmpBranches() const;
   virtual bool CheckFilters(unsigned int slot, Long64_t entry) = 0;
   virtual void Report() const        = 0;
   virtual void PartialReport() const = 0;
   void         IncrChildrenCount() { ++fNChildren; }
   virtual void StopProcessing() = 0;
};

template <typename PrevData>
class TDataFrameRange final : public TDataFrameRangeBase {
   PrevData &fPrevData;

public:
   TDataFrameRange(unsigned int start, unsigned int stop, unsigned int stride, PrevData &pd)
      : TDataFrameRangeBase(pd.GetImplPtr(), pd.GetTmpBranches(), start, stop, stride), fPrevData(pd)
   {
   }

   TDataFrameRange(const TDataFrameRange &) = delete;

   /// Ranges act as filters when it comes to selecting entries that downstream nodes should process
   bool CheckFilters(unsigned int slot, Long64_t entry) final
   {
      if (entry != fLastCheckedEntry) {
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
            if (fNProcessedEntries == fStop) fPrevData.StopProcessing();
         }
         fLastCheckedEntry = entry;
      }
      return fLastResult;
   }

   // recursive chain of `Report`s
   // TDataFrameRange simply forwards these calls to the previous node
   void Report() const final { fPrevData.PartialReport(); }

   void PartialReport() const final { fPrevData.PartialReport(); }

   void StopProcessing()
   {
      ++fNStopsReceived;
      if (fNStopsReceived == fNChildren) fPrevData.StopProcessing();
   }
};

} // end NS ROOT::Detail
} // namespace ROOT

// method implementations
template <typename T>
void ROOT::Internal::TDataFrameValue<T>::SetTmpColumn(unsigned int slot, ROOT::Detail::TDataFrameBranchBase *tmpColumn)
{
   fTmpColumn = tmpColumn;
   if (tmpColumn->GetTypeId() != typeid(T))
      throw std::runtime_error(std::string("TDataFrameValue: type specified is ") + typeid(T).name() +
                               " but temporary column has type " + tmpColumn->GetTypeId().name());
   fValuePtr = static_cast<T *>(tmpColumn->GetValuePtr(slot));
   fSlot     = slot;
}

// This method is executed inside the event-loop, many times per entry
// If need be, the if statement can be avoided using thunks
// (have both branches inside functions and have a pointer to
// the branch to be executed)
template <typename T>
template <typename U, typename std::enable_if<
                         std::is_same<typename ROOT::Internal::TDataFrameValue<U>::ProxyParam_t, U>::value, int>::type>
T &ROOT::Internal::TDataFrameValue<T>::Get(Long64_t entry)
{
   if (fReaderValue) {
      return *(fReaderValue->Get());
   } else {
      fTmpColumn->Update(fSlot, entry);
      return *fValuePtr;
   }
}

#endif // ROOT_TDFNODES
