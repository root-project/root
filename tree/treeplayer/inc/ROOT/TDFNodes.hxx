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

#include "ROOT/TActionResultProxy.hxx"
#include "ROOT/TDFUtils.hxx"
#include "TTreeReaderArray.h"
#include "TTreeReaderValue.h"

#include <map>

namespace ROOT {

// forward declaration for TDataFrameActionBase
namespace Detail {
class TDataFrameImpl;
}

namespace Internal {

// Forward declarations
template <typename T>
T &GetBranchValue(TVBPtr_t &readerValues, unsigned int slot, Long64_t entry, const std::string &branch,
                  ROOT::Detail::TDataFrameImpl *df, TDFTraitsUtils::TTypeList<T>);

template <typename T>
std::array_view<T> GetBranchValue(TVBPtr_t &readerValues, unsigned int slot, Long64_t entry, const std::string &branch,
                                  ROOT::Detail::TDataFrameImpl *df, TDFTraitsUtils::TTypeList<std::array_view<T>>);

class TDataFrameActionBase {
protected:
   ROOT::Detail::TDataFrameImpl *fImplPtr; ///< A raw pointer to the TDataFrameImpl at the root of this functional
                                           /// graph. It is only guaranteed to contain a valid address during an event
                                           /// loop.
   const BranchNames_t   fTmpBranches;
   std::vector<TVBVec_t> fReaderValues;

public:
   TDataFrameActionBase(ROOT::Detail::TDataFrameImpl *implPtr, const BranchNames_t &tmpBranches);
   virtual ~TDataFrameActionBase() {}
   virtual void Run(unsigned int slot, Long64_t entry)               = 0;
   virtual void BuildReaderValues(TTreeReader &r, unsigned int slot) = 0;
   void CreateSlots(unsigned int nSlots);
};

using ActionBasePtr_t = std::shared_ptr<TDataFrameActionBase>;
using ActionBaseVec_t = std::vector<ActionBasePtr_t>;

template <typename Helper, typename PrevDataFrame, typename BranchTypes_t = typename Helper::BranchTypes_t>
class TDataFrameAction final : public TDataFrameActionBase {
   using TypeInd_t = typename TDFTraitsUtils::TGenStaticSeq<BranchTypes_t::fgSize>::Type_t;

   Helper              fHelper;
   const BranchNames_t fBranches;
   PrevDataFrame &     fPrevData;

public:
   TDataFrameAction(Helper &&h, const BranchNames_t &bl, PrevDataFrame &pd)
      : TDataFrameActionBase(pd.GetImplPtr(), pd.GetTmpBranches()), fHelper(std::move(h)), fBranches(bl), fPrevData(pd)
   {
   }

   TDataFrameAction(const TDataFrameAction &) = delete;

   void BuildReaderValues(TTreeReader &r, unsigned int slot) final
   {
      fReaderValues[slot] = ROOT::Internal::BuildReaderValues(r, fBranches, fTmpBranches, BranchTypes_t(), TypeInd_t());
   }

   void Run(unsigned int slot, Long64_t entry) final
   {
      // check if entry passes all filters
      if (fPrevData.CheckFilters(slot, entry)) Exec(slot, entry, TypeInd_t(), BranchTypes_t());
   }

   template <int... S, typename... BranchTypes>
   void Exec(unsigned int slot, Long64_t entry, TDFTraitsUtils::TStaticSeq<S...>,
             TDFTraitsUtils::TTypeList<BranchTypes...>)
   {
      (void)entry; // avoid bogus unused-but-set-parameter warning by gcc
      // Take each pointer in tvb, cast it to a pointer to the
      // correct specialization of TTreeReaderValue, and get its content.
      // S expands to a sequence of integers 0 to sizeof...(types)-1
      // S and BranchTypes are expanded simultaneously by "..."
      fHelper.Exec(slot, GetBranchValue(fReaderValues[slot][S], slot, entry, fBranches[S], fImplPtr,
                                        TDFTraitsUtils::TTypeList<BranchTypes>())...);
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

public:
   TDataFrameBranchBase(TDataFrameImpl *df, const BranchNames_t &tmpBranches, const std::string &name);
   virtual ~TDataFrameBranchBase() {}
   virtual void BuildReaderValues(TTreeReader &r, unsigned int slot) = 0;
   virtual void CreateSlots(unsigned int nSlots) = 0;
   virtual void *GetValue(unsigned int slot, Long64_t entry) = 0;
   virtual const std::type_info &GetTypeId() const = 0;
   virtual bool CheckFilters(unsigned int slot, Long64_t entry) = 0;
   TDataFrameImpl *GetImplPtr() const;
   virtual void    Report() const        = 0;
   virtual void    PartialReport() const = 0;
   std::string     GetName() const;
   BranchNames_t   GetTmpBranches() const;
};
using TmpBranchBasePtr_t = std::shared_ptr<TDataFrameBranchBase>;

template <typename F, typename PrevData>
class TDataFrameBranch final : public TDataFrameBranchBase {
   using BranchTypes_t = typename ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::Args_t;
   using TypeInd_t     = typename ROOT::Internal::TDFTraitsUtils::TGenStaticSeq<BranchTypes_t::fgSize>::Type_t;
   using Ret_t         = typename ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::Ret_t;

   F                   fExpression;
   const BranchNames_t fBranches;

   std::vector<ROOT::Internal::TVBVec_t> fReaderValues;
   std::vector<std::shared_ptr<Ret_t>>   fLastResultPtr;
   PrevData &                            fPrevData;
   std::vector<Long64_t>                 fLastCheckedEntry = {-1};

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
      fReaderValues[slot] = ROOT::Internal::BuildReaderValues(r, fBranches, fTmpBranches, BranchTypes_t(), TypeInd_t());
   }

   void *GetValue(unsigned int slot, Long64_t entry) final
   {
      if (entry != fLastCheckedEntry[slot]) {
         // evaluate this filter, cache the result
         auto newValuePtr        = GetValueHelper(BranchTypes_t(), TypeInd_t(), slot, entry);
         fLastResultPtr[slot]    = newValuePtr;
         fLastCheckedEntry[slot] = entry;
      }
      return static_cast<void *>(fLastResultPtr[slot].get());
   }

   const std::type_info &GetTypeId() const { return typeid(Ret_t); }

   void CreateSlots(unsigned int nSlots) final
   {
      fReaderValues.resize(nSlots);
      fLastCheckedEntry.resize(nSlots, -1);
      fLastResultPtr.resize(nSlots);
   }

   bool CheckFilters(unsigned int slot, Long64_t entry) final
   {
      // dummy call: it just forwards to the previous object in the chain
      return fPrevData.CheckFilters(slot, entry);
   }

   template <int... S, typename... BranchTypes>
   std::shared_ptr<Ret_t> GetValueHelper(ROOT::Internal::TDFTraitsUtils::TTypeList<BranchTypes...>,
                                         ROOT::Internal::TDFTraitsUtils::TStaticSeq<S...>, unsigned int slot,
                                         Long64_t entry)
   {
      auto valuePtr = std::make_shared<Ret_t>(
         fExpression(ROOT::Internal::GetBranchValue(fReaderValues[slot][S], slot, entry, fBranches[S], fImplPtr,
                                                    ROOT::Internal::TDFTraitsUtils::TTypeList<BranchTypes>())...));
      return valuePtr;
   }

   // recursive chain of `Report`s
   // TDataFrameBranch simply forwards the call to the previous node
   void Report() const final { fPrevData.PartialReport(); }

   void PartialReport() const final { fPrevData.PartialReport(); }
};

class TDataFrameFilterBase {
protected:
   TDataFrameImpl *fImplPtr; ///< A raw pointer to the TDataFrameImpl at the root of this functional graph. It is only
                             /// guaranteed to contain a valid address during an event loop.
   const BranchNames_t                   fTmpBranches;
   std::vector<ROOT::Internal::TVBVec_t> fReaderValues     = {};
   std::vector<Long64_t>                 fLastCheckedEntry = {-1};
   std::vector<int>       fLastResult = {true}; // std::vector<bool> cannot be used in a MT context safely
   std::vector<ULong64_t> fAccepted   = {0};
   std::vector<ULong64_t> fRejected   = {0};
   const std::string      fName;

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
   void CreateSlots(unsigned int nSlots);
   void PrintReport() const;
};
using FilterBasePtr_t = std::shared_ptr<TDataFrameFilterBase>;
using FilterBaseVec_t = std::vector<FilterBasePtr_t>;

template <typename FilterF, typename PrevDataFrame>
class TDataFrameFilter final : public TDataFrameFilterBase {
   using BranchTypes_t = typename ROOT::Internal::TDFTraitsUtils::TFunctionTraits<FilterF>::Args_t;
   using TypeInd_t     = typename ROOT::Internal::TDFTraitsUtils::TGenStaticSeq<BranchTypes_t::fgSize>::Type_t;

   FilterF             fFilter;
   const BranchNames_t fBranches;
   PrevDataFrame &     fPrevData;

public:
   TDataFrameFilter(FilterF &&f, const BranchNames_t &bl, PrevDataFrame &pd, const std::string &name = "")
      : TDataFrameFilterBase(pd.GetImplPtr(), pd.GetTmpBranches(), name), fFilter(std::move(f)), fBranches(bl),
        fPrevData(pd)
   {
   }

   TDataFrameFilter(const TDataFrameFilter &) = delete;

   bool CheckFilters(unsigned int slot, Long64_t entry) final
   {
      if (entry != fLastCheckedEntry[slot]) {
         if (!fPrevData.CheckFilters(slot, entry)) {
            // a filter upstream returned false, cache the result
            fLastResult[slot] = false;
         } else {
            // evaluate this filter, cache the result
            auto passed = CheckFilterHelper(BranchTypes_t(), TypeInd_t(), slot, entry);
            passed ? ++fAccepted[slot] : ++fRejected[slot];
            fLastResult[slot] = passed;
         }
         fLastCheckedEntry[slot] = entry;
      }
      return fLastResult[slot];
   }

   template <int... S, typename... BranchTypes>
   bool CheckFilterHelper(ROOT::Internal::TDFTraitsUtils::TTypeList<BranchTypes...>,
                          ROOT::Internal::TDFTraitsUtils::TStaticSeq<S...>, unsigned int slot, Long64_t entry)
   {
      // Take each pointer in tvb, cast it to a pointer to the
      // correct specialization of TTreeReaderValue, and get its content.
      // S expands to a sequence of integers 0 to `sizeof...(types)-1
      // S and types are expanded simultaneously by "..."
      (void)slot;  // avoid bogus unused-but-set-parameter warning by gcc
      (void)entry; // avoid bogus unused-but-set-parameter warning by gcc
      return fFilter(ROOT::Internal::GetBranchValue(fReaderValues[slot][S], slot, entry, fBranches[S], fImplPtr,
                                                    ROOT::Internal::TDFTraitsUtils::TTypeList<BranchTypes>())...);
   }

   void BuildReaderValues(TTreeReader &r, unsigned int slot) final
   {
      fReaderValues[slot] = ROOT::Internal::BuildReaderValues(r, fBranches, fTmpBranches, BranchTypes_t(), TypeInd_t());
   }

   // recursive chain of `Report`s
   void Report() const final { PartialReport(); }

   void PartialReport() const final
   {
      fPrevData.PartialReport();
      PrintReport();
   }
};

class TDataFrameImpl : public std::enable_shared_from_this<TDataFrameImpl> {

   ROOT::Internal::ActionBaseVec_t fBookedActions;
   ROOT::Detail::FilterBaseVec_t   fBookedFilters;
   ROOT::Detail::FilterBaseVec_t   fBookedNamedFilters;
   std::map<std::string, TmpBranchBasePtr_t> fBookedBranches;
   std::vector<std::shared_ptr<bool>> fResProxyReadiness;
   ::TDirectory *                     fDirPtr{nullptr};
   TTree *                            fTree{nullptr};
   const BranchNames_t                fDefaultBranches;
   const unsigned int                 fNSlots{0};
   bool                               fHasRunAtLeastOnce{false};

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
   const TDataFrameBranchBase &GetBookedBranch(const std::string &name) const;
   void *GetTmpBranchValue(const std::string &branch, unsigned int slot, Long64_t entry);
   ::TDirectory *GetDirectory() const;
   std::string   GetTreeName() const;
   void Book(const ROOT::Internal::ActionBasePtr_t &actionPtr);
   void Book(const ROOT::Detail::FilterBasePtr_t &filterPtr);
   void Book(const ROOT::Detail::TmpBranchBasePtr_t &branchPtr);
   bool         CheckFilters(int, unsigned int);
   unsigned int GetNSlots() const;
   template <typename T>
   Experimental::TActionResultProxy<T> MakeActionResultProxy(const std::shared_ptr<T> &r)
   {
      auto        readiness = std::make_shared<bool>(false);
      const auto &df        = shared_from_this();
      auto        resPtr    = Experimental::TActionResultProxy<T>::MakeActionResultProxy(r, readiness, df);
      fResProxyReadiness.emplace_back(readiness);
      return resPtr;
   }
   bool HasRunAtLeastOnce() const { return fHasRunAtLeastOnce; }
   void Report() const;
   /// End of recursive chain of calls, does nothing
   void PartialReport() const {}
   void SetTree(TTree *tree) { fTree = tree; }
};

} // end NS ROOT::Detail

namespace Internal {

template <typename T>
T &GetBranchValue(TVBPtr_t &readerValue, unsigned int slot, Long64_t entry, const std::string &branch,
                  ROOT::Detail::TDataFrameImpl *df, TDFTraitsUtils::TTypeList<T>)
{
   if (readerValue == nullptr) {
      // temporary branch
      void *tmpBranchVal = df->GetTmpBranchValue(branch, slot, entry);
      return *static_cast<T *>(tmpBranchVal);
   } else {
      // real branch
      return **std::static_pointer_cast<TTreeReaderValue<T>>(readerValue);
   }
}

template <typename T>
std::array_view<T> GetBranchValue(TVBPtr_t &readerValue, unsigned int slot, Long64_t entry, const std::string &branch,
                                  ROOT::Detail::TDataFrameImpl *df, TDFTraitsUtils::TTypeList<std::array_view<T>>)
{
   if (readerValue == nullptr) {
      // temporary branch
      void *tmpBranchVal = df->GetTmpBranchValue(branch, slot, entry);
      auto &tra          = *static_cast<TTreeReaderArray<T> *>(tmpBranchVal);
      return std::array_view<T>(tra.begin(), tra.end());
   } else {
      // real branch
      auto &tra = *std::static_pointer_cast<TTreeReaderArray<T>>(readerValue);
      if (tra.GetSize() > 1 && 1 != (&tra[1] - &tra[0])) {
         std::string exceptionText = "Branch ";
         exceptionText += branch;
         exceptionText += " hangs from a non-split branch. For this reason, it cannot be accessed via an array_view. "
                          "Please read the top level branch instead.";
         throw std::runtime_error(exceptionText.c_str());
      }
      return std::array_view<T>(tra.begin(), tra.end());
   }
}

} // namespace Internal

} // namespace ROOT

#endif // ROOT_TDFNODES
