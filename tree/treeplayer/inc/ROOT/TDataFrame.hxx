// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
  \defgroup dataframe Data Frame
The ROOT Data Frame allows to analyse data stored in TTrees with a high level interface.
*/


#ifndef ROOT_TDATAFRAME
#define ROOT_TDATAFRAME

#include "TBranchElement.h"
#include "TH1F.h" // For Histo actions
#include "ROOT/RArrayView.hxx"
#include "ROOT/TDFOperations.hxx"
#include "ROOT/TDFTraitsUtils.hxx"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"
#include "TTreeReaderValue.h"

#include <algorithm> // std::find
#include <array>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

namespace ROOT {

using BranchNames = std::vector<std::string>;

// Fwd declarations
namespace Detail {
class TDataFrameImpl;
}

namespace Experimental {

/// Smart pointer for the return type of actions
/**
\class ROOT::Experimental::TActionResultProxy
\ingroup dataframe
\brief A wrapper around the result of TDataFrame actions able to trigger calculations lazily.
\tparam T Type of the action result

A smart pointer which allows to access the result of a TDataFrame action. The
methods of the encapsulated object can be accessed via the arrow operator.
Upon invocation of the arrow operator or dereferencing (`operator*`), the
loop on the events and calculations of all scheduled actions are executed
if needed.
It is possible to iterate on the result proxy if the proxied object is a collection.
~~~{.cpp}
for (auto& myItem : myResultProxy) { ... };
~~~
If iteration is not supported by the type of the proxied object, a compilation error is thrown.

*/
template <typename T>
class TActionResultProxy {
/// \cond HIDDEN_SYMBOLS
   template<typename V, bool isCont = ROOT::Internal::TDFTraitsUtils::TIsContainer<V>::fgValue>
   struct TIterationHelper{
      using Iterator_t = void;
      void GetBegin(const V& ){static_assert(sizeof(V) == 0, "It does not make sense to ask begin for this class.");}
      void GetEnd(const V& ){static_assert(sizeof(V) == 0, "It does not make sense to ask end for this class.");}
   };

   template<typename V>
   struct TIterationHelper<V,true>{
      using Iterator_t = decltype(std::begin(std::declval<V>()));
      static Iterator_t GetBegin(const V& v) {return std::begin(v);};
      static Iterator_t GetEnd(const V& v) {return std::end(v);};
   };
/// \endcond
   using SPT_t = std::shared_ptr<T> ;
   using SPTDFI_t = std::shared_ptr<ROOT::Detail::TDataFrameImpl>;
   using WPTDFI_t = std::weak_ptr<ROOT::Detail::TDataFrameImpl>;
   using ShrdPtrBool_t = std::shared_ptr<bool>;
   friend class ROOT::Detail::TDataFrameImpl;

   ShrdPtrBool_t fReadiness = std::make_shared<bool>(false); ///< State registered also in the TDataFrameImpl until the event loop is executed
   WPTDFI_t fFirstData;                                      ///< Original TDataFrame
   SPT_t fObjPtr;                                            ///< Shared pointer encapsulating the wrapped result
   /// Triggers the event loop in the TDataFrameImpl instance to which it's associated via the fFirstData
   void TriggerRun();
   /// Get the pointer to the encapsulated result.
   /// Ownership is not transferred to the caller.
   /// Triggers event loop and execution of all actions booked in the associated TDataFrameImpl.
   T *Get()
   {
      if (!*fReadiness) TriggerRun();
      return fObjPtr.get();
   }
   TActionResultProxy(SPT_t objPtr, ShrdPtrBool_t readiness, SPTDFI_t firstData)
      : fReadiness(readiness), fFirstData(firstData), fObjPtr(objPtr) { }
   /// Factory to allow to keep the constructor private
   static TActionResultProxy<T> MakeActionResultProxy(SPT_t objPtr, ShrdPtrBool_t readiness, SPTDFI_t firstData)
   {
      return TActionResultProxy(objPtr, readiness, firstData);
   }
public:
   TActionResultProxy() = delete;
   /// Get a reference to the encapsulated object.
   /// Triggers event loop and execution of all actions booked in the associated TDataFrameImpl.
   T &operator*() { return *Get(); }
   /// Get a pointer to the encapsulated object.
   /// Ownership is not transferred to the caller.
   /// Triggers event loop and execution of all actions booked in the associated TDataFrameImpl.
   T *operator->() { return Get(); }
   /// Return an iterator to the beginning of the contained object if this makes
   /// sense, throw a compilation error otherwise
   typename TIterationHelper<T>::Iterator_t begin()
   {
      if (!*fReadiness) TriggerRun();
      return TIterationHelper<T>::GetBegin(*fObjPtr);
   }
   /// Return an iterator to the end of the contained object if this makes
   /// sense, throw a compilation error otherwise
   typename TIterationHelper<T>::Iterator_t end()
   {
      if (!*fReadiness) TriggerRun();
      return TIterationHelper<T>::GetEnd(*fObjPtr);
   }
};

} // end NS Experimental

} // end NS ROOT

// Internal classes

namespace ROOT {

namespace Detail {
class TDataFrameImpl;
}

namespace Internal {

unsigned int GetNSlots();

using TVBPtr_t = std::shared_ptr<TTreeReaderValueBase>;
using TVBVec_t = std::vector<TVBPtr_t>;

template<typename BranchType>
std::shared_ptr<ROOT::Internal::TTreeReaderValueBase>
ReaderValueOrArray(TTreeReader& r, const std::string& branch, TDFTraitsUtils::TTypeList<BranchType>) {
   return std::make_shared<TTreeReaderValue<BranchType>>(r, branch.c_str());
}

template<typename BranchType>
std::shared_ptr<ROOT::Internal::TTreeReaderValueBase>
ReaderValueOrArray(TTreeReader& r, const std::string& branch, TDFTraitsUtils::TTypeList<std::array_view<BranchType>>) {
   return std::make_shared<TTreeReaderArray<BranchType>>(r, branch.c_str());
}

template <int... S, typename... BranchTypes>
TVBVec_t BuildReaderValues(TTreeReader &r, const BranchNames &bl, const BranchNames &tmpbl,
                           TDFTraitsUtils::TTypeList<BranchTypes...>,
                           TDFTraitsUtils::TStaticSeq<S...>)
{
   // isTmpBranch has length bl.size(). Elements are true if the corresponding
   // branch is a temporary branch created with AddBranch, false if they are
   // actual branches present in the TTree.
   std::array<bool, sizeof...(S)> isTmpBranch;
   for (unsigned int i = 0; i < isTmpBranch.size(); ++i)
      isTmpBranch[i] = std::find(tmpbl.begin(), tmpbl.end(), bl.at(i)) != tmpbl.end();

   // Build vector of pointers to TTreeReaderValueBase.
   // tvb[i] points to a TTreeReader{Value,Array} specialized for the i-th BranchType,
   // corresponding to the i-th branch in bl
   // For temporary branches (declared with AddBranch) a nullptr is created instead
   // S is expected to be a sequence of sizeof...(BranchTypes) integers
   // Note that here TTypeList only contains one single type
   TVBVec_t tvb{isTmpBranch[S] ? nullptr : ReaderValueOrArray(r, bl.at(S), TDFTraitsUtils::TTypeList<BranchTypes>())
                ...}; // "..." expands BranchTypes and S simultaneously

   return tvb;
}

template <typename Filter>
void CheckFilter(Filter)
{
   using FilterRet_t = typename TDFTraitsUtils::TFunctionTraits<Filter>::Ret_t;
   static_assert(std::is_same<FilterRet_t, bool>::value, "filter functions must return a bool");
}

void CheckTmpBranch(const std::string& branchName, TTree *treePtr);

/// Returns local BranchNames or default BranchNames according to which one should be used
const BranchNames &PickBranchNames(unsigned int nArgs, const BranchNames &bl, const BranchNames &defBl);

class TDataFrameActionBase {
protected:
   std::vector<TVBVec_t> fReaderValues;
public:
   virtual ~TDataFrameActionBase() {}
   virtual void Run(unsigned int slot, Long64_t entry) = 0;
   virtual void BuildReaderValues(TTreeReader &r, unsigned int slot) = 0;
   void CreateSlots(unsigned int nSlots);

};

using ActionBasePtr_t = std::shared_ptr<TDataFrameActionBase>;
using ActionBaseVec_t = std::vector<ActionBasePtr_t>;

// Forward declarations
template<typename T>
T &GetBranchValue(TVBPtr_t &readerValues, unsigned int slot, Long64_t entry, const std::string& branch,
                  std::weak_ptr<Detail::TDataFrameImpl> df, TDFTraitsUtils::TTypeList<T>);
template<typename T>
std::array_view<T> GetBranchValue(TVBPtr_t &readerValues, unsigned int slot, Long64_t entry, const std::string& branch,
                  std::weak_ptr<Detail::TDataFrameImpl> df, TDFTraitsUtils::TTypeList<std::array_view<T>>);


template <typename F, typename PrevDataFrame>
class TDataFrameAction final : public TDataFrameActionBase {
   using BranchTypes_t = typename TDFTraitsUtils::TRemoveFirst<typename TDFTraitsUtils::TFunctionTraits<F>::Args_t>::Types_t;
   using TypeInd_t = typename TDFTraitsUtils::TGenStaticSeq<BranchTypes_t::fgSize>::Type_t;

   F fAction;
   const BranchNames fBranches;
   const BranchNames fTmpBranches;
   PrevDataFrame &fPrevData;
   std::weak_ptr<ROOT::Detail::TDataFrameImpl> fFirstData;

public:
   TDataFrameAction(F f, const BranchNames &bl, std::shared_ptr<PrevDataFrame> pd)
      : fAction(f), fBranches(bl), fTmpBranches(pd->GetTmpBranches()), fPrevData(*pd),
        fFirstData(pd->GetDataFrame()) { }

   TDataFrameAction(const TDataFrameAction &) = delete;

   void Run(unsigned int slot, Long64_t entry)
   {
      // check if entry passes all filters
      if (CheckFilters(slot, entry)) ExecuteAction(slot, entry);
   }

   bool CheckFilters(unsigned int slot, Long64_t entry)
   {
      // start the recursive chain of CheckFilters calls
      return fPrevData.CheckFilters(slot, entry);
   }

   void ExecuteAction(unsigned int slot, Long64_t entry) { ExecuteActionHelper(slot, entry, TypeInd_t(), BranchTypes_t()); }

   void BuildReaderValues(TTreeReader &r, unsigned int slot)
   {
      fReaderValues[slot] = ROOT::Internal::BuildReaderValues(r, fBranches, fTmpBranches, BranchTypes_t(), TypeInd_t());
   }

   template <int... S, typename... BranchTypes>
   void ExecuteActionHelper(unsigned int slot, Long64_t entry,
                            TDFTraitsUtils::TStaticSeq<S...>,
                            TDFTraitsUtils::TTypeList<BranchTypes...>)
   {
      // Take each pointer in tvb, cast it to a pointer to the
      // correct specialization of TTreeReaderValue, and get its content.
      // S expands to a sequence of integers 0 to sizeof...(types)-1
      // S and types are expanded simultaneously by "..."
      (void) entry; // avoid bogus unused-but-set-parameter warning by gcc
      fAction(slot, GetBranchValue(fReaderValues[slot][S], slot, entry,
                                   fBranches[S], fFirstData, TDFTraitsUtils::TTypeList<BranchTypes>())
              ...);
   }
};

enum class EActionType : short { kHisto1D, kMin, kMax, kMean };

// Utilities to accommodate v7
namespace TDFV7Utils {

template<typename T, bool ISV7HISTO = !std::is_base_of<TH1, T>::value>
struct TIsV7Histo {
   const static bool fgValue = ISV7HISTO;
};

template<typename T, bool ISV7HISTO = TIsV7Histo<T>::fgValue>
struct Histo {
   static void SetCanExtendAllAxes(T& h)
   {
      h.SetCanExtend(::TH1::kAllAxes);
   }
   static bool HasAxisLimits(T& h)
   {
      auto xaxis = h.GetXaxis();
      return !(xaxis->GetXmin() == 0. && xaxis->GetXmax() == 0.);
   }
};

template<typename T>
struct Histo<T, true> {
   static void SetCanExtendAllAxes(T&) { }
   static bool HasAxisLimits(T&) {return true;}
};

} // end NS TDFV7Utils

} // end NS Internal

namespace Detail {

class TDataFrameGuessedType{};

// forward declarations for TDataFrameInterface
template <typename F, typename PrevData>
class TDataFrameFilter;
template <typename F, typename PrevData>
class TDataFrameBranch;
class TDataFrameImpl;
}

namespace Experimental {

class TDataFrame;

/**
* \class ROOT::Experimental::TDataFrameInterface
* \ingroup dataframe
* \brief The public interface to the TDataFrame federation of classes: TDataFrameImpl, TDataFrameFilter, TDataFrameBranch
* \tparam T One of the TDataFrameImpl, TDataFrameFilter, TDataFrameBranch classes. The user never specifies this type manually.
*/
template <typename Proxied>
class TDataFrameInterface {
   friend std::string cling::printValue(ROOT::Experimental::TDataFrame *tdf); // For a nice printing at the prompt
   template<typename T> friend class TDataFrameInterface;
public:

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] f Function, lambda expression, functor class or any other callable object. It must return a `bool` signalling whether the event has passed the selection (true) or not (false).
   /// \param[in] bl Names of the branches in input to the filter function.
   ///
   /// Append a filter node at the point of the call graph corresponding to the
   /// object this method is called on.
   /// The callable `f` should not have side-effects (e.g. modification of an
   /// external or static variable) to ensure correct results when implicit
   /// multi-threading is active.
   ///
   /// TDataFrame only evaluates filters when necessary: if multiple filters
   /// are chained one after another, they are executed in order and the first
   /// one returning false causes the event to be discarded.
   /// Even if multiple actions or transformations depend on the same filter,
   /// it is executed once per entry. If its result is requested more than
   /// once, the cached result is served.
   template <typename F>
   TDataFrameInterface<ROOT::Detail::TDataFrameFilter<F, Proxied>> Filter(F f, const BranchNames &bl = {}, const std::string& name = "")
   {
      ROOT::Internal::CheckFilter(f);
      auto df = GetDataFrameChecked();
      const BranchNames &defBl = df->GetDefaultBranches();
      auto nArgs = ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::Args_t::fgSize;
      const BranchNames &actualBl = ROOT::Internal::PickBranchNames(nArgs, bl, defBl);
      using DFF_t = ROOT::Detail::TDataFrameFilter<F, Proxied>;
      auto FilterPtr = std::make_shared<DFF_t> (f, actualBl, fProxiedPtr, name);
      TDataFrameInterface<DFF_t> tdf_f(FilterPtr);
      df->Book(FilterPtr);
      return tdf_f;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a temporary branch
   /// \param[in] name The name of the temporary branch.
   /// \param[in] expression Function, lambda expression, functor class or any other callable object producing the temporary value. Returns the value that will be assigned to the temporary branch.
   /// \param[in] bl Names of the branches in input to the producer function.
   ///
//    /// Create a temporary branch that will be visible from all subsequent nodes
   /// of the functional chain. The `expression` is only evaluated for entries that pass
   /// all the preceding filters.
   /// A new variable is created called `name`, accessible as if it was contained
   /// in the dataset from subsequent transformations/actions.
   ///
   /// Use cases include:
   ///
   /// * caching the results of complex calculations for easy and efficient multiple access
   /// * extraction of quantities of interest from complex objects
   /// * branch aliasing, i.e. changing the name of a branch
   ///
   /// An exception is thrown if the name of the new branch is already in use
   /// for another branch in the TTree.
   template <typename F>
   TDataFrameInterface<ROOT::Detail::TDataFrameBranch<F, Proxied>>
   AddBranch(const std::string &name, F expression, const BranchNames &bl = {})
   {
      auto df = GetDataFrameChecked();
      ROOT::Internal::CheckTmpBranch(name, df->GetTree());
      const BranchNames &defBl = df->GetDefaultBranches();
      auto nArgs = ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::Args_t::fgSize;
      const BranchNames &actualBl = ROOT::Internal::PickBranchNames(nArgs, bl, defBl);
      using DFB_t = ROOT::Detail::TDataFrameBranch<F, Proxied>;
      auto BranchPtr = std::make_shared<DFB_t>(name, expression, actualBl, fProxiedPtr);
      TDataFrameInterface<DFB_t> tdf_b(BranchPtr);
      df->Book(BranchPtr);
      return tdf_b;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined function on each entry (*instant action*)
   /// \param[in] f Function, lambda expression, functor class or any other callable object performing user defined calculations.
   /// \param[in] bl Names of the branches in input to the user function.
   ///
   /// The callable `f` is invoked once per entry. This is an *instant action*:
   /// upon invocation, an event loop as well as execution of all scheduled actions
   /// is triggered.
   /// Users are responsible for the thread-safety of this callable when executing
   /// with implicit multi-threading enabled (i.e. ROOT::EnableImplicitMT).
   template <typename F>
   void Foreach(F f, const BranchNames &bl = {})
   {
      namespace IU = ROOT::Internal::TDFTraitsUtils;
      using Args_t = typename IU::TFunctionTraits<decltype(f)>::ArgsNoDecay_t;
      using Ret_t = typename IU::TFunctionTraits<decltype(f)>::Ret_t;
      auto fWithSlot = IU::AddSlotParameter<Ret_t>(f, Args_t());
      ForeachSlot(fWithSlot, bl);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined function requiring a processing slot index on each entry (*instant action*)
   /// \param[in] f Function, lambda expression, functor class or any other callable object performing user defined calculations.
   /// \param[in] bl Names of the branches in input to the user function.
   ///
   /// Same as `Foreach`, but the user-defined function takes an extra
   /// `unsigned int` as its first parameter, the *processing slot index*.
   /// This *slot index* will be assigned a different value, `0` to `poolSize - 1`,
   /// for each thread of execution.
   /// This is meant as a helper in writing thread-safe `Foreach`
   /// actions when using `TDataFrame` after `ROOT::EnableImplicitMT()`.
   /// The user-defined processing callable is able to follow different
   /// *streams of processing* indexed by the first parameter.
   /// `ForeachSlot` works just as well with single-thread execution: in that
   /// case `slot` will always be `0`.
   template<typename F>
   void ForeachSlot(F f, const BranchNames &bl = {}) {
      auto df = GetDataFrameChecked();
      const BranchNames &defBl= df->GetDefaultBranches();
      auto nArgs = ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::Args_t::fgSize;
      const BranchNames &actualBl = ROOT::Internal::PickBranchNames(nArgs-1, bl, defBl);
      using DFA_t  = ROOT::Internal::TDataFrameAction<decltype(f), Proxied>;
      df->Book(std::make_shared<DFA_t>(f, actualBl, fProxiedPtr));
      df->Run();
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the number of entries processed (*lazy action*)
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   TActionResultProxy<unsigned int> Count()
   {
      auto df = GetDataFrameChecked();
      unsigned int nSlots = df->GetNSlots();
      auto cShared = std::make_shared<unsigned int>(0);
      auto c = df->MakeActionResultProxy(cShared);
      auto cPtr = cShared.get();
      auto cOp = std::make_shared<ROOT::Internal::Operations::CountOperation>(cPtr, nSlots);
      auto countAction = [cOp](unsigned int slot) mutable { cOp->Exec(slot); };
      BranchNames bl = {};
      using DFA_t = ROOT::Internal::TDataFrameAction<decltype(countAction), Proxied>;
      df->Book(std::shared_ptr<DFA_t>(new DFA_t(countAction, bl, fProxiedPtr)));
      return c;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return a collection of values of a branch (*lazy action*)
   /// \tparam T The type of the branch.
   /// \tparam COLL The type of collection used to store the values.
   /// \param[in] branchName The name of the branch of which the values are to be collected
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T, typename COLL = std::vector<T>>
   TActionResultProxy<COLL> Take(const std::string &branchName = "")
   {
      auto df = GetDataFrameChecked();
      unsigned int nSlots = df->GetNSlots();
      auto theBranchName(branchName);
      GetDefaultBranchName(theBranchName, "get the values of the branch");
      auto valuesPtr = std::make_shared<COLL>();
      auto values = df->MakeActionResultProxy(valuesPtr);
      auto getOp = std::make_shared<ROOT::Internal::Operations::TakeOperation<T,COLL>>(valuesPtr, nSlots);
      auto getAction = [getOp] (unsigned int slot , const T &v) mutable { getOp->Exec(v, slot); };
      BranchNames bl = {theBranchName};
      using DFA_t = ROOT::Internal::TDataFrameAction<decltype(getAction), Proxied>;
      df->Book(std::shared_ptr<DFA_t>(new DFA_t(getAction, bl, fProxiedPtr)));
      return values;
   }


   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the values of a branch (*lazy action*)
   /// \tparam T The type of the branch the values of which are used to fill the histogram.
   /// \param[in] branchName The name of the branch of which the values are to be collected.
   /// \param[in] model The model to be considered to build the new return value.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   /// The returned histogram is independent of the input one.
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   /// The user renounces to the ownership of the model. The value to be used is the
   /// returned one.
   template <typename T = ROOT::Detail::TDataFrameGuessedType>
   TActionResultProxy<::TH1F> Histo(const std::string &branchName, ::TH1F &&model)
   {
      auto theBranchName(branchName);
      GetDefaultBranchName(theBranchName, "fill the histogram");
      auto h = std::make_shared<::TH1F>(model);
      return CreateAction<T, ROOT::Internal::EActionType::kHisto1D>(theBranchName, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the values of a branch (*lazy action*)
   /// \tparam T The type of the branch the values of which are used to fill the histogram.
   /// \param[in] branchName The name of the branch of which the values are to be collected.
   /// \param[in] nbins The number of bins.
   /// \param[in] minVal The lower value of the xaxis.
   /// \param[in] maxVal The upper value of the xaxis.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// If no axes boundaries are specified, all entries are buffered: at the end of
   /// the loop on the entries, the histogram is filled. If the axis boundaries are
   /// specified, the histogram (or histograms in the parallel case) are filled. This
   /// latter mode may result in a reduced memory footprint.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T = ROOT::Detail::TDataFrameGuessedType>
   TActionResultProxy<::TH1F> Histo(const std::string &branchName = "", int nBins = 128, double minVal = 0.,
                                double maxVal = 0.)
   {
      auto theBranchName(branchName);
      GetDefaultBranchName(theBranchName, "fill the histogram");
      auto h = std::make_shared<::TH1F>("", "", nBins, minVal, maxVal);
      if (minVal == maxVal) {
         ROOT::Internal::TDFV7Utils::Histo<::TH1F>::SetCanExtendAllAxes(*h);
      }
      return CreateAction<T, ROOT::Internal::EActionType::kHisto1D>(theBranchName, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the minimum of processed branch values (*lazy action*)
   /// \tparam T The type of the branch.
   /// \param[in] branchName The name of the branch to be treated.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T = ROOT::Detail::TDataFrameGuessedType>
   TActionResultProxy<double> Min(const std::string &branchName = "")
   {
      auto theBranchName(branchName);
      GetDefaultBranchName(theBranchName, "calculate the minimum");
      auto minV = std::make_shared<double>(std::numeric_limits<double>::max());
      return CreateAction<T, ROOT::Internal::EActionType::kMin>(theBranchName, minV);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the maximum of processed branch values (*lazy action*)
   /// \tparam T The type of the branch.
   /// \param[in] branchName The name of the branch to be treated.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T = ROOT::Detail::TDataFrameGuessedType>
   TActionResultProxy<double> Max(const std::string &branchName = "")
   {
      auto theBranchName(branchName);
      GetDefaultBranchName(theBranchName, "calculate the maximum");
      auto maxV = std::make_shared<double>(std::numeric_limits<double>::min());
      return CreateAction<T, ROOT::Internal::EActionType::kMax>(theBranchName, maxV);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the mean of processed branch values (*lazy action*)
   /// \tparam T The type of the branch.
   /// \param[in] branchName The name of the branch to be treated.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T = ROOT::Detail::TDataFrameGuessedType>
   TActionResultProxy<double> Mean(const std::string &branchName = "")
   {
      auto theBranchName(branchName);
      GetDefaultBranchName(theBranchName, "calculate the mean");
      auto meanV = std::make_shared<double>(0);
      return CreateAction<T, ROOT::Internal::EActionType::kMean>(theBranchName, meanV);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Print filtering statistics on screen
   ///
   /// Calling `Report` on the main `TDataFrame` object prints stats for
   /// all named filters in the call graph. Calling this method on a
   /// stored chain state (i.e. a graph node different from the first) prints
   /// the stats for all named filters in the chain section between the original
   /// `TDataFrame` and that node (included). Stats are printed in the same
   /// order as the named filters have been added to the graph.
   void Report() {
      auto df = GetDataFrameChecked();
      if (!df->HasRunAtLeastOnce())
         Info("TDataFrame::Report", "Warning: the event-loop has not been run yet, all reports are empty");
      else
         fProxiedPtr->Report();
   }

private:

   /// \cond HIDDEN_SYMBOLS
   template <typename BranchType, typename ActionResultType, enum ROOT::Internal::EActionType, typename ThisType, bool isGuessedType = std::is_same<BranchType, ROOT::Detail::TDataFrameGuessedType>::value>
   struct SimpleAction {};

   template <typename BranchType, typename ActionResultType, enum ROOT::Internal::EActionType ART, typename ThisType>
   struct SimpleAction<BranchType, ActionResultType, ART, ThisType, true> {
      static TActionResultProxy<ActionResultType> BuildAndBook(ThisType thisFrame, const std::string &,
                                                               std::shared_ptr<ActionResultType> r, unsigned int)
      {
         // This code will never be executed!
         auto df = thisFrame->GetDataFrameChecked();
         return df->MakeActionResultProxy(r);
      }
   };

   template <typename BranchType, typename ThisType>
   struct SimpleAction<BranchType, ::TH1F, ROOT::Internal::EActionType::kHisto1D, ThisType, false> {
      static TActionResultProxy<::TH1F> BuildAndBook(ThisType thisFrame, const std::string &theBranchName,
                                                 std::shared_ptr<::TH1F> h, unsigned int nSlots)
      {
         // we use a shared_ptr so that the operation has the same scope of the lambda
         // and therefore of the TDataFrameAction that contains it: merging of results
         // from different threads is performed in the operation's destructor, at the
         // moment when the TDataFrameAction is deleted by TDataFrameImpl
         BranchNames bl = {theBranchName};
         auto df = thisFrame->GetDataFrameChecked();
         auto hasAxisLimits = ROOT::Internal::TDFV7Utils::Histo<::TH1F>::HasAxisLimits(*h);

         if (hasAxisLimits) {
            auto fillTOOp = std::make_shared<ROOT::Internal::Operations::FillTOOperation<::TH1F>>(h, nSlots);
            auto fillLambda = [fillTOOp](unsigned int slot, const BranchType &v) mutable { fillTOOp->Exec(v, slot); };
            using DFA_t = ROOT::Internal::TDataFrameAction<decltype(fillLambda), Proxied>;
            df->Book(std::make_shared<DFA_t>(fillLambda, bl, thisFrame->fProxiedPtr));
         } else {
            auto fillOp = std::make_shared<ROOT::Internal::Operations::FillOperation<::TH1F>>(h, nSlots);
            auto fillLambda = [fillOp](unsigned int slot, const BranchType &v) mutable { fillOp->Exec(v, slot); };
            using DFA_t = ROOT::Internal::TDataFrameAction<decltype(fillLambda), Proxied>;
            df->Book(std::make_shared<DFA_t>(fillLambda, bl, thisFrame->fProxiedPtr));
         }
         return df->MakeActionResultProxy(h);
      }
   };

   template <typename BranchType, typename ThisType, typename ActionResultType>
   struct SimpleAction<BranchType, ActionResultType, ROOT::Internal::EActionType::kMin, ThisType, false> {
      static TActionResultProxy<ActionResultType> BuildAndBook(ThisType thisFrame, const std::string &theBranchName,
                                                             std::shared_ptr<ActionResultType> minV, unsigned int nSlots)
      {
         // see "TActionResultProxy<::TH1F> BuildAndBook" for why this is a shared_ptr
         auto minOp = std::make_shared<ROOT::Internal::Operations::MinOperation>(minV.get(), nSlots);
         auto minOpLambda = [minOp](unsigned int slot, const BranchType &v) mutable { minOp->Exec(v, slot); };
         BranchNames bl = {theBranchName};
         using DFA_t = ROOT::Internal::TDataFrameAction<decltype(minOpLambda), Proxied>;
         auto df = thisFrame->GetDataFrameChecked();
         df->Book(std::make_shared<DFA_t>(minOpLambda, bl, thisFrame->fProxiedPtr));
         return df->MakeActionResultProxy(minV);
      }
   };

   template <typename BranchType, typename ThisType, typename ActionResultType>
   struct SimpleAction<BranchType, ActionResultType, ROOT::Internal::EActionType::kMax, ThisType, false> {
      static TActionResultProxy<ActionResultType> BuildAndBook(ThisType thisFrame, const std::string &theBranchName,
                                                             std::shared_ptr<ActionResultType> maxV, unsigned int nSlots)
      {
         // see "TActionResultProxy<::TH1F> BuildAndBook" for why this is a shared_ptr
         auto maxOp = std::make_shared<ROOT::Internal::Operations::MaxOperation>(maxV.get(), nSlots);
         auto maxOpLambda = [maxOp](unsigned int slot, const BranchType &v) mutable { maxOp->Exec(v, slot); };
         BranchNames bl = {theBranchName};
         using DFA_t = ROOT::Internal::TDataFrameAction<decltype(maxOpLambda), Proxied>;
         auto df = thisFrame->GetDataFrameChecked();
         df->Book(std::make_shared<DFA_t>(maxOpLambda, bl, thisFrame->fProxiedPtr));
         return df->MakeActionResultProxy(maxV);
      }
   };

   template <typename BranchType, typename ThisType, typename ActionResultType>
   struct SimpleAction<BranchType, ActionResultType, ROOT::Internal::EActionType::kMean, ThisType, false> {
      static TActionResultProxy<ActionResultType> BuildAndBook(ThisType thisFrame, const std::string &theBranchName,
                                                             std::shared_ptr<ActionResultType> meanV, unsigned int nSlots)
      {
         // see "TActionResultProxy<::TH1F> BuildAndBook" for why this is a shared_ptr
         auto meanOp = std::make_shared<ROOT::Internal::Operations::MeanOperation>(meanV.get(), nSlots);
         auto meanOpLambda = [meanOp](unsigned int slot, const BranchType &v) mutable { meanOp->Exec(v, slot); };
         BranchNames bl = {theBranchName};
         using DFA_t = ROOT::Internal::TDataFrameAction<decltype(meanOpLambda), Proxied>;
         auto df = thisFrame->GetDataFrameChecked();
         df->Book(std::make_shared<DFA_t>(meanOpLambda, bl, thisFrame->fProxiedPtr));
         return df->MakeActionResultProxy(meanV);
      }
   };

   /// \endcond

   template <typename BranchType, ROOT::Internal::EActionType ActionType, typename ActionResultType>
   TActionResultProxy<ActionResultType> CreateAction(const std::string & theBranchName,
                                                   std::shared_ptr<ActionResultType> r)
   {
      // More types can be added at will at the cost of some compilation time and size of binaries.
      using ART_t = ActionResultType;
      using TT_t = decltype(this);
      const auto at = ActionType;
      auto df = GetDataFrameChecked();
      unsigned int nSlots = df->GetNSlots();

      // In this case the type is specified by the user. We do not need to guess it.
      // Given that the boolean is known at compile time, the rest of the method will not be compiled.
      // All this would be perfectly expressed by a constexpr if.
      constexpr bool isGuessedType = std::is_same<BranchType, ROOT::Detail::TDataFrameGuessedType>::value;
      if (!isGuessedType) return SimpleAction<BranchType, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);

      auto tree = static_cast<TTree*>(df->GetDirectory()->Get(df->GetTreeName().c_str()));
      auto branch = tree->GetBranch(theBranchName.c_str());

      if (!branch) {
         // temporary branch
         const auto &type_id = df->GetBookedBranch(theBranchName).GetTypeId();
         if (type_id == typeid(char)) {
            return SimpleAction<char, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         } else if (type_id == typeid(int)) {
            return SimpleAction<int, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         } else if (type_id == typeid(double)) {
            return SimpleAction<double, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         } else if (type_id == typeid(std::vector<double>)) {
            return SimpleAction<std::vector<double>, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         } else if (type_id == typeid(std::vector<float>)) {
            return SimpleAction<std::vector<float>, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         }
      }
      // real branch
      auto branchEl = dynamic_cast<TBranchElement *>(branch);
      if (!branchEl) { // This is a fundamental type
         auto title    = branch->GetTitle();
         auto typeCode = title[strlen(title) - 1];
         if (typeCode == 'B') {
            return SimpleAction<char, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         }
         // else if (typeCode == 'b') { return SimpleAction<Uchar, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots); }
         // else if (typeCode == 'S') { return SimpleAction<Short_t, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots); }
         // else if (typeCode == 's') { return SimpleAction<UShort_t, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots); }
         else if (typeCode == 'I') {
            return SimpleAction<int, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         }
         // else if (typeCode == 'i') { return SimpleAction<unsigned int , ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots); }
         // else if (typeCode == 'F') { return SimpleAction<float, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots); }
         else if (typeCode == 'D') {
            return SimpleAction<double, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         }
         // else if (typeCode == 'L') { return SimpleAction<Long64_t, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots); }
         // else if (typeCode == 'l') { return SimpleAction<ULong64_t, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots); }
         else if (typeCode == 'O') {
            return SimpleAction<bool, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         }
      } else {
         std::string typeName = branchEl->GetTypeName();
         if (typeName == "vector<double>") {
            return SimpleAction<std::vector<double>, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         } else if (typeName == "vector<float>") {
            return SimpleAction<std::vector<float>, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         }
      }

      std::string exceptionText = "The type of branch ";
      exceptionText += theBranchName;
      exceptionText += " could not be guessed. Please specify one.";
      throw std::runtime_error(exceptionText.c_str());
   }

protected:
   /// Get the TDataFrameImpl if reachable. If not, throw.
   std::shared_ptr<ROOT::Detail::TDataFrameImpl> GetDataFrameChecked()
   {
      auto df = fProxiedPtr->GetDataFrame().lock();
      if (!df) {
         throw std::runtime_error("The main TDataFrame is not reachable: did it go out of scope?");
      }
      return df;
   }

   void GetDefaultBranchName(std::string &theBranchName, const std::string &actionNameForErr)
   {
      if (theBranchName.empty()) {
         // Try the default branch if possible
         auto df = GetDataFrameChecked();
         const BranchNames &defBl = df->GetDefaultBranches();
         if (defBl.size() == 1) {
            theBranchName = defBl[0];
         } else {
            std::string msg("No branch in input to ");
            msg += actionNameForErr;
            msg += " and default branch list has size ";
            msg += std::to_string(defBl.size());
            msg += ", need 1";
            throw std::runtime_error(msg);
         }
      }
   }
   TDataFrameInterface(std::shared_ptr<Proxied> proxied) : fProxiedPtr(proxied) {}
   std::shared_ptr<Proxied> fProxiedPtr;
};

class TDataFrame : public TDataFrameInterface<ROOT::Detail::TDataFrameImpl> {
   void Init();
public:
   TDataFrame(const std::string &treeName, ::TDirectory *dirPtr, const BranchNames &defaultBranches = {});
   TDataFrame(TTree &tree, const BranchNames &defaultBranches = {});
};

} // end NS Experimental

namespace Detail {

class TDataFrameBranchBase {
protected:
   std::weak_ptr<TDataFrameImpl> fFirstData;
   BranchNames fTmpBranches;
   const std::string fName;
public:
   TDataFrameBranchBase(std::weak_ptr<TDataFrameImpl> df, BranchNames branches, const std::string &name);
   virtual ~TDataFrameBranchBase() {}
   virtual void BuildReaderValues(TTreeReader &r, unsigned int slot) = 0;
   virtual void CreateSlots(unsigned int nSlots) = 0;
   virtual void *GetValue(unsigned int slot, Long64_t entry) = 0;
   virtual const std::type_info &GetTypeId() const = 0;
   std::string GetName() const;
   BranchNames GetTmpBranches() const;
};
using TmpBranchBasePtr_t = std::shared_ptr<TDataFrameBranchBase>;

template <typename F, typename PrevData>
class TDataFrameBranch final : public TDataFrameBranchBase {
   using BranchTypes_t = typename Internal
   ::TDFTraitsUtils::TFunctionTraits<F>::Args_t;
   using TypeInd_t = typename ROOT::Internal::TDFTraitsUtils::TGenStaticSeq<BranchTypes_t::fgSize>::Type_t;
   using Ret_t = typename ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::Ret_t;

   F fExpression;
   const BranchNames fBranches;

   std::vector<ROOT::Internal::TVBVec_t> fReaderValues;
   std::vector<std::shared_ptr<Ret_t>> fLastResultPtr;
   PrevData &fPrevData;
   std::vector<Long64_t> fLastCheckedEntry = {-1};

public:
   TDataFrameBranch(const std::string &name, F expression, const BranchNames &bl, std::shared_ptr<PrevData> pd)
      : TDataFrameBranchBase(pd->GetDataFrame(), pd->GetTmpBranches(), name), fExpression(expression), fBranches(bl), fPrevData(*pd)
   {
      fTmpBranches.emplace_back(name);
   }

   TDataFrameBranch(const TDataFrameBranch &) = delete;

   std::weak_ptr<TDataFrameImpl> GetDataFrame() const { return fFirstData; }

   void BuildReaderValues(TTreeReader &r, unsigned int slot)
   {
      fReaderValues[slot] = ROOT::Internal::BuildReaderValues(r, fBranches, fTmpBranches, BranchTypes_t(), TypeInd_t());
   }

   void *GetValue(unsigned int slot, Long64_t entry)
   {
      if (entry != fLastCheckedEntry[slot]) {
         // evaluate this filter, cache the result
         auto newValuePtr = GetValueHelper(BranchTypes_t(), TypeInd_t(), slot, entry);
         fLastResultPtr[slot] = newValuePtr;
         fLastCheckedEntry[slot] = entry;
      }
      return static_cast<void *>(fLastResultPtr[slot].get());
   }

   const std::type_info &GetTypeId() const { return typeid(Ret_t); }

   void CreateSlots(unsigned int nSlots)
   {
      fReaderValues.resize(nSlots);
      fLastCheckedEntry.resize(nSlots, -1);
      fLastResultPtr.resize(nSlots);
   }

   bool CheckFilters(unsigned int slot, Long64_t entry)
   {
      // dummy call: it just forwards to the previous object in the chain
      return fPrevData.CheckFilters(slot, entry);
   }

   template <int... S, typename... BranchTypes>
   std::shared_ptr<Ret_t> GetValueHelper(Internal::TDFTraitsUtils::TTypeList<BranchTypes...>,
                                             ROOT::Internal::TDFTraitsUtils::TStaticSeq<S...>,
                                             unsigned int slot, Long64_t entry)
   {
      auto valuePtr = std::make_shared<Ret_t>(fExpression(
         Internal::GetBranchValue(fReaderValues[slot][S], slot, entry, fBranches[S],
                                  fFirstData, Internal::TDFTraitsUtils::TTypeList<BranchTypes>())...));
      return valuePtr;
   }

   // recursive chain of `Report`s
   // TDataFrameBranch simply forwards the call to the previous node
   void Report() const {
      fPrevData.PartialReport();
   }

   void PartialReport() const {
      fPrevData.PartialReport();
   }

};

class TDataFrameFilterBase {
protected:
   std::weak_ptr<TDataFrameImpl> fFirstData;
   const BranchNames fTmpBranches;
   std::vector<ROOT::Internal::TVBVec_t> fReaderValues = {};
   std::vector<Long64_t> fLastCheckedEntry = {-1};
   std::vector<int> fLastResult = {true}; // std::vector<bool> cannot be used in a MT context safely
   std::vector<ULong64_t> fAccepted = {0};
   std::vector<ULong64_t> fRejected = {0};
   const std::string fName;

public:
   TDataFrameFilterBase(std::weak_ptr<TDataFrameImpl> df, BranchNames branches, const std::string& name);
   virtual ~TDataFrameFilterBase() {}
   virtual void BuildReaderValues(TTreeReader &r, unsigned int slot) = 0;
   std::weak_ptr<TDataFrameImpl> GetDataFrame() const;
   BranchNames GetTmpBranches() const;
   void CreateSlots(unsigned int nSlots);
   void PrintReport() const;
};
using FilterBasePtr_t = std::shared_ptr<TDataFrameFilterBase>;
using FilterBaseVec_t = std::vector<FilterBasePtr_t>;

template <typename FilterF, typename PrevDataFrame>
class TDataFrameFilter final : public TDataFrameFilterBase {
   using BranchTypes_t = typename ROOT::Internal::TDFTraitsUtils::TFunctionTraits<FilterF>::Args_t;
   using TypeInd_t = typename ROOT::Internal::TDFTraitsUtils::TGenStaticSeq<BranchTypes_t::fgSize>::Type_t;

   FilterF fFilter;
   const BranchNames fBranches;
   PrevDataFrame &fPrevData;

public:
   TDataFrameFilter(FilterF f, const BranchNames &bl,
                    std::shared_ptr<PrevDataFrame> pd, const std::string& name = "")
      : TDataFrameFilterBase(pd->GetDataFrame(), pd->GetTmpBranches(), name),
        fFilter(f), fBranches(bl), fPrevData(*pd) { }

   TDataFrameFilter(const TDataFrameFilter &) = delete;

   bool CheckFilters(unsigned int slot, Long64_t entry)
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
   bool CheckFilterHelper(Internal::TDFTraitsUtils::TTypeList<BranchTypes...>,
                          ROOT::Internal::TDFTraitsUtils::TStaticSeq<S...>,
                          unsigned int slot, Long64_t entry)
   {
      // Take each pointer in tvb, cast it to a pointer to the
      // correct specialization of TTreeReaderValue, and get its content.
      // S expands to a sequence of integers 0 to `sizeof...(types)-1
      // S and types are expanded simultaneously by "..."
      (void) slot; // avoid bogus unused-but-set-parameter warning by gcc
      (void) entry; // avoid bogus unused-but-set-parameter warning by gcc
      return fFilter(Internal::GetBranchValue(fReaderValues[slot][S], slot, entry, fBranches[S],
                     fFirstData, Internal::TDFTraitsUtils::TTypeList<BranchTypes>())...);
   }

   void BuildReaderValues(TTreeReader &r, unsigned int slot)
   {
      fReaderValues[slot] = ROOT::Internal::BuildReaderValues(r, fBranches, fTmpBranches, BranchTypes_t(), TypeInd_t());
   }


   // recursive chain of `Report`s
   void Report() const {
      PartialReport();
   }

   void PartialReport() const {
      fPrevData.PartialReport();
      PrintReport();
   }
};

class TDataFrameImpl {

   ROOT::Internal::ActionBaseVec_t fBookedActions;
   ROOT::Detail::FilterBaseVec_t fBookedFilters;
   std::map<std::string, TmpBranchBasePtr_t> fBookedBranches;
   std::vector<std::shared_ptr<bool>> fResProxyReadiness;
   std::string fTreeName;
   ::TDirectory *fDirPtr = nullptr;
   TTree *fTree = nullptr;
   const BranchNames fDefaultBranches;
   const unsigned int fNSlots;
   // TDataFrameInterface<TDataFrameImpl> calls SetFirstData to set this to a
   // weak pointer to the TDataFrameImpl object itself
   // so subsequent objects in the chain can call GetDataFrame on TDataFrameImpl
   std::weak_ptr<TDataFrameImpl> fFirstData;
   bool fHasRunAtLeastOnce = false;

public:
   TDataFrameImpl(const std::string &treeName, ::TDirectory *dirPtr, const BranchNames &defaultBranches = {});
   TDataFrameImpl(TTree &tree, const BranchNames &defaultBranches = {});
   TDataFrameImpl(const TDataFrameImpl &) = delete;
   ~TDataFrameImpl(){};
   void Run();
   void BuildAllReaderValues(TTreeReader &r, unsigned int slot);
   void CreateSlots(unsigned int nSlots);
   std::weak_ptr<ROOT::Detail::TDataFrameImpl> GetDataFrame() const;
   const BranchNames &GetDefaultBranches() const;
   const BranchNames GetTmpBranches() const { return {}; };
   TTree* GetTree() const;
   const TDataFrameBranchBase &GetBookedBranch(const std::string &name) const;
   void *GetTmpBranchValue(const std::string &branch, unsigned int slot, Long64_t entry);
   ::TDirectory *GetDirectory() const;
   std::string GetTreeName() const;
   void SetFirstData(const std::shared_ptr<TDataFrameImpl>& sp);
   void Book(Internal::ActionBasePtr_t actionPtr);
   void Book(ROOT::Detail::FilterBasePtr_t filterPtr);
   void Book(TmpBranchBasePtr_t branchPtr);
   bool CheckFilters(int, unsigned int);
   unsigned int GetNSlots() const;
   template<typename T>
   Experimental::TActionResultProxy<T> MakeActionResultProxy(std::shared_ptr<T> r)
   {
      auto readiness = std::make_shared<bool>(false);
      // since fFirstData is a weak_ptr to `this`, we are sure the lock succeeds
      auto df = fFirstData.lock();
      auto resPtr = Experimental::TActionResultProxy<T>::MakeActionResultProxy(r, readiness, df);
      fResProxyReadiness.emplace_back(readiness);
      return resPtr;
   }
   bool HasRunAtLeastOnce() const { return fHasRunAtLeastOnce; }
   void Report() const;
   /// End of recursive chain of calls, does nothing
   void PartialReport() const {}
};

} // end NS ROOT::Detail

} // end NS ROOT

// Functions and method implementations
namespace ROOT {

namespace Experimental {

template<typename T>
void Experimental::TActionResultProxy<T>::TriggerRun()
{
   auto df = fFirstData.lock();
   if (!df) {
      throw std::runtime_error("The main TDataFrame is not reachable: did it go out of scope?");
   }
   df->Run();
}

} // end NS Experimental

namespace Internal {
template <typename T>
T &GetBranchValue(TVBPtr_t &readerValue, unsigned int slot, Long64_t entry, const std::string &branch,
                  std::weak_ptr<Detail::TDataFrameImpl> df, TDFTraitsUtils::TTypeList<T>)
{
   if (readerValue == nullptr) {
      // temporary branch
      void *tmpBranchVal = df.lock()->GetTmpBranchValue(branch, slot, entry);
      return *static_cast<T *>(tmpBranchVal);
   } else {
      // real branch
      return **std::static_pointer_cast<TTreeReaderValue<T>>(readerValue);
   }
}

template<typename T>
std::array_view<T> GetBranchValue(TVBPtr_t& readerValue, unsigned int slot,
                                  Long64_t entry, const std::string& branch,
                                  std::weak_ptr<Detail::TDataFrameImpl> df,
                                  TDFTraitsUtils::TTypeList<std::array_view<T>>)
{
   if(readerValue == nullptr) {
      // temporary branch
      void* tmpBranchVal = df.lock()->GetTmpBranchValue(branch, slot, entry);
      auto& tra = *static_cast<TTreeReaderArray<T> *>(tmpBranchVal);
      return std::array_view<T>(tra.begin(), tra.end());
   } else {
      // real branch
      auto& tra = *std::static_pointer_cast<TTreeReaderArray<T>>(readerValue);
      if (tra.GetSize() > 1 &&
          1 != (&tra[1] - &tra[0])) {
         std::string exceptionText = "Branch ";
         exceptionText += branch;
         exceptionText += " hangs from a non-split branch. For this reason, it cannot be accessed via an array_view. Please read the top level branch instead.";
         throw std::runtime_error(exceptionText.c_str());
      }
      return std::array_view<T>(tra.begin(), tra.end());
   }
}

} // end NS Internal

} // end NS ROOT

////////////////////////////////////////////////////////////////////////////////
/// Print a TDataFrame at the prompt:
namespace cling {
inline std::string printValue(ROOT::Experimental::TDataFrame *tdf)
{
   auto df = tdf->GetDataFrameChecked();
   auto treeName = df->GetTreeName();
   auto defBranches = df->GetDefaultBranches();
   auto tmpBranches = df->GetTmpBranches();

   std::ostringstream ret;
   ret << "A data frame built on top of the " << treeName << " dataset.";
   if (!defBranches.empty()) {
      if(defBranches.size() == 1) ret << "\nDefault branch: " << defBranches[0];
      else {
         ret << "\nDefault branches:\n";
         for (auto&& branch : defBranches) {
            ret << " - " << branch << "\n";
         }
      }
   }

   return ret.str();
}
}
#endif // ROOT_TDATAFRAME
