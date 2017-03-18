// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDFUTILS
#define ROOT_TDFUTILS

#include "ROOT/RArrayView.hxx"
#include "TH1.h"
#include "TTreeReaderArray.h"
#include "TTreeReaderValue.h"

#include <array>
#include <cstddef> // std::size_t
#include <functional>
#include <memory>
#include <string>
#include <type_traits> // std::decay
#include <vector>
class TTree;
class TTreeReader;

/// \cond HIDDEN_SYMBOLS

namespace ROOT {

using BranchNames_t = std::vector<std::string>;
namespace Detail {
// forward declaration for ColumnName2ColumnTypeName
class TDataFrameBranchBase;
}

namespace Internal {
namespace TDFTraitsUtils {
template <typename... Types>
struct TTypeList {
   static constexpr std::size_t fgSize = sizeof...(Types);
};

// extract parameter types from a callable object
template <typename T>
struct TFunctionTraits {
   using Args_t        = typename TFunctionTraits<decltype(&T::operator())>::Args_t;
   using ArgsNoDecay_t = typename TFunctionTraits<decltype(&T::operator())>::ArgsNoDecay_t;
   using Ret_t         = typename TFunctionTraits<decltype(&T::operator())>::Ret_t;
};

// lambdas and std::function
template <typename R, typename T, typename... Args>
struct TFunctionTraits<R (T::*)(Args...) const> {
   using Args_t        = TTypeList<typename std::decay<Args>::type...>;
   using ArgsNoDecay_t = TTypeList<Args...>;
   using Ret_t         = R;
};

// mutable lambdas and functor classes
template <typename R, typename T, typename... Args>
struct TFunctionTraits<R (T::*)(Args...)> {
   using Args_t        = TTypeList<typename std::decay<Args>::type...>;
   using ArgsNoDecay_t = TTypeList<Args...>;
   using Ret_t         = R;
};

// function pointers
template <typename R, typename... Args>
struct TFunctionTraits<R (*)(Args...)> {
   using Args_t        = TTypeList<typename std::decay<Args>::type...>;
   using ArgsNoDecay_t = TTypeList<Args...>;
   using Ret_t         = R;
};

// free functions
template <typename R, typename... Args>
struct TFunctionTraits<R(Args...)> {
   using Args_t        = TTypeList<typename std::decay<Args>::type...>;
   using ArgsNoDecay_t = TTypeList<Args...>;
   using Ret_t         = R;
};

// remove first type from TTypeList
template <typename>
struct TRemoveFirst {
};

template <typename T, typename... Args>
struct TRemoveFirst<TTypeList<T, Args...>> {
   using Types_t = TTypeList<Args...>;
};

// return wrapper around f that prepends an `unsigned int slot` parameter
template <typename R, typename F, typename... Args>
std::function<R(unsigned int, Args...)> AddSlotParameter(F &f, TTypeList<Args...>)
{
   return [f](unsigned int, Args... a) -> R { return f(a...); };
}

// compile-time integer sequence generator
// e.g. calling TGenStaticSeq<3>::type() instantiates a TStaticSeq<0,1,2>
template <int...>
struct TStaticSeq {
};

template <int N, int... S>
struct TGenStaticSeq : TGenStaticSeq<N - 1, N - 1, S...> {
};

template <int... S>
struct TGenStaticSeq<0, S...> {
   using Type_t = TStaticSeq<S...>;
};

template <typename T>
struct TIsContainer {
   using Test_t = typename std::decay<T>::type;

   template <typename A>
   static constexpr bool Test(A *pt, A const *cpt = nullptr, decltype(pt->begin()) * = nullptr,
                              decltype(pt->end()) * = nullptr, decltype(cpt->begin()) * = nullptr,
                              decltype(cpt->end()) * = nullptr, typename A::iterator *pi = nullptr,
                              typename A::const_iterator *pci = nullptr)
   {
      using It_t  = typename A::iterator;
      using CIt_t = typename A::const_iterator;
      using V_t   = typename A::value_type;
      return std::is_same<Test_t, std::vector<bool>>::value ||
             (std::is_same<decltype(pt->begin()), It_t>::value && std::is_same<decltype(pt->end()), It_t>::value &&
              std::is_same<decltype(cpt->begin()), CIt_t>::value && std::is_same<decltype(cpt->end()), CIt_t>::value &&
              std::is_same<decltype(**pi), V_t &>::value && std::is_same<decltype(**pci), V_t const &>::value);
   }

   template <typename A>
   static constexpr bool Test(...)
   {
      return false;
   }

   static const bool fgValue = Test<Test_t>(nullptr);
};

} // end NS TDFTraitsUtils

using TVBPtr_t = std::shared_ptr<TTreeReaderValueBase>;
using TVBVec_t = std::vector<TVBPtr_t>;

std::string ColumnName2ColumnTypeName(const std::string &colName, TTree &, ROOT::Detail::TDataFrameBranchBase *);

const char *ToConstCharPtr(const char *s);
const char *ToConstCharPtr(const std::string s);
unsigned int GetNSlots();

template <typename BranchType>
std::shared_ptr<ROOT::Internal::TTreeReaderValueBase> ReaderValueOrArray(TTreeReader &r, const std::string &branch,
                                                                         TDFTraitsUtils::TTypeList<BranchType>)
{
   return std::make_shared<TTreeReaderValue<BranchType>>(r, branch.c_str());
}

template <typename BranchType>
std::shared_ptr<ROOT::Internal::TTreeReaderValueBase> ReaderValueOrArray(
   TTreeReader &r, const std::string &branch, TDFTraitsUtils::TTypeList<std::array_view<BranchType>>)
{
   return std::make_shared<TTreeReaderArray<BranchType>>(r, branch.c_str());
}

template <int... S, typename... BranchTypes>
TVBVec_t BuildReaderValues(TTreeReader &r, const BranchNames_t &bl, const BranchNames_t &tmpbl,
                           TDFTraitsUtils::TTypeList<BranchTypes...>, TDFTraitsUtils::TStaticSeq<S...>)
{
   // isTmpBranch has length bl.size(). Elements are true if the corresponding
   // branch is a temporary branch created with AddColumn, false if they are
   // actual branches present in the TTree.
   std::array<bool, sizeof...(S)> isTmpBranch;
   for (unsigned int i = 0; i < isTmpBranch.size(); ++i)
      isTmpBranch[i]   = std::find(tmpbl.begin(), tmpbl.end(), bl.at(i)) != tmpbl.end();

   // Build vector of pointers to TTreeReaderValueBase.
   // tvb[i] points to a TTreeReader{Value,Array} specialized for the i-th BranchType,
   // corresponding to the i-th branch in bl
   // For temporary branches (declared with AddColumn) a nullptr is created instead
   // S is expected to be a sequence of sizeof...(BranchTypes) integers
   // Note that here TTypeList only contains one single type
   TVBVec_t tvb{isTmpBranch[S]
                   ? nullptr
                   : ReaderValueOrArray(
                        r, bl.at(S),
                        TDFTraitsUtils::TTypeList<BranchTypes>())...}; // "..." expands BranchTypes and S simultaneously

   return tvb;
}

template <typename Filter>
void CheckFilter(Filter &)
{
   using FilterRet_t = typename TDFTraitsUtils::TFunctionTraits<Filter>::Ret_t;
   static_assert(std::is_same<FilterRet_t, bool>::value, "filter functions must return a bool");
}

void CheckTmpBranch(const std::string &branchName, TTree *treePtr);

///////////////////////////////////////////////////////////////////////////////
/// Check that the callable passed to TDataFrameInterface::Reduce:
/// - takes exactly two arguments of the same type
/// - has a return value of the same type as the arguments
template <typename F, typename T>
void CheckReduce(F &, ROOT::Internal::TDFTraitsUtils::TTypeList<T, T>)
{
   using Ret_t = typename ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::Ret_t;
   static_assert(std::is_same<Ret_t, T>::value, "reduce function must have return type equal to argument type");
   return;
}

///////////////////////////////////////////////////////////////////////////////
/// This overload of CheckReduce is called if T is not a TTypeList<T,T>
template <typename F, typename T>
void CheckReduce(F &, T)
{
   static_assert(sizeof(F) == 0, "reduce function must take exactly two arguments of the same type");
}

/// Returns local BranchNames or default BranchNames according to which one should be used
const BranchNames_t &PickBranchNames(unsigned int nArgs, const BranchNames_t &bl, const BranchNames_t &defBl);

namespace ActionTypes {
struct Histo1D {
};
struct Min {
};
struct Max {
};
struct Mean {
};
}

// Utilities to accommodate v7
namespace TDFV7Utils {

template <typename T, bool ISV7HISTO = !std::is_base_of<TH1, T>::value>
struct TIsV7Histo {
   const static bool fgValue = ISV7HISTO;
};

template <typename T, bool ISV7HISTO = TIsV7Histo<T>::fgValue>
struct Histo {
   static void SetCanExtendAllAxes(T &h) { h.SetCanExtend(::TH1::kAllAxes); }
   static bool HasAxisLimits(T &h)
   {
      auto xaxis = h.GetXaxis();
      return !(xaxis->GetXmin() == 0. && xaxis->GetXmax() == 0.);
   }
};

template <typename T>
struct Histo<T, true> {
   static void SetCanExtendAllAxes(T &) {}
   static bool HasAxisLimits(T &) { return true; }
};

} // end NS TDFV7Utils

} // end NS Internal

} // end NS ROOT

/// \endcond

#endif // TDFUTILS
