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

namespace Detail {
namespace TDF {
using ColumnNames_t = std::vector<std::string>;
class TCustomColumnBase; // fwd decl for ColumnName2ColumnTypeName
struct TInferType {
};
} // end ns Detail
} // end ns TDF

namespace Internal {
namespace TDF {
using namespace ROOT::Detail::TDF;

template <typename... Types>
struct TTypeList {
   static constexpr std::size_t fgSize = sizeof...(Types);
};

// extract parameter types from a callable object
template <typename T>
struct TFunctionTraits {
   using Args_t = typename TFunctionTraits<decltype(&T::operator())>::Args_t;
   using ArgsNoDecay_t = typename TFunctionTraits<decltype(&T::operator())>::ArgsNoDecay_t;
   using Ret_t = typename TFunctionTraits<decltype(&T::operator())>::Ret_t;
};

// lambdas and std::function
template <typename R, typename T, typename... Args>
struct TFunctionTraits<R (T::*)(Args...) const> {
   using Args_t = TTypeList<typename std::decay<Args>::type...>;
   using ArgsNoDecay_t = TTypeList<Args...>;
   using Ret_t = R;
};

// mutable lambdas and functor classes
template <typename R, typename T, typename... Args>
struct TFunctionTraits<R (T::*)(Args...)> {
   using Args_t = TTypeList<typename std::decay<Args>::type...>;
   using ArgsNoDecay_t = TTypeList<Args...>;
   using Ret_t = R;
};

// function pointers
template <typename R, typename... Args>
struct TFunctionTraits<R (*)(Args...)> {
   using Args_t = TTypeList<typename std::decay<Args>::type...>;
   using ArgsNoDecay_t = TTypeList<Args...>;
   using Ret_t = R;
};

// free functions
template <typename R, typename... Args>
struct TFunctionTraits<R(Args...)> {
   using Args_t = TTypeList<typename std::decay<Args>::type...>;
   using ArgsNoDecay_t = TTypeList<Args...>;
   using Ret_t = R;
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
      using It_t = typename A::iterator;
      using CIt_t = typename A::const_iterator;
      using V_t = typename A::value_type;
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

// Extract first of possibly many template parameters. For non-template types, the result is the type itself
template <typename T>
struct TExtractType {
   using type = T;
};

template <typename T, template <typename...> class U, typename... Extras>
struct TExtractType<U<T, Extras...>> {
   using type = T;
};

template <typename T>
using ExtractType_t = typename TExtractType<T>::type;

template <typename BranchType, typename... Rest>
struct TNeedJitting {
   static constexpr bool value = TNeedJitting<Rest...>::value;
};

template <typename... Rest>
struct TNeedJitting<TInferType, Rest...> {
   static constexpr bool value = true;
};

template <typename T>
struct TNeedJitting<T> {
   static constexpr bool value = false;
};

template <>
struct TNeedJitting<TInferType> {
   static constexpr bool value = true;
};

using TVBPtr_t = std::shared_ptr<TTreeReaderValueBase>;
using TVBVec_t = std::vector<TVBPtr_t>;

std::string ColumnName2ColumnTypeName(const std::string &colName, TTree &, TCustomColumnBase *);

const char *ToConstCharPtr(const char *s);
const char *ToConstCharPtr(const std::string s);
unsigned int GetNSlots();

/// Choose between TTreeReader{Array,Value} depending on whether the branch type
/// T is a `std::array_view<T>` or any other type (respectively).
template <typename T>
struct TReaderValueOrArray {
   using Proxy_t = TTreeReaderValue<T>;
};

template <typename T>
struct TReaderValueOrArray<std::array_view<T>> {
   using Proxy_t = TTreeReaderArray<T>;
};

template <typename T>
using ReaderValueOrArray_t = typename TReaderValueOrArray<T>::Proxy_t;

/// Initialize a tuple of TColumnValues.
/// For real TTree branches a TTreeReader{Array,Value} is built and passed to the
/// TColumnValue. For temporary columns a pointer to the corresponding variable
/// is passed instead.
template <typename TDFValueTuple, int... S>
void InitTDFValues(unsigned int slot, TDFValueTuple &valueTuple, TTreeReader *r, const ColumnNames_t &bn,
                   const ColumnNames_t &tmpbn,
                   const std::map<std::string, std::shared_ptr<TCustomColumnBase>> &tmpBranches, TStaticSeq<S...>)
{
   // isTmpBranch has length bn.size(). Elements are true if the corresponding
   // branch is a temporary branch created with Define, false if they are
   // actual branches present in the TTree.
   std::array<bool, sizeof...(S)> isTmpColumn;
   for (auto i = 0u; i < isTmpColumn.size(); ++i)
      isTmpColumn[i] = std::find(tmpbn.begin(), tmpbn.end(), bn.at(i)) != tmpbn.end();

   // hack to expand a parameter pack without c++17 fold expressions.
   // The statement defines a variable with type std::initializer_list<int>, containing all zeroes, and SetTmpColumn or
   // SetProxy are conditionally executed as the braced init list is expanded. The final ... expands S.
   std::initializer_list<int> expander{(isTmpColumn[S]
                                           ? std::get<S>(valueTuple).SetTmpColumn(slot, tmpBranches.at(bn.at(S)).get())
                                           : std::get<S>(valueTuple).MakeProxy(r, bn.at(S)),
                                        0)...};
   (void)expander; // avoid "unused variable" warnings for expander on gcc4.9
   (void)slot;     // avoid _bogus_ "unused variable" warnings for slot on gcc 4.9
   (void)r;        // avoid "unused variable" warnings for r on gcc5.2
}

template <typename Filter>
void CheckFilter(Filter &)
{
   using FilterRet_t = typename TDF::TFunctionTraits<Filter>::Ret_t;
   static_assert(std::is_same<FilterRet_t, bool>::value, "filter functions must return a bool");
}

void CheckTmpBranch(const std::string &branchName, TTree *treePtr);

///////////////////////////////////////////////////////////////////////////////
/// Check that the callable passed to TInterface::Reduce:
/// - takes exactly two arguments of the same type
/// - has a return value of the same type as the arguments
template <typename F, typename T>
void CheckReduce(F &, TTypeList<T, T>)
{
   using Ret_t = typename TFunctionTraits<F>::Ret_t;
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
const ColumnNames_t &PickBranchNames(unsigned int nArgs, const ColumnNames_t &bl, const ColumnNames_t &defBl);

namespace ActionTypes {
struct Histo1D {
};
struct Histo2D {
};
struct Histo3D {
};
struct Profile1D {
};
struct Profile2D {
};
struct Min {
};
struct Max {
};
struct Mean {
};
struct Fill {
};
}

template <typename T, bool ISV7HISTO = !std::is_base_of<TH1, T>::value>
struct TIsV7Histo {
   const static bool fgValue = ISV7HISTO;
};

template <typename T, bool ISV7HISTO = TIsV7Histo<T>::fgValue>
struct HistoUtils {
   static void SetCanExtendAllAxes(T &h) { h.SetCanExtend(::TH1::kAllAxes); }
   static bool HasAxisLimits(T &h)
   {
      auto xaxis = h.GetXaxis();
      return !(xaxis->GetXmin() == 0. && xaxis->GetXmax() == 0.);
   }
};

template <typename T>
struct HistoUtils<T, true> {
   static void SetCanExtendAllAxes(T &) {}
   static bool HasAxisLimits(T &) { return true; }
};

} // end NS TDF
} // end NS Internal
} // end NS ROOT

/// \endcond

#endif // TDFUTILS
