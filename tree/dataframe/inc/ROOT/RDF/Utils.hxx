// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFUTILS
#define ROOT_RDFUTILS

#include "ROOT/RDataSource.hxx" // ColumnName2ColumnTypeName
#include "ROOT/TypeTraits.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RSnapshotOptions.hxx"
#include "ROOT/RSpan.hxx" // for IsDataContainer
#include "TH1.h"

#include <array>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <type_traits> // std::decay

class TTree;
class TTreeReader;

/// \cond HIDDEN_SYMBOLS

namespace ROOT {

namespace Detail {
namespace RDF {
using ColumnNames_t = std::vector<std::string>;

// fwd decl for ColumnName2ColumnTypeName
class RCustomColumnBase;

// type used for tag dispatching
struct RInferredType {
};

} // end ns Detail
} // end ns RDF

namespace Internal {
namespace RDF {
using namespace ROOT::TypeTraits;
using namespace ROOT::Detail::RDF;
using namespace ROOT::RDF;

/// Check for container traits.
///
/// Note that we don't recognize std::string as a container.
template <typename T>
struct IsDataContainer {
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
              std::is_same<decltype(**pi), V_t &>::value && std::is_same<decltype(**pci), V_t const &>::value &&
              !std::is_same<T, std::string>::value);
   }

   template <typename A>
   static constexpr bool Test(...)
   {
      return false;
   }

   static constexpr bool value = Test<Test_t>(nullptr);
};

template<typename T>
struct IsDataContainer<std::span<T>> {
   static constexpr bool value = true;
};

/// Detect whether a type is an instantiation of vector<T,A>
template <typename>
struct IsVector_t : public std::false_type {};

template <typename T, typename A>
struct IsVector_t<std::vector<T, A>> : public std::true_type {};

const std::type_info &TypeName2TypeID(const std::string &name);

std::string TypeID2TypeName(const std::type_info &id);

std::string ColumnName2ColumnTypeName(const std::string &colName, unsigned int namespaceID, TTree *, RDataSource *,
                                      bool isCustomColumn, bool vector2rvec = true, unsigned int customColID = 0);

char TypeName2ROOTTypeName(const std::string &b);

unsigned int GetNSlots();

/// `type` is TypeList if MustRemove is false, otherwise it is a TypeList with the first type removed
template <bool MustRemove, typename TypeList>
struct RemoveFirstParameterIf {
   using type = TypeList;
};

template <typename TypeList>
struct RemoveFirstParameterIf<true, TypeList> {
   using type = RemoveFirstParameter_t<TypeList>;
};

template <bool MustRemove, typename TypeList>
using RemoveFirstParameterIf_t = typename RemoveFirstParameterIf<MustRemove, TypeList>::type;

template <bool MustRemove, typename TypeList>
struct RemoveFirstTwoParametersIf {
   using type = TypeList;
};

template <typename TypeList>
struct RemoveFirstTwoParametersIf<true, TypeList> {
   using typeTmp = typename RemoveFirstParameterIf<true, TypeList>::type;
   using type = typename RemoveFirstParameterIf<true, typeTmp>::type;
};

template <bool MustRemove, typename TypeList>
using RemoveFirstTwoParametersIf_t = typename RemoveFirstTwoParametersIf<MustRemove, TypeList>::type;

/// Detect whether a type is an instantiation of RVec<T>
template <typename>
struct IsRVec_t : public std::false_type {};

template <typename T>
struct IsRVec_t<ROOT::VecOps::RVec<T>> : public std::true_type {};

// Check the value_type type of a type with a SFINAE to allow compilation in presence
// fundamental types
template <typename T, bool IsDataContainer = IsDataContainer<typename std::decay<T>::type>::value || std::is_same<std::string, T>::value>
struct ValueType {
   using value_type = typename T::value_type;
};

template <typename T>
struct ValueType<T, false> {
   using value_type = T;
};

template <typename T>
struct ValueType<ROOT::VecOps::RVec<T>, false> {
   using value_type = T;
};

std::vector<std::string> ReplaceDotWithUnderscore(const std::vector<std::string> &columnNames);

/// Erase `that` element from vector `v`
template <typename T>
void Erase(const T &that, std::vector<T> &v)
{
   v.erase(std::remove(v.begin(), v.end(), that), v.end());
}

/// Declare code in the interpreter via the TInterpreter::Declare method, throw in case of errors
void InterpreterDeclare(const std::string &code);

/// Jit code in the interpreter with TInterpreter::Calc, throw in case of errors.
/// The optional `context` parameter, if present, is mentioned in the error message.
/// The pointer returned by the call to TInterpreter::Calc is returned in case of success.
Long64_t InterpreterCalc(const std::string &code, const std::string &context = "");

} // end NS RDF
} // end NS Internal
} // end NS ROOT

/// \endcond

#endif // RDFUTILS
