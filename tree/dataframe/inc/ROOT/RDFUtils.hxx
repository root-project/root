// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
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
#include "TH1.h"
#include "TTreeReaderArray.h"
#include "TTreeReaderValue.h"

#include <array>
#include <deque>
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
namespace RDF {
using ColumnNames_t = std::vector<std::string>;

// fwd decl for ColumnName2ColumnTypeName
class RCustomColumnBase;

// type used for tag dispatching
struct TInferType {
};

} // end ns Detail
} // end ns RDF

namespace Internal {
namespace RDF {
using namespace ROOT::TypeTraits;
using namespace ROOT::Detail::RDF;
using namespace ROOT::RDF;

/// Detect whether a type is an instantiation of vector<T,A>
template <typename>
struct IsVector_t : public std::false_type {};

template <typename T, typename A>
struct IsVector_t<std::vector<T, A>> : public std::true_type {};

const std::type_info &TypeName2TypeID(const std::string &name);

std::string TypeID2TypeName(const std::type_info &id);

std::string ColumnName2ColumnTypeName(const std::string &colName, unsigned int namespaceID, TTree *, RDataSource *,
                                      bool isCustomColumn, bool extraConversions = true);

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
struct RemoveFirstTwoParametersIf {
   using type = TypeList;
};

template <typename TypeList>
struct RemoveFirstTwoParametersIf<true, TypeList> {
   using typeTmp = typename RemoveFirstParameterIf<true, TypeList>::type;
   using type = typename RemoveFirstParameterIf<true, typeTmp>::type;
};

/// Detect whether a type is an instantiation of RVec<T>
template <typename>
struct IsRVec_t : public std::false_type {};

template <typename T>
struct IsRVec_t<ROOT::VecOps::RVec<T>> : public std::true_type {};

// Check the value_type type of a type with a SFINAE to allow compilation in presence
// fundamental types
template <typename T, bool IsContainer = IsContainer<typename std::decay<T>::type>::value>
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

} // end NS RDF
} // end NS Internal
} // end NS ROOT

/// \endcond

#endif // RDFUTILS
