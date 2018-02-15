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

#include "ROOT/TDataSource.hxx" // ColumnName2ColumnTypeName
#include "ROOT/TypeTraits.hxx"
#include "ROOT/TArrayBranch.hxx"
#include "Compression.h"
#include "TH1.h"
#include "TTreeReaderArray.h"
#include "TTreeReaderValue.h"

#include <array>
#include <cstddef> // std::size_t
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <type_traits> // std::decay
#include <vector>

class TTree;
class TTreeReader;

namespace ROOT {
namespace Experimental {
namespace TDF {
/// A collection of options to steer the creation of the dataset on file
struct TSnapshotOptions {
   using ECAlgo = ::ROOT::ECompressionAlgorithm;
   TSnapshotOptions() = default;
   TSnapshotOptions(const TSnapshotOptions &) = default;
   TSnapshotOptions(TSnapshotOptions &&) = default;
   TSnapshotOptions(std::string_view mode, ECAlgo comprAlgo, int comprLevel, int autoFlush, int splitLevel)
      : fMode(mode), fCompressionAlgorithm(comprAlgo), fCompressionLevel{comprLevel}, fAutoFlush(autoFlush),
        fSplitLevel(splitLevel)
   {
   }
   std::string fMode = "RECREATE";             //< Mode of creation of output file
   ECAlgo fCompressionAlgorithm = ROOT::kZLIB; //< Compression algorithm of output file
   int fCompressionLevel = 1;                  //< Compression level of output file
   int fAutoFlush = 0;                         //< AutoFlush value for output tree
   int fSplitLevel = 99;                       //< Split level of output tree
};
}
}
}

/// \cond HIDDEN_SYMBOLS

namespace ROOT {

// fwd declaration for IsV7Hist
namespace Experimental {
template <int D, typename P, template <int, typename, template <typename> class> class... S>
class THist;

} // ns Experimental

namespace Detail {
namespace TDF {
using ColumnNames_t = std::vector<std::string>;

// fwd decl for ColumnName2ColumnTypeName
class TCustomColumnBase;
// fwd decl for FindUnknownColumns
class TLoopManager;

// type used for tag dispatching
struct TInferType {
};

} // end ns Detail
} // end ns TDF

namespace Internal {
namespace TDF {
using namespace ROOT::TypeTraits;
using namespace ROOT::Detail::TDF;
using namespace ROOT::Experimental::TDF;

/// Compile-time integer sequence generator
/// e.g. calling GenStaticSeq<3>::type() instantiates a StaticSeq<0,1,2>
// TODO substitute all usages of StaticSeq and GenStaticSeq with std::index_sequence when it becomes available
template <int...>
struct StaticSeq {
};

template <int N, int... S>
struct GenStaticSeq : GenStaticSeq<N - 1, N - 1, S...> {
};

template <int... S>
struct GenStaticSeq<0, S...> {
   using type = StaticSeq<S...>;
};

template <int... S>
using GenStaticSeq_t = typename GenStaticSeq<S...>::type;

const std::type_info &TypeName2TypeID(const std::string &name);

std::string TypeID2TypeName(const std::type_info &id);

std::string
ColumnName2ColumnTypeName(const std::string &colName, TTree *, TCustomColumnBase *, TDataSource * = nullptr);

char TypeName2ROOTTypeName(const std::string &b);

const char *ToConstCharPtr(const char *s);
const char *ToConstCharPtr(const std::string &s);
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
struct ValueType<ROOT::Experimental::TDF::TArrayBranch<T>, false> {
   using value_type = T;
};

} // end NS TDF
} // end NS Internal
} // end NS ROOT

/// \endcond

#endif // TDFUTILS
