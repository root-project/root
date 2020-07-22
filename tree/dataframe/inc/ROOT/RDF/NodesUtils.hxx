// Author: Enrico Guiraud, Danilo Piparo CERN  02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFNODES_UTILS
#define ROOT_RDFNODES_UTILS

#include "ROOT/RIntegerSequence.hxx"
#include "ROOT/RDF/RBookedCustomColumns.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/Utils.hxx" // ColumnNames_t

/// \cond
template <typename T>
class TTreeReaderValue;

template <typename T>
class TTreeReaderArray;
/// \endcond

namespace ROOT {
namespace Internal {
namespace RDF {
using namespace ROOT::VecOps;
using namespace ROOT::Detail::RDF;
using namespace ROOT::RDF;

/// Choose between TTreeReader{Array,Value} depending on whether the branch type
/// T is a `RVec<T>` or any other type (respectively).
template <typename T>
struct TReaderValueOrArray {
   using Proxy_t = TTreeReaderValue<T>;
};

template <typename T>
struct TReaderValueOrArray<RVec<T>> {
   using Proxy_t = TTreeReaderArray<T>;
};

template <typename T>
using ReaderValueOrArray_t = typename TReaderValueOrArray<T>::Proxy_t;

/// Initialize a tuple of RColumnValues.
/// For real TTree branches a TTreeReader{Array,Value} is built and passed to the
/// RColumnValue. For temporary columns a pointer to the corresponding variable
/// is passed instead.
template <typename RDFValueTuple, std::size_t... S>
void InitRDFValues(unsigned int slot, RDFValueTuple &valueTuple, TTreeReader *r, const ColumnNames_t &bn,
                   const RBookedCustomColumns &customCols, std::index_sequence<S...>,
                   const std::array<bool, sizeof...(S)> &isCustomColumn)
{
   // hack to expand a parameter pack without c++17 fold expressions.
   // The statement defines a variable with type std::initializer_list<int>, containing all zeroes, and Init
   // is executed as the braced init list is expanded. The final ... expands S.
   int expander[] = {(std::get<S>(valueTuple)
                         .Init(slot, isCustomColumn[S] ? customCols.GetColumns().at(bn[S]).get() : nullptr, r, bn[S]),
                      0)...,
                     0};
   (void)expander; // avoid "unused variable" warnings for expander on gcc4.9
   (void)slot;     // avoid _bogus_ "unused variable" warnings for slot on gcc 4.9
   (void)r;        // avoid "unused variable" warnings for r on gcc5.2
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif
