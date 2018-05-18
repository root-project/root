// Author: Enrico Guiraud, Danilo Piparo CERN  02/2018

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFNODES_UTILS
#define ROOT_RDFNODES_UTILS

#include "ROOT/RIntegerSequence.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDFUtils.hxx" // ColumnNames_t

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

/// Initialize a tuple of TColumnValues.
/// For real TTree branches a TTreeReader{Array,Value} is built and passed to the
/// TColumnValue. For temporary columns a pointer to the corresponding variable
/// is passed instead.
template <typename RDFValueTuple, std::size_t... S>
void InitRDFValues(unsigned int slot, RDFValueTuple &valueTuple, TTreeReader *r, const ColumnNames_t &bn,
                   const ColumnNames_t &tmpbn,
                   const std::map<std::string, std::shared_ptr<RCustomColumnBase>> &customCols,
                   std::index_sequence<S...>)
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
   int expander[] = {(isTmpColumn[S] ? std::get<S>(valueTuple).SetTmpColumn(slot, customCols.at(bn.at(S)).get())
                                     : std::get<S>(valueTuple).MakeProxy(r, bn.at(S)),
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
