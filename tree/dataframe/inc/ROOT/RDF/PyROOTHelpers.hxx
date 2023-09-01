// Author: Danilo Piparo, Enrico Guiraud, Stefan Wunsch CERN  04/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_PyROOTHelpers
#define ROOT_PyROOTHelpers

#include "ROOT/RDataFrame.hxx"

#include <vector>
#include <string>
#include <utility>

namespace ROOT {
namespace Internal {
namespace RDF {

template <typename dtype>
ULong64_t GetVectorAddress(std::vector<dtype> &p)
{
   return reinterpret_cast<ULong64_t>(&p);
}

inline ULong64_t GetAddress(std::vector<std::string> &p)
{
   return reinterpret_cast<ULong64_t>(&p);
}
inline ULong64_t GetAddress(TTree &p)
{
   return reinterpret_cast<ULong64_t>(&p);
}

template <typename BufType, typename... ColTypes, std::size_t... Idx>
void TTreeAsFlatMatrix(std::index_sequence<Idx...>, TTree &tree, std::vector<BufType> &matrix,
                       std::vector<std::string> &columns)
{
   auto buffer = matrix.data();

   auto fillMatrix = [buffer](ColTypes... cols, ULong64_t entry) {
      int expander[] = {(buffer[entry * sizeof...(Idx) + Idx] = cols, 0)...};
      (void)expander;
   };

   auto columnsWithEntry = columns;
   columnsWithEntry.emplace_back("tdfentry_");

   ROOT::RDataFrame dataframe(tree, columns);
   dataframe.Foreach(fillMatrix, columnsWithEntry);
}

template <typename BufType, typename... ColTypes>
void TTreeAsFlatMatrixHelper(TTree &tree, std::vector<BufType> &matrix, std::vector<std::string> &columns)
{
   TTreeAsFlatMatrix<BufType, ColTypes...>(std::index_sequence_for<ColTypes...>(), tree, matrix, columns);
}

// RDataFrame.AsNumpy helpers

// NOTE: This is a workaround for the missing Take action in the PyROOT interface
template <typename T>
ROOT::RDF::RResultPtr<std::vector<T>> RDataFrameTake(ROOT::RDF::RNode df, std::string_view column)
{
   return df.Take<T>(column);
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif
