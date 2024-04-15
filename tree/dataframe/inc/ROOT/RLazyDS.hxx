// Author: Enrico Guiraud, Danilo Piparo CERN  02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RLAZYDS
#define ROOT_RLAZYDS

#include "ROOT/RDF/RLazyDSImpl.hxx"
#include "ROOT/RDataFrame.hxx"

#include <memory>

namespace ROOT {

namespace RDF {

// clang-format off
////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Factory method to create a Lazy RDataFrame.
/// \param[in] colNameProxyPairs the series of pairs to describe the columns of the data source, first element of the pair is the name of the column and the second is the RResultPtr to the column in the parent data frame.
// clang-format on
template <typename... ColumnTypes>
RDataFrame MakeLazyDataFrame(std::pair<std::string, RResultPtr<std::vector<ColumnTypes>>> &&... colNameProxyPairs)
{
   RDataFrame tdf(std::make_unique<RLazyDS<ColumnTypes...>>(
      std::forward<std::pair<std::string, RResultPtr<std::vector<ColumnTypes>>>>(colNameProxyPairs)...));
   return tdf;
}

} // ns RDF

} // ns ROOT

#endif
