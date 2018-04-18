// Author: Enrico Guiraud, Danilo Piparo CERN  02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLAZYDS
#define ROOT_TLAZYDS

#include "ROOT/TLazyDSImpl.hxx"
#include "ROOT/RMakeUnique.hxx"
#include "ROOT/TDataFrame.hxx"

namespace ROOT {
namespace Experimental {
namespace TDF {

// clang-format off
////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Factory method to create a Lazy TDataFrame.
/// \param[in] colNameProxyPairs the series of pairs to describe the columns of the data source, first element of the pair is the name of the column and the second is the TResultPtr to the column in the parent data frame.
// clang-format on
template <typename... ColumnTypes>
TDataFrame MakeLazyDataFrame(std::pair<std::string, TResultPtr<std::vector<ColumnTypes>>> &&... colNameProxyPairs)
{
   TDataFrame tdf(std::make_unique<TLazyDS<ColumnTypes...>>(
      std::forward<std::pair<std::string, TResultPtr<std::vector<ColumnTypes>>>>(colNameProxyPairs)...));
   return tdf;
}

} // ns TDF
} // ns Experimental
} // ns ROOT

#endif
