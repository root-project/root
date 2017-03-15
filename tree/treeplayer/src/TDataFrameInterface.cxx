// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TDataFrameInterface.hxx"

namespace ROOT {
namespace Experimental {

// extern templates
template class TDataFrameInterface<ROOT::Detail::TDataFrameImpl>;
template class TDataFrameInterface<ROOT::Detail::TDataFrameFilterBase>;
template class TDataFrameInterface<ROOT::Detail::TDataFrameBranchBase>;

} // namespace Experimental
} // namespace ROOT
