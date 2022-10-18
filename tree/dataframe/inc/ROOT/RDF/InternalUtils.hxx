// Author: Ivan Kabadzhov CERN, Vincenzo Eduardo Padulano CERN/UPV 06/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_INTERNALUTILS
#define ROOT_RDF_INTERNALUTILS

#include "ROOT/RDF/RDatasetSpec.hxx"
#include "ROOT/RDataFrame.hxx"

namespace ROOT {
namespace Internal {
namespace RDF {

ROOT::RDataFrame MakeDataFrameFromSpec(const RDatasetSpec &spec);

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif // ROOT_RDF_INTERNALUTILS
