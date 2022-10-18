// Author: Ivan Kabadzhov CERN, Vincenzo Eduardo Padulano CERN/UPV 06/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/InternalUtils.hxx"

namespace ROOT {
namespace Internal {
namespace RDF {

ROOT::RDataFrame MakeDataFrameFromSpec(const RDatasetSpec &spec)
{
   return ROOT::RDataFrame(std::move(spec));
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT
