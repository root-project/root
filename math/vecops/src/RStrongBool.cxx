// Author: Danilo Piparo CERN  06/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RStrongBool.hxx"

#include <string>

namespace ROOT {
namespace VecOps {

RStrongBool::RStrongBool() : fVal(false){};
RStrongBool::~RStrongBool() = default;
RStrongBool::RStrongBool(bool val) : fVal{val}
{
}

} // End NS VecOps
} // End NS ROOT

namespace cling {

std::string printValue(::ROOT::VecOps::RStrongBool *sb)
{
   return bool(*sb) ? "true" : "false";
}

} // End NS cling