/// \file ROOT/StringUtils.hxx
/// \ingroup Base StdExt
/// \author Jonas Rembser <jonas.rembser@cern.ch>
/// \date 2021-08-09

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_StringUtils
#define ROOT_StringUtils

#include "ROOT/RStringView.hxx"

#include <string>
#include <vector>

namespace ROOT {

std::vector<std::string> Split(std::string_view str, std::string_view delims, bool skipEmpty = false);

} // namespace ROOT

#endif
