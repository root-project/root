/// \file RNTupleUtil.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch> & Max Orok <maxwellorok@gmail.com>
/// \date 2020-07-14
/// \author Vincenzo Eduardo Padulano, CERN
/// \date 2024-11-08
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RNTupleUtil.hxx"

#include "ROOT/RLogger.hxx"
#include "ROOT/RMiniFile.hxx"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>

ROOT::RLogChannel &ROOT::Internal::NTupleLog()
{
   static RLogChannel sLog("ROOT.NTuple");
   return sLog;
}

ROOT::RResult<void> ROOT::Internal::EnsureValidNameForRNTuple(std::string_view name, std::string_view where)
{
   using codeAndRepr = std::pair<const char *, const char *>;
   constexpr static std::array<codeAndRepr, 4> forbiddenChars{codeAndRepr{"\u002E", "."}, codeAndRepr{"\u002F", "/"},
                                                              codeAndRepr{"\u0020", "space"},
                                                              codeAndRepr{"\u005C", "\\"}};

   for (auto &&[code, repr] : forbiddenChars) {
      if (name.find(code) != std::string_view::npos)
         return R__FAIL(std::string(where) + " name '" + std::string(name) + "' cannot contain character '" + repr +
                        "'.");
   }

   if (std::count_if(name.begin(), name.end(), [](unsigned char c) { return std::iscntrl(c); }))
      return R__FAIL(std::string(where) + " name '" + std::string(name) +
                     "' cannot contain character classified as control character. These notably include newline, tab, "
                     "carriage return.");

   return RResult<void>::Success();
}
