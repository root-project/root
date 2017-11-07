/// \file TStyle.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-10-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TStyle.hxx"

#include "ROOT/TLogger.hxx"
#include "ROOT/TPadExtent.hxx"
#include "ROOT/TPadPos.hxx"

#include <cassert>
#include <limits>

using namespace ROOT::Experimental;

namespace {
static std::unordered_map<std::string, TStyle> ReadGlobalDefaultStyles()
{
   // TODO: use AttrReader
   return {};
}

static std::unordered_map<std::string, TStyle> &GetGlobalStyles()
{
   static std::unordered_map<std::string, TStyle> sStyles = ReadGlobalDefaultStyles();
   return sStyles;
}
}

void TStyle::Register(const TStyle& style)
{
   GetGlobalStyles()[style.GetName()] = style;
}

TStyle *TStyle::Get(std::string_view name)
{
   auto iStyle = GetGlobalStyles().find(std::string(name));
   if (iStyle != GetGlobalStyles().end())
      return &iStyle->second;
   return nullptr;
}


namespace {
static TStyle GetInitialCurrent()
{
   static constexpr const char* kDefaultStyleName = "plain";
   auto iDefStyle = GetGlobalStyles().find(std::string(kDefaultStyleName));
   if (iDefStyle == GetGlobalStyles().end()) {
      R__ERROR_HERE("Gpad") << "Cannot find initial default style named \"" << kDefaultStyleName
      << "\", using an empty one.";
      TStyle defStyle(kDefaultStyleName);
      TStyle::Register(defStyle);
      return defStyle;
   } else {
      return iDefStyle->second;
   }
}
}

TStyle &TStyle::GetCurrent()
{
   static TStyle sCurrentStyle = GetInitialCurrent();
   return sCurrentStyle;
}
