/// \file RStyle.cxx
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

#include "ROOT/RStyle.hxx"

#include "ROOT/TLogger.hxx"
#include "ROOT/RPadExtent.hxx"
#include "ROOT/RPadPos.hxx"

#include "RStyleReader.hxx" // in src/

#include <ROOT/RStringView.hxx>

#include <cassert>
#include <limits>
#include <string>
#include <sstream>

using namespace ROOT::Experimental;

namespace {
static Internal::RStyleReader::AllStyles_t ReadGlobalDefaultStyles()
{
   Internal::RStyleReader::AllStyles_t target;
   Internal::RStyleReader reader(target);
   reader.ReadDefaults();
   return target;
}

static Internal::RStyleReader::AllStyles_t &GetGlobalStyles()
{
   static Internal::RStyleReader::AllStyles_t sStyles = ReadGlobalDefaultStyles();
   return sStyles;
}
} // unnamed namespace

RStyle &RStyle::Register(RStyle&& style)
{
   RStyle& ret = GetGlobalStyles()[style.GetName()];
   ret = style;
   return ret;
}

RStyle *RStyle::Get(std::string_view name)
{
   auto iStyle = GetGlobalStyles().find(std::string(name));
   if (iStyle != GetGlobalStyles().end())
      return &iStyle->second;
   return nullptr;
}


namespace {
static RStyle GetInitialCurrent()
{
   static constexpr const char* kDefaultStyleName = "plain";
   auto iDefStyle = GetGlobalStyles().find(std::string(kDefaultStyleName));
   if (iDefStyle == GetGlobalStyles().end()) {
      R__INFO_HERE("Gpad") << "Cannot find initial default style named \"" << kDefaultStyleName
      << "\", using an empty one.";
      RStyle defStyle(kDefaultStyleName);
      return RStyle::Register(std::move(defStyle));
   } else {
      return iDefStyle->second;
   }
}
}

RStyle &RStyle::GetCurrent()
{
   static RStyle sCurrentStyle = GetInitialCurrent();
   return sCurrentStyle;
}

std::string RStyle::GetAttribute(const std::string &attrName, const std::string &/*className*/) const {
   std::string trailingPart(attrName);
   while (!trailingPart.empty()) {
      auto iter = fAttrs.find(trailingPart);
      if (iter != fAttrs.end())
         return iter->second;
      auto posDot = trailingPart.find('.');
      if (posDot != std::string::npos) {
         trailingPart.erase(0, posDot + 1);
      } else {
         return {};
      }
   }
   return {};
}
