/// \file RPadExtent.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2018-02-08
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RPadExtent.hxx"

#include <ROOT/TLogger.hxx>
#include <ROOT/RDrawingAttr.hxx>


////////////////////////////////////////////////////////////////////////////////
/// Initialize a RPadExtent from a style string.
/// Syntax: X, Y
/// where X and Y are a series of numbers separated by "+", where each number is
/// followed by one of `px`, `user`, `normal` to specify an extent in pixel,
/// user or normal coordinates. Spaces between any part is allowed.
/// Example: `100 px + 0.1 user, 0.5 normal` is a `RPadExtent{100_px + 0.1_user, 0.5_normal}`.

ROOT::Experimental::RPadExtent ROOT::Experimental::FromAttributeString(const std::string &val, const RDrawingAttrBase& attr, const std::string &name, RPadExtent*)
{
   RPadExtent ret;
   ret.SetFromAttrString(val, attr, name);
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert a RPadExtent to a style string, matching what ExtentFromString can parse.

std::string ROOT::Experimental::ToAttributeString(const RPadExtent &extent)
{
   std::string ret = ToAttributeString(extent.fHoriz);
   ret += ", ";
   ret += ToAttributeString(extent.fVert);
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize a RPadHorizVert from a style string.
/// Syntax: X, Y
/// where X and Y are a series of numbers separated by "+", where each number is
/// followed by one of `px`, `user`, `normal` to specify an extent in pixel,
/// user or normal coordinates. Spaces between any part is allowed.
/// Example: `100 px + 0.1 user, 0.5 normal` is a `RPadExtent{100_px + 0.1_user, 0.5_normal}`.

void ROOT::Experimental::Internal::RPadHorizVert::SetFromAttrString(const std::string &val, const RDrawingAttrBase& attr, const std::string &name)
{
   if (val.empty()) {
      // Leave it at its default value.
      return;
   }

   auto buildName = [&]() {
      RDrawingAttrBase::Name_t fullName(attr.GetName());
      fullName.emplace_back(name);
      return RDrawingAttrBase::NameToDottedDiagName(fullName);
   };

   auto posComma = val.find(',');
   if (posComma == std::string::npos) {
      R__ERROR_HERE("Gpad") << "Parsing attribute for " << buildName() << ": "
         << "expected two coordinate dimensions but found only one in " << val;
      return;
   }
   if (val.find(',', posComma + 1) != std::string::npos) {
      R__ERROR_HERE("Gpad") << "Parsing attribute for " << buildName() << ": "
         << "found more than the expected two coordinate dimensions in " << val;
      return;
   }
   fHoriz.SetFromAttrString(val.substr(0, posComma), attr, name);
   fVert.SetFromAttrString(val.substr(posComma + 1), attr, name);
}
