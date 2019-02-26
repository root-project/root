/// \file RStringEnumAttr.cxx
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

#include "ROOT/RStringEnumAttr.hxx"

#include "ROOT/TLogger.hxx"

using namespace ROOT::Experimental;

////////////////////////////////////////////////////////////////////////////////
/// Initialize an attribute `val` from a string value.
///
///\param[in] name - the attribute name (for diagnostic purposes).
///\param[in] strval - the attribute value as a string.
///\param[out] val - the value to be initialized.

void ROOT::Experimental::InitializeAttrFromString(const std::string & /*name*/, const std::string &strval,
                                                  ROOT::Experimental::RStringEnumAttrBase & /*val*/)
{
   if (strval.empty())
      return;

   R__WARNING_HERE("Graf2d") << "Not implemented!";
}
