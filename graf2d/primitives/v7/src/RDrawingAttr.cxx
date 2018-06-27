/// \file RDrawingAttrBase.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-09-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDrawingAttr.hxx"

#include "ROOT/RDrawingOptsBase.hxx"
#include "ROOT/TLogger.hxx"


// pin vtable.
ROOT::Experimental::RDrawingAttrBase::~RDrawingAttrBase() = default;

/// Get the style class currently active in the RDrawingOptsBase.
const std::string &ROOT::Experimental::RDrawingAttrBase::GetStyleClass(const RDrawingOptsBase& opts) const
{
   return opts.GetStyleClass();
}


////////////////////////////////////////////////////////////////////////////////
/// Initialize an attribute `val` from a string value.
///
///\param[in] name - the attribute name (for diagnostic purposes).
///\param[in] strval - the attribute value as a string.
///\param[out] val - the value to be initialized.

void ROOT::Experimental::InitializeAttrFromString(const std::string &name, const std::string &strval, int& val)
{
   if (strval.empty())
      return;

   std::size_t pos;
   val = std::stoi(strval, &pos, /*base*/ 0);
   if (pos != strval.length()) {
      R__WARNING_HERE("Graf2d") << "Leftover characters while parsing default style value for " << name
         << " with value \"" << strval << "\", remainder: \"" << strval.substr(pos) << "\"";
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize an attribute `val` from a string value.
///
///\param[in] name - the attribute name (for diagnostic purposes).
///\param[in] strval - the attribute value as a string.
///\param[out] val - the value to be initialized.

void ROOT::Experimental::InitializeAttrFromString(const std::string &name, const std::string &strval, long long& val)
{
   if (strval.empty())
      return;

   std::size_t pos;
   val = std::stoll(strval, &pos, /*base*/ 0);
   if (pos != strval.length()) {
      R__WARNING_HERE("Graf2d") << "Leftover characters while parsing default style value for " << name
         << " with value \"" << strval << "\", remainder: \"" << strval.substr(pos) << "\"";
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize an attribute `val` from a string value.
///
///\param[in] name - the attribute name (for diagnostic purposes).
///\param[in] strval - the attribute value as a string.
///\param[out] val - the value to be initialized.

void ROOT::Experimental::InitializeAttrFromString(const std::string &name, const std::string &strval, float& val)
{
   if (strval.empty())
      return;

   std::size_t pos;
   val = std::stof(strval, &pos);
   if (pos != strval.length()) {
      R__WARNING_HERE("Graf2d") << "Leftover characters while parsing default style value for " << name
         << " with value \"" << strval << "\", remainder: \"" << strval.substr(pos) << "\"";
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize an attribute `val` from a string value.
///
///\param[in] name - the attribute name (for diagnostic purposes).
///\param[in] strval - the attribute value as a string.
///\param[out] val - the value to be initialized.

void ROOT::Experimental::InitializeAttrFromString(const std::string & /*name*/, const std::string &strval, std::string& val)
{
   if (strval.empty())
      return;
   val = strval;
}
