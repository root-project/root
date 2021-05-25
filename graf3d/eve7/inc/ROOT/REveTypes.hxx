// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT7_REveTypes
#define ROOT7_REveTypes

#include "GuiTypes.h" // For Pixel_t only, to be changed.

#include "TString.h"

class TGeoManager;

namespace ROOT {
namespace Experimental {

typedef unsigned int ElementId_t;


//==============================================================================
// Exceptions, string functions
//==============================================================================

bool operator==(const TString &t, const std::string &s);
bool operator==(const std::string &s, const TString &t);

////////////////////////////////////////////////////////////////////////////////
/// REveException
/// Exception-type thrown by Eve classes.
////////////////////////////////////////////////////////////////////////////////

class REveException : public std::exception {
   std::string fWhat;
public:
   REveException() = default;
   explicit REveException(const std::string &s) : fWhat(s) {}
   virtual ~REveException() noexcept {}
   void append(const std::string &s) { fWhat.append(s); }

   const char *what() const noexcept override { return fWhat.c_str(); }
};

REveException operator+(const REveException &s1, const std::string &s2);
REveException operator+(const REveException &s1, const TString &s2);
REveException operator+(const REveException &s1, const char *s2);
REveException operator+(const REveException &s1, ElementId_t x);

} // namespace Experimental
} // namespace ROOT

#endif
