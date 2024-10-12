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

#include "RtypesCore.h"

#include <string>
#include <ostream>

typedef ULong_t Pixel_t; // from GuiTypes.h

class TString;

namespace ROOT {
namespace Experimental {
typedef unsigned int ElementId_t;

class RLogChannel;

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
   explicit REveException(std::string_view s) : fWhat(s) {}
   ~REveException() noexcept override {}
   void append(std::string_view s) { fWhat.append(s); }

   operator const std::string&() const noexcept { return fWhat; }
   const std::string &str() const noexcept { return fWhat; }
   const char *what() const noexcept override { return fWhat.c_str(); }
};

REveException operator+(const REveException &s1, const std::string &s2);
REveException operator+(const REveException &s1, const TString &s2);
REveException operator+(const REveException &s1, const char *s2);
REveException operator+(const REveException &s1, ElementId_t x);

inline std::ostream& operator <<(std::ostream &s, const REveException &e)
{ s << e.what(); return s; }

/// Log channel for Eve diagnostics.
RLogChannel &REveLog();

} // namespace Experimental
} // namespace ROOT

#endif
