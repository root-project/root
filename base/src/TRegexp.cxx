// @(#)root/base:$Name$:$Id$
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRegexp                                                              //
//                                                                      //
// Regular expression class.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRegexp.h"
#include "TString.h"
#include "TError.h"

const unsigned TRegexp::fgMaxpat = 128;


ClassImp(TRegexp)

//______________________________________________________________________________
TRegexp::TRegexp(const char *re, Bool_t wildcard)
{
   // Create a regular expression from the input string. If wildcard is true
   // then the input string contains a wildcard expression (see MakeWildcard()).

   if (wildcard)
      GenPattern(MakeWildcard(re));
   else
      GenPattern(re);
}

//______________________________________________________________________________
TRegexp::TRegexp(const TString& re)
{
   // Create a regular expression from a TString.

   GenPattern(re.Data());
}

//______________________________________________________________________________
TRegexp::TRegexp(const TRegexp& r)
{
   // Copy ctor.

   CopyPattern(r);
}

//______________________________________________________________________________
TRegexp::~TRegexp()
{
   delete [] fPattern;
}

//______________________________________________________________________________
TRegexp& TRegexp::operator=(const TRegexp& r)
{
   // Assignment operator.

   if (this != &r) {
      delete [] fPattern;
      CopyPattern(r);
   }
   return *this;
}

//______________________________________________________________________________
TRegexp& TRegexp::operator=(const char *str)
{
   // Assignment operator taking a char* and assigning it to a regexp.

   delete [] fPattern;
   GenPattern(str);
   return *this;
}

//______________________________________________________________________________
TRegexp& TRegexp::operator=(const TString &str)
{
   // Assignment operator taking a TString.

   delete [] fPattern;
   GenPattern(str.Data());
   return *this;
}

//______________________________________________________________________________
void TRegexp::GenPattern(const char *str)
{
   // Generate the regular expression pattern.

   fPattern = new Pattern_t[fgMaxpat];
   int error = ::Makepat(str, fPattern, fgMaxpat);
   fStat = (error < 3) ? (EStatVal) error : kToolong;
}

//______________________________________________________________________________
void TRegexp::CopyPattern(const TRegexp& r)
{
   // Copy the regular expression pattern.

   fPattern = new Pattern_t[fgMaxpat];
   memcpy(fPattern, r.fPattern, fgMaxpat * sizeof(Pattern_t));
   fStat = r.fStat;
}

//______________________________________________________________________________
const char *TRegexp::MakeWildcard(const char *re)
{
   // This routine transforms a wildcarding regular expression into
   // a general regular expression used for pattern matching.
   // When using wildcards the regular expression is assumed to be
   // preceded by a "^" (BOL) and terminated by a "$" (EOL). Also, all
   // "*"'s (closures) are assumed to be preceded by a "." (any character)
   // and all .'s are escaped (so *.ps is different from *.eps).

   static char buf[100];
   char *s = buf;
   int   len = strlen(re);

   if (!re || !len) return "";

   for (int i = 0; i < len; i++) {
      if (i == 0 && re[i] != '^')
         *s++ = '^';
      if (re[i] == '*')
         *s++ = '.';
      if (re[i] == '.')
         *s++ = '\\';
      *s++ = re[i];
      if (i == len-1 && re[i] != '$')
         *s++ = '$';
   }
   *s = '\0';
   return buf;
}

//______________________________________________________________________________
Ssiz_t TRegexp::Index(const TString& string, Ssiz_t* len, Ssiz_t i) const
{
   // Find the first occurance of the regexp in string and return the position.
   // Len is length of the matched string and i is the offset at which the
   // matching should start.

   if (fStat != kOK)
      Error("TRegexp::Index", "Bad Regular Expression");

   const char* startp;
   const char* s = string.Data();
   Ssiz_t slen = string.Length();
   if (slen < i) return kNPOS;
   const char* endp = ::Matchs(s+i, slen-i, fPattern, &startp);
   if (endp) {
      *len = endp - startp;
      return startp - s;
   } else {
      *len = 0;
      return kNPOS;
   }
}

//______________________________________________________________________________
TRegexp::EStatVal TRegexp::Status()
{
   // Check status of regexp.

   EStatVal temp = fStat;
   fStat = kOK;
   return temp;
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TString member functions, put here so the linker will include        //
// them only if regular expressions are used.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
Ssiz_t TString::Index(const TRegexp& r, Ssiz_t start) const
{
   // Find the first occurance of the regexp in string and return the position.
   // Start is the offset at which the search should start.

   Ssiz_t len;
   return r.Index(*this, &len, start); // len not used
}

//______________________________________________________________________________
Ssiz_t TString::Index(const TRegexp& r, Ssiz_t* extent, Ssiz_t start) const
{
   // Find the first occurance of the regexp in string and return the position.
   // Extent is length of the matched string and start is the offset at which
   // the matching should start.

   return r.Index(*this, extent, start);
}

//______________________________________________________________________________
TSubString TString::operator()(const TRegexp& r, Ssiz_t start)
{
   // Return the substring found by applying the regexp starting at start.

   Ssiz_t len;
   Ssiz_t begin = Index(r, &len, start);
   return TSubString(*this, begin, len);
}

//______________________________________________________________________________
TSubString TString::operator()(const TRegexp& r)
{
   // Return the substring found by applying the regexp.

   return (*this)(r,0);
}

//______________________________________________________________________________
TSubString TString::operator()(const TRegexp& r, Ssiz_t start) const
{
   // Return the substring found by applying the regexp starting at start.

   Ssiz_t len;
   Ssiz_t begin = Index(r, &len, start);
   return TSubString(*this, begin, len);
}

//______________________________________________________________________________
TSubString TString::operator()(const TRegexp& r) const
{
   // Return the substring found by applying the regexp.

   return (*this)(r,0);
}

