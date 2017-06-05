// @(#)root/base:$Id$
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TRegexp
\ingroup Base

Regular expression class.

~~~ {.cpp}
  '^'             // start-of-line anchor
  '$'             // end-of-line anchor
  '.'             // matches any character
  '['             // start a character class
  ']'             // end a character class
  '^'             // negates character class if 1st character
  '*'             // Kleene closure (matches 0 or more)
  '+'             // Positive closure (1 or more)
  '?'             // Optional closure (0 or 1)
~~~
Note that the '|' operator (union) is not supported, nor are
parentheses (grouping). Therefore "a|b" does not match "a".

Standard classes like [:alnum:], [:alpha:], etc. are not supported,
only [a-zA-Z], [^ntf] and so on.
*/

#include "TRegexp.h"
#include "TString.h"
#include "TError.h"
#include "ThreadLocalStorage.h"

const unsigned TRegexp::fgMaxpat = 2048;


ClassImp(TRegexp);

////////////////////////////////////////////////////////////////////////////////
/// Create a regular expression from the input string. If wildcard is
/// true then the input string will first be interpreted as a wildcard
/// expression by MakeWildcard(), and the result then interpreted as a
/// regular expression.

TRegexp::TRegexp(const char *re, Bool_t wildcard)
{
   if (wildcard)
      GenPattern(MakeWildcard(re));
   else
      GenPattern(re);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a regular expression from a TString.

TRegexp::TRegexp(const TString& re)
{
   GenPattern(re.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Copy ctor.

TRegexp::TRegexp(const TRegexp& r)
{
   CopyPattern(r);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TRegexp::~TRegexp()
{
   delete [] fPattern;
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator.

TRegexp& TRegexp::operator=(const TRegexp& r)
{
   if (this != &r) {
      delete [] fPattern;
      CopyPattern(r);
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator taking a char* and assigning it to a regexp.

TRegexp& TRegexp::operator=(const char *str)
{
   delete [] fPattern;
   GenPattern(str);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator taking a TString.

TRegexp& TRegexp::operator=(const TString &str)
{
   delete [] fPattern;
   GenPattern(str.Data());
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Generate the regular expression pattern.

void TRegexp::GenPattern(const char *str)
{
   fPattern = new Pattern_t[fgMaxpat];
   int error = ::Makepat(str, fPattern, fgMaxpat);
   fStat = (error < 3) ? (EStatVal) error : kToolong;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy the regular expression pattern.

void TRegexp::CopyPattern(const TRegexp& r)
{
   fPattern = new Pattern_t[fgMaxpat];
   memcpy(fPattern, r.fPattern, fgMaxpat * sizeof(Pattern_t));
   fStat = r.fStat;
}

////////////////////////////////////////////////////////////////////////////////
/// This routine transforms a wildcarding regular expression into
/// a general regular expression used for pattern matching.
/// When using wildcards the regular expression is assumed to be
/// preceded by a "^" (BOL) and terminated by a "$" (EOL). Also, all
/// "*"'s and "?"'s (closures) are assumed to be preceded by a "." (i.e. any
/// character, except "/"'s) and all .'s are escaped (so *.ps is different
/// from *.eps). The special treatment of "/" allows the easy matching of
/// pathnames, e.g. "*.root" will match "aap.root", but not "pipo/aap.root".

const char *TRegexp::MakeWildcard(const char *re)
{
   TTHREAD_TLS_ARRAY(char,fgMaxpat,buf);
   char *s = buf;
   if (!re) return "";
   int len = strlen(re);
   int slen = 0;

   if (!len) return "";

   for (int i = 0; i < len; i++) {
      if ((unsigned)slen > fgMaxpat - 10) {
         Error("MakeWildcard", "regexp too large");
         break;
      }
      if (i == 0 && re[i] != '^') {
         *s++ = '^';
         slen++;
      }
      if (re[i] == '*') {
#ifndef R__WIN32
         //const char *wc = "[a-zA-Z0-9-+_\\.,: []<>]";
         const char *wc = "[^/]";
#else
         //const char *wc = "[a-zA-Z0-9-+_., []<>]";
         const char *wc = "[^\\/:]";
#endif
         strcpy(s, wc);
         s += strlen(wc);
         slen += strlen(wc);
      }
      if (re[i] == '.') {
         *s++ = '\\';
         slen++;
      }
      if (re[i] == '?') {
#ifndef R__WIN32
         //const char *wc = "[a-zA-Z0-9-+_\\.,: []<>]";
         const char *wc = "[^/]";
#else
         //const char *wc = "[a-zA-Z0-9-+_., []<>]";
         const char *wc = "[^\\/:]";
#endif
         strcpy(s, wc);
         s += strlen(wc);
         slen += strlen(wc);
      } else {
         *s++ = re[i];
         slen++;
      }
      if (i == len-1 && re[i] != '$') {
         *s++ = '$';
         slen++;
      }
   }
   *s = '\0';
   return buf;
}

////////////////////////////////////////////////////////////////////////////////
/// Find the first occurrence of the regexp in string and return the
/// position, or -1 if there is no match. Len is length of the matched
/// string and i is the offset at which the matching should start.

Ssiz_t TRegexp::Index(const TString& string, Ssiz_t* len, Ssiz_t i) const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Check status of regexp.

TRegexp::EStatVal TRegexp::Status()
{
   EStatVal temp = fStat;
   fStat = kOK;
   return temp;
}

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TString member functions, put here so the linker will include              //
// them only if regular expressions are used.                                 //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// Find the first occurrence of the regexp in string and return the
/// position, or -1 if there is no match. Start is the offset at which
/// the search should start.

Ssiz_t TString::Index(const TRegexp& r, Ssiz_t start) const
{
   Ssiz_t len;
   return r.Index(*this, &len, start); // len not used
}

////////////////////////////////////////////////////////////////////////////////
/// Find the first occurrence of the regexp in string and return the
/// position, or -1 if there is no match. Extent is length of the matched
/// string and start is the offset at which the matching should start.

Ssiz_t TString::Index(const TRegexp& r, Ssiz_t* extent, Ssiz_t start) const
{
   return r.Index(*this, extent, start);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the substring found by applying the regexp starting at start.

TSubString TString::operator()(const TRegexp& r, Ssiz_t start) const
{
   Ssiz_t len;
   Ssiz_t begin = Index(r, &len, start);
   return TSubString(*this, begin, len);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the substring found by applying the regexp.

TSubString TString::operator()(const TRegexp& r) const
{
   return (*this)(r,0);
}

////////////////////////////////////////////////////////////////////////////////
/// Search for tokens delimited by regular expression 'delim' (default " ")
/// in this string; search starts at 'from' and the token is returned in 'tok'.
/// Returns in 'from' the next position after the delimiter.
/// Returns kTRUE if a token is found, kFALSE if not or if some inconsistency
/// occurred.
/// This method allows to loop over tokens in this way:
/// ~~~ {.cpp}
///    TString myl = "tok1 tok2|tok3";
///    TString tok;
///    Ssiz_t from = 0;
///    while (myl.Tokenize(tok, from, "[ |]")) {
///       // Analyse tok
///       ...
///    }
/// ~~~
/// more convenient of the other Tokenize method when saving the tokens is not
/// needed.

Bool_t TString::Tokenize(TString &tok, Ssiz_t &from, const char *delim) const
{
   Bool_t found = kFALSE;

   // Reset the token
   tok = "";

   // Make sure inputs make sense
   Int_t len = Length();
   if (len <= 0 || from > (len - 1) || from < 0)
      return found;

   // Ensure backward compatibility to allow one or more times the delimiting character
   TString rdelim(delim);
   if(rdelim.Length() == 1) {
      rdelim = "[" + rdelim + "]+";
   }
   TRegexp rg(rdelim);

   // Find delimiter
   Int_t ext = 0;
   Int_t pos = Index(rg, &ext, from);

   // Assign to token
   if (pos == kNPOS || pos > from) {
      Ssiz_t last = (pos != kNPOS) ? (pos - 1) : len;
      tok = (*this)(from, last-from+1);
   }
   found = kTRUE;

   // Update start-of-search index
   from = pos + ext;
   if (pos == kNPOS) {
      from = pos;
      if (tok.IsNull()) {
         // Empty, last token
         found = kFALSE;
      }
   }
   // Make sure that 'from' has a meaningful value
   from = (from < len) ? from : len;

   // Done
   return found;
}
