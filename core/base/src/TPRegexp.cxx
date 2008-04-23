// @(#)root/base:$Id$
// Author: Eddy Offermann   24/06/05

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPRegexp                                                             //
//                                                                      //
// C++ Wrapper for the "Perl Compatible Regular Expressions" library    //
//  The PCRE lib can be found at:                                       //
//              http://www.pcre.org/                                    //
//                                                                      //
// Extensive documentation about Regular expressions in Perl can be     //
// found at :                                                           //
//              http://perldoc.perl.org/perlre.html                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Riostream.h"
#include "TPRegexp.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TError.h"

#include <pcre.h>


struct PCREPriv_t {
   pcre       *fPCRE;
   pcre_extra *fPCREExtra;

   PCREPriv_t() { fPCRE = 0; fPCREExtra = 0; }
};


ClassImp(TPRegexp)

//______________________________________________________________________________
TPRegexp::TPRegexp()
{
   // Default ctor.

   fPriv     = new PCREPriv_t;
   fPCREOpts = 0;
}

//______________________________________________________________________________
TPRegexp::TPRegexp(const TString &pat)
{
   // Create and initialize with pat.

   fPattern  = pat;
   fPriv     = new PCREPriv_t;
   fPCREOpts = 0;
}

//______________________________________________________________________________
TPRegexp::TPRegexp(const TPRegexp &p)
{
   // Copy ctor.

   fPattern  = p.fPattern;
   fPriv     = new PCREPriv_t;
   fPCREOpts = p.fPCREOpts;
}

//______________________________________________________________________________
TPRegexp::~TPRegexp()
{
   // Cleanup.

   if (fPriv->fPCRE)
      pcre_free(fPriv->fPCRE);
   if (fPriv->fPCREExtra)
      pcre_free(fPriv->fPCREExtra);
   delete fPriv;
}

//______________________________________________________________________________
TPRegexp &TPRegexp::operator=(const TPRegexp &p)
{
   // Assignement operator.

   if (this != &p) {
      fPattern = p.fPattern;
      if (fPriv->fPCRE)
         pcre_free(fPriv->fPCRE);
      fPriv->fPCRE = 0;
      if (fPriv->fPCREExtra)
         pcre_free(fPriv->fPCREExtra);
      fPriv->fPCREExtra = 0;
      fPCREOpts  = p.fPCREOpts;
   }
   return *this;
}

//______________________________________________________________________________
UInt_t TPRegexp::ParseMods(const TString &modStr) const
{
   // Translate Perl modifier flags into pcre flags.

   UInt_t opts = 0;

   if (modStr.Length() <= 0)
      return fPCREOpts;

   //translate perl flags into pcre flags
   const char *m = modStr;
   while (*m) {
      switch (*m) {
         case 'g':
            opts |= kPCRE_GLOBAL;
            break;
         case 'i':
            opts |= PCRE_CASELESS;
            break;
         case 'm':
            opts |= PCRE_MULTILINE;
            break;
         case 'o':
            opts |= kPCRE_OPTIMIZE;
            break;
         case 's':
            opts |= PCRE_DOTALL;
            break;
         case 'x':
            opts |= PCRE_EXTENDED;
            break;
         case 'd': // special flag to enable debug printing (not Perl compat.)
            opts |= kPCRE_DEBUG_MSGS;
            break;
         default:
            Error("ParseMods", "illegal pattern modifier: %c", *m);
	    opts = 0;
      }
      ++m;
   }
   return opts;
}

//______________________________________________________________________________
void TPRegexp::Compile()
{
   // Compile the fPattern.

   if (fPriv->fPCRE)
      pcre_free(fPriv->fPCRE);

   if (fPCREOpts & kPCRE_DEBUG_MSGS)
      Info("Compile", "PREGEX compiling %s", fPattern.Data());

   const char *errstr;
   Int_t patIndex;
   fPriv->fPCRE = pcre_compile(fPattern.Data(), fPCREOpts & kPCRE_INTMASK,
                               &errstr, &patIndex, 0);

   if (!fPriv->fPCRE) {
      Error("Compile", "compilation of TPRegexp(%s) failed at: %d because %s",
            fPattern.Data(), patIndex, errstr);
   }

   if (fPriv->fPCREExtra || (fPCREOpts & kPCRE_OPTIMIZE))
      Optimize();
}

//______________________________________________________________________________
void TPRegexp::Optimize()
{
   // Send the pattern through the optimizer.

   if (fPriv->fPCREExtra)
      pcre_free(fPriv->fPCREExtra);

   if (fPCREOpts & kPCRE_DEBUG_MSGS)
      Info("Optimize", "PREGEX studying %s", fPattern.Data());

   const char *errstr;
   fPriv->fPCREExtra = pcre_study(fPriv->fPCRE, fPCREOpts & kPCRE_INTMASK, &errstr);

   if (!fPriv->fPCREExtra && errstr) {
      Error("Optimize", "Optimization of TPRegexp(%s) failed: %s",
            fPattern.Data(), errstr);
   }
}

//______________________________________________________________________________
Int_t TPRegexp::ReplaceSubs(const TString &s, TString &final,
                            const TString &replacePattern,
                            Int_t *offVec, Int_t nrMatch) const
{
   // Return the number of substitutions.

   Int_t nrSubs = 0;
   const char *p = replacePattern;

   Int_t state = 0;
   Int_t subnum = 0;
   while (state != -1) {
      switch (state) {
         case 0:
            if (!*p) {
               state = -1;
               break;
            }
            if (*p == '$') {
               state = 1;
               subnum = 0;
               if (p[1] == '&') {
                  p++;
                  if (isdigit(p[1]))
                     p++;
               } else if (!isdigit(p[1])) {
                  Error("ReplaceSubs", "badly formed replacement pattern: %s",
                        replacePattern.Data());
               }
            } else
               final += *p;
            break;
         case 1:
            if (isdigit(*p)) {
               subnum *= 10;
               subnum += (*p)-'0';
            } else {
               if (fPCREOpts & kPCRE_DEBUG_MSGS)
                  Info("ReplaceSubs", "PREGEX appending substr #%d", subnum);
               if (subnum < 0 || subnum > nrMatch-1) {
                  Error("ReplaceSubs","bad string number :%d",subnum);
               }
               const TString subStr = s(offVec[2*subnum],offVec[2*subnum+1]-offVec[2*subnum]);
               final += subStr;
               nrSubs++;

               state = 0;
               continue;  // send char to start state
            }
      }
      p++;
   }
   return nrSubs;
}

//______________________________________________________________________________
Int_t TPRegexp::Match(const TString &s, const TString &mods, Int_t start,
                      Int_t nMaxMatch, TArrayI *pos)
{
   // The number of matches is returned, this equals the full match +
   // sub-pattern matches.
   // nMaxmatch is the maximum allowed number of matches.
   // pos contains the string indices of the matches. Its usage is
   // shown in the routine MatchS.

   UInt_t opts = ParseMods(mods);

   if (!fPriv->fPCRE || opts != fPCREOpts) {
      fPCREOpts = opts;
      Compile();
   }

   Int_t *offVec = new Int_t[nMaxMatch];
   Int_t nrMatch = pcre_exec(fPriv->fPCRE, fPriv->fPCREExtra, s.Data(),
                             s.Length(), start, fPCREOpts & kPCRE_INTMASK,
                             offVec, nMaxMatch);

   if (nrMatch == PCRE_ERROR_NOMATCH)
      nrMatch = 0;
   else if (nrMatch <= 0) {
      Error("Match","pcre_exec error = %d", nrMatch);
      delete [] offVec;
      return 0;
   }

   if (pos)
      pos->Adopt(2*nrMatch, offVec);
   else
      delete [] offVec;

   return nrMatch;
}

//______________________________________________________________________________
TObjArray *TPRegexp::MatchS(const TString &s, const TString &mods,
                            Int_t start, Int_t nMaxMatch)
{
   // Returns a TObjArray of matched substrings as TObjString's.
   // The TObjArray is owner of the objects. The first entry is the full
   // matched pattern, followed by the subpatterns.
   // If a pattern was not matched, it will return an empty substring:
   //
   // TObjArray *subStrL = TPRegexp("(a|(z))(bc)").MatchS("abc");
   // for (Int_t i = 0; i < subStrL->GetLast()+1; i++) {
   //    const TString subStr = ((TObjString *)subStrL->At(i))->GetString();
   //    cout << "\"" << subStr << "\" ";
   // }
   // cout << subStr << endl;
   //
   // produces:  "abc" "a" "" "bc"

   TArrayI pos;
   Int_t nrMatch = Match(s, mods, start, nMaxMatch, &pos);

   TObjArray *subStrL = new TObjArray();
   subStrL->SetOwner();

   for (Int_t i = 0; i < nrMatch; i++) {
      Int_t startp = pos[2*i];
      Int_t stopp  = pos[2*i+1];
      if (startp >= 0 && stopp >= 0) {
         const TString subStr = s(pos[2*i], pos[2*i+1]-pos[2*i]);
         subStrL->Add(new TObjString(subStr));
      } else
         subStrL->Add(new TObjString());
   }

   return subStrL;
}

//______________________________________________________________________________
Int_t TPRegexp::Substitute(TString &s, const TString &replacePattern,
                           const TString &mods, Int_t start, Int_t nMaxMatch)
{
   // Substitute replaces the string s by a new string in which matching
   // patterns are replaced by the replacePattern string. The number of
   // substitutions are returned.
   //
   // TString s("aap noot mies");
   // const Int_t nrSub = TPRegexp("(\\w*) noot (\\w*)").Substitute(s,"$2 noot $1");
   // cout << nrSub << " \"" << s << "\"" <<endl;
   //
   // produces: 2 "mies noot aap"

   UInt_t opts = ParseMods(mods);
   Int_t nrSubs = 0;
   TString final;

   if (!fPriv->fPCRE || opts != fPCREOpts) {
      fPCREOpts = opts;
      Compile();
   }

   Int_t *offVec = new Int_t[nMaxMatch];

   Int_t offset = start;
   Int_t last = 0;

   while (kTRUE) {

      // find next matching subs
      Int_t nrMatch = pcre_exec(fPriv->fPCRE, fPriv->fPCREExtra, s.Data(),
                                s.Length(), offset, fPCREOpts & kPCRE_INTMASK,
                                offVec, nMaxMatch);

      if (nrMatch == PCRE_ERROR_NOMATCH) {
         nrMatch = 0;
         break;
      } else if (nrMatch <= 0) {
         Error("Substitute", "pcre_exec error = %d", nrMatch);
         break;
      }

      // append anything previously unmatched, but not substituted
      if (last <= offVec[0]) {
         final += s(last,offVec[0]-last);
         last = offVec[1];
      }

      // replace stuff in s
      nrSubs += ReplaceSubs(s, final, replacePattern, offVec, nrMatch);

      // if global gotta check match at every pos
      if (!(fPCREOpts & kPCRE_GLOBAL))
         break;

      if (offVec[0] != offVec[1])
         offset = offVec[1];
      else {
         // matched empty string
         if (offVec[1] == s.Length())
         break;
         offset = offVec[1]+1;
      }
   }

   delete [] offVec;

   final += s(last,s.Length()-last);
   s = final;

   return nrSubs;
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TString member functions, put here so the linker will include        //
// them only if regular expressions are used.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
Ssiz_t TString::Index(TPRegexp& r, Ssiz_t start) const
{
   // Find the first occurance of the regexp in string and return the position.
   // Start is the offset at which the search should start.

   TArrayI pos;
   Int_t nrMatch = r.Match(*this,"",start,30,&pos);
   if (nrMatch > 0)
      return pos[0];
   else
      return -1;
}

//______________________________________________________________________________
Ssiz_t TString::Index(TPRegexp& r, Ssiz_t* extent, Ssiz_t start) const
{
   // Find the first occurance of the regexp in string and return the position.
   // Extent is length of the matched string and start is the offset at which
   // the matching should start.

   TArrayI pos;
   const Int_t nrMatch = r.Match(*this,"",start,30,&pos);
   if (nrMatch > 0) {
      *extent = pos[1]-pos[0];
      return pos[0];
   } else {
      *extent = 0;
      return -1;
   }
}

//______________________________________________________________________________
TSubString TString::operator()(TPRegexp& r, Ssiz_t start)
{
   // Return the substring found by applying the regexp starting at start.

   Ssiz_t len;
   Ssiz_t begin = Index(r, &len, start);
   return TSubString(*this, begin, len);
}

//______________________________________________________________________________
TSubString TString::operator()(TPRegexp& r)
{
   // Return the substring found by applying the regexp.

   return (*this)(r,0);
}

//______________________________________________________________________________
TSubString TString::operator()(TPRegexp& r, Ssiz_t start) const
{
   // Return the substring found by applying the regexp starting at start.

   Ssiz_t len;
   Ssiz_t begin = Index(r, &len, start);
   return TSubString(*this, begin, len);
}

//______________________________________________________________________________
TSubString TString::operator()(TPRegexp& r) const
{
   // Return the substring found by applying the regexp.

   return (*this)(r, 0);
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStringToken                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
//
// Provides iteration through tokens of a given string:
//
// - fFullStr     stores the string to be split. It is never modified.
// - fSplitRe     is the perl-re that is used to separete the tokens.
// - fReturnVoid  if true, empty strings will be returned.
//
// Current token is stored in the TString base-class.
// During construction no match is done, use NextToken() to get the first
// and all subsequent tokens.
//

ClassImp(TStringToken)

//______________________________________________________________________________
TStringToken::TStringToken(const TString& fullStr, const TString& splitRe, Bool_t retVoid) :
   fFullStr    (fullStr),
   fSplitRe    (splitRe),
   fReturnVoid (retVoid),
   fPos        (0)
{
   // Constructor.
}

//______________________________________________________________________________
Bool_t TStringToken::NextToken()
{
   // Get the next token, it is stored in this TString.
   // Returns true if new token is available, false otherwise.

   TArrayI x;
   while (fPos < fFullStr.Length())
   {
      if (fSplitRe.Match(fFullStr, "", fPos, 2, &x))
      {
         TString::operator=(fFullStr(fPos, x[0] - fPos));
         fPos = x[1];
      } else {
         TString::operator=(fFullStr(fPos, fFullStr.Length() - fPos));
         fPos = fFullStr.Length() + 1;
      }
      if (Length() || fReturnVoid)
         return kTRUE;
   }

   // Special case: void-strings are requested and the full-string
   // ends with the separator. Thus we return another empty string.
   if (fPos == fFullStr.Length() && fReturnVoid) {
      TString::operator=("");
      fPos = fFullStr.Length() + 1;
      return kTRUE;
   }

   return kFALSE;
}
