// @(#)root/base:$Name:  $:$Id: TPRegexp.h,v 1.1 2005/12/02 16:17:48 rdm Exp $
// Author: Eddy Offermann   24/06/05

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPRegexp
#define ROOT_TPRegexp

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

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TArrayI
#include "TArrayI.h"
#endif

struct PCREPriv_t;


class TPRegexp {

private:
   enum {
      kPCRE_GLOBAL     = 0x80000000,
      kPCRE_OPTIMIZE   = 0x40000000,
      kPCRE_DEBUG_MSGS = 0x20000000,
      kPCRE_INTMASK    = 0x0FFF
   };

   TString     fPattern;
   PCREPriv_t *fPriv;
   UInt_t      fPCREOpts;

   void     Compile();
   void     Optimize();
   UInt_t   ParseMods(const TString &mods) const;
   Int_t    ReplaceSubs(const TString &s, TString &final,
                        const TString &replacePattern,
                        Int_t *ovec, Int_t nmatch) const;

public:
   TPRegexp();
   TPRegexp(const TString &pat);
   TPRegexp(const TPRegexp &p);
   virtual ~TPRegexp();

   Int_t      Match(const TString &s, const TString &mods="",
                    Int_t offset=0, Int_t nMatchMax=30, TArrayI *pos=0);
   TObjArray *MatchS(const TString &s, const TString &mods="",
                     Int_t offset=0, Int_t nMaxMatch=30);
   Bool_t     MatchB(const TString &s, const TString &mods="",
                     Int_t offset=0, Int_t nMaxMatch=30) {
                           return (Match(s,mods,offset,nMaxMatch) > 0); }
   Int_t      Substitute(TString &s, const TString &replace,
                         const TString &mods="", Int_t offset=0,
                         Int_t nMatchMax=30);

   TPRegexp &operator=(const TPRegexp &p);

   ClassDef(TPRegexp,0)  // Perl Compatible Regular Expression Class
};

#endif
