// @(#)root/base:$Id$
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRegexp
#define ROOT_TRegexp


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TRegexp                                                             //
//                                                                      //
//  Declarations for regular expression class.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef ROOT_Match
#include "Match.h"
#endif

class TString;


class TRegexp {

public:
   enum EStatVal { kOK = 0, kIllegal, kNomem, kToolong };

private:
   Pattern_t            *fPattern;       // Compiled pattern
   EStatVal              fStat;          // Status
   static const unsigned fgMaxpat;       // Max length of compiled pattern

   void                  CopyPattern(const TRegexp& re);
   void                  GenPattern(const char *re);
   const char           *MakeWildcard(const char *re);

public:
   TRegexp(const char *re, Bool_t wildcard = kFALSE);
   TRegexp(const TString& re);
   TRegexp(const TRegexp& re);
   virtual ~TRegexp();

   TRegexp&              operator=(const TRegexp& re);
   TRegexp&              operator=(const TString& re);   // Recompiles pattern
   TRegexp&              operator=(const char *re);      // Recompiles pattern
   Ssiz_t                Index(const TString& str, Ssiz_t *len, Ssiz_t start=0) const;
   EStatVal              Status();                       // Return & clear status

   ClassDef(TRegexp,0)  // Regular expression class
};

#endif
