// @(#)root/base:$Id$
// Author: Rene Brun   15/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TStringLong
#define ROOT_TStringLong

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStringLong                                                          //
//                                                                      //
// ATTENTION: this class is obsolete. It's functionality has been taken //
// over by TString.                                                     //
//                                                                      //
// The long string class (unlimited number of chars in I/O).            //
// Class TString can contain long strings, but it can read/write only   //
// 255 characters.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"

class TStringLong : public TString {

public:
   TStringLong();                       // Null string
   TStringLong(Ssiz_t ic);              // Suggested capacity
   TStringLong(const TString& s);       // Copy constructor

   TStringLong(const char *s);              // Copy to embedded null
   TStringLong(const char *s, Ssiz_t n);    // Copy past any embedded nulls
   TStringLong(char c);

   TStringLong(char c, Ssiz_t s);

   TStringLong(const TSubString& sub);
   virtual ~TStringLong();

   // ROOT I/O interface
   virtual void     FillBuffer(char *&buffer) const;
   virtual void     ReadBuffer(char *&buffer);
   virtual Int_t    Sizeof() const;

   ClassDef(TStringLong,1)  //Long string class (more than 255 chars)
} R__ALWAYS_SUGGEST_ALTERNATIVE("TString");

#endif
