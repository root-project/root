// @(#)root/base:$Name$:$Id$
// Author: Fons Rademakers   12/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TObjString
#define ROOT_TObjString


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjString                                                           //
//                                                                      //
// Collectable string class. This is a TObject containing a TString.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif


class TObjString : public TObject {

private:
   TString    fString;       // wrapped TString

public:
   TObjString(const char *s = "") : fString(s) { }
   TObjString(const TObjString &s) : fString(s.fString) { }
   ~TObjString() { }
   Int_t     Compare(TObject *obj);
   const char *GetName() const { return fString.Data(); }
   ULong_t   Hash() { return fString.Hash(); }
   void      FillBuffer(char *&buffer) { fString.FillBuffer(buffer); }
   void      Print(Option_t *) { Printf("TObjString = %s", (const char*)fString); }
   Bool_t    IsSortable() const { return kTRUE; }
   Bool_t    IsEqual(TObject *obj);
   void      ReadBuffer(char *&buffer) { fString.ReadBuffer(buffer); }
   void      SetString(char *s) { fString = s; }
   TString   GetString() const { return fString; }
   Int_t     Sizeof() const { return fString.Sizeof(); }
   TString  &String() { return fString; }

   ClassDef(TObjString,1)  //Collectable string class
};

#endif

