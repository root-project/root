// @(#)root/gui:$Name:  $:$Id: TGString.h,v 1.2 2001/05/29 14:26:17 rdm Exp $
// Author: Fons Rademakers   05/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGString
#define ROOT_TGString


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGString and TGHotString                                             //
//                                                                      //
// TGString wraps a TString and adds some graphics routines like        //
// drawing, size of string on screen depending on font, etc.            //
// TGHotString is a string with a "hot" character unerlined.            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_GuiTypes
#include "GuiTypes.h"
#endif


class TGString {

protected:
   TString   fString;   // embedded string

public:
   TGString() { }
   TGString(TString s) { fString = s; }
   TGString(const char *s) { fString = s; }
   TGString(const TGString *s) { fString = s->fString; }
   TGString(Int_t number) { fString += number; }
   virtual ~TGString() { }

   Int_t GetLength() const { return fString.Length(); }
   const char *GetString() const { return (const char *)fString; }
   void SetString(const char *s) { fString = s; }

   virtual void Draw(Drawable_t id, GContext_t gc, Int_t x, Int_t y);
   virtual void DrawWrapped(Drawable_t id, GContext_t gc,
                            Int_t x, Int_t y, UInt_t w, FontStruct_t font);
   virtual Int_t GetLines(FontStruct_t font, UInt_t w);
   TGString     &operator=(const char *s) { fString = s; return *this; }

   ClassDef(TGString,0)  // Graphics string
};


class TGHotString : public TGString {

protected:
   char        fHotChar;      // hot character
   Int_t       fHotPos;       // position of hot character

   GContext_t  fLastGC;       // context used during last drawing
   Int_t       fOff1;         // variable used during drawing (cache)
   Int_t       fOff2;         // variable used during drawing (cache)

   void DrawHotChar(Drawable_t id, GContext_t gc, Int_t x, Int_t y);

public:
   TGHotString(const char *s);

   Int_t GetHotChar() const { return fHotChar; }
   virtual void Draw(Drawable_t id, GContext_t gc, Int_t x, Int_t y);
   virtual void DrawWrapped(Drawable_t id, GContext_t gc,
                            Int_t x, Int_t y, UInt_t w, FontStruct_t font);

   ClassDef(TGHotString,0)  // Graphics string with hot character
};

#endif
