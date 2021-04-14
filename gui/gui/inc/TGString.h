// @(#)root/gui:$Id$
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


#include "TString.h"
#include "GuiTypes.h"


class TGString : public TString {

public:
   TGString() : TString() { }
   TGString(const char *s) : TString(s) { }
   TGString(Int_t number) : TString() { *this += number; }
   TGString(const TGString *s);
   virtual ~TGString() { }

   Int_t GetLength() const { return Length(); }
   const char  *GetString() const { return Data(); }
   virtual void SetString(const char *s) { *this = s; }

   virtual void Draw(Drawable_t id, GContext_t gc, Int_t x, Int_t y);
   virtual void DrawWrapped(Drawable_t id, GContext_t gc,
                            Int_t x, Int_t y, UInt_t w, FontStruct_t font);
   virtual Int_t GetLines(FontStruct_t font, UInt_t w);

   ClassDef(TGString,0)  // Graphics string
};


class TGHotString : public TGString {

protected:
   char        fHotChar;      ///< hot character
   Int_t       fHotPos;       ///< position of hot character

   GContext_t  fLastGC;       ///< context used during last drawing
   Int_t       fOff1;         ///< variable used during drawing (cache)
   Int_t       fOff2;         ///< variable used during drawing (cache)

   void DrawHotChar(Drawable_t id, GContext_t gc, Int_t x, Int_t y);

public:
   TGHotString(const char *s);

   Int_t GetHotChar() const { return fHotChar; }
   Int_t GetHotPos() const { return fHotPos; }
   virtual void Draw(Drawable_t id, GContext_t gc, Int_t x, Int_t y);
   virtual void DrawWrapped(Drawable_t id, GContext_t gc,
                            Int_t x, Int_t y, UInt_t w, FontStruct_t font);

   ClassDef(TGHotString,0)  // Graphics string with hot character
};

#endif
