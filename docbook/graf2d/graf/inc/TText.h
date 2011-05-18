// @(#)root/graf:$Id$
// Author: Nicolas Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TText
#define ROOT_TText


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TText                                                                //
//                                                                      //
// Text.                                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TAttText
#include "TAttText.h"
#endif

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif


class TText : public TNamed, public TAttText {

protected:
   Double_t     fX;           // X position of text (left,center,etc..)
   Double_t     fY;           // Y position of text (left,center,etc..)

public:
   // TText status bits
   enum { kTextNDC = BIT(14) };

   TText();
   TText(Double_t x, Double_t y, const char *text);
   TText(const TText &text);
   virtual ~TText();
   void             Copy(TObject &text) const;
   virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
   virtual TText   *DrawText(Double_t x, Double_t y, const char *text);
   virtual TText   *DrawTextNDC(Double_t x, Double_t y, const char *text);
   virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);

   virtual void     GetControlBox(Int_t x, Int_t y, Double_t theta,
                                  Int_t cBoxX[4], Int_t cBoxY[4]);
   Double_t         GetX() const  { return fX; }
   virtual void     GetBoundingBox(UInt_t &w, UInt_t &h, Bool_t angle = kFALSE);
   virtual void     GetTextAscentDescent(UInt_t &a, UInt_t &d, const char *text) const;
   virtual void     GetTextExtent(UInt_t &w, UInt_t &h, const char *text) const;
   virtual void     GetTextAdvance(UInt_t &a, const char *text, const Bool_t kern=kTRUE) const;
   Double_t         GetY() const  { return fY; }

   virtual void     ls(Option_t *option="") const;
   virtual void     Paint(Option_t *option="");
   virtual void     PaintControlBox(Int_t x, Int_t y, Double_t theta);
   virtual void     PaintText(Double_t x, Double_t y, const char *text);
   virtual void     PaintTextNDC(Double_t u, Double_t v, const char *text);
   virtual void     Print(Option_t *option="") const;
   virtual void     SavePrimitive(ostream &out, Option_t *option = "");
   virtual void     SetNDC(Bool_t isNDC=kTRUE);
   virtual void     SetText(Double_t x, Double_t y, const char *text) {fX=x; fY=y; SetTitle(text);} // *MENU* *ARGS={x=>fX,y=>fY,text=>fTitle}
   virtual void     SetX(Double_t x) { fX = x; } // *MENU*
   virtual void     SetY(Double_t y) { fY = y; } // *MENU*

   ClassDef(TText,2)  //Text
};

#endif
