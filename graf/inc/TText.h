// @(#)root/graf:$Name:  $:$Id: TText.h,v 1.2 2000/06/13 11:20:32 brun Exp $
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
        Double_t     fX;           //X position of text (left,center,etc..)
        Double_t     fY;           //Y position of text (left,center,etc..)

public:
        // TText status bits
        enum { kTextNDC = BIT(14) };

        TText();
        TText(Double_t x, Double_t y, const char *text);
        TText(const TText &text);
        virtual ~TText();
                void     Copy(TObject &text);
        virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
        virtual TText   *DrawText(Double_t x, Double_t y, const char *text);
        virtual TText   *DrawTextNDC(Double_t x, Double_t y, const char *text);
        virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);

        Double_t         GetX() const  {return fX;}
        Double_t         GetY() const  {return fY;}

        virtual void     ls(Option_t *option="") const;
        virtual void     Paint(Option_t *option="");
        virtual void     PaintText(Double_t x, Double_t y, const char *text);
        virtual void     PaintTextNDC(Double_t u, Double_t v, const char *text);
        virtual void     Print(Option_t *option="") const;
        virtual void     SavePrimitive(ofstream &out, Option_t *option);
        virtual void     SetNDC(Bool_t isNDC=kTRUE);
        virtual void     SetText(Double_t x, Double_t y, const char *text) {fX=x; fY=y; SetTitle(text);} // *MENU* *ARGS={x=>fX,y=>fY,text=>fTitle}
        virtual void     SetX(Double_t x) { fX = x;} // *MENU*
        virtual void     SetY(Double_t y) { fY = y;} // *MENU*

        ClassDef(TText,2)  //Text
};

#endif

