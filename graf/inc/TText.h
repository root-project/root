// @(#)root/graf:$Name$:$Id$
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
        Coord_t     fX;           //X position of text (left,center,etc..)
        Coord_t     fY;           //Y position of text (left,center,etc..)

public:
        // TText status bits
        enum { kTextNDC = BIT(14) };

        TText();
        TText(Coord_t x, Coord_t y, const char *text);
        TText(const TText &text);
        virtual ~TText();
                void     Copy(TObject &text);
        virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
        virtual TText   *DrawText(Coord_t x, Coord_t y, const char *text);
        virtual TText   *DrawTextNDC(Coord_t x, Coord_t y, const char *text);
        virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);

         Coord_t         GetX()  {return fX;}
         Coord_t         GetY()  {return fY;}

        virtual void     ls(Option_t *option="");
        virtual void     Paint(Option_t *option="");
        virtual void     PaintText(Coord_t x, Coord_t y, const char *text);
        virtual void     PaintTextNDC(Coord_t u, Coord_t v, const char *text);
        virtual void     Print(Option_t *option="");
        virtual void     SavePrimitive(ofstream &out, Option_t *option);
        virtual void     SetNDC(Bool_t isNDC=kTRUE);
        virtual void     SetText(Coord_t x, Coord_t y, const char *text) {fX=x; fY=y; SetTitle(text);} // *MENU* *ARGS={x=>fX,y=>fY,text=>fTitle}
        virtual void     SetX(Coord_t x) { fX = x;} // *MENU*
        virtual void     SetY(Coord_t y) { fY = y;} // *MENU*

        ClassDef(TText,1)  //Text
};

#endif

