// @(#)root/graf:$Name$:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPolyLine
#define ROOT_TPolyLine


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPolyLine                                                            //
//                                                                      //
// A PolyLine.                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif
#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif


class TPolyLine : public TObject, public TAttLine, public TAttFill {

protected:
        Int_t        fN;            //Number of points
        Float_t      *fX;           //[fN] Array of X coordinates
        Float_t      *fY;           //[fN] Array of Y coordinates
        TString      fOption;       //options

public:
        TPolyLine();
        TPolyLine(Int_t n, Option_t *option="");
        TPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
        TPolyLine(const TPolyLine &polyline);
        virtual ~TPolyLine();
        virtual void    Copy(TObject &polyline);
        virtual Int_t   DistancetoPrimitive(Int_t px, Int_t py);
        virtual void    Draw(Option_t *option="");
        virtual void    DrawPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
        virtual void    ExecuteEvent(Int_t event, Int_t px, Int_t py);
        Int_t           GetN() {return fN;}
        Float_t         *GetX() {return fX;}
        Float_t         *GetY() {return fY;}
        Option_t        *GetOption() const {return fOption.Data();}
        virtual void    ls(Option_t *option="");
        virtual void    Paint(Option_t *option="");
        virtual void    PaintPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
        virtual void    PaintPolyLineNDC(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
        virtual void    Print(Option_t *option="");
        virtual void    SavePrimitive(ofstream &out, Option_t *option);
        virtual void    SetOption(Option_t *option="") {fOption = option;}
        virtual void    SetPoint(Int_t point, Float_t x, Float_t y); // *MENU*
        virtual void    SetPolyLine(Int_t n, Float_t *x=0, Float_t *y=0, Option_t *option="");

        ClassDef(TPolyLine,1)  //A PolyLine
};

#endif

