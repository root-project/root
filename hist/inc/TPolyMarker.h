// @(#)root/hist:$Name$:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPolyMarker
#define ROOT_TPolyMarker


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPolyMarker                                                          //
//                                                                      //
// An array of points with the same marker.                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TAttMarker
#include "TAttMarker.h"
#endif

class TPolyMarker : public TObject, public TAttMarker {
protected:
        Int_t       fN;            //number of points
        Float_t     *fX;           //[fN] Array of X coordinates
        Float_t     *fY;           //[fN] Array of Y coordinates
        TString     fOption;       //options

public:
        TPolyMarker();
        TPolyMarker(Int_t n, Option_t *option="");
        TPolyMarker(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
        TPolyMarker(const TPolyMarker &polymarker);
        virtual ~TPolyMarker();
        virtual void     Copy(TObject &polymarker);
        virtual void     Draw(Option_t *option="");
        virtual void     DrawPolyMarker(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
        virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
        virtual Int_t    GetN() {return fN;}
        Option_t         *GetOption() const {return fOption.Data();}
        Float_t          *GetX() {return fX;}
        Float_t          *GetY() {return fY;}
        virtual void     ls(Option_t *option="");
        virtual void     Paint(Option_t *option="");
        virtual void     PaintPolyMarker(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
        virtual void     Print(Option_t *option="");
        virtual void     SavePrimitive(ofstream &out, Option_t *option);
        virtual void     SetPoint(Int_t point, Float_t x, Float_t y); // *MENU*
        virtual void     SetPolyMarker(Int_t n, Float_t *x=0, Float_t *y=0, Option_t *option="");

        ClassDef(TPolyMarker,1)  //An array of points with the same marker
};

#endif
