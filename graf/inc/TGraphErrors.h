// @(#)root/graf:$Name$:$Id$
// Author: Rene Brun   15/09/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraphErrors
#define ROOT_TGraphErrors


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraphErrors                                                         //
//                                                                      //
// a Graph with error bars                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGraph
#include "TGraph.h"
#endif

class TGraphErrors : public TGraph {

protected:
    Float_t    *fEX;        //[fNpoints] array of X errors
    Float_t    *fEY;        //[fNpoints] array of Y errors

public:
        TGraphErrors();
        TGraphErrors(Int_t n, Float_t *x=0, Float_t *y=0, Float_t *ex=0, Float_t *ey=0);
        virtual ~TGraphErrors();
        virtual void    ComputeRange(Float_t &xmin, Float_t &ymin, Float_t &xmax, Float_t &ymax);
        Float_t         GetErrorX(Int_t bin);
        Float_t         GetErrorY(Int_t bin);
        Float_t         *GetEX() {return fEX;}
        Float_t         *GetEY() {return fEY;}
        virtual void    Paint(Option_t *chopt="");
        virtual void    Print(Option_t *chopt="");
        virtual void    SavePrimitive(ofstream &out, Option_t *option);
        virtual void    SetPoint(Int_t i, Float_t x, Float_t y);
        virtual void    SetPointError(Int_t i, Float_t ex, Float_t ey);

        ClassDef(TGraphErrors,1)  //a Graph with error bars
};

#endif


