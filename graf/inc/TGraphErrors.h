// @(#)root/graf:$Name:  $:$Id: TGraphErrors.h,v 1.1.1.1 2000/05/16 17:00:50 rdm Exp $
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
    Double_t    *fEX;        //[fNpoints] array of X errors
    Double_t    *fEY;        //[fNpoints] array of Y errors

public:
        TGraphErrors();
        TGraphErrors(Int_t n);
        TGraphErrors(Int_t n, Float_t *x, Float_t *y, Float_t *ex=0, Float_t *ey=0);
        TGraphErrors(Int_t n, Double_t *x, Double_t *y, Double_t *ex=0, Double_t *ey=0);
        virtual ~TGraphErrors();
        virtual void    ComputeRange(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax);
        Double_t        GetErrorX(Int_t bin);
        Double_t        GetErrorY(Int_t bin);
        Double_t       *GetEX() {return fEX;}
        Double_t       *GetEY() {return fEY;}
        virtual void    Paint(Option_t *chopt="");
        virtual void    Print(Option_t *chopt="");
        virtual void    SavePrimitive(ofstream &out, Option_t *option);
        virtual void    SetPoint(Int_t i, Double_t x, Double_t y);
        virtual void    SetPointError(Int_t i, Double_t ex, Double_t ey);

        ClassDef(TGraphErrors,2)  //a Graph with error bars
};

#endif


