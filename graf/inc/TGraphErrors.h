// @(#)root/graf:$Name:  $:$Id: TGraphErrors.h,v 1.8 2001/12/19 14:21:54 brun Exp $
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
        TGraphErrors(Int_t n, const Float_t *x, const Float_t *y, const Float_t *ex=0, const Float_t *ey=0);
        TGraphErrors(Int_t n, const Double_t *x, const Double_t *y, const Double_t *ex=0, const Double_t *ey=0);
        virtual ~TGraphErrors();
        virtual void    Apply(TF1 *f);
        virtual void    ComputeRange(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax);
        Double_t        GetErrorX(Int_t bin) const;
        Double_t        GetErrorY(Int_t bin) const;
        Double_t       *GetEX() const {return fEX;}
        Double_t       *GetEY() const {return fEY;}
        virtual Int_t   InsertPoint(); // *MENU*
        virtual void    Paint(Option_t *chopt="");
        virtual void    Print(Option_t *chopt="") const;
        virtual Int_t   RemovePoint(); // *MENU*
        virtual void    SavePrimitive(ofstream &out, Option_t *option);
        virtual void    Set(Int_t n);
        virtual void    SetPoint(Int_t i, Double_t x, Double_t y);
        virtual void    SetPointError(Double_t ex, Double_t ey);  // *MENU
        virtual void    SetPointError(Int_t i, Double_t ex, Double_t ey);

        ClassDef(TGraphErrors,3)  //A graph with error bars
};

#endif


