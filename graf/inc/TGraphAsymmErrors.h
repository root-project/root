// @(#)root/graf:$Name:  $:$Id: TGraphAsymmErrors.h,v 1.1.1.1 2000/05/16 17:00:50 rdm Exp $
// Author: Rene Brun   03/03/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraphAsymmErrors
#define ROOT_TGraphAsymmErrors


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraphAsymmErrors                                                    //
//                                                                      //
// a Graph with asymmetric error bars                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGraph
#include "TGraph.h"
#endif

class TGraphAsymmErrors : public TGraph {

protected:
    Double_t    *fEXlow;        //[fNpoints] array of X low errors
    Double_t    *fEXhigh;       //[fNpoints] array of X high errors
    Double_t    *fEYlow;        //[fNpoints] array of Y low errors
    Double_t    *fEYhigh;       //[fNpoints] array of Y high errors

public:
        TGraphAsymmErrors();
        TGraphAsymmErrors(Int_t n);
        TGraphAsymmErrors(Int_t n, Float_t *x, Float_t *y, Float_t *exl=0, Float_t *exh=0, Float_t *eyl=0, Float_t *eyh=0);
        TGraphAsymmErrors(Int_t n, Double_t *x, Double_t *y, Double_t *exl=0, Double_t *exh=0, Double_t *eyl=0, Double_t *eyh=0);
        virtual ~TGraphAsymmErrors();
        virtual void    ComputeRange(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax);
        Double_t        GetErrorX(Int_t bin);
        Double_t        GetErrorY(Int_t bin);
        Double_t       *GetEXlow()  {return fEXlow;}
        Double_t       *GetEXhigh() {return fEXhigh;}
        Double_t       *GetEYlow()  {return fEYlow;}
        Double_t       *GetEYhigh() {return fEYhigh;}
        virtual void    Paint(Option_t *chopt="");
        virtual void    Print(Option_t *chopt="");
        virtual void    SavePrimitive(ofstream &out, Option_t *option);
        virtual void    SetPoint(Int_t i, Double_t x, Double_t y);
        virtual void    SetPointError(Int_t i, Double_t exl, Double_t exh, Double_t eyl, Double_t eyh);

        ClassDef(TGraphAsymmErrors,2)  //a Graph with asymmetric error bars
};

#endif
