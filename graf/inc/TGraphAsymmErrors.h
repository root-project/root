// @(#)root/graf:$Name:  $:$Id: TGraphAsymmErrors.h,v 1.8 2002/02/23 15:45:56 rdm Exp $
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
        TGraphAsymmErrors(Int_t n, const Float_t *x, const Float_t *y, const Float_t *exl=0, const Float_t *exh=0, const Float_t *eyl=0, const Float_t *eyh=0);
        TGraphAsymmErrors(Int_t n, const Double_t *x, const Double_t *y, const Double_t *exl=0, const Double_t *exh=0, const Double_t *eyl=0, const Double_t *eyh=0);
        virtual ~TGraphAsymmErrors();
        virtual void    Apply(TF1 *f);
        virtual void    ComputeRange(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax);
        Double_t        GetErrorX(Int_t bin) const;
        Double_t        GetErrorY(Int_t bin) const;
        Double_t       *GetEXlow()  const {return fEXlow;}
        Double_t       *GetEXhigh() const {return fEXhigh;}
        Double_t       *GetEYlow()  const {return fEYlow;}
        Double_t       *GetEYhigh() const {return fEYhigh;}
        virtual Int_t   InsertPoint(); // *MENU*
        virtual void    Paint(Option_t *chopt="");
        virtual void    Print(Option_t *chopt="") const;
        virtual Int_t   RemovePoint(); // *MENU*
        virtual Int_t   RemovePoint(Int_t ipoint);
        virtual void    SavePrimitive(ofstream &out, Option_t *option);
        virtual void    SetPoint(Int_t i, Double_t x, Double_t y);
        virtual void    SetPointError(Double_t exl, Double_t exh, Double_t eyl, Double_t eyh); // *MENU*
        virtual void    SetPointError(Int_t i, Double_t exl, Double_t exh, Double_t eyl, Double_t eyh);

        ClassDef(TGraphAsymmErrors,3)  //A graph with asymmetric error bars
};

#endif
