// @(#)root/graf:$Name$:$Id$
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
    Float_t    *fEXlow;        //[fNpoints] array of X low errors
    Float_t    *fEXhigh;       //[fNpoints] array of X high errors
    Float_t    *fEYlow;        //[fNpoints] array of Y low errors
    Float_t    *fEYhigh;       //[fNpoints] array of Y high errors

public:
        TGraphAsymmErrors();
        TGraphAsymmErrors(Int_t n, Float_t *x=0, Float_t *y=0, Float_t *exl=0, Float_t *exh=0, Float_t *eyl=0, Float_t *eyh=0);
        virtual ~TGraphAsymmErrors();
        virtual void    ComputeRange(Float_t &xmin, Float_t &ymin, Float_t &xmax, Float_t &ymax);
        Float_t         GetErrorX(Int_t bin);
        Float_t         GetErrorY(Int_t bin);
        Float_t         *GetEXlow()  {return fEXlow;}
        Float_t         *GetEXhigh() {return fEXhigh;}
        Float_t         *GetEYlow()  {return fEYlow;}
        Float_t         *GetEYhigh() {return fEYhigh;}
        virtual void    Paint(Option_t *chopt="");
        virtual void    Print(Option_t *chopt="");
        virtual void    SavePrimitive(ofstream &out, Option_t *option);
        virtual void    SetPoint(Int_t i, Float_t x, Float_t y);
        virtual void    SetPointError(Int_t i, Float_t exl, Float_t exh, Float_t eyl, Float_t eyh);

        ClassDef(TGraphAsymmErrors,1)  //a Graph with asymmetric error bars
};

#endif
