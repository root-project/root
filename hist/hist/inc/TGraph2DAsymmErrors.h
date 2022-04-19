// @(#)root/hist:$Id: TGraph2DAsymmErrors.h,v 1.00
// Author: Olivier Couet 07/04/2022

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraph2DAsymmErrors
#define ROOT_TGraph2DAsymmErrors


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraph2DAsymmErrors                                                  //
//                                                                      //
// a 2D Graph with asymmetric error bars                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGraph2D.h"

class TGraph2DAsymmErrors : public TGraph2D {

private:


protected:
   Double_t    *fEXlow;        ///<[fNpoints] array of X low errors
   Double_t    *fEXhigh;       ///<[fNpoints] array of X high errors
   Double_t    *fEYlow;        ///<[fNpoints] array of Y low errors
   Double_t    *fEYhigh;       ///<[fNpoints] array of Y high errors
   Double_t    *fEZlow;        ///<[fNpoints] array of Z low errors
   Double_t    *fEZhigh;       ///<[fNpoints] array of Z high errors

public:
   TGraph2DAsymmErrors();
   TGraph2DAsymmErrors(Int_t n);
   TGraph2DAsymmErrors(Int_t n, Double_t *x, Double_t *y, Double_t *z, Double_t *exl=0, Double_t *exh=0, Double_t *eyl=0, Double_t *eyh=0, Double_t *ezl=0, Double_t *ezh=0, Option_t *option="");
   TGraph2DAsymmErrors(const TGraph2DAsymmErrors&);
   TGraph2DAsymmErrors& operator=(const TGraph2DAsymmErrors&);
   ~TGraph2DAsymmErrors() override;
   Double_t        GetErrorX(Int_t bin) const override;
   Double_t        GetErrorY(Int_t bin) const override;
   Double_t        GetErrorZ(Int_t bin) const override;
   Double_t        GetErrorXlow(Int_t i)  const;
   Double_t        GetErrorXhigh(Int_t i) const;
   Double_t        GetErrorYlow(Int_t i)  const;
   Double_t        GetErrorYhigh(Int_t i) const;
   Double_t        GetErrorZlow(Int_t i)  const;
   Double_t        GetErrorZhigh(Int_t i) const;
   Double_t       *GetEXlow()  const override {return fEXlow;}
   Double_t       *GetEXhigh() const override {return fEXhigh;}
   Double_t       *GetEYlow()  const override {return fEYlow;}
   Double_t       *GetEYhigh() const override {return fEYhigh;}
   Double_t       *GetEZlow()  const override {return fEZlow;}
   Double_t       *GetEZhigh() const override {return fEZhigh;}
   Double_t        GetXmaxE() const override;
   Double_t        GetXminE() const override;
   Double_t        GetYmaxE() const override;
   Double_t        GetYminE() const override;
   Double_t        GetZmaxE() const override;
   Double_t        GetZminE() const override;
   void    Print(Option_t *chopt="") const override;
   Int_t           RemovePoint(Int_t ipoint); // *MENU*
   void    Scale(Double_t c1=1., Option_t *option="z") override; // *MENU*
   void    Set(Int_t n) override;
   void    SetPoint(Int_t i, Double_t x, Double_t y, Double_t z) override;
   virtual void    SetPointError(Int_t i, Double_t exl, Double_t exh, Double_t eyl, Double_t eyh, Double_t ezl, Double_t ezh);

   ClassDefOverride(TGraph2DAsymmErrors,1)  //A 2D graph with error bars
};

#endif
