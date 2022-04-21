// @(#)root/hist:$Id: TGraph2DErrors.h,v 1.00
// Author: Olivier Couet 26/11/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraph2DErrors
#define ROOT_TGraph2DErrors


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraph2DErrors                                                       //
//                                                                      //
// a 2D Graph with error bars                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGraph2D.h"

class TGraph2DErrors : public TGraph2D {

protected:
   Double_t *fEX{nullptr}; ///<[fNpoints] array of X errors
   Double_t *fEY{nullptr}; ///<[fNpoints] array of Y errors
   Double_t *fEZ{nullptr}; ///<[fNpoints] array of Z errors

public:
   TGraph2DErrors();
   TGraph2DErrors(Int_t n);
   TGraph2DErrors(Int_t n, Double_t *x, Double_t *y, Double_t *z,
                  Double_t *ex = nullptr, Double_t *ey = nullptr, Double_t *ez = nullptr, Option_t *option = "");
   TGraph2DErrors(const TGraph2DErrors&);
   TGraph2DErrors& operator=(const TGraph2DErrors&);
   ~TGraph2DErrors() override;
   Double_t        GetErrorX(Int_t bin) const override;
   Double_t        GetErrorY(Int_t bin) const override;
   Double_t        GetErrorZ(Int_t bin) const override;
   Double_t       *GetEX() const override {return fEX;}
   Double_t       *GetEY() const override {return fEY;}
   Double_t       *GetEZ() const override {return fEZ;}
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
   virtual void    SetPointError(Int_t i, Double_t ex, Double_t ey, Double_t ez);

   ClassDefOverride(TGraph2DErrors,1)  //A 2D graph with error bars
};

#endif


