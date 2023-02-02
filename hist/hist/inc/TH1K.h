// @(#)root/hist:$Id$
// Author: Victor Perevoztchikov <perev@bnl.gov>  21/02/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TH1K
#define ROOT_TH1K


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TH1K                                                                 //
//                                                                      //
// 1-Dim histogram nearest K Neighbour class.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TH1.h"

class TH1K : public TH1, public TArrayF {

private:
   void Sort();
protected:
   Int_t fReady;  //!
   Int_t fNIn;
   Int_t fKOrd;   //!
   Int_t fKCur;   //!
public:
   TH1K();
   TH1K(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup,Int_t k=0);
   ~TH1K() override;

   void      Copy(TObject &obj) const override;
   Int_t     Fill(Double_t x) override;
   Int_t     Fill(Double_t x,Double_t w) override{return TH1::Fill(x,w);}
   Int_t     Fill(const char *name,Double_t w) override{return TH1::Fill(name,w);}
   Double_t  GetBinContent(Int_t bin) const override;
   Double_t  GetBinContent(Int_t bin,Int_t) const override {return GetBinContent(bin);}
   Double_t  GetBinContent(Int_t bin,Int_t,Int_t) const override {return GetBinContent(bin);}

   Double_t  GetBinError(Int_t bin) const override;
   Double_t  GetBinError(Int_t bin,Int_t) const override {return GetBinError(bin);}
   Double_t  GetBinError(Int_t bin,Int_t,Int_t) const override {return GetBinError(bin);}


   void      Reset(Option_t *option="") override;
   void      SavePrimitive(std::ostream &out, Option_t *option = "") override;

   void    SetKOrd(Int_t k){fKOrd=k;}

   ClassDefOverride(TH1K,2)  //1-Dim Nearest Kth neighbour method
};

#endif
