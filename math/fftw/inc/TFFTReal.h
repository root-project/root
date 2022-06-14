// @(#)root/fft:$Id$
// Author: Anna Kreshuk   07/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFFTReal
#define ROOT_TFFTReal

#include "TVirtualFFT.h"
#include "TString.h"

class TComplex;

class TFFTReal: public TVirtualFFT{
 protected:
   void     *fIn;         //input array
   void     *fOut;        //output array
   void     *fPlan;       //fftw plan (the plan how to compute the transform)
   Int_t     fNdim;       //number of dimensions
   Int_t     fTotalSize;  //total size of the transform
   Int_t    *fN;          //transform sizes in each dimension
   void     *fKind;       //transform kinds in each dimension
   TString   fFlags;      //transform flags

   Int_t  MapOptions(const Int_t *kind);
   UInt_t MapFlag(Option_t *flag);

 public:
   TFFTReal();
   TFFTReal(Int_t n, Bool_t inPlace=kFALSE);
   TFFTReal(Int_t ndim, Int_t *n, Bool_t inPlace=kFALSE);
   ~TFFTReal() override;

   void      Init( Option_t *flags,Int_t sign, const Int_t *kind) override;

   virtual Int_t     GetSize() const {return fTotalSize;}
   Int_t    *GetN()    const override {return fN;}
   Int_t     GetNdim() const override {return fNdim;}
   Option_t *GetType() const override;
   Int_t     GetSign() const override {return 0;}
   Option_t *GetTransformFlag() const override {return fFlags;}
   Bool_t    IsInplace() const override {if (fOut) return kTRUE; else return kFALSE;}

   void      GetPoints(Double_t *data, Bool_t fromInput = kFALSE) const override;
   Double_t  GetPointReal(Int_t ipoint, Bool_t fromInput = kFALSE) const override;
   Double_t  GetPointReal(const Int_t *ipoint, Bool_t fromInput = kFALSE) const override;
   void      GetPointComplex(const Int_t *ipoint, Double_t &re, Double_t &im, Bool_t fromInput=kFALSE) const override;

   void      GetPointComplex(Int_t ipoint, Double_t &re, Double_t &im, Bool_t fromInput=kFALSE) const override;

   Double_t *GetPointsReal(Bool_t fromInput=kFALSE) const override;
   void      GetPointsComplex(Double_t* /*re*/, Double_t* /*im*/, Bool_t /*fromInput = kFALSE*/) const override{};
    void     GetPointsComplex(Double_t* /*data*/, Bool_t /*fromInput = kFALSE*/) const override {};

   void      SetPoint(Int_t ipoint, Double_t re, Double_t im = 0) override;
   void      SetPoint(const Int_t *ipoint, Double_t re, Double_t /*im=0*/) override;
   void      SetPoints(const Double_t *data) override;
   void      SetPointComplex(Int_t /*ipoint*/, TComplex &/*c*/) override{};
   void      SetPointsComplex(const Double_t* /*re*/, const Double_t* /*im*/) override{};
   void      Transform() override;


   ClassDefOverride(TFFTReal,0);
};

#endif
