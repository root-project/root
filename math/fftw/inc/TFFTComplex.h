// @(#)root/fft:$Id$
// Author: Anna Kreshuk   07/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFFTComplex
#define ROOT_TFFTComplex

//////////////////////////////////////////////////////////////////////////
//
// TFFTComplex
// One of the interface classes to the FFTW package, can be used directly
// or via the TVirtualFFT class. Only the basic interface of FFTW is implemented.
// Computes complex input/output discrete Fourier transforms (DFT)
// in one or more dimensions. For the detailed information on the computed
// transforms please refer to the FFTW manual, chapter "What FFTW really computes".
//
// How to use it:
// 1) Create an instance of TFFTComplex - this will allocate input and output
//    arrays (unless an in-place transform is specified)
// 2) Run the Init() function with the desired flags and settings
// 3) Set the data (via SetPoints(), SetPoint() or SetPointComplex() functions)
// 4) Run the Transform() function
// 5) Get the output (via GetPoints(), GetPoint() or GetPointComplex() functions)
// 6) Repeat steps 3)-5) as needed
//
// For a transform of the same size, but with different flags or sign, rerun the Init()
// function and continue with steps 3)-5)
// NOTE: 1) running Init() function will overwrite the input array! Don't set any data
//          before running the Init() function
//       2) FFTW computes unnormalized transform, so doing a transform followed by
//          its inverse will lead to the original array scaled by the transform size
//
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualFFT
#include "TVirtualFFT.h"
#endif

class TComplex;

class TFFTComplex : public TVirtualFFT{
protected:
   void     *fIn;        //input array
   void     *fOut;       //output array
   void     *fPlan;      //fftw plan (the plan how to compute the transform)
   Int_t     fNdim;      //number of dimensions
   Int_t     fTotalSize; //total size of the transform
   Int_t    *fN;         //transform sizes in each dimension
   Int_t     fSign;      //sign of the exponent of the transform (-1 is FFTW_FORWARD and +1 FFTW_BACKWARD)
   Option_t *fFlags;     //transform flags

   UInt_t MapFlag(Option_t *flag);

public:
   TFFTComplex();
   TFFTComplex(Int_t n, Bool_t inPlace);
   TFFTComplex(Int_t ndim, Int_t *n, Bool_t inPlace = kFALSE);
   virtual ~TFFTComplex();

   virtual void       Init(Option_t *flags, Int_t sign, const Int_t* /*kind*/);

   virtual Int_t     *GetN()    const {return fN;}
   virtual Int_t      GetNdim() const {return fNdim;}
   virtual Int_t      GetSize() const {return fTotalSize;}
   virtual Option_t  *GetType() const {if (fSign==-1) return "C2CBackward"; else return "C2CForward";}
   virtual Int_t      GetSign() const {return fSign;}
   virtual Option_t  *GetTransformFlag() const {return fFlags;}
   virtual Bool_t     IsInplace() const {if (fOut) return kTRUE; else return kFALSE;};

   virtual void       GetPoints(Double_t *data, Bool_t fromInput = kFALSE) const;
   virtual Double_t   GetPointReal(Int_t /*ipoint*/, Bool_t /*fromInput = kFALSE*/) const {return 0;};
   virtual Double_t   GetPointReal(const Int_t* /*ipoint*/, Bool_t /*fromInput=kFALSE*/) const{return 0;}
   virtual void       GetPointComplex(Int_t ipoint, Double_t &re, Double_t &im, Bool_t fromInput=kFALSE) const;
   virtual void       GetPointComplex(const Int_t *ipoint, Double_t &re, Double_t &im, Bool_t fromInput=kFALSE) const;
   virtual Double_t*  GetPointsReal(Bool_t /*fromInput=kFALSE*/) const {return 0;};
   virtual void       GetPointsComplex(Double_t *re, Double_t *im, Bool_t fromInput = kFALSE) const ;
   virtual void       GetPointsComplex(Double_t *data, Bool_t fromInput = kFALSE) const ;

   virtual void       SetPoint(Int_t ipoint, Double_t re, Double_t im = 0);
   virtual void       SetPoint(const Int_t *ipoint, Double_t re, Double_t im = 0);
   virtual void       SetPoints(const Double_t *data);
   virtual void       SetPointComplex(Int_t ipoint, TComplex &c);
   virtual void       SetPointsComplex(const Double_t *re, const Double_t *im);
   virtual void       Transform();

   ClassDef(TFFTComplex,0);
};

#endif
