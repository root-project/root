// @(#)root/fft:$Id$
// Author: Anna Kreshuk   07/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFFTRealComplex
#define ROOT_TFFTRealComplex

//////////////////////////////////////////////////////////////////////////
//                                                                      
// TFFTRealComplex                                                       
//                                                                      
// One of the interface classes to the FFTW package, can be used directly
// or via the TVirtualFFT class. Only the basic interface of FFTW is implemented.
//
// Computes a real input/complex output discrete Fourier transform in 1 or more
// dimensions. However, only out-of-place transforms are now supported for transforms
// in more than 1 dimension. For detailed information about the computed transforms,
// please refer to the FFTW manual
//
// How to use it:
// 1) Create an instance of TFFTRealComplex - this will allocate input and output
//    arrays (unless an in-place transform is specified)
// 2) Run the Init() function with the desired flags and settings (see function
//    comments for possible kind parameters)
// 3) Set the data (via SetPoints()or SetPoint() functions)
// 4) Run the Transform() function
// 5) Get the output (via GetPoints() or GetPoint() functions)
// 6) Repeat steps 3)-5) as needed
// For a transform of the same size, but with different flags, 
// rerun the Init() function and continue with steps 3)-5)
//
// NOTE: 1) running Init() function will overwrite the input array! Don't set any data
//          before running the Init() function
//       2) FFTW computes unnormalized transform, so doing a transform followed by 
//          its inverse will lead to the original array scaled by the transform size
// 
//
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualFFT
#include "TVirtualFFT.h"
#endif

class TComplex;

class TFFTRealComplex: public TVirtualFFT {
 protected:
   void     *fIn;        //input array
   void     *fOut;       //output array
   void     *fPlan;      //fftw plan (the plan how to compute the transform)
   Int_t     fNdim;      //number of dimensions
   Int_t     fTotalSize; //total size of the transform
   Int_t    *fN;         //transform sizes in each dimension
   Option_t *fFlags;     //transform flags

   UInt_t MapFlag(Option_t *flag);

 public:
   TFFTRealComplex();
   TFFTRealComplex(Int_t n, Bool_t inPlace);
   TFFTRealComplex(Int_t ndim, Int_t *n, Bool_t inPlace);
   virtual ~TFFTRealComplex();

   virtual void       Init( Option_t *flags, Int_t /*sign*/,const Int_t* /*kind*/);

   virtual Int_t      GetSize() const {return fTotalSize;}
   virtual Int_t     *GetN()    const {return fN;}
   virtual Int_t      GetNdim() const {return fNdim;}
   virtual Option_t  *GetType() const {return "R2C";}
   virtual Int_t      GetSign() const {return 1;}
   virtual Option_t  *GetTransformFlag() const {return fFlags;}
   virtual Bool_t     IsInplace() const {if (fOut) return kTRUE; else return kFALSE;};

   virtual void       GetPoints(Double_t *data, Bool_t fromInput = kFALSE) const;
   virtual Double_t   GetPointReal(Int_t ipoint, Bool_t fromInput = kFALSE) const;
   virtual Double_t   GetPointReal(const Int_t *ipoint, Bool_t fromInput = kFALSE) const;
   virtual void       GetPointComplex(Int_t ipoint, Double_t &re, Double_t &im, Bool_t fromInput=kFALSE) const;
   virtual void       GetPointComplex(const Int_t *ipoint, Double_t &re, Double_t &im, Bool_t fromInput=kFALSE) const;
   virtual Double_t  *GetPointsReal(Bool_t fromInput=kFALSE) const;
   virtual void       GetPointsComplex(Double_t *re, Double_t *im, Bool_t fromInput = kFALSE) const ;
   virtual  void      GetPointsComplex(Double_t *data, Bool_t fromInput = kFALSE) const ;

   virtual void       SetPoint(Int_t ipoint, Double_t re, Double_t im = 0);
   virtual void       SetPoint(const Int_t *ipoint, Double_t re, Double_t im = 0);
   virtual void       SetPoints(const Double_t *data);
   virtual void       SetPointComplex(Int_t ipoint, TComplex &c);
   virtual void       SetPointsComplex(const Double_t *re, const Double_t *im);
   virtual void       Transform();

   ClassDef(TFFTRealComplex,0);
};

#endif
