// @(#)root/base:$Id$
// Author: Anna Kreshuk  10/04/2006

#ifndef ROOT_TVirtualFFT
#define ROOT_TVirtualFFT

//////////////////////////////////////////////////////////////////////////
//
// TVirtualFFT
//
// TVirtualFFT is an interface class for Fast Fourier Transforms.
//
//
//
// The default FFT library is FFTW. To use it, FFTW3 library should already
// be installed, and ROOT should be have fftw3 module enabled, with the directories
// of fftw3 include file and library specified (see installation instructions).
// Function SetDefaultFFT() allows to change the default library.
//
// Available transform types:
// FFT:
// - "C2CFORWARD" - a complex input/output discrete Fourier transform (DFT)
//                  in one or more dimensions, -1 in the exponent
// - "C2CBACKWARD"- a complex input/output discrete Fourier transform (DFT)
//                  in one or more dimensions, +1 in the exponent
// - "R2C"        - a real-input/complex-output discrete Fourier transform (DFT)
//                  in one or more dimensions,
// - "C2R"        - inverse transforms to "R2C", taking complex input
//                  (storing the non-redundant half of a logically Hermitian array)
//                  to real output
// - "R2HC"       - a real-input DFT with output in "halfcomplex" format,
//                  i.e. real and imaginary parts for a transform of size n stored as
//                  r0, r1, r2, ..., rn/2, i(n+1)/2-1, ..., i2, i1
// - "HC2R"       - computes the reverse of FFTW_R2HC, above
// - "DHT"        - computes a discrete Hartley transform
//
// Sine/cosine transforms:
// Different types of transforms are specified by parameter kind of the SineCosine() static
// function. 4 different kinds of sine and cosine transforms are available
//  DCT-I  (REDFT00 in FFTW3 notation)- kind=0
//  DCT-II (REDFT10 in FFTW3 notation)- kind=1
//  DCT-III(REDFT01 in FFTW3 notation)- kind=2
//  DCT-IV (REDFT11 in FFTW3 notation)- kind=3
//  DST-I  (RODFT00 in FFTW3 notation)- kind=4
//  DST-II (RODFT10 in FFTW3 notation)- kind=5
//  DST-III(RODFT01 in FFTW3 notation)- kind=6
//  DST-IV (RODFT11 in FFTW3 notation)- kind=7
// Formulas and detailed descriptions can be found in the chapter
// "What FFTW really computes" of the FFTW manual
//
// NOTE: FFTW computes unnormalized transforms, so doing a transform, followed by its
//       inverse will give the original array, multiplied by normalization constant
//       (transform size(N) for FFT, 2*(N-1) for DCT-I, 2*(N+1) for DST-I, 2*N for
//       other sine/cosine transforms)
//
// How to use it:
// Call to the static function FFT returns a pointer to a fast fourier transform
// with requested parameters. Call to the static function SineCosine returns a
// pointer to a sine or cosine transform with requested parameters. Example:
// {
//    Int_t N = 10; Double_t *in = new Double_t[N];
//    TVirtualFFT *fftr2c = TVirtualFFT::FFT(1, &N, "R2C");
//    fftr2c->SetPoints(in);
//    fftr2c->Transform();
//    Double_t re, im;
//    for (Int_t i=0; i<N; i++)
//       fftr2c->GetPointComplex(i, re, im);
//    ...
//    fftr2c->SetPoints(in2);
//    ...
//    fftr2c->SetPoints(in3);
//    ...
// }
// Different options are explained in the function comments
//
//
//
//
//
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"

#include "TString.h"

class TComplex;

class TVirtualFFT: public TObject {

 protected:
   static TVirtualFFT *fgFFT;      //current transformer
   static TString      fgDefault;  //default transformer

 public:

   TVirtualFFT(){};
   virtual ~TVirtualFFT();

   virtual Int_t     *GetN()    const = 0;

   virtual Int_t      GetNdim() const = 0;
   virtual Option_t  *GetType() const = 0;
   virtual Int_t      GetSign() const = 0;
   virtual Option_t  *GetTransformFlag() const = 0;
   virtual void       Init(Option_t *flag,Int_t sign, const Int_t *kind) = 0;
   virtual Bool_t     IsInplace() const = 0;

   virtual void       GetPoints(Double_t *data, Bool_t fromInput = kFALSE) const = 0;
   virtual Double_t   GetPointReal(Int_t ipoint, Bool_t fromInput = kFALSE) const = 0;
   virtual Double_t   GetPointReal(const Int_t *ipoint, Bool_t fromInput = kFALSE) const = 0;
   virtual void       GetPointComplex(Int_t ipoint, Double_t &re, Double_t &im, Bool_t fromInput=kFALSE) const = 0;
   virtual void       GetPointComplex(const Int_t *ipoint, Double_t &re, Double_t &im, Bool_t fromInput=kFALSE) const = 0;
   virtual Double_t*  GetPointsReal(Bool_t fromInput=kFALSE) const = 0;
   virtual void       GetPointsComplex(Double_t *re, Double_t *im, Bool_t fromInput = kFALSE) const = 0;
   virtual void       GetPointsComplex(Double_t *data, Bool_t fromInput = kFALSE) const = 0;

   virtual void       SetPoint(Int_t ipoint, Double_t re, Double_t im = 0) = 0;
   virtual void       SetPoint(const Int_t *ipoint, Double_t re, Double_t im = 0) = 0;
   virtual void       SetPoints(const Double_t *data) = 0;
   virtual void       SetPointComplex(Int_t ipoint, TComplex &c) = 0;
   virtual void       SetPointsComplex(const Double_t *re, const Double_t *im) =0;
   virtual void       Transform() = 0;

   static TVirtualFFT* FFT(Int_t ndim, Int_t *n, Option_t *option);
   static TVirtualFFT* SineCosine(Int_t ndim, Int_t *n, Int_t *r2rkind, Option_t *option);
   static TVirtualFFT* GetCurrentTransform();

   static void         SetTransform(TVirtualFFT *fft);
   static const char*  GetDefaultFFT();
   static void         SetDefaultFFT(const char *name ="");

   ClassDefOverride(TVirtualFFT, 0); //abstract interface for FFT calculations
};

#endif
