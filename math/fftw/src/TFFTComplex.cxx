// @(#)root/fft:$Id$
// Author: Anna Kreshuk   07/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
/// \class TFFTComplex
///
/// One of the interface classes to the FFTW package, can be used directly
/// or via the TVirtualFFT class. Only the basic interface of FFTW is implemented.
/// Computes complex input/output discrete Fourier transforms (DFT)
/// in one or more dimensions. For the detailed information on the computed
/// transforms please refer to the FFTW manual, chapter "What FFTW really computes".
///
/// How to use it:
///
/// 1. Create an instance of TFFTComplex - this will allocate input and output
///    arrays (unless an in-place transform is specified)
/// 2. Run the Init() function with the desired flags and settings
/// 3. Set the data (via SetPoints(), SetPoint() or SetPointComplex() functions)
/// 4. Run the Transform() function
/// 5. Get the output (via GetPoints(), GetPoint() or GetPointComplex() functions)
/// 6. Repeat steps 3)-5) as needed
///
/// For a transform of the same size, but with different flags or sign, rerun the Init()
/// function and continue with steps 3)-5)
///
/// NOTE:
///       1. running Init() function will overwrite the input array! Don't set any data
///          before running the Init() function
///       2. FFTW computes unnormalized transform, so doing a transform followed by
///          its inverse will lead to the original array scaled by the transform size
///
////////////////////////////////////////////////////////////////////////////////

#include "TFFTComplex.h"
#include "fftw3.h"
#include "TComplex.h"


ClassImp(TFFTComplex);

////////////////////////////////////////////////////////////////////////////////
///default

TFFTComplex::TFFTComplex()
{
   fIn   = 0;
   fOut  = 0;
   fPlan = 0;
   fN    = 0;
   fNdim = 0;
   fTotalSize = 0;
   fSign = 1;
}

////////////////////////////////////////////////////////////////////////////////
///For 1d transforms
///Allocates memory for the input array, and, if inPlace = kFALSE, for the output array

TFFTComplex::TFFTComplex(Int_t n, Bool_t inPlace)
{
   fIn = fftw_malloc(sizeof(fftw_complex) *n);
   if (!inPlace)
      fOut = fftw_malloc(sizeof(fftw_complex) * n);
   else
      fOut = 0;
   fN    = new Int_t[1];
   fN[0] = n;
   fTotalSize = n;
   fNdim = 1;
   fSign = 1;
   fPlan = 0;
}

////////////////////////////////////////////////////////////////////////////////
///For multidim. transforms
///Allocates memory for the input array, and, if inPlace = kFALSE, for the output array

TFFTComplex::TFFTComplex(Int_t ndim, Int_t *n, Bool_t inPlace)
{
   fNdim = ndim;
   fTotalSize = 1;
   fN = new Int_t[fNdim];
   for (Int_t i=0; i<fNdim; i++){
      fN[i] = n[i];
      fTotalSize*=n[i];
   }
   fIn = fftw_malloc(sizeof(fftw_complex)*fTotalSize);
   if (!inPlace)
      fOut = fftw_malloc(sizeof(fftw_complex) * fTotalSize);
   else
      fOut = 0;
   fSign = 1;
   fPlan = 0;
}

////////////////////////////////////////////////////////////////////////////////
///Destroys the data arrays and the plan. However, some plan information stays around
///until the root session is over, and is reused if other plans of the same size are
///created

TFFTComplex::~TFFTComplex()
{
   fftw_destroy_plan((fftw_plan)fPlan);
   fPlan = 0;
   fftw_free((fftw_complex*)fIn);
   if (fOut)
      fftw_free((fftw_complex*)fOut);
   if (fN)
      delete [] fN;
}

////////////////////////////////////////////////////////////////////////////////
///Creates the fftw-plan
///
///NOTE:  input and output arrays are overwritten during initialisation,
///       so don't set any points, before running this function!!!!!
///
///2nd parameter: +1
///
///Argument kind is dummy and doesn't need to be specified
///
///Possible flag_options:
/// - "ES" (from "estimate") - no time in preparing the transform, but probably sub-optimal
///   performance
/// - "M" (from "measure") - some time spend in finding the optimal way to do the transform
/// - "P" (from "patient") - more time spend in finding the optimal way to do the transform
/// - "EX" (from "exhaustive") - the most optimal way is found
///This option should be chosen depending on how many transforms of the same size and
///type are going to be done. Planning is only done once, for the first transform of this
///size and type.

void TFFTComplex::Init( Option_t *flags, Int_t sign,const Int_t* /*kind*/)
{
   fSign = sign;
   fFlags = flags;

   if (fPlan)
      fftw_destroy_plan((fftw_plan)fPlan);
   fPlan = 0;

   if (fOut)
      fPlan = (void*)fftw_plan_dft(fNdim, fN, (fftw_complex*)fIn, (fftw_complex*)fOut, sign,MapFlag(flags));
   else
      fPlan = (void*)fftw_plan_dft(fNdim, fN, (fftw_complex*)fIn, (fftw_complex*)fIn, sign, MapFlag(flags));
}

////////////////////////////////////////////////////////////////////////////////
///Computes the transform, specified in Init() function

void TFFTComplex::Transform()
{
   if (fPlan)
      fftw_execute((fftw_plan)fPlan);
   else {
      Error("Transform", "transform not initialised");
      return;
   }
}

////////////////////////////////////////////////////////////////////////////////
///Copies the output(or input) into the argument array

void TFFTComplex::GetPoints(Double_t *data, Bool_t fromInput) const
{
   if (!fromInput){
      for (Int_t i=0; i<2*fTotalSize; i+=2){
         data[i] = ((fftw_complex*)fOut)[i/2][0];
         data[i+1] = ((fftw_complex*)fOut)[i/2][1];
      }
   } else {
      for (Int_t i=0; i<2*fTotalSize; i+=2){
         data[i] = ((fftw_complex*)fIn)[i/2][0];
         data[i+1] = ((fftw_complex*)fIn)[i/2][1];
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
///returns real and imaginary parts of the point #ipoint

void TFFTComplex::GetPointComplex(Int_t ipoint, Double_t &re, Double_t &im, Bool_t fromInput) const
{
   if (fOut && !fromInput){
      re = ((fftw_complex*)fOut)[ipoint][0];
      im = ((fftw_complex*)fOut)[ipoint][1];
   } else {
      re = ((fftw_complex*)fIn)[ipoint][0];
      im = ((fftw_complex*)fIn)[ipoint][1];
   }
}

////////////////////////////////////////////////////////////////////////////////
///For multidimensional transforms. Returns real and imaginary parts of the point #ipoint

void TFFTComplex::GetPointComplex(const Int_t *ipoint, Double_t &re, Double_t &im, Bool_t fromInput) const
{
   Int_t ireal = ipoint[0];
   for (Int_t i=0; i<fNdim-1; i++)
      ireal=fN[i+1]*ireal + ipoint[i+1];

   if (fOut && !fromInput){
      re = ((fftw_complex*)fOut)[ireal][0];
      im = ((fftw_complex*)fOut)[ireal][1];
   } else {
      re = ((fftw_complex*)fIn)[ireal][0];
      im = ((fftw_complex*)fIn)[ireal][1];
   }
}

////////////////////////////////////////////////////////////////////////////////
///Copies real and imaginary parts of the output (input) into the argument arrays

void TFFTComplex::GetPointsComplex(Double_t *re, Double_t *im, Bool_t fromInput) const
{
   if (fOut && !fromInput){
      for (Int_t i=0; i<fTotalSize; i++){
         re[i] = ((fftw_complex*)fOut)[i][0];
         im[i] = ((fftw_complex*)fOut)[i][1];
      }
   } else {
      for (Int_t i=0; i<fTotalSize; i++){
         re[i] = ((fftw_complex*)fIn)[i][0];
         im[i] = ((fftw_complex*)fIn)[i][1];
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
///Copies the output(input) into the argument array

void TFFTComplex::GetPointsComplex(Double_t *data, Bool_t fromInput) const
{
   if (fOut && !fromInput){
      for (Int_t i=0; i<fTotalSize; i+=2){
         data[i] = ((fftw_complex*)fOut)[i/2][0];
         data[i+1] = ((fftw_complex*)fOut)[i/2][1];
      }
   } else {
      for (Int_t i=0; i<fTotalSize; i+=2){
         data[i] = ((fftw_complex*)fIn)[i/2][0];
         data[i+1] = ((fftw_complex*)fIn)[i/2][1];
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
///sets real and imaginary parts of point # ipoint

void TFFTComplex::SetPoint(Int_t ipoint, Double_t re, Double_t im)
{
   ((fftw_complex*)fIn)[ipoint][0]=re;
   ((fftw_complex*)fIn)[ipoint][1]=im;
}

////////////////////////////////////////////////////////////////////////////////
///For multidim. transforms. Sets real and imaginary parts of point # ipoint

void TFFTComplex::SetPoint(const Int_t *ipoint, Double_t re, Double_t im)
{
   Int_t ireal = ipoint[0];
   for (Int_t i=0; i<fNdim-1; i++)
      ireal=fN[i+1]*ireal + ipoint[i+1];

   ((fftw_complex*)fIn)[ireal][0]=re;
   ((fftw_complex*)fIn)[ireal][1]=im;
}

////////////////////////////////////////////////////////////////////////////////

void TFFTComplex::SetPointComplex(Int_t ipoint, TComplex &c)
{
   ((fftw_complex*)fIn)[ipoint][0] = c.Re();
   ((fftw_complex*)fIn)[ipoint][1] = c.Im();
}

////////////////////////////////////////////////////////////////////////////////
///set all points. the values are copied. points should be ordered as follows:
///[re_0, im_0, re_1, im_1, ..., re_n, im_n)

void TFFTComplex::SetPoints(const Double_t *data)
{
   for (Int_t i=0; i<2*fTotalSize-1; i+=2){
      ((fftw_complex*)fIn)[i/2][0]=data[i];
      ((fftw_complex*)fIn)[i/2][1]=data[i+1];
   }
}

////////////////////////////////////////////////////////////////////////////////
///set all points. the values are copied

void TFFTComplex::SetPointsComplex(const Double_t *re_data, const Double_t *im_data)
{
   if (!fIn){
      Error("SetPointsComplex", "Size is not set yet");
      return;
   }
   for (Int_t i=0; i<fTotalSize; i++){
      ((fftw_complex*)fIn)[i][0]=re_data[i];
      ((fftw_complex*)fIn)[i][1]=im_data[i];
   }
}

////////////////////////////////////////////////////////////////////////////////
///allowed options:
/// - "ES" - FFTW_ESTIMATE
/// - "M" - FFTW_MEASURE
/// - "P" - FFTW_PATIENT
/// - "EX" - FFTW_EXHAUSTIVE

UInt_t TFFTComplex::MapFlag(Option_t *flag)
{
   TString opt = flag;
   opt.ToUpper();
   if (opt.Contains("ES"))
      return FFTW_ESTIMATE;
   if (opt.Contains("M"))
      return FFTW_MEASURE;
   if (opt.Contains("P"))
      return FFTW_PATIENT;
   if (opt.Contains("EX"))
      return FFTW_EXHAUSTIVE;
   return FFTW_ESTIMATE;
}
