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
///
/// \class TFFTRealComplex
///
/// One of the interface classes to the FFTW package, can be used directly
/// or via the TVirtualFFT class. Only the basic interface of FFTW is implemented.
///
/// Computes a real input/complex output discrete Fourier transform in 1 or more
/// dimensions. However, only out-of-place transforms are now supported for transforms
/// in more than 1 dimension. For detailed information about the computed transforms,
/// please refer to the FFTW manual
///
/// How to use it:
/// 1. Create an instance of TFFTRealComplex - this will allocate input and output
///    arrays (unless an in-place transform is specified)
/// 2. Run the Init() function with the desired flags and settings (see function
///    comments for possible kind parameters)
/// 3. Set the data (via SetPoints()or SetPoint() functions)
/// 4. Run the Transform() function
/// 5. Get the output (via GetPoints() or GetPoint() functions)
/// 6. Repeat steps 3)-5) as needed
/// For a transform of the same size, but with different flags,
/// rerun the Init() function and continue with steps 3)-5)
///
/// NOTE:
///       1. running Init() function will overwrite the input array! Don't set any data
///          before running the Init() function
///       2. FFTW computes unnormalized transform, so doing a transform followed by
///          its inverse will lead to the original array scaled by the transform size
///
///
/////////////////////////////////////////////////////////////////////////////////

#include "TFFTRealComplex.h"
#include "fftw3.h"
#include "TComplex.h"


ClassImp(TFFTRealComplex);

////////////////////////////////////////////////////////////////////////////////
///default

TFFTRealComplex::TFFTRealComplex()
{
   fIn   = 0;
   fOut  = 0;
   fPlan = 0;
   fN    = 0;
   fNdim = 0;
   fTotalSize = 0;
}

////////////////////////////////////////////////////////////////////////////////
///For 1d transforms
///Allocates memory for the input array, and, if inPlace = kFALSE, for the output array

TFFTRealComplex::TFFTRealComplex(Int_t n, Bool_t inPlace)
{

   if (!inPlace){
      fIn = fftw_malloc(sizeof(Double_t)*n);
      fOut = fftw_malloc(sizeof(fftw_complex)*(n/2+1));
   } else {
      fIn = fftw_malloc(sizeof(Double_t)*(2*(n/2+1)));
      fOut = 0;
   }
   fN = new Int_t[1];
   fN[0] = n;
   fTotalSize = n;
   fNdim = 1;
   fPlan = 0;
}

////////////////////////////////////////////////////////////////////////////////
///For ndim-dimensional transforms
///Second argument contains sizes of the transform in each dimension

TFFTRealComplex::TFFTRealComplex(Int_t ndim, Int_t *n, Bool_t inPlace)
{
   if (ndim>1 && inPlace==kTRUE){
      Error("TFFTRealComplex", "multidimensional in-place r2c transforms are not implemented yet");
      return;
   }
   fNdim = ndim;
   fTotalSize = 1;
   fN = new Int_t[fNdim];
   for (Int_t i=0; i<fNdim; i++){
      fN[i] = n[i];
      fTotalSize*=n[i];
   }
   Int_t sizeout = Int_t(Double_t(fTotalSize)*(n[ndim-1]/2+1)/n[ndim-1]);
   if (!inPlace){
      fIn = fftw_malloc(sizeof(Double_t)*fTotalSize);
      fOut = fftw_malloc(sizeof(fftw_complex)*sizeout);
   } else {
      fIn = fftw_malloc(sizeof(Double_t)*(2*sizeout));
      fOut = 0;
   }
   fPlan = 0;
}

////////////////////////////////////////////////////////////////////////////////
///Destroys the data arrays and the plan. However, some plan information stays around
///until the root session is over, and is reused if other plans of the same size are
///created

TFFTRealComplex::~TFFTRealComplex()
{
   fftw_destroy_plan((fftw_plan)fPlan);
   fPlan = 0;
   fftw_free(fIn);
   fIn = 0;
   if (fOut)
      fftw_free((fftw_complex*)fOut);
   fOut = 0;
   if (fN)
      delete [] fN;
   fN = 0;
}

////////////////////////////////////////////////////////////////////////////////
///Creates the fftw-plan
///
///NOTE:  input and output arrays are overwritten during initialisation,
///       so don't set any points, before running this function!!!!!
///
///Arguments sign and kind are dummy and not need to be specified
///Possible flag_options:
///
/// - "ES" (from "estimate") - no time in preparing the transform, but probably sub-optimal
///   performance
/// - "M" (from "measure") - some time spend in finding the optimal way to do the transform
/// - "P" (from "patient") - more time spend in finding the optimal way to do the transform
/// - "EX" (from "exhaustive") - the most optimal way is found
///
///This option should be chosen depending on how many transforms of the same size and
///type are going to be done. Planning is only done once, for the first transform of this
///size and type.

void TFFTRealComplex::Init(Option_t *flags,Int_t /*sign*/, const Int_t* /*kind*/)
{
   fFlags = flags;

   if (fPlan)
      fftw_destroy_plan((fftw_plan)fPlan);
   fPlan = 0;

   if (fOut)
      fPlan = (void*)fftw_plan_dft_r2c(fNdim, fN, (Double_t*)fIn, (fftw_complex*)fOut,MapFlag(flags));
   else
      fPlan = (void*)fftw_plan_dft_r2c(fNdim, fN, (Double_t*)fIn, (fftw_complex*)fIn, MapFlag(flags));
}

////////////////////////////////////////////////////////////////////////////////
///Computes the transform, specified in Init() function

void TFFTRealComplex::Transform()
{

   if (fPlan){
      fftw_execute((fftw_plan)fPlan);
   }
   else {
      Error("Transform", "transform hasn't been initialised");
      return;
   }
}

////////////////////////////////////////////////////////////////////////////////
///Fills the array data with the computed transform.
///Only (roughly) a half of the transform is copied (exactly the output of FFTW),
///the rest being Hermitian symmetric with the first half

void TFFTRealComplex::GetPoints(Double_t *data, Bool_t fromInput) const
{
   if (fromInput){
      for (Int_t i=0; i<fTotalSize; i++)
         data[i] = ((Double_t*)fIn)[i];
   } else {
      Int_t realN = 2*Int_t(Double_t(fTotalSize)*(fN[fNdim-1]/2+1)/fN[fNdim-1]);
      if (fOut){
         for (Int_t i=0; i<realN; i+=2){
            data[i] = ((fftw_complex*)fOut)[i/2][0];
            data[i+1] = ((fftw_complex*)fOut)[i/2][1];
         }
      }
      else {
         for (Int_t i=0; i<realN; i++)
            data[i] = ((Double_t*)fIn)[i];
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
///Returns the real part of the point `#ipoint` from the output or the point `#ipoint`
///from the input

Double_t TFFTRealComplex::GetPointReal(Int_t ipoint, Bool_t fromInput) const
{
   if (fOut && !fromInput){
      Warning("GetPointReal", "Output is complex. Only real part returned");
      return ((fftw_complex*)fOut)[ipoint][0];
   }
   else
      return ((Double_t*)fIn)[ipoint];
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the real part of the point `#ipoint` from the output or the point
/// `#ipoint` from the input

Double_t TFFTRealComplex::GetPointReal(const Int_t *ipoint, Bool_t fromInput) const
{
   Int_t ireal = ipoint[0];
   for (Int_t i=0; i<fNdim-1; i++)
      ireal=fN[i+1]*ireal + ipoint[i+1];

    if (fOut && !fromInput){
      Warning("GetPointReal", "Output is complex. Only real part returned");
      return ((fftw_complex*)fOut)[ireal][0];
   }
   else
      return ((Double_t*)fIn)[ireal];
}


////////////////////////////////////////////////////////////////////////////////
///Returns the point `#ipoint`.
///For 1d, if ipoint > fN/2+1 (the point is in the Hermitian symmetric part), it is still
///returned. For >1d, only the first (roughly)half of points can be returned
///For 2d, see function GetPointComplex(Int_t *ipoint,...)

void TFFTRealComplex::GetPointComplex(Int_t ipoint, Double_t &re, Double_t &im, Bool_t fromInput) const
{
   if (fromInput){
      re = ((Double_t*)fIn)[ipoint];
   } else {
      if (fNdim==1){
         if (fOut){
            if (ipoint<fN[0]/2+1){
               re = ((fftw_complex*)fOut)[ipoint][0];
               im = ((fftw_complex*)fOut)[ipoint][1];
            } else {
               re = ((fftw_complex*)fOut)[fN[0]-ipoint][0];
               im = -((fftw_complex*)fOut)[fN[0]-ipoint][1];
            }
         } else {
            if (ipoint<fN[0]/2+1){
               re = ((Double_t*)fIn)[2*ipoint];
               im = ((Double_t*)fIn)[2*ipoint+1];
            } else {
               re = ((Double_t*)fIn)[2*(fN[0]-ipoint)];
               im = ((Double_t*)fIn)[2*(fN[0]-ipoint)+1];
            }
         }
      }
      else {
         Int_t realN = 2*Int_t(Double_t(fTotalSize)*(fN[fNdim-1]/2+1)/fN[fNdim-1]);
         if (ipoint>realN/2){
            Error("GetPointComplex", "Illegal index value");
            return;
         }
         if (fOut){
            re = ((fftw_complex*)fOut)[ipoint][0];
            im = ((fftw_complex*)fOut)[ipoint][1];
         } else {
            re = ((Double_t*)fIn)[2*ipoint];
            im = ((Double_t*)fIn)[2*ipoint+1];
         }
      }
   }
}
////////////////////////////////////////////////////////////////////////////////
///For multidimensional transforms. Returns the point `#ipoint`.
///In case of transforms of more than 2 dimensions,
///only points from the first (roughly)half are returned, the rest being Hermitian symmetric

void TFFTRealComplex::GetPointComplex(const Int_t *ipoint, Double_t &re, Double_t &im, Bool_t fromInput) const
{

   Int_t ireal = ipoint[0];
   for (Int_t i=0; i<fNdim-2; i++)
      ireal=fN[i+1]*ireal + ipoint[i+1];
   //special treatment of the last dimension
   ireal = (fN[fNdim-1]/2+1)*ireal + ipoint[fNdim-1];

   if (fromInput){
      re = ((Double_t*)fIn)[ireal];
      return;
   }
   if (fNdim==1){
      if (fOut){
         if (ipoint[0] <fN[0]/2+1){
            re = ((fftw_complex*)fOut)[ipoint[0]][0];
            im = ((fftw_complex*)fOut)[ipoint[0]][1];
         } else {
            re = ((fftw_complex*)fOut)[fN[0]-ipoint[0]][0];
            im = -((fftw_complex*)fOut)[fN[0]-ipoint[0]][1];
         }
      } else {
         if (ireal <fN[0]/2+1){
            re = ((Double_t*)fIn)[2*ipoint[0]];
            im = ((Double_t*)fIn)[2*ipoint[0]+1];
         } else {
            re = ((Double_t*)fIn)[2*(fN[0]-ipoint[0])];
            im = ((Double_t*)fIn)[2*(fN[0]-ipoint[0])+1];
         }
      }
   }
   else if (fNdim==2){
      if (fOut){
         if (ipoint[1]<fN[1]/2+1){
            re = ((fftw_complex*)fOut)[ipoint[1]+(fN[1]/2+1)*ipoint[0]][0];
            im = ((fftw_complex*)fOut)[ipoint[1]+(fN[1]/2+1)*ipoint[0]][1];
         } else {
            if (ipoint[0]==0){
               re = ((fftw_complex*)fOut)[fN[1]-ipoint[1]][0];
               im = -((fftw_complex*)fOut)[fN[1]-ipoint[1]][1];
            } else {
               re = ((fftw_complex*)fOut)[(fN[1]-ipoint[1])+(fN[1]/2+1)*(fN[0]-ipoint[0])][0];
               im = -((fftw_complex*)fOut)[(fN[1]-ipoint[1])+(fN[1]/2+1)*(fN[0]-ipoint[0])][1];
            }
         }
      } else {
         if (ipoint[1]<fN[1]/2+1){
            re = ((Double_t*)fIn)[2*(ipoint[1]+(fN[1]/2+1)*ipoint[0])];
            im = ((Double_t*)fIn)[2*(ipoint[1]+(fN[1]/2+1)*ipoint[0])+1];
         } else {
            if (ipoint[0]==0){
               re = ((Double_t*)fIn)[2*(fN[1]-ipoint[1])];
               im = -((Double_t*)fIn)[2*(fN[1]-ipoint[1])+1];
            } else {
               re = ((Double_t*)fIn)[2*((fN[1]-ipoint[1])+(fN[1]/2+1)*(fN[0]-ipoint[0]))];
               im = -((Double_t*)fIn)[2*((fN[1]-ipoint[1])+(fN[1]/2+1)*(fN[0]-ipoint[0]))+1];
            }
         }
      }
   }
   else {

      if (fOut){
         re = ((fftw_complex*)fOut)[ireal][0];
         im = ((fftw_complex*)fOut)[ireal][1];
      } else {
         re = ((Double_t*)fIn)[2*ireal];
         im = ((Double_t*)fIn)[2*ireal+1];
      }
   }
}


////////////////////////////////////////////////////////////////////////////////
///Returns the input array// One of the interface classes to the FFTW package, can be used directly
/// or via the TVirtualFFT class. Only the basic interface of FFTW is implemented.

Double_t* TFFTRealComplex::GetPointsReal(Bool_t fromInput) const
{
   if (!fromInput){
      Error("GetPointsReal", "Output array is complex");
      return 0;
   }
   return (Double_t*)fIn;
}

////////////////////////////////////////////////////////////////////////////////
///Fills the argument arrays with the real and imaginary parts of the computed transform.
///Only (roughly) a half of the transform is copied, the rest being Hermitian
///symmetric with the first half

void TFFTRealComplex::GetPointsComplex(Double_t *re, Double_t *im, Bool_t fromInput) const
{
   Int_t nreal;
   if (fOut && !fromInput){
      nreal = Int_t(Double_t(fTotalSize)*(fN[fNdim-1]/2+1)/fN[fNdim-1]);
      for (Int_t i=0; i<nreal; i++){
         re[i] = ((fftw_complex*)fOut)[i][0];
         im[i] = ((fftw_complex*)fOut)[i][1];
      }
   } else if (fromInput) {
      for (Int_t i=0; i<fTotalSize; i++){
         re[i] = ((Double_t *)fIn)[i];
         im[i] = 0;
      }
   }
   else {
      nreal = 2*Int_t(Double_t(fTotalSize)*(fN[fNdim-1]/2+1)/fN[fNdim-1]);
      for (Int_t i=0; i<nreal; i+=2){
         re[i/2] = ((Double_t*)fIn)[i];
         im[i/2] = ((Double_t*)fIn)[i+1];
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
///Fills the argument arrays with the real and imaginary parts of the computed transform.
///Only (roughly) a half of the transform is copied, the rest being Hermitian
///symmetric with the first half

void TFFTRealComplex::GetPointsComplex(Double_t *data, Bool_t fromInput) const
{
   Int_t nreal;

   if (fOut && !fromInput){
      nreal = Int_t(Double_t(fTotalSize)*(fN[fNdim-1]/2+1)/fN[fNdim-1]);

      for (Int_t i=0; i<nreal; i+=2){
         data[i] = ((fftw_complex*)fOut)[i/2][0];
         data[i+1] = ((fftw_complex*)fOut)[i/2][1];
      }
   } else if (fromInput){
      for (Int_t i=0; i<fTotalSize; i+=2){
         data[i] = ((Double_t*)fIn)[i/2];
         data[i+1] = 0;
      }
   }
   else {

      nreal = 2*Int_t(Double_t(fTotalSize)*(fN[fNdim-1]/2+1)/fN[fNdim-1]);
      for (Int_t i=0; i<nreal; i++)
         data[i] = ((Double_t*)fIn)[i];
   }
}

////////////////////////////////////////////////////////////////////////////////
///Set the point `#ipoint`

void TFFTRealComplex::SetPoint(Int_t ipoint, Double_t re, Double_t /*im*/)
{
   ((Double_t *)fIn)[ipoint] = re;
}

////////////////////////////////////////////////////////////////////////////////
///For multidimensional transforms. Set the point `#ipoint`

void TFFTRealComplex::SetPoint(const Int_t *ipoint, Double_t re, Double_t /*im*/)
{
   Int_t ireal = ipoint[0];
   for (Int_t i=0; i<fNdim-1; i++)
      ireal=fN[i+1]*ireal + ipoint[i+1];
   ((Double_t*)fIn)[ireal]=re;
}

////////////////////////////////////////////////////////////////////////////////
///Set all input points

void TFFTRealComplex::SetPoints(const Double_t *data)
{
   for (Int_t i=0; i<fTotalSize; i++){
      ((Double_t*)fIn)[i]=data[i];
   }
}

////////////////////////////////////////////////////////////////////////////////
///Sets the point `#ipoint` (only the real part of the argument is taken)

void TFFTRealComplex::SetPointComplex(Int_t ipoint, TComplex &c)
{
   ((Double_t *)fIn)[ipoint]=c.Re();
}

////////////////////////////////////////////////////////////////////////////////
///Set all points. Only the real array is used

void TFFTRealComplex::SetPointsComplex(const Double_t *re, const Double_t* /*im*/)
{
   for (Int_t i=0; i<fTotalSize; i++)
      ((Double_t *)fIn)[i] = re[i];
}

////////////////////////////////////////////////////////////////////////////////
///allowed options:
///"ES"
///"M"
///"P"
///"EX"

UInt_t TFFTRealComplex::MapFlag(Option_t *flag)
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
