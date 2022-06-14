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
/// \class TFFTComplexReal
///
/// One of the interface classes to the FFTW package, can be used directly
/// or via the TVirtualFFT class. Only the basic interface of FFTW is implemented.
///
/// Computes the inverse of the real-to-complex transforms (class TFFTRealComplex)
/// taking complex input (storing the non-redundant half of a logically Hermitian array)
/// to real output (see FFTW manual for more details)
///
/// How to use it:
/// 1. Create an instance of TFFTComplexReal - this will allocate input and output
///    arrays (unless an in-place transform is specified)
/// 2. Run the Init() function with the desired flags and settings
/// 3. Set the data (via SetPoints(), SetPoint() or SetPointComplex() functions)
/// 4. Run the Transform() function
/// 5. Get the output (via GetPoints(), GetPoint() or GetPointReal() functions)
/// 6. Repeat steps 3)-5) as needed
///
/// For a transform of the same size, but with different flags, rerun the Init()
/// function and continue with steps 3)-5)
///
/// NOTE:
///       1. running Init() function will overwrite the input array! Don't set any data
///          before running the Init() function
///       2. FFTW computes unnormalized transform, so doing a transform followed by
///          its inverse will lead to the original array scaled by the transform size
///       3. In Complex to Real transform the input array is destroyed. It cannot then
///          be retrieved when using the Get's methods.
///
////////////////////////////////////////////////////////////////////////////////

#include "TFFTComplexReal.h"
#include "fftw3.h"
#include "TComplex.h"


ClassImp(TFFTComplexReal);

////////////////////////////////////////////////////////////////////////////////
///default

TFFTComplexReal::TFFTComplexReal()
{
   fIn   = 0;
   fOut  = 0;
   fPlan = 0;
   fN    = 0;
   fTotalSize = 0;
   fNdim = 0;
}

////////////////////////////////////////////////////////////////////////////////
///For 1d transforms
///Allocates memory for the input array, and, if inPlace = kFALSE, for the output array

TFFTComplexReal::TFFTComplexReal(Int_t n, Bool_t inPlace)
{
   if (!inPlace){
      fIn = fftw_malloc(sizeof(fftw_complex)*(n/2+1));
      fOut = fftw_malloc(sizeof(Double_t)*n);

   } else {
      fIn = fftw_malloc(sizeof(fftw_complex)*(n/2+1));
      fOut = 0;
   }
   fNdim = 1;
   fN = new Int_t[1];
   fN[0] = n;
   fPlan = 0;
   fTotalSize = n;
}

////////////////////////////////////////////////////////////////////////////////
///For ndim-dimensional transforms
///Second argument contains sizes of the transform in each dimension

TFFTComplexReal::TFFTComplexReal(Int_t ndim, Int_t *n, Bool_t inPlace)
{
   fNdim = ndim;
   fTotalSize = 1;
   fN = new Int_t[fNdim];
   for (Int_t i=0; i<fNdim; i++){
      fN[i] = n[i];
      fTotalSize*=n[i];
   }
   Int_t sizein = Int_t(Double_t(fTotalSize)*(n[ndim-1]/2+1)/n[ndim-1]);
   if (!inPlace){
      fIn = fftw_malloc(sizeof(fftw_complex)*sizein);
      fOut = fftw_malloc(sizeof(Double_t)*fTotalSize);
   } else {
      fIn = fftw_malloc(sizeof(fftw_complex)*sizein);
      fOut = 0;
   }
   fPlan = 0;
}


////////////////////////////////////////////////////////////////////////////////
///Destroys the data arrays and the plan. However, some plan information stays around
///until the root session is over, and is reused if other plans of the same size are
///created

TFFTComplexReal::~TFFTComplexReal()
{
   fftw_destroy_plan((fftw_plan)fPlan);
   fPlan = 0;
   fftw_free((fftw_complex*)fIn);
   if (fOut)
      fftw_free(fOut);
   fIn = 0;
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

void TFFTComplexReal::Init( Option_t *flags, Int_t /*sign*/,const Int_t* /*kind*/)
{
   fFlags = flags;

   if (fPlan)
      fftw_destroy_plan((fftw_plan)fPlan);
   fPlan = 0;

   if (fOut)
      fPlan = (void*)fftw_plan_dft_c2r(fNdim, fN,(fftw_complex*)fIn,(Double_t*)fOut, MapFlag(flags));
   else
      fPlan = (void*)fftw_plan_dft_c2r(fNdim, fN, (fftw_complex*)fIn, (Double_t*)fIn, MapFlag(flags));
}

////////////////////////////////////////////////////////////////////////////////
///Computes the transform, specified in Init() function

void TFFTComplexReal::Transform()
{
   if (fPlan)
      fftw_execute((fftw_plan)fPlan);
   else {
      Error("Transform", "transform was not initialized");
      return;
   }
}

////////////////////////////////////////////////////////////////////////////////
///Fills the argument array with the computed transform
/// Works only for output (input array is destroyed in a C2R transform)

void TFFTComplexReal::GetPoints(Double_t *data, Bool_t fromInput) const
{
   if (fromInput){
      Error("GetPoints", "Input array has been destroyed");
      return;
   }
   const Double_t * array =  (const Double_t*) ( (fOut) ?  fOut :  fIn );
   std::copy(array,array+fTotalSize, data);
}

////////////////////////////////////////////////////////////////////////////////
///Returns the point `#ipoint`
/// Works only for output (input array is destroyed in a C2R transform)

Double_t TFFTComplexReal::GetPointReal(Int_t ipoint, Bool_t fromInput) const
{
   if (fromInput){
      Error("GetPointReal", "Input array has been destroyed");
      return 0;
   }
   const Double_t * array =  (const Double_t*) ( (fOut) ?  fOut :  fIn );
   return array[ipoint];
}

////////////////////////////////////////////////////////////////////////////////
///For multidimensional transforms. Returns the point `#ipoint`
/// Works only for output (input array is destroyed in a C2R transform)

Double_t TFFTComplexReal::GetPointReal(const Int_t *ipoint, Bool_t fromInput) const
{
   if (fromInput){
      Error("GetPointReal", "Input array has been destroyed");
      return 0;
   }
   Int_t ireal = ipoint[0];
   for (Int_t i=0; i<fNdim-1; i++)
      ireal=fN[i+1]*ireal + ipoint[i+1];

   const Double_t * array =  (const Double_t*) ( (fOut) ?  fOut :  fIn );
   return array[ireal];
}


////////////////////////////////////////////////////////////////////////////////
/// Works only for output (input array is destroyed in a C2R transform)

void TFFTComplexReal::GetPointComplex(Int_t ipoint, Double_t &re, Double_t &im, Bool_t fromInput) const
{
   if (fromInput){
      Error("GetPointComplex", "Input array has been destroyed");
      return;
   }
   const Double_t * array =  (const Double_t*) ( (fOut) ?  fOut :  fIn );
   re = array[ipoint];
   im = 0;
}

////////////////////////////////////////////////////////////////////////////////
///For multidimensional transforms. Returns the point `#ipoint`
/// Works only for output (input array is destroyed in a C2R transform)

void TFFTComplexReal::GetPointComplex(const Int_t *ipoint, Double_t &re, Double_t &im, Bool_t fromInput) const
{
   if (fromInput){
      Error("GetPointComplex", "Input array has been destroyed");
      return;
   }
   const Double_t * array =  (const Double_t*) ( (fOut) ?  fOut :  fIn );

   Int_t ireal = ipoint[0];
   for (Int_t i=0; i<fNdim-1; i++)
      ireal=fN[i+1]*ireal + ipoint[i+1];

   re = array[ireal];
   im = 0;
}
////////////////////////////////////////////////////////////////////////////////
///Returns the array of computed transform
/// Works only for output (input array is destroyed in a C2R transform)

Double_t* TFFTComplexReal::GetPointsReal(Bool_t fromInput) const
{
   // we have 2 different cases
   // fromInput = false; fOut = !NULL (transformed is not in place) : return fOut
   // fromInput = false; fOut = NULL (transformed is in place) : return fIn

   if (fromInput) {
      Error("GetPointsReal","Input array was destroyed");
      return 0;
   }
   return  (Double_t*) ( (fOut) ?  fOut :  fIn );
}

////////////////////////////////////////////////////////////////////////////////
///Fills the argument array with the computed transform
/// Works only for output (input array is destroyed in a C2R transform)

void TFFTComplexReal::GetPointsComplex(Double_t *re, Double_t *im, Bool_t fromInput) const
{
   if (fromInput){
      Error("GetPointsComplex", "Input array has been destroyed");
      return;
   }
   const Double_t * array =  (const Double_t*) ( (fOut) ?  fOut :  fIn );
   for (Int_t i=0; i<fTotalSize; i++){
      re[i] = array[i];
      im[i] = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
///Fills the argument array with the computed transform.
/// Works only for output (input array is destroyed in a C2R transform)

void TFFTComplexReal::GetPointsComplex(Double_t *data, Bool_t fromInput) const
{
   if (fromInput){
      Error("GetPointsComplex", "Input array has been destroyed");
      return;
   }
   const Double_t * array =  (const Double_t*) ( (fOut) ?  fOut :  fIn );
   for (Int_t i=0; i<fTotalSize; i+=2){
      data[i] = array[i/2];
      data[i+1]=0;
   }
}

////////////////////////////////////////////////////////////////////////////////
///since the input must be complex-Hermitian, if the ipoint > n/2, the according
///point before n/2 is set to (re, -im)

void TFFTComplexReal::SetPoint(Int_t ipoint, Double_t re, Double_t im)
{
   if (ipoint <= fN[0]/2){
      ((fftw_complex*)fIn)[ipoint][0] = re;
      ((fftw_complex*)fIn)[ipoint][1] = im;
   } else {
      ((fftw_complex*)fOut)[2*(fN[0]/2)-ipoint][0] = re;
      ((fftw_complex*)fOut)[2*(fN[0]/2)-ipoint][1] = -im;
   }
}

////////////////////////////////////////////////////////////////////////////////
///Set the point `#ipoint`. Since the input is Hermitian, only the first (roughly) half of
///the points have to be set.

void TFFTComplexReal::SetPoint(const Int_t *ipoint, Double_t re, Double_t im)
{
   Int_t ireal = ipoint[0];
   for (Int_t i=0; i<fNdim-2; i++){
      ireal=fN[i+1]*ireal + ipoint[i+1];
   }
   ireal = (fN[fNdim-1]/2+1)*ireal+ipoint[fNdim-1];
   Int_t realN = Int_t(Double_t(fTotalSize)*(fN[fNdim-1]/2+1)/fN[fNdim-1]);

   if (ireal > realN){
      Error("SetPoint", "Illegal index value");
      return;
   }
   ((fftw_complex*)fIn)[ireal][0] = re;
   ((fftw_complex*)fIn)[ireal][1] = im;
}

////////////////////////////////////////////////////////////////////////////////
///since the input must be complex-Hermitian, if the ipoint > n/2, the according
///point before n/2 is set to (re, -im)

void TFFTComplexReal::SetPointComplex(Int_t ipoint, TComplex &c)
{
   if (ipoint <= fN[0]/2){
      ((fftw_complex*)fIn)[ipoint][0] = c.Re();
      ((fftw_complex*)fIn)[ipoint][1] = c.Im();
   } else {
      ((fftw_complex*)fIn)[2*(fN[0]/2)-ipoint][0] = c.Re();
      ((fftw_complex*)fIn)[2*(fN[0]/2)-ipoint][1] = -c.Im();
   }
}

////////////////////////////////////////////////////////////////////////////////
///set all points. the values are copied. points should be ordered as follows:
///[re_0, im_0, re_1, im_1, ..., re_n, im_n)

void TFFTComplexReal::SetPoints(const Double_t *data)
{
   Int_t sizein = Int_t(Double_t(fTotalSize)*(fN[fNdim-1]/2+1)/fN[fNdim-1]);

   for (Int_t i=0; i<2*(sizein); i+=2){
      ((fftw_complex*)fIn)[i/2][0]=data[i];
      ((fftw_complex*)fIn)[i/2][1]=data[i+1];
   }
}

////////////////////////////////////////////////////////////////////////////////
///Set all points. The values are copied.

void TFFTComplexReal::SetPointsComplex(const Double_t *re, const Double_t *im)
{
   Int_t sizein = Int_t(Double_t(fTotalSize)*(fN[fNdim-1]/2+1)/fN[fNdim-1]);
   for (Int_t i=0; i<sizein; i++){
      ((fftw_complex*)fIn)[i][0]=re[i];
      ((fftw_complex*)fIn)[i][1]=im[i];
   }
}

////////////////////////////////////////////////////////////////////////////////
///allowed options:
///"ES" - FFTW_ESTIMATE
///"M" - FFTW_MEASURE
///"P" - FFTW_PATIENT
///"EX" - FFTW_EXHAUSTIVE

UInt_t TFFTComplexReal::MapFlag(Option_t *flag)
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
