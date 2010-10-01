// @(#)root/fft:$Id$
// Author: Anna Kreshuk   07/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      
// TFFTReal                                                       
// One of the interface classes to the FFTW package, can be used directly
// or via the TVirtualFFT class. Only the basic interface of FFTW is implemented.
//
// Computes transforms called r2r in FFTW manual: 
// - transforms of real input and output in "halfcomplex" format i.e. 
//   real and imaginary parts for a transform of size n stored as 
//   (r0, r1, r2, ..., rn/2, i(n+1)/2-1, ..., i2, i1)
// - discrete Hartley transform
// - sine and cosine transforms (DCT-I,II,III,IV and DST-I,II,III,IV)
// For the detailed information on the computed
// transforms please refer to the FFTW manual, chapter "What FFTW really computes".
//
// How to use it:
// 1) Create an instance of TFFTReal - this will allocate input and output
//    arrays (unless an in-place transform is specified)
// 2) Run the Init() function with the desired flags and settings (see function
//    comments for possible kind parameters)
// 3) Set the data (via SetPoints()or SetPoint() functions)
// 4) Run the Transform() function
// 5) Get the output (via GetPoints() or GetPoint() functions)
// 6) Repeat steps 3)-5) as needed
// For a transform of the same size, but of different kind (or with different flags), 
// rerun the Init() function and continue with steps 3)-5)
//
// NOTE: 1) running Init() function will overwrite the input array! Don't set any data
//          before running the Init() function!
//       2) FFTW computes unnormalized transform, so doing a transform followed by 
//          its inverse will lead to the original array scaled BY:
//          - transform size (N) for R2HC, HC2R, DHT transforms
//          - 2*(N-1) for DCT-I (REDFT00)
//          - 2*(N+1) for DST-I (RODFT00)
//          - 2*N for the remaining transforms
// Transform inverses:
// R2HC<-->HC2R
// DHT<-->DHT
// DCT-I<-->DCT-I
// DCT-II<-->DCT-III
// DCT-IV<-->DCT-IV
// DST-I<-->DST-I
// DST-II<-->DST-III
// DST-IV<-->DST-IV
// 
//////////////////////////////////////////////////////////////////////////

#include "TFFTReal.h"
#include "fftw3.h"

ClassImp(TFFTReal)

//_____________________________________________________________________________
TFFTReal::TFFTReal()
{
//default

   fIn    = 0;
   fOut   = 0;
   fPlan  = 0;
   fN     = 0;
   fKind  = 0;
   fFlags = 0;
   fNdim = 0; 
   fTotalSize = 0;
}

//_____________________________________________________________________________
TFFTReal::TFFTReal(Int_t n, Bool_t inPlace)
{
//For 1d transforms
//n here is the physical size of the transform (see FFTW manual for more details)

   fIn = fftw_malloc(sizeof(Double_t)*n);
   if (inPlace) fOut = 0;
   else fOut = fftw_malloc(sizeof(Double_t)*n);

   fPlan = 0;
   fNdim = 1;
   fN = new Int_t[1];
   fN[0] = n;
   fKind = 0;
   fTotalSize = n;
   fFlags = 0;
}

//_____________________________________________________________________________
TFFTReal::TFFTReal(Int_t ndim, Int_t *n, Bool_t inPlace)
{
//For multidimensional transforms
//1st parameter is the # of dimensions,
//2nd is the sizes (physical) of the transform in each dimension

   fTotalSize = 1;
   fNdim = ndim;
   fN = new Int_t[ndim];
   fKind = 0;
   fPlan = 0;
   fFlags = 0; 
   for (Int_t i=0; i<ndim; i++){
      fTotalSize*=n[i];
      fN[i] = n[i];
   }
   fIn = fftw_malloc(sizeof(Double_t)*fTotalSize);
   if (!inPlace)
      fOut = fftw_malloc(sizeof(Double_t)*fTotalSize);
   else
      fOut = 0;
}

//_____________________________________________________________________________
TFFTReal::~TFFTReal()
{
//clean-up

   fftw_destroy_plan((fftw_plan)fPlan);
   fPlan = 0;
   fftw_free(fIn);
   fIn = 0;
   if (fOut){
      fftw_free(fOut);
   }
   fOut = 0;
   if (fN)
      delete [] fN;
   fN = 0;
   if (fKind)
      fftw_free((fftw_r2r_kind*)fKind);
   fKind = 0;
}

//_____________________________________________________________________________
void TFFTReal::Init( Option_t* flags,Int_t /*sign*/, const Int_t *kind)
{
//Creates the fftw-plan
//
//NOTE:  input and output arrays are overwritten during initialisation,
//       so don't set any points, before running this function!!!!!
//
//1st parameter:
//  Possible flag_options:
//  "ES" (from "estimate") - no time in preparing the transform, but probably sub-optimal
//       performance
//  "M" (from "measure") - some time spend in finding the optimal way to do the transform
//  "P" (from "patient") - more time spend in finding the optimal way to do the transform
//  "EX" (from "exhaustive") - the most optimal way is found
//  This option should be chosen depending on how many transforms of the same size and
//  type are going to be done. Planning is only done once, for the first transform of this
//  size and type.
//2nd parameter is dummy and doesn't need to be specified
//3rd parameter- transform kind for each dimension
//     4 different kinds of sine and cosine transforms are available
//     DCT-I   - kind=0
//     DCT-II  - kind=1
//     DCT-III - kind=2
//     DCT-IV  - kind=3
//     DST-I   - kind=4
//     DST-II  - kind=5
//     DSTIII  - kind=6
//     DSTIV   - kind=7

   if (!fKind)
      fKind = (fftw_r2r_kind*)fftw_malloc(sizeof(fftw_r2r_kind)*fNdim);

   if (MapOptions(kind)){
      if (fOut)
         fPlan = (void*)fftw_plan_r2r(fNdim, fN, (Double_t*)fIn, (Double_t*)fOut, (fftw_r2r_kind*)fKind, MapFlag(flags));
      else
         fPlan = (void*)fftw_plan_r2r(fNdim, fN, (Double_t*)fIn, (Double_t*)fIn, (fftw_r2r_kind*)fKind, MapFlag(flags));
      fFlags = flags;
   }
}

//_____________________________________________________________________________
void TFFTReal::Transform()
{
//Computes the transform, specified in Init() function

   if (fPlan)
      fftw_execute((fftw_plan)fPlan);
   else {
      Error("Transform", "transform hasn't been initialised");
      return;
   }
}

//_____________________________________________________________________________
Option_t *TFFTReal::GetType() const
{
//Returns the type of the transform

   if (!fKind) {
      Error("GetType", "Type not defined yet (kind not set)");
      return "";
   }
   if (((fftw_r2r_kind*)fKind)[0]==FFTW_R2HC) return "R2HC";
   if (((fftw_r2r_kind*)fKind)[0]==FFTW_HC2R) return "HC2R";
   if (((fftw_r2r_kind*)fKind)[0]==FFTW_DHT) return "DHT";
   else return "R2R";
}

//_____________________________________________________________________________
void TFFTReal::GetPoints(Double_t *data, Bool_t fromInput) const
{
//Copies the output (or input) points into the provided array, that should
//be big enough

   const Double_t * array = GetPointsReal(fromInput);
   if (!array) return; 
   std::copy(array, array+fTotalSize, data);
}

//_____________________________________________________________________________
Double_t TFFTReal::GetPointReal(Int_t ipoint, Bool_t fromInput) const
{
//For 1d tranforms. Returns point #ipoint

   if (ipoint<0 || ipoint>fTotalSize){
      Error("GetPointReal", "No such point");
      return 0;
   }
   const Double_t * array = GetPointsReal(fromInput);
   return ( array ) ? array[ipoint] : 0; 
}

//_____________________________________________________________________________
Double_t TFFTReal::GetPointReal(const Int_t *ipoint, Bool_t fromInput) const
{
//For multidim.transforms. Returns point #ipoint

   Int_t ireal = ipoint[0];
   for (Int_t i=0; i<fNdim-1; i++)
      ireal=fN[i+1]*ireal + ipoint[i+1];

   const Double_t * array = GetPointsReal(fromInput);
   return ( array ) ? array[ireal] : 0; 
}

//_____________________________________________________________________________
void TFFTReal::GetPointComplex(Int_t ipoint, Double_t &re, Double_t &im, Bool_t fromInput) const
{
//Only for input of HC2R and output of R2HC

   const Double_t * array = GetPointsReal(fromInput);
   if (!array) return; 
   if ( ( ((fftw_r2r_kind*)fKind)[0]==FFTW_R2HC && !fromInput ) ||
        ( ((fftw_r2r_kind*)fKind)[0]==FFTW_HC2R &&  fromInput ) )
   {
      if (ipoint<fN[0]/2+1){
         re = array[ipoint];
         im = array[fN[0]-ipoint];
      } else {
         re = array[fN[0]-ipoint];
         im = -array[ipoint];
      }
      if ((fN[0]%2)==0 && ipoint==fN[0]/2) im = 0;
   }
}
//_____________________________________________________________________________
void TFFTReal::GetPointComplex(const Int_t *ipoint, Double_t &re, Double_t &im, Bool_t fromInput) const
{
//Only for input of HC2R and output of R2HC and for 1d

   GetPointComplex(ipoint[0], re, im, fromInput);
}

//_____________________________________________________________________________
Double_t* TFFTReal::GetPointsReal(Bool_t fromInput) const
{
//Returns the output (or input) array

   // we have 4 different cases
   // fromInput = false; fOut = !NULL (transformed is not in place) : return fOut 
   // fromInput = false; fOut = NULL (transformed is in place) : return fIn
   // fromInput = true; fOut = !NULL :   return fIn
   // fromInput = true; fOut = NULL return an error since input array is overwritten

   if (!fromInput && fOut) 
      return (Double_t*)fOut;
   else if (fromInput && !fOut) { 
      Error("GetPointsReal","Input array was destroyed"); 
      return 0; 
   }
   //R__ASSERT(fIn);
   return (Double_t*)fIn;
}

//_____________________________________________________________________________
void TFFTReal::SetPoint(Int_t ipoint, Double_t re, Double_t im)
{
   if (ipoint<0 || ipoint>fTotalSize){
      Error("SetPoint", "illegal point index");
      return;
   }
   if (((fftw_r2r_kind*)fKind)[0]==FFTW_HC2R){
      if ((fN[0]%2)==0 && ipoint==fN[0]/2)
         ((Double_t*)fIn)[ipoint] = re;
      else {
         ((Double_t*)fIn)[ipoint] = re;
         ((Double_t*)fIn)[fN[0]-ipoint]=im;
      }
   }
   else
      ((Double_t*)fIn)[ipoint]=re;
}

//_____________________________________________________________________________
void TFFTReal::SetPoint(const Int_t *ipoint, Double_t re, Double_t /*im*/)
{
//Since multidimensional R2HC and HC2R transforms are not supported,
//third parameter is dummy

   Int_t ireal = ipoint[0];
   for (Int_t i=0; i<fNdim-1; i++)
      ireal=fN[i+1]*ireal + ipoint[i+1];
   if (ireal < 0 || ireal >fTotalSize){
      Error("SetPoint", "illegal point index");
      return;
   }
   ((Double_t*)fIn)[ireal]=re;
}

//_____________________________________________________________________________
void TFFTReal::SetPoints(const Double_t *data)
{
//Sets all points

   for (Int_t i=0; i<fTotalSize; i++)
      ((Double_t*)fIn)[i] = data[i];
}

//_____________________________________________________________________________
Int_t TFFTReal::MapOptions(const Int_t *kind)
{
//transfers the r2r_kind parameters to fftw type

   if (kind[0] == 10){
      if (fNdim>1){
         Error("Init", "Multidimensional R2HC transforms are not supported, use R2C interface instead");
         return 0;
      }
      ((fftw_r2r_kind*)fKind)[0] = FFTW_R2HC;
   }
   else if (kind[0] == 11) {
      if (fNdim>1){
         Error("Init", "Multidimensional HC2R transforms are not supported, use C2R interface instead");
         return 0;
      }
      ((fftw_r2r_kind*)fKind)[0] = FFTW_HC2R;
   }
   else if (kind[0] == 12) {
      for (Int_t i=0; i<fNdim; i++)
         ((fftw_r2r_kind*)fKind)[i] = FFTW_DHT;
   }
   else {
      for (Int_t i=0; i<fNdim; i++){
         switch (kind[i]) {
         case 0: ((fftw_r2r_kind*)fKind)[i] = FFTW_REDFT00;  break;
         case 1: ((fftw_r2r_kind*)fKind)[i] = FFTW_REDFT01;  break;
         case 2: ((fftw_r2r_kind*)fKind)[i] = FFTW_REDFT10;  break;
         case 3: ((fftw_r2r_kind*)fKind)[i] = FFTW_REDFT11;  break;
         case 4: ((fftw_r2r_kind*)fKind)[i] = FFTW_RODFT00;  break;
         case 5: ((fftw_r2r_kind*)fKind)[i] = FFTW_RODFT01;  break;
         case 6: ((fftw_r2r_kind*)fKind)[i] = FFTW_RODFT10;  break;
         case 7: ((fftw_r2r_kind*)fKind)[i] = FFTW_RODFT11;  break;
         default:
         ((fftw_r2r_kind*)fKind)[i] = FFTW_R2HC; break;
         }
      }
   }
   return 1;
}

//_____________________________________________________________________________
UInt_t TFFTReal::MapFlag(Option_t *flag)
{
//allowed options:
//"ES" - FFTW_ESTIMATE
//"M" - FFTW_MEASURE
//"P" - FFTW_PATIENT
//"EX" - FFTW_EXHAUSTIVE

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
