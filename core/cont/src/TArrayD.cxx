// @(#)root/cont:$Id$
// Author: Rene Brun   06/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TArrayD                                                              //
//                                                                      //
// Array of doubles (64 bits per element).                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TArrayD.h"
#include "TBuffer.h"


ClassImp(TArrayD)

//______________________________________________________________________________
TArrayD::TArrayD()
{
   // Default TArrayD ctor.

   fArray = 0;
}

//______________________________________________________________________________
TArrayD::TArrayD(Int_t n)
{
   // Create TArrayD object and set array size to n doubles.

   fArray = 0;
   if (n > 0) Set(n);
}

//______________________________________________________________________________
TArrayD::TArrayD(Int_t n, const Double_t *array)
{
   // Create TArrayD object and initialize it with values of array.

   fArray = 0;
   Set(n, array);
}

//______________________________________________________________________________
TArrayD::TArrayD(const TArrayD &array) : TArray(array)
{
   // Copy constructor.

   fArray = 0;
   Set(array.fN, array.fArray);
}

//______________________________________________________________________________
TArrayD &TArrayD::operator=(const TArrayD &rhs)
{
   // TArrayD assignment operator.

   if (this != &rhs)
      Set(rhs.fN, rhs.fArray);
   return *this;
}

//______________________________________________________________________________
TArrayD::~TArrayD()
{
   // Delete TArrayD object.

   delete [] fArray;
   fArray = 0;
}

//______________________________________________________________________________
void TArrayD::Adopt(Int_t n, Double_t *arr)
{
   // Adopt array arr into TArrayD, i.e. don't copy arr but use it directly
   // in TArrayD. User may not delete arr, TArrayD dtor will do it.

   if (fArray)
      delete [] fArray;

   fN     = n;
   fArray = arr;
}

//______________________________________________________________________________
void TArrayD::AddAt(Double_t c, Int_t i)
{
   // Add double c at position i. Check for out of bounds.

   if (!BoundsOk("TArrayD::AddAt", i)) return;
   fArray[i] = c;
}

//______________________________________________________________________________
void TArrayD::Set(Int_t n)
{
   // Set size of this array to n doubles.
   // A new array is created, the old contents copied to the new array,
   // then the old array is deleted.
   // This function should not be called if the array was declared via Adopt.

   if (n < 0) return;
   if (n != fN) {
      Double_t *temp = fArray;
      if (n != 0) {
         fArray = new Double_t[n];
         if (n < fN) memcpy(fArray,temp, n*sizeof(Double_t));
         else {
            memcpy(fArray,temp,fN*sizeof(Double_t));
            memset(&fArray[fN],0,(n-fN)*sizeof(Double_t));
         }
      } else {
         fArray = 0;
      }
      if (fN) delete [] temp;
      fN = n;
   }
}

//______________________________________________________________________________
void TArrayD::Set(Int_t n, const Double_t *array)
{
   // Set size of this array to n doubles and set the contents
   // This function should not be called if the array was declared via Adopt.

   if (fArray && fN != n) {
      delete [] fArray;
      fArray = 0;
   }
   fN = n;
   if (fN == 0) return;
   if (array == 0) return;
   if (!fArray) fArray = new Double_t[fN];
   memmove(fArray, array, n*sizeof(Double_t));
}

//_______________________________________________________________________
void TArrayD::Streamer(TBuffer &b)
{
   // Stream a TArrayD object.

   if (b.IsReading()) {
      Int_t n;
      b >> n;
      Set(n);
      b.ReadFastArray(fArray,n);
   } else {
      b << fN;
      b.WriteFastArray(fArray, fN);
   }
}

