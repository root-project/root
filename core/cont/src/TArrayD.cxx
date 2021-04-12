// @(#)root/cont:$Id$
// Author: Rene Brun   06/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TArrayD
\ingroup Containers
Array of doubles (64 bits per element).
*/

#include "TArrayD.h"
#include "TBuffer.h"


ClassImp(TArrayD);

////////////////////////////////////////////////////////////////////////////////
/// Default TArrayD ctor.

TArrayD::TArrayD()
{
   fArray = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Create TArrayD object and set array size to n doubles.

TArrayD::TArrayD(Int_t n)
{
   fArray = nullptr;
   if (n > 0) Set(n);
}

////////////////////////////////////////////////////////////////////////////////
/// Create TArrayD object and initialize it with values of array.

TArrayD::TArrayD(Int_t n, const Double_t *array)
{
   fArray = nullptr;
   Set(n, array);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TArrayD::TArrayD(const TArrayD &array) : TArray(array)
{
   fArray = nullptr;
   Set(array.fN, array.fArray);
}

////////////////////////////////////////////////////////////////////////////////
/// TArrayD assignment operator.

TArrayD &TArrayD::operator=(const TArrayD &rhs)
{
   if (this != &rhs)
      Set(rhs.fN, rhs.fArray);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete TArrayD object.

TArrayD::~TArrayD()
{
   delete [] fArray;
   fArray = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Adopt array arr into TArrayD, i.e. don't copy arr but use it directly
/// in TArrayD. User may not delete arr, TArrayD dtor will do it.

void TArrayD::Adopt(Int_t n, Double_t *arr)
{
   if (fArray)
      delete [] fArray;

   fN     = n;
   fArray = arr;
}

////////////////////////////////////////////////////////////////////////////////
/// Set size of this array to n doubles.
/// A new array is created, the old contents copied to the new array,
/// then the old array is deleted.
/// This function should not be called if the array was declared via Adopt.

void TArrayD::Set(Int_t n)
{
   if (n < 0) return;
   if (n != fN) {
      Double_t *temp = fArray;
      if (n != 0) {
         fArray = new Double_t[n];
         if (n < fN) {
            memcpy(fArray, temp, n*sizeof(Double_t));
         } else if (temp) {
            memcpy(fArray, temp, fN*sizeof(Double_t));
            memset(&fArray[fN], 0, (n-fN)*sizeof(Double_t));
         } else {
            memset(fArray, 0, n*sizeof(Double_t));
         }
      } else {
         fArray = nullptr;
      }
      if (fN) delete [] temp;
      fN = n;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set size of this array to n doubles and set the contents
/// This function should not be called if the array was declared via Adopt.

void TArrayD::Set(Int_t n, const Double_t *array)
{
   if (fArray && fN != n) {
      delete [] fArray;
      fArray = nullptr;
   }
   fN = n;
   if ((fN == 0) || !array)
      return;
   if (!fArray) fArray = new Double_t[fN];
   memmove(fArray, array, n*sizeof(Double_t));
}

////////////////////////////////////////////////////////////////////////////////
/// Stream a TArrayD object.

void TArrayD::Streamer(TBuffer &b)
{
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

