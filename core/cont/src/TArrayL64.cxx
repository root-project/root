// @(#)root/cont:$Id$
// Author: Fons Rademakers   20/11/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TArrayL64
\ingroup Containers
Array of long64s (64 bits per element).
*/

#include "TArrayL64.h"
#include "TBuffer.h"


ClassImp(TArrayL64);

////////////////////////////////////////////////////////////////////////////////
/// Default TArrayL64 ctor.

TArrayL64::TArrayL64()
{
   fArray = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Create TArrayL64 object and set array size to n long64s.

TArrayL64::TArrayL64(Int_t n)
{
   fArray = nullptr;
   if (n > 0) Set(n);
}

////////////////////////////////////////////////////////////////////////////////
/// Create TArrayL object and initialize it with values of array.

TArrayL64::TArrayL64(Int_t n, const Long64_t *array)
{
   fArray = nullptr;
   Set(n, array);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TArrayL64::TArrayL64(const TArrayL64 &array) : TArray(array)
{
   fArray = nullptr;
   Set(array.fN, array.fArray);
}

////////////////////////////////////////////////////////////////////////////////
/// TArrayL64 assignment operator.

TArrayL64 &TArrayL64::operator=(const TArrayL64 &rhs)
{
   if (this != &rhs)
      Set(rhs.fN, rhs.fArray);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete TArrayL64 object.

TArrayL64::~TArrayL64()
{
   delete [] fArray;
   fArray = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Adopt array arr into TArrayL64, i.e. don't copy arr but use it directly
/// in TArrayL64. User may not delete arr, TArrayL64 dtor will do it.

void TArrayL64::Adopt(Int_t n, Long64_t *arr)
{
   if (fArray)
      delete [] fArray;

   fN     = n;
   fArray = arr;
}

////////////////////////////////////////////////////////////////////////////////
/// Add long64 c at position i. Check for out of bounds.

void TArrayL64::AddAt(Long64_t c, Int_t i)
{
   if (!BoundsOk("TArrayL64::AddAt", i)) return;
   fArray[i] = c;
}

////////////////////////////////////////////////////////////////////////////////
/// Set size of this array to n long64s.
/// A new array is created, the old contents copied to the new array,
/// then the old array is deleted.
/// This function should not be called if the array was declared via Adopt.

void TArrayL64::Set(Int_t n)
{
   if (n < 0) return;
   if (n != fN) {
      Long64_t *temp = fArray;
      if (n != 0) {
         fArray = new Long64_t[n];
         if (n < fN) {
            memcpy(fArray, temp, n*sizeof(Long64_t));
         } else if (temp) {
            memcpy(fArray, temp, fN*sizeof(Long64_t));
            memset(&fArray[fN], 0, (n-fN)*sizeof(Long64_t));
         } else {
            memset(fArray, 0, n*sizeof(Long64_t));
         }
      } else {
         fArray = nullptr;
      }
      if (fN) delete [] temp;
      fN = n;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set size of this array to n long64s and set the contents.
/// This function should not be called if the array was declared via Adopt.

void TArrayL64::Set(Int_t n, const Long64_t *array)
{
   if (fArray && fN != n) {
      delete [] fArray;
      fArray = nullptr;
   }
   fN = n;
   if ((fN == 0) || !array)
      return;
   if (!fArray) fArray = new Long64_t[fN];
   memmove(fArray, array, n*sizeof(Long64_t));
}

////////////////////////////////////////////////////////////////////////////////
/// Stream a TArrayL64 object.

void TArrayL64::Streamer(TBuffer &b)
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
