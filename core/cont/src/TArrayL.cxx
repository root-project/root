// @(#)root/cont:$Id$
// Author: Rene Brun   06/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TArrayL
\ingroup Containers
Array of longs (32 or 64 bits per element).
*/

#include "TArrayL.h"
#include "TBuffer.h"


ClassImp(TArrayL);

////////////////////////////////////////////////////////////////////////////////
/// Default TArrayL ctor.

TArrayL::TArrayL()
{
   fArray = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Create TArrayL object and set array size to n longs.

TArrayL::TArrayL(Int_t n)
{
   fArray = nullptr;
   if (n > 0) Set(n);
}

////////////////////////////////////////////////////////////////////////////////
/// Create TArrayL object and initialize it with values of array.

TArrayL::TArrayL(Int_t n, const Long_t *array)
{
   fArray = nullptr;
   Set(n, array);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TArrayL::TArrayL(const TArrayL &array) : TArray(array)
{
   fArray = nullptr;
   Set(array.fN, array.fArray);
}

////////////////////////////////////////////////////////////////////////////////
/// TArrayL assignment operator.

TArrayL &TArrayL::operator=(const TArrayL &rhs)
{
   if (this != &rhs)
      Set(rhs.fN, rhs.fArray);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete TArrayL object.

TArrayL::~TArrayL()
{
   delete [] fArray;
   fArray = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Adopt array arr into TArrayL, i.e. don't copy arr but use it directly
/// in TArrayL. User may not delete arr, TArrayL dtor will do it.

void TArrayL::Adopt(Int_t n, Long_t *arr)
{
   if (fArray)
      delete [] fArray;

   fN     = n;
   fArray = arr;
}

////////////////////////////////////////////////////////////////////////////////
/// Set size of this array to n longs.
/// A new array is created, the old contents copied to the new array,
/// then the old array is deleted.
/// This function should not be called if the array was declared via Adopt.

void TArrayL::Set(Int_t n)
{
   if (n < 0) return;
   if (n != fN) {
      Long_t *temp = fArray;
      if (n != 0) {
         fArray = new Long_t[n];
         if (n < fN) {
            memcpy(fArray, temp, n*sizeof(Long_t));
         } else if (temp) {
            memcpy(fArray, temp, fN*sizeof(Long_t));
            memset(&fArray[fN], 0, (n-fN)*sizeof(Long_t));
         } else {
            memset(fArray, 0, n*sizeof(Long_t));
         }
      } else {
         fArray = nullptr;
      }
      if (fN) delete [] temp;
      fN = n;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set size of this array to n longs and set the contents.
/// This function should not be called if the array was declared via Adopt.

void TArrayL::Set(Int_t n, const Long_t *array)
{
   if (fArray && fN != n) {
      delete [] fArray;
      fArray = nullptr;
   }
   fN = n;
   if ((fN == 0) || !array)
      return;
   if (!fArray) fArray = new Long_t[fN];
   memmove(fArray, array, n*sizeof(Long_t));
}

////////////////////////////////////////////////////////////////////////////////
/// Stream a TArrayL object.

void TArrayL::Streamer(TBuffer &b)
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

