// @(#)root/cont:$Id$
// Author: Rene Brun   06/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TArrayI
\ingroup Containers
Array of integers (32 bits per element).
*/

#include "TArrayI.h"
#include "TBuffer.h"


ClassImp(TArrayI);

////////////////////////////////////////////////////////////////////////////////
/// Default TArrayI ctor.

TArrayI::TArrayI()
{
   fArray = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Create TArrayI object and set array size to n integers.

TArrayI::TArrayI(Int_t n)
{
   fArray = nullptr;
   if (n > 0) Set(n);
}

////////////////////////////////////////////////////////////////////////////////
/// Create TArrayI object and initialize it with values of array.

TArrayI::TArrayI(Int_t n, const Int_t *array)
{
   fArray = nullptr;
   Set(n, array);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TArrayI::TArrayI(const TArrayI &array) : TArray(array)
{
   fArray = nullptr;
   Set(array.fN, array.fArray);
}

////////////////////////////////////////////////////////////////////////////////
/// TArrayI assignment operator.

TArrayI &TArrayI::operator=(const TArrayI &rhs)
{
   if (this != &rhs)
      Set(rhs.fN, rhs.fArray);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete TArrayI object.

TArrayI::~TArrayI()
{
   delete [] fArray;
   fArray = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Adopt array arr into TArrayI, i.e. don't copy arr but use it directly
/// in TArrayI. User may not delete arr, TArrayI dtor will do it.

void TArrayI::Adopt(Int_t n, Int_t *arr)
{
   if (fArray)
      delete [] fArray;

   fN     = n;
   fArray = arr;
}

////////////////////////////////////////////////////////////////////////////////
/// Set size of this array to n ints.
/// A new array is created, the old contents copied to the new array,
/// then the old array is deleted.
/// This function should not be called if the array was declared via Adopt.

void TArrayI::Set(Int_t n)
{
   if (n < 0) return;
   if (n != fN) {
      Int_t *temp = fArray;
      if (n != 0) {
         fArray = new Int_t[n];
         if (n < fN) {
            memcpy(fArray, temp, n*sizeof(Int_t));
         } else if(temp) {
            memcpy(fArray, temp, fN*sizeof(Int_t));
            memset(&fArray[fN], 0, (n-fN)*sizeof(Int_t));
         } else {
            memset(fArray, 0, n*sizeof(Int_t));
         }
      } else {
         fArray = nullptr;
      }
      if (fN) delete [] temp;
      fN = n;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set size of this array to n ints and set the contents.
/// This function should not be called if the array was declared via Adopt.

void TArrayI::Set(Int_t n, const Int_t *array)
{
   if (fArray && fN != n) {
      delete [] fArray;
      fArray = nullptr;
   }
   fN = n;
   if ((fN == 0) || !array)
      return;
   if (!fArray) fArray = new Int_t[fN];
   memmove(fArray, array, n*sizeof(Int_t));
}

////////////////////////////////////////////////////////////////////////////////
/// Stream a TArrayI object.

void TArrayI::Streamer(TBuffer &b)
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

