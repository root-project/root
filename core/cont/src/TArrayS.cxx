// @(#)root/cont:$Id$
// Author: Rene Brun   06/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TArrayS
\ingroup Containers
Array of shorts (16 bits per element).
*/

#include "TArrayS.h"
#include "TBuffer.h"


ClassImp(TArrayS);

////////////////////////////////////////////////////////////////////////////////
/// Default TArrayS ctor.

TArrayS::TArrayS()
{
   fArray = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Create TArrayS object and set array size to n shorts.

TArrayS::TArrayS(Int_t n)
{
   fArray = nullptr;
   if (n > 0) Set(n);
}

////////////////////////////////////////////////////////////////////////////////
/// Create TArrayS object and initialize it with values of array.

TArrayS::TArrayS(Int_t n, const Short_t *array)
{
   fArray = nullptr;
   Set(n, array);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TArrayS::TArrayS(const TArrayS &array) : TArray(array)
{
   fArray = nullptr;
   Set(array.fN, array.fArray);
}

////////////////////////////////////////////////////////////////////////////////
/// TArrayS assignment operator.

TArrayS &TArrayS::operator=(const TArrayS &rhs)
{
   if (this != &rhs)
      Set(rhs.fN, rhs.fArray);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete TArrayS object.

TArrayS::~TArrayS()
{
   delete [] fArray;
   fArray = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Adopt array arr into TArrayS, i.e. don't copy arr but use it directly
/// in TArrayS. User may not delete arr, TArrayS dtor will do it.

void TArrayS::Adopt(Int_t n, Short_t *arr)
{
   if (fArray)
      delete [] fArray;

   fN     = n;
   fArray = arr;
}

////////////////////////////////////////////////////////////////////////////////
/// Add short c at position i. Check for out of bounds.

void TArrayS::AddAt(Short_t c, Int_t i)
{
   if (!BoundsOk("TArrayS::AddAt", i)) return;
   fArray[i] = c;
}

////////////////////////////////////////////////////////////////////////////////
/// Set size of this array to n shorts.
/// A new array is created, the old contents copied to the new array,
/// then the old array is deleted.
/// This function should not be called if the array was declared via Adopt.

void TArrayS::Set(Int_t n)
{
   if (n < 0) return;
   if (n != fN) {
      Short_t *temp = fArray;
      if (n != 0) {
         fArray = new Short_t[n];
         if (n < fN) {
            memcpy(fArray, temp, n*sizeof(Short_t));
         } else if (temp) {
            memcpy(fArray, temp, fN*sizeof(Short_t));
            memset(&fArray[fN], 0, (n-fN)*sizeof(Short_t));
         } else {
            memset(fArray, 0, n*sizeof(Short_t));
         }
      } else {
         fArray = nullptr;
      }
      if (fN) delete [] temp;
      fN = n;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set size of this array to n shorts and set the contents.
/// This function should not be called if the array was declared via Adopt.

void TArrayS::Set(Int_t n, const Short_t *array)
{
   if (fArray && fN != n) {
      delete [] fArray;
      fArray = nullptr;
   }
   fN = n;
   if ((fN == 0) || !array)
      return;
   if (!fArray) fArray = new Short_t[fN];
   memmove(fArray, array, n*sizeof(Short_t));
}

////////////////////////////////////////////////////////////////////////////////
/// Stream a TArrayS object.

void TArrayS::Streamer(TBuffer &b)
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

