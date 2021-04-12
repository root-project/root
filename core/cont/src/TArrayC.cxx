// @(#)root/cont:$Id$
// Author: Rene Brun   06/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TArrayC
\ingroup Containers
Array of chars or bytes (8 bits per element).
*/

#include "TArrayC.h"
#include "TBuffer.h"


ClassImp(TArrayC);

////////////////////////////////////////////////////////////////////////////////
/// Default TArrayC ctor.

TArrayC::TArrayC()
{
   fArray = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Create TArrayC object and set array size to n chars.

TArrayC::TArrayC(Int_t n)
{
   fArray = nullptr;
   if (n > 0) Set(n);
}

////////////////////////////////////////////////////////////////////////////////
/// Create TArrayC object and initialize it with values of array.

TArrayC::TArrayC(Int_t n, const Char_t *array)
{
   fArray = nullptr;
   Set(n, array);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TArrayC::TArrayC(const TArrayC &array) : TArray(array)
{
   fArray = nullptr;
   Set(array.fN, array.fArray);
}

////////////////////////////////////////////////////////////////////////////////
/// TArrayC assignment operator.

TArrayC &TArrayC::operator=(const TArrayC &rhs)
{
   if (this != &rhs)
      Set(rhs.fN, rhs.fArray);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete TArrayC object.

TArrayC::~TArrayC()
{
   delete [] fArray;
   fArray = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Adopt array arr into TArrayC, i.e. don't copy arr but use it directly
/// in TArrayC. User may not delete arr, TArrayC dtor will do it.

void TArrayC::Adopt(Int_t n, Char_t *arr)
{
   if (fArray)
      delete [] fArray;

   fN     = n;
   fArray = arr;
}

////////////////////////////////////////////////////////////////////////////////
/// Set size of this array to n chars.
/// A new array is created, the old contents copied to the new array,
/// then the old array is deleted.
/// This function should not be called if the array was declared via Adopt.

void TArrayC::Set(Int_t n)
{
   if (n < 0) return;
   if (n != fN) {
      Char_t *temp = fArray;
      if (n != 0) {
         fArray = new Char_t[n];
         if (n < fN) {
            memcpy(fArray, temp, n*sizeof(Char_t));
         } else if (temp) {
            memcpy(fArray, temp, fN*sizeof(Char_t));
            memset(&fArray[fN], 0, (n-fN)*sizeof(Char_t));
         } else {
            memset(fArray, 0, n*sizeof(Char_t));
         }
      } else {
         fArray = nullptr;
      }
      if (fN) delete [] temp;
      fN = n;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set size of this array to n chars and set the contents.
/// This function should not be called if the array was declared via Adopt.

void TArrayC::Set(Int_t n, const Char_t *array)
{
   if (fArray && fN != n) {
      delete [] fArray;
      fArray = nullptr;
   }
   fN = n;
   if ((fN == 0) || !array)
      return;
   if (!fArray) fArray = new Char_t[fN];
   memmove(fArray, array, n*sizeof(Char_t));
}

////////////////////////////////////////////////////////////////////////////////
/// Stream a TArrayC object.

void TArrayC::Streamer(TBuffer &b)
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

