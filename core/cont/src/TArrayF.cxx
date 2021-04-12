// @(#)root/cont:$Id$
// Author: Rene Brun   06/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TArrayF
\ingroup Containers
Array of floats (32 bits per element).
*/

#include "TArrayF.h"
#include "TBuffer.h"


ClassImp(TArrayF);

////////////////////////////////////////////////////////////////////////////////
/// Default TArrayF ctor.

TArrayF::TArrayF()
{
   fArray = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Create TArrayF object and set array size to n floats.

TArrayF::TArrayF(Int_t n)
{
   fArray = nullptr;
   if (n > 0) Set(n);
}

////////////////////////////////////////////////////////////////////////////////
/// Create TArrayF object and initialize it with values of array.

TArrayF::TArrayF(Int_t n, const Float_t *array)
{
   fArray = nullptr;
   Set(n, array);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TArrayF::TArrayF(const TArrayF &array) : TArray(array)
{
   fArray = nullptr;
   Set(array.fN, array.fArray);
}

////////////////////////////////////////////////////////////////////////////////
/// TArrayF assignment operator.

TArrayF &TArrayF::operator=(const TArrayF &rhs)
{
   if (this != &rhs)
      Set(rhs.fN, rhs.fArray);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete TArrayF object.

TArrayF::~TArrayF()
{
   delete [] fArray;
   fArray = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Adopt array arr into TArrayF, i.e. don't copy arr but use it directly
/// in TArrayF. User may not delete arr, TArrayF dtor will do it.

void TArrayF::Adopt(Int_t n, Float_t *arr)
{
   if (fArray)
      delete [] fArray;

   fN     = n;
   fArray = arr;
}

////////////////////////////////////////////////////////////////////////////////
/// Add float c at position i. Check for out of bounds.

void TArrayF::AddAt(Float_t c, Int_t i)
{
   if (!BoundsOk("TArrayF::AddAt", i)) return;
   fArray[i] += c;
}

////////////////////////////////////////////////////////////////////////////////
/// Set size of this array to n floats.
/// A new array is created, the old contents copied to the new array,
/// then the old array is deleted.
/// This function should not be called if the array was declared via Adopt.

void TArrayF::Set(Int_t n)
{
   if (n < 0) return;
   if (n != fN) {
      Float_t *temp = fArray;
      if (n != 0) {
         fArray = new Float_t[n];
         if (n < fN) {
            memcpy(fArray, temp, n*sizeof(Float_t));
         } else if (temp) {
            memcpy(fArray, temp, fN*sizeof(Float_t));
            memset(&fArray[fN], 0, (n-fN)*sizeof(Float_t));
         } else {
            memset(fArray, 0, n*sizeof(Float_t));
         }
      } else {
         fArray = nullptr;
      }
      if (fN) delete [] temp;
      fN = n;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set size of this array to n floats and set the contents.
/// This function should not be called if the array was declared via Adopt.

void TArrayF::Set(Int_t n, const Float_t *array)
{
   if (fArray && fN != n) {
      delete [] fArray;
      fArray = nullptr;
   }
   fN = n;
   if ((fN == 0) || !array)
      return;
   if (!fArray) fArray = new Float_t[fN];
   memmove(fArray, array, n*sizeof(Float_t));
}

////////////////////////////////////////////////////////////////////////////////
/// Stream a TArrayF object.

void TArrayF::Streamer(TBuffer &b)
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

