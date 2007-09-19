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
// TArrayS                                                              //
//                                                                      //
// Array of shorts (16 bits per element).                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TArrayS.h"
#include "TBuffer.h"


ClassImp(TArrayS)

//______________________________________________________________________________
TArrayS::TArrayS()
{
   // Default TArrayS ctor.

   fArray = 0;
}

//______________________________________________________________________________
TArrayS::TArrayS(Int_t n)
{
   // Create TArrayS object and set array size to n shorts.

   fArray = 0;
   if (n > 0) Set(n);
}

//______________________________________________________________________________
TArrayS::TArrayS(Int_t n, const Short_t *array)
{
   // Create TArrayS object and initialize it with values of array.

   fArray = 0;
   Set(n, array);
}

//______________________________________________________________________________
TArrayS::TArrayS(const TArrayS &array) : TArray(array)
{
   // Copy constructor.

   fArray = 0;
   Set(array.fN, array.fArray);
}

//______________________________________________________________________________
TArrayS &TArrayS::operator=(const TArrayS &rhs)
{
   // TArrayS assignment operator.

   if (this != &rhs)
      Set(rhs.fN, rhs.fArray);
   return *this;
}

//______________________________________________________________________________
TArrayS::~TArrayS()
{
   // Delete TArrayS object.

   delete [] fArray;
   fArray = 0;
}

//______________________________________________________________________________
void TArrayS::Adopt(Int_t n, Short_t *arr)
{
   // Adopt array arr into TArrayS, i.e. don't copy arr but use it directly
   // in TArrayS. User may not delete arr, TArrayS dtor will do it.

   if (fArray)
      delete [] fArray;

   fN     = n;
   fArray = arr;
}

//______________________________________________________________________________
void TArrayS::AddAt(Short_t c, Int_t i)
{
   // Add short c at position i. Check for out of bounds.

   if (!BoundsOk("TArrayS::AddAt", i)) return;
   fArray[i] = c;
}

//______________________________________________________________________________
void TArrayS::Set(Int_t n)
{
   // Set size of this array to n shorts.
   // A new array is created, the old contents copied to the new array,
   // then the old array is deleted.
   // This function should not be called if the array was declared via Adopt.

   if (n < 0) return;
   if (n != fN) {
      Short_t *temp = fArray;
      if (n != 0) {
         fArray = new Short_t[n];
         if (n < fN) memcpy(fArray,temp, n*sizeof(Short_t));
         else {
            memcpy(fArray,temp,fN*sizeof(Short_t));
            memset(&fArray[fN],0,(n-fN)*sizeof(Short_t));
         }
      } else {
         fArray = 0;
      }
      if (fN) delete [] temp;
      fN = n;
   }
}

//______________________________________________________________________________
void TArrayS::Set(Int_t n, const Short_t *array)
{
   // Set size of this array to n shorts and set the contents.
   // This function should not be called if the array was declared via Adopt.

   if (fArray && fN != n) {
      delete [] fArray;
      fArray = 0;
   }
   fN = n;
   if (fN == 0) return;
   if (array == 0) return;
   if (!fArray) fArray = new Short_t[fN];
   memcpy(fArray,array, n*sizeof(Short_t));
}

//_______________________________________________________________________
void TArrayS::Streamer(TBuffer &b)
{
   // Stream a TArrayS object.

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

