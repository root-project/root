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
// TArrayL                                                              //
//                                                                      //
// Array of longs (32 or 64 bits per element).                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TArrayL.h"
#include "TBuffer.h"


ClassImp(TArrayL)

//______________________________________________________________________________
TArrayL::TArrayL()
{
   // Default TArrayL ctor.

   fArray = 0;
}

//______________________________________________________________________________
TArrayL::TArrayL(Int_t n)
{
   // Create TArrayL object and set array size to n longs.

   fArray = 0;
   if (n > 0) Set(n);
}

//______________________________________________________________________________
TArrayL::TArrayL(Int_t n, const Long_t *array)
{
   // Create TArrayL object and initialize it with values of array.

   fArray = 0;
   Set(n, array);
}

//______________________________________________________________________________
TArrayL::TArrayL(const TArrayL &array) : TArray(array)
{
   // Copy constructor.

   fArray = 0;
   Set(array.fN, array.fArray);
}

//______________________________________________________________________________
TArrayL &TArrayL::operator=(const TArrayL &rhs)
{
   // TArrayL assignment operator.

   if (this != &rhs)
      Set(rhs.fN, rhs.fArray);
   return *this;
}

//______________________________________________________________________________
TArrayL::~TArrayL()
{
   // Delete TArrayL object.

   delete [] fArray;
   fArray = 0;
}

//______________________________________________________________________________
void TArrayL::Adopt(Int_t n, Long_t *arr)
{
   // Adopt array arr into TArrayL, i.e. don't copy arr but use it directly
   // in TArrayL. User may not delete arr, TArrayL dtor will do it.

   if (fArray)
      delete [] fArray;

   fN     = n;
   fArray = arr;
}

//______________________________________________________________________________
void TArrayL::AddAt(Long_t c, Int_t i)
{
   // Add long c at position i. Check for out of bounds.

   if (!BoundsOk("TArrayL::AddAt", i)) return;
   fArray[i] = c;
}

//______________________________________________________________________________
void TArrayL::Set(Int_t n)
{
   // Set size of this array to n longs.
   // A new array is created, the old contents copied to the new array,
   // then the old array is deleted.
   // This function should not be called if the array was declared via Adopt.

   if (n < 0) return;
   if (n != fN) {
      Long_t *temp = fArray;
      if (n != 0) {
         fArray = new Long_t[n];
         if (n < fN) memcpy(fArray,temp, n*sizeof(Long_t));
         else {
            memcpy(fArray,temp,fN*sizeof(Long_t));
            memset(&fArray[fN],0,(n-fN)*sizeof(Long_t));
         }
      } else {
         fArray = 0;
      }
      if (fN) delete [] temp;
      fN = n;
   }
}

//______________________________________________________________________________
void TArrayL::Set(Int_t n, const Long_t *array)
{
   // Set size of this array to n longs and set the contents.
   // This function should not be called if the array was declared via Adopt.

   if (fArray && fN != n) {
      delete [] fArray;
      fArray = 0;
   }
   fN = n;
   if (fN == 0) return;
   if (array == 0) return;
   if (!fArray) fArray = new Long_t[fN];
   memmove(fArray, array, n*sizeof(Long_t));
}

//_______________________________________________________________________
void TArrayL::Streamer(TBuffer &b)
{
   // Stream a TArrayL object.

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

