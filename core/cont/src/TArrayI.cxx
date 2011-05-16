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
// TArrayI                                                              //
//                                                                      //
// Array of integers (32 bits per element).                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TArrayI.h"
#include "TBuffer.h"


ClassImp(TArrayI)

//______________________________________________________________________________
TArrayI::TArrayI()
{
   // Default TArrayI ctor.

   fArray = 0;
}

//______________________________________________________________________________
TArrayI::TArrayI(Int_t n)
{
   // Create TArrayI object and set array size to n integers.

   fArray = 0;
   if (n > 0) Set(n);
}

//______________________________________________________________________________
TArrayI::TArrayI(Int_t n, const Int_t *array)
{
   // Create TArrayI object and initialize it with values of array.

   fArray = 0;
   Set(n, array);
}

//______________________________________________________________________________
TArrayI::TArrayI(const TArrayI &array) : TArray(array)
{
   // Copy constructor.

   fArray = 0;
   Set(array.fN, array.fArray);
}

//______________________________________________________________________________
TArrayI &TArrayI::operator=(const TArrayI &rhs)
{
   // TArrayI assignment operator.

   if (this != &rhs)
      Set(rhs.fN, rhs.fArray);
   return *this;
}

//______________________________________________________________________________
TArrayI::~TArrayI()
{
   // Delete TArrayI object.

   delete [] fArray;
   fArray = 0;
}

//______________________________________________________________________________
void TArrayI::Adopt(Int_t n, Int_t *arr)
{
   // Adopt array arr into TArrayI, i.e. don't copy arr but use it directly
   // in TArrayI. User may not delete arr, TArrayI dtor will do it.

   if (fArray)
      delete [] fArray;

   fN     = n;
   fArray = arr;
}

//______________________________________________________________________________
void TArrayI::AddAt(Int_t c, Int_t i)
{
   // Add Int_t c at position i. Check for out of bounds.

   if (!BoundsOk("TArrayI::AddAt", i)) return;
   fArray[i] = c;
}

//______________________________________________________________________________
void TArrayI::Set(Int_t n)
{
   // Set size of this array to n ints.
   // A new array is created, the old contents copied to the new array,
   // then the old array is deleted.
   // This function should not be called if the array was declared via Adopt.

   if (n < 0) return;
   if (n != fN) {
      Int_t *temp = fArray;
      if (n != 0) {
         fArray = new Int_t[n];
         if (n < fN) memcpy(fArray,temp, n*sizeof(Int_t));
         else {
            memcpy(fArray,temp,fN*sizeof(Int_t));
            memset(&fArray[fN],0,(n-fN)*sizeof(Int_t));
         }
      } else {
         fArray = 0;
      }
      if (fN) delete [] temp;
      fN = n;
   }
}

//______________________________________________________________________________
void TArrayI::Set(Int_t n, const Int_t *array)
{
   // Set size of this array to n ints and set the contents.
   // This function should not be called if the array was declared via Adopt.

   if (fArray && fN != n) {
      delete [] fArray;
      fArray = 0;
   }
   fN = n;
   if (fN == 0) return;
   if (array == 0) return;
   if (!fArray) fArray = new Int_t[fN];
   memmove(fArray, array, n*sizeof(Int_t));
}

//_______________________________________________________________________
void TArrayI::Streamer(TBuffer &b)
{
   // Stream a TArrayI object.

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

