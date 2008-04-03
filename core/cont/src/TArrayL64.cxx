// @(#)root/cont:$Id$
// Author: Fons Rademakers   20/11/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TArrayL64                                                            //
//                                                                      //
// Array of long64s (64 bits per element).                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TArrayL64.h"
#include "TBuffer.h"


ClassImp(TArrayL64)

//______________________________________________________________________________
TArrayL64::TArrayL64()
{
   // Default TArrayL64 ctor.

   fArray = 0;
}

//______________________________________________________________________________
TArrayL64::TArrayL64(Int_t n)
{
   // Create TArrayL64 object and set array size to n long64s.

   fArray = 0;
   if (n > 0) Set(n);
}

//______________________________________________________________________________
TArrayL64::TArrayL64(Int_t n, const Long64_t *array)
{
   // Create TArrayL object and initialize it with values of array.

   fArray = 0;
   Set(n, array);
}

//______________________________________________________________________________
TArrayL64::TArrayL64(const TArrayL64 &array) : TArray(array)
{
   // Copy constructor.

   fArray = 0;
   Set(array.fN, array.fArray);
}

//______________________________________________________________________________
TArrayL64 &TArrayL64::operator=(const TArrayL64 &rhs)
{
   // TArrayL64 assignment operator.

   if (this != &rhs)
      Set(rhs.fN, rhs.fArray);
   return *this;
}

//______________________________________________________________________________
TArrayL64::~TArrayL64()
{
   // Delete TArrayL64 object.

   delete [] fArray;
   fArray = 0;
}

//______________________________________________________________________________
void TArrayL64::Adopt(Int_t n, Long64_t *arr)
{
   // Adopt array arr into TArrayL64, i.e. don't copy arr but use it directly
   // in TArrayL64. User may not delete arr, TArrayL64 dtor will do it.

   if (fArray)
      delete [] fArray;

   fN     = n;
   fArray = arr;
}

//______________________________________________________________________________
void TArrayL64::AddAt(Long64_t c, Int_t i)
{
   // Add long64 c at position i. Check for out of bounds.

   if (!BoundsOk("TArrayL64::AddAt", i)) return;
   fArray[i] = c;
}

//______________________________________________________________________________
void TArrayL64::Set(Int_t n)
{
   // Set size of this array to n long64s.
   // A new array is created, the old contents copied to the new array,
   // then the old array is deleted.
   // This function should not be called if the array was declared via Adopt.

   if (n < 0) return;
   if (n != fN) {
      Long64_t *temp = fArray;
      if (n != 0) {
         fArray = new Long64_t[n];
         if (n < fN) memcpy(fArray,temp, n*sizeof(Long64_t));
         else {
            memcpy(fArray,temp,fN*sizeof(Long64_t));
            memset(&fArray[fN],0,(n-fN)*sizeof(Long64_t));
         }
      } else {
         fArray = 0;
      }
      if (fN) delete [] temp;
      fN = n;
   }
}

//______________________________________________________________________________
void TArrayL64::Set(Int_t n, const Long64_t *array)
{
   // Set size of this array to n long64s and set the contents.
   // This function should not be called if the array was declared via Adopt.

   if (fArray && fN != n) {
      delete [] fArray;
      fArray = 0;
   }
   fN = n;
   if (fN == 0) return;
   if (array == 0) return;
   if (!fArray) fArray = new Long64_t[fN];
   memcpy(fArray,array, n*sizeof(Long64_t));
}

//_______________________________________________________________________
void TArrayL64::Streamer(TBuffer &b)
{
   // Stream a TArrayL64 object.

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
