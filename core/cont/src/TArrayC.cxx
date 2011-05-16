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
// TArrayC                                                              //
//                                                                      //
// Array of chars or bytes (8 bits per element).                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TArrayC.h"
#include "TBuffer.h"


ClassImp(TArrayC)

//______________________________________________________________________________
TArrayC::TArrayC()
{
   // Default TArrayC ctor.

   fArray = 0;
}

//______________________________________________________________________________
TArrayC::TArrayC(Int_t n)
{
   // Create TArrayC object and set array size to n chars.

   fArray = 0;
   if (n > 0) Set(n);
}

//______________________________________________________________________________
TArrayC::TArrayC(Int_t n, const Char_t *array)
{
   // Create TArrayC object and initialize it with values of array.

   fArray = 0;
   Set(n, array);
}

//______________________________________________________________________________
TArrayC::TArrayC(const TArrayC &array) : TArray(array)
{
   // Copy constructor.

   fArray = 0;
   Set(array.fN, array.fArray);
}

//______________________________________________________________________________
TArrayC &TArrayC::operator=(const TArrayC &rhs)
{
   // TArrayC assignment operator.

   if (this != &rhs)
      Set(rhs.fN, rhs.fArray);
   return *this;
}

//______________________________________________________________________________
TArrayC::~TArrayC()
{
   // Delete TArrayC object.

   delete [] fArray;
   fArray = 0;
}

//______________________________________________________________________________
void TArrayC::Adopt(Int_t n, Char_t *arr)
{
   // Adopt array arr into TArrayC, i.e. don't copy arr but use it directly
   // in TArrayC. User may not delete arr, TArrayC dtor will do it.

   if (fArray)
      delete [] fArray;

   fN     = n;
   fArray = arr;
}

//______________________________________________________________________________
void TArrayC::AddAt(Char_t c, Int_t i)
{
   // Add char c at position i. Check for out of bounds.

   if (!BoundsOk("TArrayC::AddAt", i)) return;
   fArray[i] = c;
}

//______________________________________________________________________________
void TArrayC::Set(Int_t n)
{
   // Set size of this array to n chars.
   // A new array is created, the old contents copied to the new array,
   // then the old array is deleted.
   // This function should not be called if the array was declared via Adopt.

   if (n < 0) return;
   if (n != fN) {
      Char_t *temp = fArray;
      if (n != 0) {
         fArray = new Char_t[n];
         if (n < fN) memcpy(fArray,temp, n*sizeof(Char_t));
         else {
            memcpy(fArray,temp,fN*sizeof(Char_t));
            memset(&fArray[fN],0,(n-fN)*sizeof(Char_t));
         }
      } else {
         fArray = 0;
      }
      if (fN) delete [] temp;
      fN = n;
   }
}

//______________________________________________________________________________
void TArrayC::Set(Int_t n, const Char_t *array)
{
   // Set size of this array to n chars and set the contents.
   // This function should not be called if the array was declared via Adopt.

   if (fArray && fN != n) {
      delete [] fArray;
      fArray = 0;
   }
   fN = n;
   if (fN == 0) return;
   if (array == 0) return;
   if (!fArray) fArray = new Char_t[fN];
   memmove(fArray, array, n*sizeof(Char_t));
}

//_______________________________________________________________________
void TArrayC::Streamer(TBuffer &b)
{
   // Stream a TArrayC object.

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

