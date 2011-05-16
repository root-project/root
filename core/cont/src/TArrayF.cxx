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
// TArrayF                                                              //
//                                                                      //
// Array of floats (32 bits per element).                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TArrayF.h"
#include "TBuffer.h"


ClassImp(TArrayF)

//______________________________________________________________________________
TArrayF::TArrayF()
{
   // Default TArrayF ctor.

   fArray = 0;
}

//______________________________________________________________________________
TArrayF::TArrayF(Int_t n)
{
   // Create TArrayF object and set array size to n floats.

   fArray = 0;
   if (n > 0) Set(n);
}

//______________________________________________________________________________
TArrayF::TArrayF(Int_t n, const Float_t *array)
{
   // Create TArrayF object and initialize it with values of array.

   fArray = 0;
   Set(n, array);
}

//______________________________________________________________________________
TArrayF::TArrayF(const TArrayF &array) : TArray(array)
{
   // Copy constructor.

   fArray = 0;
   Set(array.fN, array.fArray);
}

//______________________________________________________________________________
TArrayF &TArrayF::operator=(const TArrayF &rhs)
{
   // TArrayF assignment operator.

   if (this != &rhs)
      Set(rhs.fN, rhs.fArray);
   return *this;
}

//______________________________________________________________________________
TArrayF::~TArrayF()
{
   // Delete TArrayF object.

   delete [] fArray;
   fArray = 0;
}

//______________________________________________________________________________
void TArrayF::Adopt(Int_t n, Float_t *arr)
{
   // Adopt array arr into TArrayF, i.e. don't copy arr but use it directly
   // in TArrayF. User may not delete arr, TArrayF dtor will do it.

   if (fArray)
      delete [] fArray;

   fN     = n;
   fArray = arr;
}

//______________________________________________________________________________
void TArrayF::AddAt(Float_t c, Int_t i)
{
   // Add float c at position i. Check for out of bounds.

   if (!BoundsOk("TArrayF::AddAt", i)) return;
   fArray[i] = c;
}

//______________________________________________________________________________
void TArrayF::Set(Int_t n)
{
   // Set size of this array to n floats.
   // A new array is created, the old contents copied to the new array,
   // then the old array is deleted.
   // This function should not be called if the array was declared via Adopt.

   if (n < 0) return;
   if (n != fN) {
      Float_t *temp = fArray;
      if (n != 0) {
         fArray = new Float_t[n];
         if (n < fN) memcpy(fArray,temp, n*sizeof(Float_t));
         else {
            memcpy(fArray,temp,fN*sizeof(Float_t));
            memset(&fArray[fN],0,(n-fN)*sizeof(Float_t));
         }
      } else {
         fArray = 0;
      }
      if (fN) delete [] temp;
      fN = n;
   }
}

//______________________________________________________________________________
void TArrayF::Set(Int_t n, const Float_t *array)
{
   // Set size of this array to n floats and set the contents.
   // This function should not be called if the array was declared via Adopt.

   if (fArray && fN != n) {
      delete [] fArray;
      fArray = 0;
   }
   fN = n;
   if (fN == 0) return;
   if (array == 0) return;
   if (!fArray) fArray = new Float_t[fN];
   memmove(fArray, array, n*sizeof(Float_t));
}

//_______________________________________________________________________
void TArrayF::Streamer(TBuffer &b)
{
   // Stream a TArrayF object.

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

