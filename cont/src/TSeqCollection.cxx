// @(#)root/cont:$Name$:$Id$
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSeqCollection                                                       //
//                                                                      //
// Sequenceable collection abstract base class. TSeqCollection's have   //
// an ordering relation, i.e. there is a first and last element.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSeqCollection.h"


ClassImp(TSeqCollection)

//______________________________________________________________________________
Int_t TSeqCollection::IndexOf(TObject *obj) const
{
   // Return index of object in collection. Returns -1 when object not found.
   // Uses member IsEqual() to find object.

   Int_t   idx = 0;
   TIter   next(this);
   TObject *ob;

   while ((ob = next())) {
      if (ob->IsEqual(obj)) return idx;
      idx++;
   }
   return -1;
}

//______________________________________________________________________________
Int_t TSeqCollection::ObjCompare(TObject *a, TObject *b)
{
   // Compare to objects in the collection. Use member Compare() of object a.

   if (a == 0 && b == 0) return 0;
   if (a == 0) return 1;
   if (b == 0) return -1;
   return a->Compare(b);
}

//______________________________________________________________________________
void TSeqCollection::QSort(TObject **a, Int_t first, Int_t last)
{
   // Sort array of TObject pointers using a quicksort algorithm.
   // Uses ObjCompare() to compare objects.

   static TObject *tmp;
   static int i;           // "static" to save stack space
   int j;

   while (last - first > 1) {
      i = first;
      j = last;
      for (;;) {
         while (++i < last && ObjCompare(a[i], a[first]) < 0)
            ;
         while (--j > first && ObjCompare(a[j], a[first]) > 0)
            ;
         if (i >= j)
            break;

         tmp  = a[i];
         a[i] = a[j];
         a[j] = tmp;
      }
      if (j == first) {
         ++first;
         continue;
      }
      tmp = a[first];
      a[first] = a[j];
      a[j] = tmp;
      if (j - first < last - (j + 1)) {
         QSort(a, first, j);
         first = j + 1;   // QSort(j + 1, last);
      } else {
         QSort(a, j + 1, last);
         last = j;        // QSort(first, j);
      }
   }
}

//______________________________________________________________________________
void TSeqCollection::QSort(TObject **a, TObject **b, Int_t first, Int_t last)
{
   // Sort array a of TObject pointers using a quicksort algorithm.
   // Array b will be sorted just like a (a determines the sort).
   // Uses ObjCompare() to compare objects.

   static TObject *tmp1, *tmp2;
   static int i;           // "static" to save stack space
   int j;

   while (last - first > 1) {
      i = first;
      j = last;
      for (;;) {
         while (++i < last && ObjCompare(a[i], a[first]) < 0)
            ;
         while (--j > first && ObjCompare(a[j], a[first]) > 0)
            ;
         if (i >= j)
            break;

         tmp1 = a[i]; tmp2 = b[i];
         a[i] = a[j]; b[i] = b[j];
         a[j] = tmp1; b[j] = tmp2;
      }
      if (j == first) {
         ++first;
         continue;
      }
      tmp1 = a[first]; tmp2 = b[first];
      a[first] = a[j]; b[first] = b[j];
      a[j] = tmp1;     b[j] = tmp2;
      if (j - first < last - (j + 1)) {
         QSort(a, b, first, j);
         first = j + 1;   // QSort(j + 1, last);
      } else {
         QSort(a, b, j + 1, last);
         last = j;        // QSort(first, j);
      }
   }
}
