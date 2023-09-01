// @(#)root/cont:$Id$
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TSeqCollection
\ingroup Containers
Sequenceable collection abstract base class. TSeqCollection's have
an ordering relation, i.e. there is a first and last element.
*/

#include "TSeqCollection.h"
#include "TCollection.h"
#include "TVirtualMutex.h"
#include "TClass.h"
#include "TMethodCall.h"

ClassImp(TSeqCollection);

////////////////////////////////////////////////////////////////////////////////
/// Return index of object in collection. Returns -1 when object not found.
/// Uses member IsEqual() to find object.

Int_t TSeqCollection::IndexOf(const TObject *obj) const
{
   Int_t   idx = 0;
   TIter   next(this);
   TObject *ob;

   while ((ob = next())) {
      if (ob->IsEqual(obj)) return idx;
      idx++;
   }
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns index of last object in collection. Returns -1 when no
/// objects in collection.

Int_t TSeqCollection::GetLast() const
{
   TObject *tmp = Last();
   return tmp ? IndexOf(tmp) : -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Compare to objects in the collection. Use member Compare() of object a.

Int_t TSeqCollection::ObjCompare(TObject *a, TObject *b)
{
   if (!a && !b) return 0;
   if (!a) return 1;
   if (!b) return -1;
   return a->Compare(b);
}

////////////////////////////////////////////////////////////////////////////////
/// Sort array of TObject pointers using a quicksort algorithm.
/// The algorithm used is a non stable sort (i.e. already sorted
/// elements might switch/change places).
/// Uses ObjCompare() to compare objects.

void TSeqCollection::QSort(TObject **a, Int_t first, Int_t last)
{
   R__LOCKGUARD2(gCollectionMutex);

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

////////////////////////////////////////////////////////////////////////////////
/// Sort array a of TObject pointers using a quicksort algorithm.
/// Arrays b will be sorted just like a (a determines the sort).
/// Argument nBs is the number of TObject** arrays in b.
/// The algorithm used is a non stable sort (i.e. already sorted
/// elements might switch/change places).
/// Uses ObjCompare() to compare objects.

void TSeqCollection::QSort(TObject **a, Int_t nBs, TObject ***b, Int_t first, Int_t last)
{
   R__LOCKGUARD2(gCollectionMutex);

   static TObject *tmp1, **tmp2;
   static int i; // "static" to save stack space
   int j,k;

   static int depth = 0;
   if (depth == 0 && nBs > 0) tmp2 = new TObject*[nBs];
   depth++;

   while (last - first > 1) {
      i = first;
      j = last;
      for (;;) {
         while (++i < last && ObjCompare(a[i], a[first]) < 0) {}
         while (--j > first && ObjCompare(a[j], a[first]) > 0) {}
         if (i >= j) break;

         tmp1 = a[i]; for(k=0;k<nBs;k++) tmp2[k] = b[k][i];
         a[i] = a[j]; for(k=0;k<nBs;k++) b[k][i] = b[k][j];
         a[j] = tmp1; for(k=0;k<nBs;k++) b[k][j] = tmp2[k];
      }
      if (j == first) {
         ++first;
         continue;
      }
      tmp1 = a[first]; for(k=0;k<nBs;k++) tmp2[k] = b[k][first];
      a[first] = a[j]; for(k=0;k<nBs;k++) b[k][first] = b[k][j];
      a[j] = tmp1; for(k=0;k<nBs;k++) b[k][j] = tmp2[k];
      if (j - first < last - (j + 1)) {
         QSort(a, nBs, b, first, j);
         first = j + 1; // QSort(j + 1, last);
      } else {
         QSort(a, nBs, b, j + 1, last);
         last = j; // QSort(first, j);
      }
   }
   depth--;

   if (depth == 0 && nBs > 0) delete [] tmp2;
}

////////////////////////////////////////////////////////////////////////////////
/// Merge this collection with all collections coming in the input list. The
/// input list must contain other collections of objects compatible with the
/// ones in this collection and ordered in the same manner. For example, if this
/// collection contains a TH1 object and a tree, all collections in the input
/// list have to contain a histogram and a tree. In case the list contains
/// collections, the objects in the input lists must also be collections with
/// the same structure and number of objects.
/// If some objects inside the collection are instances of a class that do not
/// have a Merge function (like TObjString), rather than merging, a copy of each
/// instance (via a call to Clone) is appended to the output.
///
/// ### Example
/// ~~~ {.cpp}
///   this                          list
/// ____________                  ---------------------|
/// | A (TH1F) |  __________      | L1 (TSeqCollection)|- [A1, B1(C1,D1,E1)]
/// | B (TList)|-| C (TTree)|     | L1 (TSeqCollection)|- [A2, B2(C2,D2,E2)]
/// |__________| | D (TH1F) |     | ...                |- [...]
///              | E (TH1F) |     |____________________|
///              |__________|
/// ~~~

Long64_t TSeqCollection::Merge(TCollection *list)
{
   Long64_t nmerged = 0;
   if (IsEmpty() || !list) {
      Warning("Merge", "list is empty - nothing to merge");
      return 0;
   }
   if (list->IsEmpty()) {
      Warning("Merge", "input list is empty - nothing to merge with");
      return 0;
   }
   TIter nextobject(this);
   TIter nextlist(list);
   TObject *object;
   TObject *objtomerge;
   TObject *collcrt;
   TSeqCollection *templist = nullptr;
   TMethodCall callEnv;
   Int_t indobj = 0;
   TSeqCollection *notmergeable = nullptr;
   Bool_t mergeable = kTRUE;
   while ((object = nextobject())) {   // loop objects in this collection
      mergeable = kTRUE;
      // If current object has not dictionary just add it
      if (!object->IsA()) {
         mergeable = kFALSE;
      } else {
         // If current object is not mergeable just add it
         callEnv.InitWithPrototype(object->IsA(), "Merge", "TCollection*");
         if (!callEnv.IsValid()) mergeable = kFALSE;
      }
      if (mergeable) {
         // Current object mergeable - get corresponding objects in input lists
         templist = (TSeqCollection*)IsA()->New();
         // Make sure original objects are not deleted; some containers, e.g. TSelectorList, maybe owners
         templist->SetOwner(kFALSE);
      } else {
         templist = nullptr;
      }
      nextlist.Reset();
      Int_t indcoll = 0;
      while ((collcrt = nextlist())) {      // loop input lists
         if (!collcrt->InheritsFrom(TSeqCollection::Class())) {
            Error("Merge", "some objects in the input list are not collections - merging aborted");
            SafeDelete(templist);
            return 0;
         }
         if (indobj > ((TSeqCollection*)collcrt)->LastIndex()) {
            // We reached the end of this collection.
            continue;
         }
         // The next object to be merged with is a collection
         // the iterator skips the 'holes' the collections, we also need to do so.
         objtomerge = ((TSeqCollection*)collcrt)->At(indobj);
         if (!objtomerge) {
            Warning("Merge", "object of type %s (position %d in list) not found in list %d. Continuing...",
                             object->ClassName(), indobj, indcoll);
            continue;
         }
/*
         // Dangerous - may try to merge non-corresponding histograms (A.G)
         while (objtomerge == 0
                && indobj < ((TSeqCollection*)collcrt)->LastIndex()
               ) {
            ++indobj;
            objtomerge = ((TSeqCollection*)collcrt)->At(indobj);
         }
*/
         if (object->IsA() != objtomerge->IsA()) {
            Error("Merge", "object of type %s at index %d not matching object of type %s in input list",
                           object->ClassName(), indobj, objtomerge->ClassName());
            SafeDelete(templist);
            return 0;
         }
         // Add object at index indobj in the temporary list
         if (mergeable) {
            templist->Add(objtomerge);
            nmerged++;
         } else {
            // Just add it to the dedicated temp list for later addition to the current list
            if (!notmergeable && IsA())
               notmergeable = (TSeqCollection*)IsA()->New();
            if (notmergeable)
               notmergeable->Add(objtomerge);
            else
               Warning("Merge", "temp list for non mergeable objects not created!");
         }
      }
      // Merge current object with objects in the temporary list
      if (mergeable) {
         callEnv.SetParam((Longptr_t) templist);
         callEnv.Execute(object);
         SafeDelete(templist);
      }
      indobj++;
   }

   // Add the non-mergeable objects, if any
   if (notmergeable && notmergeable->GetSize() > 0) {
      TIter nxnm(notmergeable);
      while (auto onm = nxnm())
         Add(onm->Clone());
      SafeDelete(notmergeable);
   }

   return nmerged;
}
