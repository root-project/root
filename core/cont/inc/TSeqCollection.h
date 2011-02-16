// @(#)root/cont:$Id$
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSeqCollection
#define ROOT_TSeqCollection


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSeqCollection                                                       //
//                                                                      //
// Sequenceable collection abstract base class. TSeqCollection's have   //
// an ordering relation, i.e. there is a first and last element.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TCollection
#include "TCollection.h"
#endif


class TSeqCollection : public TCollection {

protected:
   Bool_t            fSorted;    // true if collection has been sorted

   TSeqCollection() : fSorted(kFALSE) { }
   virtual void      Changed() { fSorted = kFALSE; }

public:
   virtual           ~TSeqCollection() { }
   virtual void      Add(TObject *obj) { AddLast(obj); }
   virtual void      AddFirst(TObject *obj) = 0;
   virtual void      AddLast(TObject *obj) = 0;
   virtual void      AddAt(TObject *obj, Int_t idx) = 0;
   virtual void      AddAfter(const TObject *after, TObject *obj) = 0;
   virtual void      AddBefore(const TObject *before, TObject *obj) = 0;
   virtual void      RemoveFirst() { Remove(First()); }
   virtual void      RemoveLast() { Remove(Last()); }
   virtual TObject  *RemoveAt(Int_t idx) { return Remove(At(idx)); }
   virtual void      RemoveAfter(TObject *after) { Remove(After(after)); }
   virtual void      RemoveBefore(TObject *before) { Remove(Before(before)); }

   virtual TObject  *At(Int_t idx) const = 0;
   virtual TObject  *Before(const TObject *obj) const = 0;
   virtual TObject  *After(const TObject *obj) const = 0;
   virtual TObject  *First() const = 0;
   virtual TObject  *Last() const = 0;
   Int_t             LastIndex() const { return GetSize() - 1; }
   virtual Int_t     GetLast() const;
   virtual Int_t     IndexOf(const TObject *obj) const;
   virtual Bool_t    IsSorted() const { return fSorted; }
   void              UnSort() { fSorted = kFALSE; }
   Long64_t          Merge(TCollection *list);

   static Int_t       ObjCompare(TObject *a, TObject *b);
   static void        QSort(TObject **a, Int_t first, Int_t last);
   static inline void QSort(TObject **a, TObject **b, Int_t first, Int_t last) { QSort(a, 1, &b, first, last); }
   static void        QSort(TObject **a, Int_t nBs, TObject ***b, Int_t first, Int_t last);

   ClassDef(TSeqCollection,0)  //Sequenceable collection ABC
};

#endif
