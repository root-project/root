// @(#)root/cont:$Name:  $:$Id: TObjArray.h,v 1.8 2001/05/08 14:21:36 brun Exp $
// Author: Fons Rademakers   11/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TObjArray
#define ROOT_TObjArray


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjArray                                                            //
//                                                                      //
// An array of TObjects. The array expands automatically when adding    //
// elements (shrinking can be done by hand).                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSeqCollection
#include "TSeqCollection.h"
#endif


class TObjArray : public TSeqCollection {

friend class TObjArrayIter;
friend class TClonesArray;

protected:
   TObject     **fCont;        //!Array contents
   Int_t         fLowerBound;  //Lower bound of the array
   Int_t         fLast;        //Last element in array containing an object

   Bool_t        BoundsOk(const char *where, Int_t at) const;
   void          Init(Int_t s, Int_t lowerBound);
   Bool_t        OutOfBoundsError(const char *where, Int_t i) const;
   Int_t         GetAbsLast() const;

public:
   TObjArray(Int_t s = TCollection::kInitCapacity, Int_t lowerBound = 0);
   TObjArray(const TObjArray &a);
   virtual          ~TObjArray();
   virtual void     Clear(Option_t *option="");
   virtual void     Compress();
   virtual void     Delete(Option_t *option="");
   virtual void     Expand(Int_t newSize);   // expand or shrink an array
   Int_t            GetEntries() const;
   Int_t            GetEntriesFast() const {return GetAbsLast()+1;}  //only OK when no gaps
   Int_t            GetLast() const;
   TObject        **GetObjectRef(TObject *obj) const;
   Bool_t           IsEmpty() const { return GetAbsLast() == -1; }
   TIterator       *MakeIterator(Bool_t dir = kIterForward) const;

   void             Add(TObject *obj) { AddLast(obj); }
   virtual void     AddFirst(TObject *obj);
   virtual void     AddLast(TObject *obj);
   virtual void     AddAt(TObject *obj, Int_t idx);
   virtual void     AddAtAndExpand(TObject *obj, Int_t idx);
   virtual Int_t    AddAtFree(TObject *obj);
   virtual void     AddAfter(TObject *after, TObject *obj);
   virtual void     AddBefore(TObject *before, TObject *obj);
   virtual TObject *RemoveAt(Int_t idx);
   virtual TObject *Remove(TObject *obj);

   TObject         *At(Int_t idx) const;
   TObject         *UncheckedAt(Int_t i) const { return fCont[i]; }
   TObject         *Before(TObject *obj) const;
   TObject         *After(TObject *obj) const;
   TObject         *First() const;
   TObject         *Last() const;
   virtual TObject *&operator[](Int_t i);
   Int_t            LowerBound() const { return fLowerBound; }
   Int_t            IndexOf(const TObject *obj) const;
   void             SetLast(Int_t last);

   virtual void     Sort(Int_t upto = kMaxInt);
   virtual Int_t    BinarySearch(TObject *obj, Int_t upto = kMaxInt); // the TObjArray has to be sorted, -1 == not found !!

   ClassDef(TObjArray,3)  //An array of objects
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjArrayIter                                                        //
//                                                                      //
// Iterator of object array.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TObjArrayIter : public TIterator {

private:
   const TObjArray  *fArray;     //array being iterated
   Int_t             fCursor;    //current position in array
   Bool_t            fDirection; //iteration direction

   TObjArrayIter() : fArray(0) { }

public:
   TObjArrayIter(const TObjArray *arr, Bool_t dir = kIterForward);
   TObjArrayIter(const TObjArrayIter &iter);
   ~TObjArrayIter() { }
   TIterator     &operator=(const TIterator &rhs);
   TObjArrayIter &operator=(const TObjArrayIter &rhs);

   const TCollection *GetCollection() const { return fArray; }
   TObject           *Next();
   void              Reset();

   ClassDef(TObjArrayIter,0)  //Object array iterator
};


//---- inlines -----------------------------------------------------------------

inline Bool_t TObjArray::BoundsOk(const char *where, Int_t at) const
{
   return (at < fLowerBound || at-fLowerBound >= fSize)
                  ? OutOfBoundsError(where, at)
                  : kTRUE;
}

inline TObject *&TObjArray::operator[](Int_t at)
{
   int j = at-fLowerBound;
   if (j >= 0 && j < fSize) return fCont[j];
   BoundsOk("operator[]", at);
   fLast = -2; // invalidate fLast since the result may be used as an lvalue
   return fCont[0];
}

inline TObject *TObjArray::At(Int_t i) const
{
   // Return the object at position i. Returns 0 if i is out of bounds.
   int j = i-fLowerBound;
   if (j >= 0 && j < fSize) return fCont[j];
   BoundsOk("At", i);
   return 0;
}

#endif
