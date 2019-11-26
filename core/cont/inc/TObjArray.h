// @(#)root/cont:$Id$
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

#include "TSeqCollection.h"

#include <iterator>

#if (__GNUC__ >= 3) && !defined(__INTEL_COMPILER)
// Prevent -Weffc++ from complaining about the inheritance
// TObjArrayIter from std::iterator.
#pragma GCC system_header
#endif

class TObjArrayIter;

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
   typedef TObjArrayIter Iterator_t;

   TObjArray(Int_t s = TCollection::kInitCapacity, Int_t lowerBound = 0);
   TObjArray(const TObjArray &a);
   virtual          ~TObjArray();
   TObjArray& operator=(const TObjArray&);
   virtual void     Clear(Option_t *option="");
   virtual void     Compress();
   virtual void     Delete(Option_t *option="");
   virtual void     Expand(Int_t newSize);   // expand or shrink an array
   Int_t            GetEntries() const;
   Int_t            GetEntriesFast() const {
      return GetAbsLast() + 1;   //only OK when no gaps
   }
   Int_t            GetEntriesUnsafe() const;
   Int_t            GetLast() const;
   TObject        **GetObjectRef() const { return fCont; };
   TObject        **GetObjectRef(const TObject *obj) const;
   Bool_t           IsEmpty() const { return GetAbsLast() == -1; }
   TIterator       *MakeIterator(Bool_t dir = kIterForward) const;

   void             Add(TObject *obj) { AddLast(obj); }
   virtual void     AddFirst(TObject *obj);
   virtual void     AddLast(TObject *obj);
   virtual void     AddAt(TObject *obj, Int_t idx);
   virtual void     AddAtAndExpand(TObject *obj, Int_t idx);
   virtual Int_t    AddAtFree(TObject *obj);
   virtual void     AddAfter(const TObject *after, TObject *obj);
   virtual void     AddBefore(const TObject *before, TObject *obj);
   virtual TObject *FindObject(const char *name) const;
   virtual TObject *FindObject(const TObject *obj) const;
   virtual TObject *RemoveAt(Int_t idx);
   virtual TObject *Remove(TObject *obj);
   virtual void     RemoveRange(Int_t idx1, Int_t idx2);
   virtual void     RecursiveRemove(TObject *obj);

   TObject         *At(Int_t idx) const;
   TObject         *UncheckedAt(Int_t i) const { return fCont[i-fLowerBound]; }
   TObject         *Before(const TObject *obj) const;
   TObject         *After(const TObject *obj) const;
   TObject         *First() const;
   TObject         *Last() const;
   virtual TObject *&operator[](Int_t i);
   virtual TObject *operator[](Int_t i) const;
   Int_t            LowerBound() const { return fLowerBound; }
   Int_t            IndexOf(const TObject *obj) const;
   void             SetLast(Int_t last);

   virtual void     Randomize(Int_t ntimes=1);
   virtual void     Sort(Int_t upto = kMaxInt);
   virtual Int_t    BinarySearch(TObject *obj, Int_t upto = kMaxInt); // the TObjArray has to be sorted, -1 == not found !!

   ClassDef(TObjArray,3)  //An array of objects
};


// Preventing warnings with -Weffc++ in GCC since it is a false positive for the TObjArrayIter destructor.
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjArrayIter                                                        //
//                                                                      //
// Iterator of object array.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TObjArrayIter : public TIterator,
                      public std::iterator<std::bidirectional_iterator_tag, // TODO: ideally it should be a  randomaccess_iterator_tag
                                           TObject*, std::ptrdiff_t,
                                           const TObject**, const TObject*&> {

private:
   const TObjArray  *fArray;     //array being iterated
   Int_t             fCurCursor; //current position in array
   Int_t             fCursor;    //next position in array
   Bool_t            fDirection; //iteration direction

   TObjArrayIter() : fArray(0), fCurCursor(0), fCursor(0), fDirection(kIterForward) { }

public:
   TObjArrayIter(const TObjArray *arr, Bool_t dir = kIterForward);
   TObjArrayIter(const TObjArrayIter &iter);
   ~TObjArrayIter() { }
   TIterator     &operator=(const TIterator &rhs);
   TObjArrayIter &operator=(const TObjArrayIter &rhs);

   const TCollection *GetCollection() const { return fArray; }
   TObject           *Next();
   void               Reset();
   Bool_t             operator!=(const TIterator &aIter) const;
   Bool_t             operator!=(const TObjArrayIter &aIter) const;
   TObject           *operator*() const;

   ClassDef(TObjArrayIter,0)  //Object array iterator
};

#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic pop
#endif

//---- inlines -----------------------------------------------------------------

inline Bool_t TObjArray::BoundsOk(const char *where, Int_t at) const
{
   return (at < fLowerBound || at-fLowerBound >= fSize)
                  ? OutOfBoundsError(where, at)
                  : kTRUE;
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
