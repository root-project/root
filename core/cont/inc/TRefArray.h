// @(#)root/cont:$Id$
// Author: Rene Brun    02/10/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRefArray
#define ROOT_TRefArray


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRefArray                                                            //
//                                                                      //
// An array of references to TObjects.                                  //
// The array expands automatically when adding elements.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSeqCollection
#include "TSeqCollection.h"
#endif
#ifndef ROOT_TProcessID
#include "TProcessID.h"
#endif

#include <iterator>

#if (__GNUC__ >= 3) && !defined(__INTEL_COMPILER)
// Prevent -Weffc++ from complaining about the inheritance
// TRefArrayIter from std::iterator.
#pragma GCC system_header
#endif

class TSystem;
class TRefArrayIter;

class TRefArray : public TSeqCollection {

friend class TRefArrayIter;

protected:
   TProcessID   *fPID;         //Pointer to Process Unique Identifier
   UInt_t       *fUIDs;        //[fSize] To store uids of referenced objects
   Int_t         fLowerBound;  //Lower bound of the array
   Int_t         fLast;        //Last element in array containing an object

   Bool_t        BoundsOk(const char *where, Int_t at) const;
   void          Init(Int_t s, Int_t lowerBound);
   Bool_t        OutOfBoundsError(const char *where, Int_t i) const;
   Int_t         GetAbsLast() const;
   TObject      *GetFromTable(Int_t idx) const;

public:
   typedef TRefArrayIter Iterator_t;

   TRefArray(TProcessID *pid = 0);
   TRefArray(Int_t s, TProcessID *pid);
   TRefArray(Int_t s, Int_t lowerBound = 0, TProcessID *pid = 0);
   TRefArray(const TRefArray &a);
   TRefArray& operator=(const TRefArray &a);
   virtual          ~TRefArray();
   virtual void     Clear(Option_t *option="");
   virtual void     Compress();
   virtual void     Delete(Option_t *option="");
   virtual void     Expand(Int_t newSize);   // expand or shrink an array
   Int_t            GetEntries() const;
   Int_t            GetEntriesFast() const {
      return GetAbsLast() + 1;   //only OK when no gaps
   }
   Int_t            GetLast() const;
   TObject        **GetObjectRef(const TObject *obj) const;
   TProcessID      *GetPID() const {return fPID;}
   UInt_t           GetUID(Int_t at) const;
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
   virtual TObject *RemoveAt(Int_t idx);
   virtual TObject *Remove(TObject *obj);

   TObject         *At(Int_t idx) const;
   TObject         *Before(const TObject *obj) const;
   TObject         *After(const TObject *obj) const;
   TObject         *First() const;
   TObject         *Last() const;
   virtual TObject *operator[](Int_t i) const;
   Int_t            LowerBound() const { return fLowerBound; }
   Int_t            IndexOf(const TObject *obj) const;
   void             SetLast(Int_t last);

   virtual void     Sort(Int_t upto = kMaxInt);
   virtual Int_t    BinarySearch(TObject *obj, Int_t upto = kMaxInt); // the TRefArray has to be sorted, -1 == not found !!

   ClassDef(TRefArray,1)  //An array of references to TObjects
};


// Preventing warnings with -Weffc++ in GCC since it is a false positive for the TRefArrayIter destructor.
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRefArrayIter                                                        //
//                                                                      //
// Iterator of object array.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TRefArrayIter : public TIterator,
                      public std::iterator<std::bidirectional_iterator_tag, // TODO: ideally it should be a  randomaccess_iterator_tag
                                           TObject*, std::ptrdiff_t,
                                           const TObject**, const TObject*&> {

private:
   const TRefArray  *fArray;      //array being iterated
   Int_t             fCurCursor;  //current position in array
   Int_t             fCursor;     //next position in array
   Bool_t            fDirection;  //iteration direction

   TRefArrayIter() : fArray(0), fCurCursor(0), fCursor(0), fDirection(kIterForward) { }

public:
   TRefArrayIter(const TRefArray *arr, Bool_t dir = kIterForward);
   TRefArrayIter(const TRefArrayIter &iter);
   ~TRefArrayIter() { }
   TIterator         &operator=(const TIterator &rhs);
   TRefArrayIter     &operator=(const TRefArrayIter &rhs);

   const TCollection *GetCollection() const { return fArray; }
   TObject           *Next();
   void               Reset();
   bool               operator!=(const TIterator &aIter) const;
   bool               operator!=(const TRefArrayIter &aIter) const;
   TObject           *operator*() const;

   ClassDef(TRefArrayIter,0)  //Object array iterator
};

#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic pop
#endif


//---- inlines -----------------------------------------------------------------

inline Bool_t TRefArray::BoundsOk(const char *where, Int_t at) const
{
   return (at < fLowerBound || at-fLowerBound >= fSize)
                  ? OutOfBoundsError(where, at)
                  : kTRUE;
}

inline TObject *TRefArray::operator[](Int_t at) const
{
   int j = at-fLowerBound;
   if (j >= 0 && j < fSize) {
      if (!fPID) return 0;
      if (!TProcessID::IsValid(fPID)) return 0;
      TObject *obj = fPID->GetObjectWithID(fUIDs[j]);
      if (obj==0) obj = GetFromTable(j);
      return obj;
   }
   BoundsOk("At", at);
   return 0;
}

inline TObject *TRefArray::At(Int_t at) const
{
   // Return the object at position i. Returns 0 if i is out of bounds.
   int j = at-fLowerBound;
   if (j >= 0 && j < fSize) {
      if (!fPID) return 0;
      if (!TProcessID::IsValid(fPID)) return 0;
      TObject *obj = fPID->GetObjectWithID(fUIDs[j]);
      if (obj==0) obj = GetFromTable(j);
      return obj;
   }
   BoundsOk("At", at);
   return 0;
}

#endif
