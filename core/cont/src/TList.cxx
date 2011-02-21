// @(#)root/cont:$Id$
// Author: Fons Rademakers   10/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TList                                                                //
//                                                                      //
// A doubly linked list. All classes inheriting from TObject can be     //
// inserted in a TList. Before being inserted into the list the object  //
// pointer is wrapped in a TObjLink object which contains, besides      //
// the object pointer also a previous and next pointer.                 //
//                                                                      //
// There are basically four ways to iterate over a TList (in order      //
// of preference, if not forced by other constraints):                  //
//    1) Using the R__FOR_EACH macro:                                   //
//         GetListOfPrimitives()->R__FOR_EACH(TObject,Paint)(option);   //
//                                                                      //
//    2) Using the TList iterator TListIter (via the wrapper class      //
//       TIter):                                                        //
//         TIter next(GetListOfPrimitives());                           //
//         while ((TObject *obj = next()))                              //
//            obj->Draw(next.GetOption());                              //
//                                                                      //
//    3) Using the TList iterator TListIter and std::for_each           //
//       algorithm:                                                     //
//         // A function object, which will be applied to each element  //
//         // of the given range.                                       //
//         struct STestFunctor {                                        //
//            bool operator()(TObject *aObj) {                          //
//               ...                                                    //
//               return true;                                           //
//            }                                                         //
//        }                                                             //
//        ...                                                           //
//        ...                                                           //
//        TIter iter(mylist);                                           //
//        for_each( iter.Begin(), TIter::End(), STestFunctor() );       //
//                                                                      //
//    4) Using the TObjLink list entries (that wrap the TObject*):      //
//         TObjLink *lnk = GetListOfPrimitives()->FirstLink();          //
//         while (lnk) {                                                //
//            lnk->GetObject()->Draw(lnk->GetOption());                 //
//            lnk = lnk->Next();                                        //
//         }                                                            //
//                                                                      //
//    5) Using the TList's After() and Before() member functions:       //
//         TFree *idcur = this;                                         //
//         while (idcur) {                                              //
//            ...                                                       //
//            ...                                                       //
//            idcur = (TFree*)GetListOfFree()->After(idcur);            //
//         }                                                            //
//                                                                      //
//   Methods 2, 3 and 4 can also easily iterate backwards using either  //
//   a backward TIter (using argument kIterBackward) or by using        //
//   LastLink() and lnk->Prev() or by using the Before() member.        //
//Begin_Html <img src=gif/tlist.gif> End_Html                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TList.h"
#include "TClass.h"

#include <string>
namespace std {} using namespace std;

ClassImp(TList)

//______________________________________________________________________________
TList::~TList()
{
   // Delete the list. Objects are not deleted unless the TList is the
   // owner (set via SetOwner()).

   Clear();
}

//______________________________________________________________________________
void TList::AddFirst(TObject *obj)
{
   // Add object at the beginning of the list.

   if (IsArgNull("AddFirst", obj)) return;

   if (!fFirst) {
      fFirst = NewLink(obj);
      fLast = fFirst;
   } else {
      TObjLink *t = NewLink(obj);
      t->fNext = fFirst;
      fFirst->fPrev = t;
      fFirst = t;
   }
   fSize++;
   Changed();
}

//______________________________________________________________________________
void TList::AddFirst(TObject *obj, Option_t *opt)
{
   // Add object at the beginning of the list and also store option.
   // Storing an option is useful when one wants to change the behaviour
   // of an object a little without having to create a complete new
   // copy of the object. This feature is used, for example, by the Draw()
   // method. It allows the same object to be drawn in different ways.

   if (IsArgNull("AddFirst", obj)) return;

   if (!fFirst) {
      fFirst = NewOptLink(obj, opt);
      fLast = fFirst;
   } else {
      TObjLink *t = NewOptLink(obj, opt);
      t->fNext = fFirst;
      fFirst->fPrev = t;
      fFirst = t;
   }
   fSize++;
   Changed();
}

//______________________________________________________________________________
void TList::AddLast(TObject *obj)
{
   // Add object at the end of the list.

   if (IsArgNull("AddLast", obj)) return;

   if (!fFirst) {
      fFirst = NewLink(obj);
      fLast  = fFirst;
   } else
      fLast = NewLink(obj, fLast);
   fSize++;
   Changed();
}

//______________________________________________________________________________
void TList::AddLast(TObject *obj, Option_t *opt)
{
   // Add object at the end of the list and also store option.
   // Storing an option is useful when one wants to change the behaviour
   // of an object a little without having to create a complete new
   // copy of the object. This feature is used, for example, by the Draw()
   // method. It allows the same object to be drawn in different ways.

   if (IsArgNull("AddLast", obj)) return;

   if (!fFirst) {
      fFirst = NewOptLink(obj, opt);
      fLast  = fFirst;
   } else
      fLast = NewOptLink(obj, opt, fLast);
   fSize++;
   Changed();
}

//______________________________________________________________________________
void TList::AddBefore(const TObject *before, TObject *obj)
{
   // Insert object before object before in the list.

   if (IsArgNull("AddBefore", obj)) return;

   if (!before)
      TList::AddFirst(obj);
   else {
      Int_t    idx;
      TObjLink *t = FindLink(before, idx);
      if (!t) {
         Error("AddBefore", "before not found, object not added");
         return;
      }
      if (t == fFirst)
         TList::AddFirst(obj);
      else {
         NewLink(obj, t->Prev());
         fSize++;
         Changed();
      }
   }
}

//______________________________________________________________________________
void TList::AddBefore(TObjLink *before, TObject *obj)
{
   // Insert object before the specified ObjLink object. If before = 0 then add
   // to the head of the list. An ObjLink can be obtained by looping over a list
   // using the above describe iterator method 3.

   if (IsArgNull("AddBefore", obj)) return;

   if (!before)
      TList::AddFirst(obj);
   else {
      if (before == fFirst)
         TList::AddFirst(obj);
      else {
         NewLink(obj, before->Prev());
         fSize++;
         Changed();
      }
   }
}

//______________________________________________________________________________
void TList::AddAfter(const TObject *after, TObject *obj)
{
   // Insert object after object after in the list.

   if (IsArgNull("AddAfter", obj)) return;

   if (!after)
      TList::AddLast(obj);
   else {
      Int_t    idx;
      TObjLink *t = FindLink(after, idx);
      if (!t) {
         Error("AddAfter", "after not found, object not added");
         return;
      }
      if (t == fLast)
         TList::AddLast(obj);
      else {
         NewLink(obj, t);
         fSize++;
         Changed();
      }
   }
}

//______________________________________________________________________________
void TList::AddAfter(TObjLink *after, TObject *obj)
{
   // Insert object after the specified ObjLink object. If after = 0 then add
   // to the tail of the list. An ObjLink can be obtained by looping over a list
   // using the above describe iterator method 3.

   if (IsArgNull("AddAfter", obj)) return;

   if (!after)
      TList::AddLast(obj);
   else {
      if (after == fLast)
         TList::AddLast(obj);
      else {
         NewLink(obj, after);
         fSize++;
         Changed();
      }
   }
}

//______________________________________________________________________________
void TList::AddAt(TObject *obj, Int_t idx)
{
   // Insert object at position idx in the list.

   if (IsArgNull("AddAt", obj)) return;

   TObjLink *lnk = LinkAt(idx);
   if (!lnk)
      TList::AddLast(obj);
   else if (lnk == fFirst)
      TList::AddFirst(obj);
   else {
      NewLink(obj, lnk->Prev());
      fSize++;
      Changed();
   }
}

//______________________________________________________________________________
TObject *TList::After(const TObject *obj) const
{
   // Returns the object after object obj. Obj is found using the
   // object's IsEqual() method.  Returns 0 if obj is last in list.

   TObjLink *t;

   if (fCache && fCache->GetObject() && fCache->GetObject()->IsEqual(obj)) {
      t = fCache;
      ((TList*)this)->fCache = fCache->Next();  //cast const away, fCache should be mutable
   } else {
      Int_t idx;
      t = FindLink(obj, idx);
      if (t) ((TList*)this)->fCache = t->Next();
   }

   if (t && t->Next())
      return t->Next()->GetObject();
   else
      return 0;
}

//______________________________________________________________________________
TObject *TList::At(Int_t idx) const
{
   // Returns the object at position idx. Returns 0 if idx is out of range.

   TObjLink *lnk = LinkAt(idx);
   if (lnk) return lnk->GetObject();
   return 0;
}

//______________________________________________________________________________
TObject *TList::Before(const TObject *obj) const
{
   // Returns the object before object obj. Obj is found using the
   // object's IsEqual() method.  Returns 0 if obj is first in list.

   TObjLink *t;

   if (fCache && fCache->GetObject() && fCache->GetObject()->IsEqual(obj)) {
      t = fCache;
      ((TList*)this)->fCache = fCache->Prev();  //cast const away, fCache should be mutable
   } else {
      Int_t idx;
      t = FindLink(obj, idx);
      if (t) ((TList*)this)->fCache = t->Prev();
   }

   if (t && t->Prev())
      return t->Prev()->GetObject();
   else
      return 0;
}

//______________________________________________________________________________
void TList::Clear(Option_t *option)
{
   // Remove all objects from the list. Does not delete the objects
   // unless the TList is the owner (set via SetOwner()) and option
   // "nodelete" is not set.
   // If option="nodelete" then don't delete any heap objects that were
   // marked with the kCanDelete bit, otherwise these objects will be
   // deleted (this option is used by THashTable::Clear()).

   Bool_t nodel = option ? (!strcmp(option, "nodelete") ? kTRUE : kFALSE) : kFALSE;

   if (!nodel && IsOwner()) {
      Delete(option);
      return;
   }

   while (fFirst) {
      TObjLink *tlk = fFirst;
      fFirst = fFirst->Next();
      fSize--;
      // delete only heap objects marked OK to clear
      if (!nodel && tlk->GetObject() && tlk->GetObject()->IsOnHeap()) {
         if (tlk->GetObject()->TestBit(kCanDelete)) {
            if(tlk->GetObject()->TestBit(kNotDeleted)) {
               TCollection::GarbageCollect(tlk->GetObject());
            }
         }
      }
      delete tlk;
   }
   fFirst = fLast = fCache = 0;
   fSize  = 0;
   Changed();
}

//______________________________________________________________________________
void TList::Delete(Option_t *option)
{
   // Remove all objects from the list AND delete all heap based objects.
   // If option="slow" then keep list consistent during delete. This allows
   // recursive list operations during the delete (e.g. during the dtor
   // of an object in this list one can still access the list to search for
   // other not yet deleted objects).

   Bool_t slow = option ? (!strcmp(option, "slow") ? kTRUE : kFALSE) : kFALSE;

   TList removeDirectory; // need to deregistere these from their directory

   if (slow) {

      while (fFirst) {
         TObjLink *tlk = fFirst;
         fFirst = fFirst->Next();
         fSize--;
         // delete only heap objects
         if (tlk->GetObject() && tlk->GetObject()->IsOnHeap())
            TCollection::GarbageCollect(tlk->GetObject());
         else if (tlk->GetObject() && tlk->GetObject()->IsA()->GetDirectoryAutoAdd())
            removeDirectory.Add(tlk->GetObject());

         delete tlk;
      }
      fFirst = fLast = fCache = 0;
      fSize  = 0;

   } else {

      TObjLink *first = fFirst;    //pointer to first entry in linked list
      fFirst = fLast = fCache = 0;
      fSize  = 0;
      while (first) {
         TObjLink *tlk = first;
         first = first->Next();
         // delete only heap objects
         if (tlk->GetObject() && tlk->GetObject()->IsOnHeap())
            TCollection::GarbageCollect(tlk->GetObject());
         else if (tlk->GetObject() && tlk->GetObject()->IsA()->GetDirectoryAutoAdd())
            removeDirectory.Add(tlk->GetObject());

         delete tlk;
      }
   }

   // These objects cannot expect to have a valid TDirectory anymore;
   // e.g. because *this is the TDirectory's list of objects. Even if
   // not, they are supposed to be deleted, so we can as well unregister
   // them from their directory, even if they are stack-based:
   TIter iRemDir(&removeDirectory);
   TObject* dirRem = 0;
   while ((dirRem = iRemDir())) {
      (*dirRem->IsA()->GetDirectoryAutoAdd())(dirRem, 0);
   }
   Changed();
}

//______________________________________________________________________________
void TList::DeleteLink(TObjLink *lnk)
{
   // Delete a TObjLink object.

   lnk->fNext = lnk->fPrev = 0;
   lnk->fObject = 0;
   delete lnk;
}

//______________________________________________________________________________
TObject *TList::FindObject(const char *name) const
{
   // Find an object in this list using its name. Requires a sequential
   // scan till the object has been found. Returns 0 if object with specified
   // name is not found. This method overrides the generic FindObject()
   // of TCollection for efficiency reasons.

   if (!name) return 0;
   TObjLink *lnk = FirstLink();
   while (lnk) {
      TObject *obj = lnk->GetObject();
      const char *objname = obj->GetName();
      if (objname && !strcmp(name, objname)) return obj;
      lnk = lnk->Next();
   }
   return 0;
}

//______________________________________________________________________________
TObject *TList::FindObject(const TObject *obj) const
{
   // Find an object in this list using the object's IsEqual()
   // member function. Requires a sequential scan till the object has
   // been found. Returns 0 if object is not found.
   // This method overrides the generic FindObject() of TCollection for
   // efficiency reasons.

   TObjLink *lnk = FirstLink();

   while (lnk) {
      TObject *ob = lnk->GetObject();
      if (ob->IsEqual(obj)) return ob;
      lnk = lnk->Next();
   }
   return 0;
}

//______________________________________________________________________________
TObjLink *TList::FindLink(const TObject *obj, Int_t &idx) const
{
   // Returns the TObjLink object that contains object obj. In idx it returns
   // the position of the object in the list.

   if (!fFirst) return 0;

   TObject *object;
   TObjLink *lnk = fFirst;
   idx = 0;

   while (lnk) {
      object = lnk->GetObject();
      if (object) {
         if (object->TestBit(kNotDeleted)) {
            if (object->IsEqual(obj)) return lnk;
         }
      }
      lnk = lnk->Next();
      idx++;
   }
   return 0;
}

//______________________________________________________________________________
TObject *TList::First() const
{
   // Return the first object in the list. Returns 0 when list is empty.

   if (fFirst) return fFirst->GetObject();
   return 0;
}

//______________________________________________________________________________
TObject **TList::GetObjectRef(const TObject *obj) const
{
   // Return address of pointer to obj

   TObjLink *lnk = FirstLink();

   while (lnk) {
      TObject *ob = lnk->GetObject();
      if (ob->IsEqual(obj)) return lnk->GetObjectRef();
      lnk = lnk->Next();
   }
   return 0;
}

//______________________________________________________________________________
TObject *TList::Last() const
{
   // Return the last object in the list. Returns 0 when list is empty.

   if (fLast) return fLast->GetObject();
   return 0;
}

//______________________________________________________________________________
TObjLink *TList::LinkAt(Int_t idx) const
{
   // Return the TObjLink object at index idx.

   Int_t    i = 0;
   TObjLink *lnk = fFirst;
   while (i < idx && lnk) {
      i++;
      lnk = lnk->Next();
   }
   return lnk;
}

//______________________________________________________________________________
TIterator *TList::MakeIterator(Bool_t dir) const
{
   // Return a list iterator.

   return new TListIter(this, dir);
}

//______________________________________________________________________________
TObjLink *TList::NewLink(TObject *obj, TObjLink *prev)
{
   // Return a new TObjLink.

   if (prev)
      return new TObjLink(obj, prev);
   else
      return new TObjLink(obj);
}

//______________________________________________________________________________
TObjLink *TList::NewOptLink(TObject *obj, Option_t *opt, TObjLink *prev)
{
   // Return a new TObjOptLink (a TObjLink that also stores the option).

   if (prev)
      return new TObjOptLink(obj, prev, opt);
   else
      return new TObjOptLink(obj, opt);
}

//______________________________________________________________________________
void TList::RecursiveRemove(TObject *obj)
{
   // Remove object from this collection and recursively remove the object
   // from all other objects (and collections).

   if (!obj) return;

   TObjLink *lnk  = fFirst;
   TObjLink *next = 0;
   while (lnk) {
      next = lnk->Next();
      TObject *ob = lnk->GetObject();
      if (ob->TestBit(kNotDeleted)) {
         if (ob->IsEqual(obj)) {
            if (lnk == fFirst) {
               fFirst = next;
               if (lnk == fLast)
                  fLast = fFirst;
               else
                  fFirst->fPrev = 0;
               DeleteLink(lnk);
            } else if (lnk == fLast) {
               fLast = lnk->Prev();
               fLast->fNext = 0;
               DeleteLink(lnk);
            } else {
               lnk->Prev()->fNext = next;
               lnk->Next()->fPrev = lnk->Prev();
               DeleteLink(lnk);
            }
            fSize--;
            fCache = 0;
            Changed();
         } else
            ob->RecursiveRemove(obj);
      }
      lnk = next;
   }
}

//______________________________________________________________________________
TObject *TList::Remove(TObject *obj)
{
   // Remove object from the list.

   if (!obj) return 0;

   Int_t    idx;
   TObjLink *lnk = FindLink(obj, idx);

   if (!lnk) return 0;

   // return object found, which may be (pointer wise) different than the
   // input object (depending on what IsEqual() is doing)
   TObject *ob = lnk->GetObject();

   if (lnk == fFirst) {
      fFirst = lnk->Next();
      if (lnk == fLast)
         fLast = fFirst;
      else
         fFirst->fPrev = 0;
      DeleteLink(lnk);
   } else if (lnk == fLast) {
      fLast = lnk->Prev();
      fLast->fNext = 0;
      DeleteLink(lnk);
   } else {
      lnk->Prev()->fNext = lnk->Next();
      lnk->Next()->fPrev = lnk->Prev();
      DeleteLink(lnk);
   }
   fSize--;
   fCache = 0;
   Changed();

   return ob;
}

//______________________________________________________________________________
TObject *TList::Remove(TObjLink *lnk)
{
   // Remove object link (and therefore the object it contains)
   // from the list.

   if (!lnk) return 0;

   TObject *obj = lnk->GetObject();

   if (lnk == fFirst) {
      fFirst = lnk->Next();
      if (lnk == fLast)
         fLast = fFirst;
      else
         fFirst->fPrev = 0;
      DeleteLink(lnk);
   } else if (lnk == fLast) {
      fLast = lnk->Prev();
      fLast->fNext = 0;
      DeleteLink(lnk);
   } else {
      lnk->Prev()->fNext = lnk->Next();
      lnk->Next()->fPrev = lnk->Prev();
      DeleteLink(lnk);
   }
   fSize--;
   fCache = 0;
   Changed();

   return obj;
}

//______________________________________________________________________________
void TList::RemoveLast()
{
   // Remove the last object of the list.

   TObjLink *lnk = fLast;
   if (!lnk) return;

   if (lnk == fFirst) {
      fFirst = 0;
      fLast = 0;
   } else {
      fLast = lnk->Prev();
      fLast->fNext = 0;
   }
   DeleteLink(lnk);

   fSize--;
   fCache = 0;
   Changed();
}

//______________________________________________________________________________
void TList::Sort(Bool_t order)
{
   // Sort linked list. Real sorting is done in private function DoSort().
   // The list can only be sorted when is contains objects of a sortable
   // class.

   if (!fFirst) return;

   fAscending = order;

   if (!fFirst->GetObject()->IsSortable()) {
      Error("Sort", "objects in list are not sortable");
      return;
   }

   DoSort(&fFirst, fSize);

   // correct back links
   TObjLink *ol, *lnk = fFirst;

   if (lnk) lnk->fPrev = 0;
   while ((ol = lnk)) {
      lnk = lnk->fNext;
      if (lnk)
         lnk->fPrev = ol;
      else
         fLast = ol;
   }
   fSorted = kTRUE;
}

//______________________________________________________________________________
Bool_t TList::LnkCompare(TObjLink *l1, TObjLink *l2)
{
   // Compares the objects stored in the TObjLink objects.
   // Depending on the flag IsAscending() the function returns
   // true if the object in l1 <= l2 (ascending) or l2 <= l1 (descending).

   Int_t cmp = l1->GetObject()->Compare(l2->GetObject());

   if ((IsAscending() && cmp <=0) || (!IsAscending() && cmp > 0))
      return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
TObjLink **TList::DoSort(TObjLink **head, Int_t n)
{
   // Sort linked list.

   TObjLink *p1, *p2, **h2, **t2;

   switch (n) {
      case 0:
         return head;

      case 1:
         return &((*head)->fNext);

      case 2:
         p2 = (p1 = *head)->fNext;
         if (LnkCompare(p1, p2)) return &(p2->fNext);
         p1->fNext = (*head = p2)->fNext;
         return &((p2->fNext = p1)->fNext);
   }

   int m;
   n -= m = n / 2;

   t2 = DoSort(h2 = DoSort(head, n), m);

   if (LnkCompare((p1 = *head), (p2 = *h2))) {
      do {
         if (!--n) return *h2 = p2, t2;
      } while (LnkCompare((p1 = *(head = &(p1->fNext))), p2));
   }

   while (1) {
      *head = p2;
      do {
         if (!--m) return *h2 = *t2, *t2 = p1, h2;
      } while (!LnkCompare(p1, (p2 = *(head = &(p2->fNext)))));
      *head = p1;
      do {
         if (!--n) return *h2 = p2, t2;
      } while (LnkCompare((p1 = *(head = &(p1->fNext))), p2));
   }
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjLink                                                             //
//                                                                      //
// Wrapper around a TObject so it can be stored in a TList.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TObjLink::TObjLink(TObject *obj, TObjLink *prev)
          : fNext(prev->fNext), fPrev(prev), fObject(obj)
{
   // Create a new TObjLink.

   fPrev->fNext = this;
   if (fNext) fNext->fPrev = this;
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListIter                                                            //
//                                                                      //
// Iterator of linked list.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TListIter)

//______________________________________________________________________________
TListIter::TListIter(const TList *l, Bool_t dir)
        : fList(l), fCurCursor(0), fCursor(0), fDirection(dir), fStarted(kFALSE)
{
   // Create a new list iterator. By default the iteration direction
   // is kIterForward. To go backward use kIterBackward.
}

//______________________________________________________________________________
TListIter::TListIter(const TListIter &iter) : TIterator(iter)
{
   // Copy ctor.

   fList      = iter.fList;
   fCurCursor = iter.fCurCursor;
   fCursor    = iter.fCursor;
   fDirection = iter.fDirection;
   fStarted   = iter.fStarted;
}

//______________________________________________________________________________
TIterator &TListIter::operator=(const TIterator &rhs)
{
   // Overridden assignment operator.

   if (this != &rhs && rhs.IsA() == TListIter::Class()) {
      const TListIter &rhs1 = (const TListIter &)rhs;
      fList      = rhs1.fList;
      fCurCursor = rhs1.fCurCursor;
      fCursor    = rhs1.fCursor;
      fDirection = rhs1.fDirection;
      fStarted   = rhs1.fStarted;
   }
   return *this;
}

//______________________________________________________________________________
TListIter &TListIter::operator=(const TListIter &rhs)
{
   // Overloaded assignment operator.

   if (this != &rhs) {
      fList      = rhs.fList;
      fCurCursor = rhs.fCurCursor;
      fCursor    = rhs.fCursor;
      fDirection = rhs.fDirection;
      fStarted   = rhs.fStarted;
   }
   return *this;
}

//______________________________________________________________________________
TObject *TListIter::Next()
{
   // Return next object in the list. Returns 0 when no more objects in list.

   if (!fList) return 0;

   if (fDirection == kIterForward) {
      if (!fStarted) {
         fCursor = fList->fFirst;
         fStarted = kTRUE;
      }
      fCurCursor = fCursor;
      if (fCursor) fCursor = fCursor->Next();
   } else {
      if (!fStarted) {
         fCursor = fList->fLast;
         fStarted = kTRUE;
      }
      fCurCursor = fCursor;
      if (fCursor) fCursor = fCursor->Prev();
   }

   if (fCurCursor) return fCurCursor->GetObject();
   return 0;
}

//______________________________________________________________________________
Option_t *TListIter::GetOption() const
{
   // Returns the object option stored in the list.

   if (fCurCursor) return fCurCursor->GetOption();
   return "";
}

//______________________________________________________________________________
void TListIter::SetOption(Option_t *option)
{
   // Sets the object option stored in the list.

   if (fCurCursor) fCurCursor->SetOption(option);
}

//______________________________________________________________________________
void TListIter::Reset()
{
   // Reset list iterator.

   fStarted = kFALSE;
}

//______________________________________________________________________________
bool TListIter::operator!=(const TIterator &aIter) const
{
   // This operator compares two TIterator objects.

   if (nullptr == (&aIter))
      return fCurCursor;

   if ((aIter.IsA() == TListIter::Class())) {
      const TListIter &iter(dynamic_cast<const TListIter &>(aIter));
      return (fCurCursor != iter.fCurCursor);
   }
   return false; // for base class we don't implement a comparison
}

//______________________________________________________________________________
bool TListIter::operator!=(const TListIter &aIter) const
{
   // This operator compares two TListIter objects.

   if (nullptr == (&aIter))
      return fCurCursor;

   return (fCurCursor != aIter.fCurCursor);
}

//_______________________________________________________________________
void TList::Streamer(TBuffer &b)
{
   // Stream all objects in the collection to or from the I/O buffer.

   Int_t nobjects;
   UChar_t nch;
   Int_t nbig;
   TObject *obj;
   UInt_t R__s, R__c;

   if (b.IsReading()) {
      Version_t v = b.ReadVersion(&R__s, &R__c);
      if (v > 3) {
         TObject::Streamer(b);
         fName.Streamer(b);
         b >> nobjects;
         string readOption;
         for (Int_t i = 0; i < nobjects; i++) {
            b >> obj;
            b >> nch;
            if (v > 4 && nch == 255)  {
               b >> nbig;
            } else {
               nbig = nch;
            }
            readOption.resize(nbig,'\0');
            b.ReadFastArray((char*) readOption.data(),nbig);
            if (obj) { // obj can be null if the class had a custom streamer and we do not have the shared library nor a streamerInfo.
               if (nch) {
                  Add(obj,readOption.c_str());
               } else {
                  Add(obj);
               }
            }
         }
         b.CheckByteCount(R__s, R__c,TList::IsA());
         return;
      }

      //  process old versions when TList::Streamer was in TCollection::Streamer
      if (v > 2)
         TObject::Streamer(b);
      if (v > 1)
         fName.Streamer(b);
      b >> nobjects;
      for (Int_t i = 0; i < nobjects; i++) {
         b >> obj;
         Add(obj);
      }
      b.CheckByteCount(R__s, R__c,TList::IsA());

   } else {
      R__c = b.WriteVersion(TList::IsA(), kTRUE);
      TObject::Streamer(b);
      fName.Streamer(b);
      nobjects = GetSize();
      b << nobjects;

      TObjLink *lnk = fFirst;
      while (lnk) {
         obj = lnk->GetObject();
         b << obj;

         nbig = strlen(lnk->GetAddOption());
         if (nbig > 254) {
            nch = 255;
            b << nch;
            b << nbig;
         } else {
            nch = UChar_t(nbig);
            b << nch;
         }
         b.WriteFastArray(lnk->GetAddOption(),nbig);

         lnk = lnk->Next();
      }
      b.SetByteCount(R__c, kTRUE);
   }
}
