// @(#)root/cont:$Name:  $:$Id: TClonesArray.cxx,v 1.62 2007/01/28 19:43:48 brun Exp $
// Author: Rene Brun   11/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// An array of clone (identical) objects. Memory for the objects        //
// stored in the array is allocated only once in the lifetime of the    //
// clones array. All objects must be of the same class. For the rest    //
// this class has the same properties as TObjArray.                     //
//                                                                      //
// To reduce the very large number of new and delete calls in large     //
// loops like this (O(100000) x O(10000) times new/delete):             //
//                                                                      //
//   TObjArray a(10000);                                                //
//   while (TEvent *ev = (TEvent *)next()) {      // O(100000) events   //
//      for (int i = 0; i < ev->Ntracks; i++) {   // O(10000) tracks    //
//         a[i] = new TTrack(x,y,z,...);                                //
//         ...                                                          //
//         ...                                                          //
//      }                                                               //
//      ...                                                             //
//      a.Delete();                                                     //
//   }                                                                  //
//                                                                      //
// One better uses a TClonesArray which reduces the number of           //
// new/delete calls to only O(10000):                                   //
//                                                                      //
//   TClonesArray a("TTrack", 10000);                                   //
//   while (TEvent *ev = (TEvent *)next()) {      // O(100000) events   //
//      for (int i = 0; i < ev->Ntracks; i++) {   // O(10000) tracks    //
//         new(a[i]) TTrack(x,y,z,...);                                 //
//         ...                                                          //
//         ...                                                          //
//      }                                                               //
//      ...                                                             //
//      a.Delete();                                                     //
//   }                                                                  //
//                                                                      //
// Note: the only supported way to add objects to a TClonesArray is     //
// via the new with placement method. The diffrent Add() methods of     //
// TObjArray and its base classes are not allowed.                      //
//                                                                      //
// Considering that a new/delete costs about 70 mus on a 300 MHz HP,    //
// O(10^9) new/deletes will save about 19 hours.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include "TClonesArray.h"
#include "TError.h"
#include "TROOT.h"
#include "TClass.h"
#include "TObjectTable.h"


ClassImp(TClonesArray)

//______________________________________________________________________________
TClonesArray::TClonesArray() : TObjArray()
{
   // Default Constructor.

   fClass      = 0;
   fKeep       = 0;
}

//______________________________________________________________________________
TClonesArray::TClonesArray(const char *classname, Int_t s, Bool_t) : TObjArray(s)
{
   // Create an array of clone objects of classname. The class must inherit from
   // TObject. If the class defines an own operator delete(), make sure that
   // it looks like this:
   //
   //    void MyClass::operator delete(void *vp)
   //    {
   //       if ((Long_t) vp != TObject::GetDtorOnly())
   //          ::operator delete(vp);       // delete space
   //       else
   //          TObject::SetDtorOnly(0);
   //    }
   //
   // The second argument s indicates an approximate number of objects
   // that will be entered in the array. If more than s objects are entered,
   // the array will be automatically expanded.
   //
   // The third argument is not used anymore and only there for backward
   // compatibility reasons.
   //
   // In case you want to send a TClonesArray (or object containing a
   // TClonesArray) via a TMessage over a TSocket don't forget to call
   // BypassStreamer(kFALSE). See TClonesArray::BypassStreamer().

   if (!gROOT)
      ::Fatal("TClonesArray::TClonesArray", "ROOT system not initialized");

   fClass = gROOT->GetClass(classname);
   char *name = new char[strlen(classname)+2];
   sprintf(name, "%ss", classname);
   SetName(name);
   delete [] name;

   fKeep = new TObjArray(s);

   BypassStreamer(kTRUE);

   if (!fClass) {
      MakeZombie();
      Error("TClonesArray", "%s is not a valid class name", classname);
      return;
   }
   if (!fClass->InheritsFrom(TObject::Class())) {
      MakeZombie();
      Error("TClonesArray", "%s does not inherit from TObject", classname);
      return;
   }
}

//______________________________________________________________________________
TClonesArray::TClonesArray(const TClass *cl, Int_t s, Bool_t) : TObjArray(s)
{
   // Create an array of clone objects of class cl. The class must inherit from
   // TObject. If the class defines an own operator delete(), make sure that
   // it looks like this:
   //
   //    void MyClass::operator delete(void *vp)
   //    {
   //       if ((Long_t) vp != TObject::GetDtorOnly())
   //          ::operator delete(vp);       // delete space
   //       else
   //          TObject::SetDtorOnly(0);
   //    }
   //
   // The second argument s indicates an approximate number of objects
   // that will be entered in the array. If more than s objects are entered,
   // the array will be automatically expanded.
   //
   // The third argument is not used anymore and only there for backward
   // compatibility reasons.
   //
   // In case you want to send a TClonesArray (or object containing a
   // TClonesArray) via a TMessage over a TSocket don't forget to call
   // BypassStreamer(kFALSE). See TClonesArray::BypassStreamer().

   if (!gROOT)
      ::Fatal("TClonesArray::TClonesArray", "ROOT system not initialized");

   fKeep  = 0;
   fClass = (TClass*)cl;
   if (!fClass) {
      MakeZombie();
      Error("TClonesArray", "called with a null pointer");
      return;
   }
   const char *classname = fClass->GetName();
   if (!fClass->InheritsFrom(TObject::Class())) {
      MakeZombie();
      Error("TClonesArray", "%s does not inherit from TObject", classname);
      return;
   }
   char *name = new char[strlen(classname)+2];
   sprintf(name, "%ss", classname);
   SetName(name);
   delete [] name;

   fKeep = new TObjArray(s);

   BypassStreamer(kTRUE);
}

//______________________________________________________________________________
TClonesArray::TClonesArray(const TClonesArray& tc): TObjArray(tc)
{
   // Copy ctor.

   fKeep = new TObjArray(tc.fSize);
   fClass = tc.fClass;

   BypassStreamer(kTRUE);

   for (Int_t i = 0; i < fSize; i++) {
      if (tc.fCont[i]) fCont[i] = tc.fCont[i]->Clone();
      fKeep->fCont[i] = fCont[i];
   }
}

//______________________________________________________________________________
TClonesArray& TClonesArray::operator=(const TClonesArray& tc)
{
   // Assignment operator.

   if (this == &tc) return *this;

   if (fClass != tc.fClass) {
      Error("operator=", "cannot copy TClonesArray's when classes are different");
      return *this;
   }

   if (tc.fSize > fSize)
      Expand(TMath::Max(tc.fSize, GrowBy(fSize)));

   Int_t i;

   for (i = 0; i < fSize; i++)
      if (fKeep->fCont[i]) {
         if (TObject::GetObjectStat() && gObjectTable)
            gObjectTable->RemoveQuietly(fKeep->fCont[i]);
         ::operator delete(fKeep->fCont[i]);
         fKeep->fCont[i] = 0;
         fCont[i] = 0;
      }

   BypassStreamer(kTRUE);

   for (i = 0; i < tc.fSize; i++) {
      if (tc.fCont[i]) fKeep->fCont[i] = tc.fCont[i]->Clone();
      fCont[i] = fKeep->fCont[i];
   }

   fLast = tc.fSize - 1;
   Changed();
   return *this;
}

//______________________________________________________________________________
TClonesArray::~TClonesArray()
{
   // Delete a clones array.

   if (fKeep) {
      for (Int_t i = 0; i < fKeep->fSize; i++) {
         TObject* p = fKeep->fCont[i];
         if (p && p->TestBit(kNotDeleted)) {
            // -- The TObject destructor has not been called.
            fClass->Destructor(p);
            fKeep->fCont[i] = 0;
         } else {
            // -- The TObject destructor was called, just free memory.
            //
            // remove any possible entries from the ObjectTable
            if (TObject::GetObjectStat() && gObjectTable) {
               gObjectTable->RemoveQuietly(p);
            }
            ::operator delete(p);
            fKeep->fCont[i] = 0;
         }
      }
   }
   SafeDelete(fKeep);

   // Protect against erroneously setting of owner bit
   SetOwner(kFALSE);
}

//______________________________________________________________________________
void TClonesArray::BypassStreamer(Bool_t bypass)
{
   // When the kBypassStreamer bit is set, the automatically
   // generated Streamer can call directly TClass::WriteBuffer.
   // Bypassing the Streamer improves the performance when writing/reading
   // the objects in the TClonesArray. However there is a drawback:
   // When a TClonesArray is written with split=0 bypassing the Streamer,
   // the StreamerInfo of the class in the array being optimized,
   // one cannot use later the TClonesArray with split>0. For example,
   // there is a problem with the following scenario:
   //  1- A class Foo has a TClonesArray of Bar objects
   //  2- The Foo object is written with split=0 to Tree T1.
   //     In this case the StreamerInfo for the class Bar is created
   //     in optimized mode in such a way that data members of the same type
   //     are written as an array improving the I/O performance.
   //  3- In a new program, T1 is read and a new Tree T2 is created
   //     with the object Foo in split>1
   //  4- When the T2 branch is created, the StreamerInfo for the class Bar
   //     is created with no optimization (mandatory for the split mode).
   //     The optimized Bar StreamerInfo is going to be used to read
   //     the TClonesArray in T1. The result will be Bar objects with
   //     data member values not in the right sequence.
   // The solution to this problem is to call BypassStreamer(kFALSE)
   // for the TClonesArray. In this case, the normal Bar::Streamer function
   // will be called. The Bar::Streamer function works OK independently
   // if the Bar StreamerInfo had been generated in optimized mode or not.
   // In case you want to send Foo via a TMessage over a TSocket you also
   // need to disable the streamer bypass.

   if (bypass)
      SetBit(kBypassStreamer);
   else
      ResetBit(kBypassStreamer);
}

//______________________________________________________________________________
void TClonesArray::Compress()
{
   // Remove empty slots from array.

   Int_t j = 0, je = 0;

   TObject **tmp = new TObject* [fSize];

   for (Int_t i = 0; i < fSize; i++) {
      if (fCont[i]) {
         fCont[j] = fCont[i];
         fKeep->fCont[j] = fKeep->fCont[i];
         j++;
      } else {
         tmp[je] = fKeep->fCont[i];
         je++;
      }
   }

   fLast = j - 1;

   Int_t jf = 0;
   for ( ; j < fSize; j++) {
      fCont[j] = 0;
      fKeep->fCont[j] = tmp[jf];
      jf++;
   }

   delete [] tmp;

   R__ASSERT(je == jf);
}

//______________________________________________________________________________
void TClonesArray::Clear(Option_t *option)
{
   // Clear the clones array. Only use this routine when your objects don't
   // allocate memory since it will not call the object dtors.
   // However, if the class in the TClonesArray implements the function
   // Clear(Option_t *option) and if option = "C" the function Clear()
   // is called for all objects in the array. In the function Clear(), one
   // can delete objects or dynamic arrays allocated in the class.
   // This procedure is much faster than calling TClonesArray::Delete().

   if (option && option[0] == 'C') {
      Int_t n = GetEntriesFast();
      for (Int_t i = 0; i < n; i++) {
         TObject *obj = UncheckedAt(i);
         if (obj) obj->Clear();
      }
   }

   // Protect against erroneously setting of owner bit
   SetOwner(kFALSE);

   TObjArray::Clear();
}

//______________________________________________________________________________
void TClonesArray::Delete(Option_t *)
{
   // Clear the clones array. Use this routine when your objects allocate
   // memory (e.g. objects inheriting from TNamed or containing TStrings
   // allocate memory). If not you better use Clear() since if is faster.

   Long_t dtoronly = TObject::GetDtorOnly();
   for (Int_t i = 0; i < fSize; i++) {
      if (fCont[i] && fCont[i]->TestBit(kNotDeleted)) {
         // Tell custom operator delete() not to delete space when
         // object fCont[i] is deleted. Only destructors are called
         // for this object.
         TObject::SetDtorOnly(fCont[i]);
         delete fCont[i];
      }
   }
   // Restore the state.
   TObject::SetDtorOnly((void*)dtoronly);

   // Protect against erroneously setting of owner bit.
   SetOwner(kFALSE);

   TObjArray::Clear();
}

//______________________________________________________________________________
void TClonesArray::Expand(Int_t newSize)
{
   // Expand or shrink the array to newSize elements.

   if (newSize < 0) {
      Error ("Expand", "newSize must be positive (%d)", newSize);
      return;
   }
   if (newSize == fSize)
      return;
   if (newSize < fSize) {
      // release allocated space in fKeep and set to 0 so
      // Expand() will shrink correctly
      for (int i = newSize; i < fSize; i++)
         if (fKeep->fCont[i]) {
            if (TObject::GetObjectStat() && gObjectTable)
               gObjectTable->RemoveQuietly(fKeep->fCont[i]);
            ::operator delete(fKeep->fCont[i]);
            fKeep->fCont[i] = 0;
         }
   }

   TObjArray::Expand(newSize);
   fKeep->Expand(newSize);
}

//______________________________________________________________________________
void TClonesArray::ExpandCreate(Int_t n)
{
   // Expand or shrink the array to n elements and create the clone
   // objects by calling their default ctor. If n is less than the current size
   // the array is shrinked and the allocated space is freed.
   // This routine is typically used to create a clonesarray into which
   // one can directly copy object data without going via the
   // "new (arr[i]) MyObj()" (i.e. the vtbl is already set correctly).

   if (n < 0) {
      Error("ExpandCreate", "n must be positive (%d)", n);
      return ;
   }
   if (n > fSize)
      Expand(TMath::Max(n, GrowBy(fSize)));

   Int_t i;
   for (i = 0; i < n; i++) {
      if (!fKeep->fCont[i]) {
         fKeep->fCont[i] = (TObject*)fClass->New();
      } else if (!fKeep->fCont[i]->TestBit(kNotDeleted)) {
         // The object has been delete (or never initilized)
         fClass->New(fKeep->fCont[i]);
      }
      fCont[i] = fKeep->fCont[i];
   }

   for (i = n; i < fSize; i++)
      if (fKeep->fCont[i]) {
         if (TObject::GetObjectStat() && gObjectTable)
            gObjectTable->RemoveQuietly(fKeep->fCont[i]);
         ::operator delete(fKeep->fCont[i]);
         fKeep->fCont[i] = 0;
         fCont[i] = 0;
      }

   fLast = n - 1;
   Changed();
}

//______________________________________________________________________________
void TClonesArray::ExpandCreateFast(Int_t n)
{
   // Expand or shrink the array to n elements and create the clone
   // objects by calling their default ctor. If n is less than the current size
   // the array is shrinked and the allocated space is freed.
   // This routine is typically used to create a clonesarray into which
   // one can directly copy object data without going via the
   // "new (arr[i]) MyObj()" (i.e. the vtbl is already set correctly).
   // This is a simplified version of ExpandCreate used in the TTree mechanism.

   if (n > fSize)
      Expand(TMath::Max(n, GrowBy(fSize)));

   Int_t i;
   for (i = 0; i < n; i++) {
      if (!fKeep->fCont[i]) {
         fKeep->fCont[i] = (TObject*)fClass->New();
      } else if (!fKeep->fCont[i]->TestBit(kNotDeleted)) {
         // The object has been delete (or never initilized)
         fClass->New(fKeep->fCont[i]);
      }
      fCont[i] = fKeep->fCont[i];
   }
   fLast = n - 1;
   Changed();
}

//______________________________________________________________________________
TObject *TClonesArray::RemoveAt(Int_t idx)
{
   // Remove object at index idx.

   if (!BoundsOk("RemoveAt", idx)) return 0;

   int i = idx-fLowerBound;

   if (fCont[i] && fCont[i]->TestBit(kNotDeleted)) {
      // Tell custom operator delete() not to delete space when
      // object fCont[i] is deleted. Only destructors are called
      // for this object.
      Long_t dtoronly = TObject::GetDtorOnly();
      TObject::SetDtorOnly(fCont[i]);
      delete fCont[i];
      TObject::SetDtorOnly((void*)dtoronly);
   }

   if (fCont[i]) {
      fCont[i] = 0;
      // recalculate array size
      if (i == fLast)
         do { fLast--; } while (fLast >= 0 && fCont[fLast] == 0);
      Changed();
   }

   return 0;
}

//______________________________________________________________________________
TObject *TClonesArray::Remove(TObject *obj)
{
   // Remove object from array.

   if (!obj) return 0;

   Int_t i = IndexOf(obj) - fLowerBound;

   if (i == -1) return 0;

   if (fCont[i] && fCont[i]->TestBit(kNotDeleted)) {
      // Tell custom operator delete() not to delete space when
      // object fCont[i] is deleted. Only destructors are called
      // for this object.
      Long_t dtoronly = TObject::GetDtorOnly();
      TObject::SetDtorOnly(fCont[i]);
      delete fCont[i];
      TObject::SetDtorOnly((void*)dtoronly);
   }

   fCont[i] = 0;
   // recalculate array size
   if (i == fLast)
      do { fLast--; } while (fLast >= 0 && fCont[fLast] == 0);
   Changed();
   return obj;
}

//______________________________________________________________________________
void TClonesArray::SetOwner(Bool_t /* enable */)
{
   // A TClonesArray is always the owner of the object it contains.
   // However the collection its inherits from (TObjArray) does not.
   // Hence this member function needs to be a nop for TClonesArray.

   // Nothing to be done.
}

//______________________________________________________________________________
void TClonesArray::Sort(Int_t upto)
{
   // If objects in array are sortable (i.e. IsSortable() returns true
   // for all objects) then sort array.

   Int_t nentries = GetAbsLast()+1;
   if (nentries <= 0 || fSorted) return;
   for (Int_t i = 0; i < fSize; i++)
      if (fCont[i]) {
         if (!fCont[i]->IsSortable()) {
            Error("Sort", "objects in array are not sortable");
            return;
         }
      }

   QSort(fCont, fKeep->fCont, 0, TMath::Min(nentries, upto-fLowerBound));

   fLast   = -2;
   fSorted = kTRUE;
}

//_______________________________________________________________________
void TClonesArray::Streamer(TBuffer &b)
{
   // Write all objects in array to the I/O buffer. ATTENTION: empty slots
   // are also stored (using one byte per slot). If you don't want this
   // use a TOrdCollection or TList.

   // Important Note: if you modify this function, remember to also modify
   // TConvertClonesArrayToProxy accordingly

   Int_t   nobjects;
   char    nch;
   TString s;
   char classv[256];
   UInt_t R__s, R__c;

   if (b.IsReading()) {
      Version_t v = b.ReadVersion(&R__s, &R__c);
      if (v == 3) {
         const Int_t kOldBypassStreamer = BIT(14);
         if (TestBit(kOldBypassStreamer)) BypassStreamer();
      }
      if (v > 2)
         TObject::Streamer(b);
      if (v > 1)
         fName.Streamer(b);
      s.Streamer(b);
      strcpy(classv,s.Data());
      Int_t clv = 0;
      char *semicolon = strchr(classv,';');
      if (semicolon) {
         *semicolon = 0;
         clv = atoi(semicolon+1);
      }
      TClass *cl = gROOT->GetClass(classv);
      if (!cl) {
         printf("TClonesArray::Streamer expecting class %s\n", classv);
         b.CheckByteCount(R__s, R__c,TClonesArray::IsA());
         return;
      }

      b >> nobjects;
      if (nobjects < 0)
         nobjects = -nobjects;  // still there for backward compatibility
      b >> fLowerBound;
      if (fClass == 0 && fKeep == 0) {
         fClass = cl;
         fKeep  = new TObjArray(fSize);
         Expand(nobjects);
      }
      if (cl != fClass) {
         fClass = cl;
         //this case may happen when switching from an emulated class to the real class
         //may not be an error. fClass may point to a deleted object
         //Error("Streamer", "expecting objects of type %s, finding objects"
         //   " of type %s", fClass->GetName(), cl->GetName());
         //return;
      }

      // make sure there are enough slots in the fKeep array
      if (fKeep->GetSize() < nobjects)
         Expand(nobjects);

      //reset fLast. nobjects may be 0
      Int_t oldLast = fLast;
      fLast = nobjects-1;

      //TStreamerInfo *sinfo = fClass->GetStreamerInfo(clv);
      if (CanBypassStreamer() && !b.TestBit(TBuffer::kCannotHandleMemberWiseStreaming)) {
         for (Int_t i = 0; i < nobjects; i++) {
            if (!fKeep->fCont[i]) {
               fKeep->fCont[i] = (TObject*)fClass->New();
            } else if (!fKeep->fCont[i]->TestBit(kNotDeleted)) {
               // The object has been delete (or never initilized)
               fClass->New(fKeep->fCont[i]);
            }

            fCont[i] = fKeep->fCont[i];
         }
         //sinfo->ReadBufferClones(b,this,nobjects,-1,0);
         b.ReadClones(this,nobjects);

      } else {
         for (Int_t i = 0; i < nobjects; i++) {
            b >> nch;
            if (nch) {
               if (!fKeep->fCont[i])
                  fKeep->fCont[i] = (TObject*)fClass->New();
               else
                  fClass->New(fKeep->fCont[i]);

               fCont[i] = fKeep->fCont[i];
               b.StreamObject(fKeep->fCont[i]);
            }
         }
      }
      for (Int_t i = TMath::Max(nobjects,0); i < oldLast+1; ++i) fCont[i] = 0;
      Changed();
      b.CheckByteCount(R__s, R__c,TClonesArray::IsA());
   } else {
      //Make sure TStreamerInfo is not optimized, otherwise it will not be
      //possible to support schema evolution in read mode.
      //In case the StreamerInfo has already been computed and optimized,
      //one must disable the option BypassStreamer
      //Bool_t optim = TStreamerInfo::CanOptimize();
      //if (optim) TStreamerInfo::Optimize(kFALSE);
      //TStreamerInfo *sinfo = fClass->GetStreamerInfo();
      //sinfo->ForceWriteInfo((TFile *)b.GetParent());
      //if (optim) TStreamerInfo::Optimize(kTRUE);
      //if (sinfo->IsOptimized()) BypassStreamer(kFALSE);
      b.ForceWriteInfo(this);
      
      R__c = b.WriteVersion(TClonesArray::IsA(), kTRUE);
      TObject::Streamer(b);
      fName.Streamer(b);
      sprintf(classv,"%s;%d",fClass->GetName(),fClass->GetClassVersion());
      s = classv;
      s.Streamer(b);
      nobjects = GetEntriesFast();
      b << nobjects;
      b << fLowerBound;
      if (CanBypassStreamer() && !b.TestBit(TBuffer::kCannotHandleMemberWiseStreaming)) {
         b.WriteClones(this,nobjects);
      } else {
         for (Int_t i = 0; i < nobjects; i++) {
            if (!fCont[i]) {
               nch = 0;
               b << nch;
            } else {
               nch = 1;
               b << nch;
               b.StreamObject(fCont[i]);
            }
         }
      }
      b.SetByteCount(R__c, kTRUE);
   }
}

//______________________________________________________________________________
TObject *&TClonesArray::operator[](Int_t idx)
{
   // Return pointer to reserved area in which a new object of clones
   // class can be constructed. This operator should not be used for
   // lefthand side assignments, like a[2] = xxx. Only like,
   // new (a[2]) myClass, or xxx = a[2]. Of course right hand side usage
   // is only legal after the object has been constructed via the
   // new operator or via the New() method. To remove elements from
   // the clones array use Remove() or RemoveAt().

   if (idx < 0) {
      Error("operator[]", "out of bounds at %d in %x", idx, this);
      return fCont[0];
   }
   if (!fClass) {
      Error("operator[]", "invalid class specified in TClonesArray ctor");
      return fCont[0];
   }
   if (idx >= fSize)
      Expand(TMath::Max(idx+1, GrowBy(fSize)));

   if (!fKeep->fCont[idx])
      fKeep->fCont[idx] = (TObject*) TStorage::ObjectAlloc(fClass->Size());

   fCont[idx] = fKeep->fCont[idx];

   fLast = TMath::Max(idx, GetAbsLast());
   Changed();

   return fCont[idx];
}

//______________________________________________________________________________
TObject *TClonesArray::operator[](Int_t idx) const
{
   // Return the object at position idx. Returns 0 if idx is out of bounds.

   if (idx < 0 || idx >= fSize) {
      Error("operator[]", "out of bounds at %d in %x", idx, this);
      return 0;
   }

   return fCont[idx];
}

//______________________________________________________________________________
TObject *TClonesArray::New(Int_t idx)
{
   // Create an object of type fClass with the default ctor at the specified
   // index. Returns 0 in case of error.

   if (idx < 0) {
      Error("New", "out of bounds at %d in %x", idx, this);
      return 0;
   }
   if (!fClass) {
      Error("New", "invalid class specified in TClonesArray ctor");
      return 0;
   }

   return (TObject *)fClass->New(operator[](idx));
}
