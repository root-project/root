// @(#)root/cont:$Name:  $:$Id: TClonesArray.cxx,v 1.20 2001/10/22 08:46:49 brun Exp $
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
//   TCloneArray a("TTrack", 10000);                                    //
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
// Considering that a new/delete costs about 70 mus, O(10^9)            //
// new/deletes will save about 19 hours.                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include "TClonesArray.h"
#include "TMath.h"
#include "TError.h"
#include "TClass.h"
#include "TROOT.h"
#include "TStreamerInfo.h"
#include "TObjectTable.h"


ClassImp(TClonesArray)

//______________________________________________________________________________
TClonesArray::TClonesArray() : TObjArray()
{
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
   // The third argument is not used anymore and only there for backward
   // compatibility reasons.

   if (!gROOT)
      ::Fatal("TClonesArray::TClonesArray", "ROOT system not initialized");

   fKeep  = 0;
   fClass = gROOT->GetClass(classname);
   if (!fClass) {
      Error("TClonesArray", "%s is not a valid class name", classname);
      return;
   }
   if (!fClass->InheritsFrom(TObject::Class())) {
      Error("TClonesArray", "%s does not inherit from TObject", classname);
      return;
   }
   char name[100];
   sprintf(name,"%ss",classname);
   SetName(name);

   fKeep = new TObjArray(s);

   BypassStreamer(kTRUE);
}

//______________________________________________________________________________
TClonesArray::~TClonesArray()
{
   // Delete a clones array.

   if (fKeep) {
      for (Int_t i = 0; i < fKeep->fSize; i++) {
         // remove any possible entries from the ObjectTable
         if (fClass->GetClassInfo()) {
            if (TObject::GetObjectStat() && gObjectTable)
               gObjectTable->RemoveQuietly(fKeep->fCont[i]);
            ::operator delete(fKeep->fCont[i]);
         }
      }
   }
   SafeDelete(fKeep);
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
   //  1- a class Foo has a TClonesArray of Bar objects
   //  2- the Foo object is written with split=0 to Tree T1.
   //     In this case the StreamerInfo for the class Bar is created
   //     in optimized mode in such a way that data members of the same type
   //     are written as an array improving the I/O performance.
   //  3- in a new program, T1 is read and a new Tree T2 is created
   //      with the object Foo in split>1
   //  4- When the T2 branch is created, the StreamerInfo for the class Bar
   //     is created with no optimization (mandatory for the split mode).
   //     The optimized Bar StreamerInfo is going to be used to read
   //     the TClonesArray in T1. The result will be Bar objects with
   //     data member values not in the right sequence.
   // The solution to this problem is to call BypassStreamer(kFALSE)
   // for the TClonesArray. In this case, the normal Bar::Streamer function 
   // will be called. The BAR::Streamer function works OK independently
   // if the Bar StreamerInfo had been generated in optimized mode or not.
   
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

   Assert(je == jf);
}

//______________________________________________________________________________
void TClonesArray::Clear(Option_t *)
{
   // Clear the clones array. Only use this routine when your objects don't
   // allocate memory since it will not call the object dtors.

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

   for (Int_t i = 0; i < fSize; i++)
      if (fCont[i] && fCont[i]->TestBit(kNotDeleted)) {
         // Tell custom operator delete() not to delete space when
         // object fCont[i] is deleted. Only destructors are called
         // for this object.
         TObject::SetDtorOnly(fCont[i]);
         delete fCont[i];
      }

   // Protect against erroneously setting of owne bit.
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
   // objects by caling their default ctor. If n is less than the current size
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
      if (!fKeep->fCont[i])
         fKeep->fCont[i] = (TObject*)fClass->New();

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
      if (!fKeep->fCont[i])
         fKeep->fCont[i] = (TObject*)fClass->New();

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
      TObject::SetDtorOnly(fCont[i]);
      delete fCont[i];
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
      TObject::SetDtorOnly(fCont[i]);
      delete fCont[i];
   }

   fCont[i] = 0;
   // recalculate array size
   if (i == fLast)
      do { fLast--; } while (fLast >= 0 && fCont[fLast] == 0);
   Changed();
   return obj;
}

//______________________________________________________________________________
void TClonesArray::Sort(Int_t upto)
{
   // If objects in array are sortable (i.e. IsSortable() returns true
   // for all objects) then sort array.

   if (GetAbsLast() == -1 || fSorted) return;
   for (Int_t i = 0; i < fSize; i++)
      if (fCont[i]) {
         if (!fCont[i]->IsSortable()) {
            Error("Sort", "objects in array are not sortable");
            return;
         }
      }

   QSort(fCont, fKeep->fCont, 0, TMath::Min(fSize, upto-fLowerBound));

   fLast   = -2;
   fSorted = kTRUE;
}

//_______________________________________________________________________
void TClonesArray::Streamer(TBuffer &b)
{
   // Write all objects in array to the I/O buffer. ATTENTION: empty slots
   // are also stored (using one byte per slot). If you don't want this
   // use a TOrdCollection or TList.

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
         //this case may happen when switching from a fake class to the real class
         //may not be an error. fClass may point to a deleted object
         //Error("Streamer", "expecting objects of type %s, finding objects"
         //   " of type %s", fClass->GetName(), cl->GetName());
         //return;
      }

      // make sure there are enough slots in the fKeep array
      if (fKeep->GetSize() < nobjects)
         Expand(nobjects);

      TStreamerInfo *sinfo = fClass->GetStreamerInfo(clv);
      //must test on sinfo and not on fClass (OK when writing)
      if (CanBypassStreamer()) {
         for (Int_t i = 0; i < nobjects; i++) {
            if (!fKeep->fCont[i])
               fKeep->fCont[i] = (TObject*)fClass->New();
            else
               fClass->New(fKeep->fCont[i]);

            fCont[i] = fKeep->fCont[i];
            fLast = i;
         }
         sinfo->ReadBufferClones(b,this,nobjects,-1,0);

      } else {
         for (Int_t i = 0; i < nobjects; i++) {
            b >> nch;
            if (nch) {
               if (!fKeep->fCont[i])
                  fKeep->fCont[i] = (TObject*)fClass->New();
               else
                  fClass->New(fKeep->fCont[i]);

               fCont[i] = fKeep->fCont[i];
               fKeep->fCont[i]->Streamer(b);
               fLast = i;
            }
         }
      }
      Changed();
      b.CheckByteCount(R__s, R__c,TClonesArray::IsA());
   } else {
      R__c = b.WriteVersion(TClonesArray::IsA(), kTRUE);
      TObject::Streamer(b);
      fName.Streamer(b);
      sprintf(classv,"%s;%d",fClass->GetName(),fClass->GetClassVersion());
      s = classv;
      s.Streamer(b);
      nobjects = GetEntriesFast();
      b << nobjects;
      b << fLowerBound;
      TStreamerInfo *sinfo = fClass->GetStreamerInfo();
      sinfo->ForceWriteInfo();
      if (CanBypassStreamer()) {
         sinfo->WriteBufferClones(b,this,nobjects,-1,0);
      } else {
         for (Int_t i = 0; i < nobjects; i++) {
            if (!fCont[i]) {
               nch = 0;
               b << nch;
            } else {
               nch = 1;
               b << nch;
               fCont[i]->Streamer(b);
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
      fKeep->fCont[idx] = (TObject*)::operator new(fClass->Size());

   fCont[idx] = fKeep->fCont[idx];

   fLast = TMath::Max(idx, GetAbsLast());
   Changed();

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
