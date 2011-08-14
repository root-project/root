// @(#)root/cont:$Id$
// Author: Rene Brun   28/09/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//
// TProcessID
//
// A TProcessID identifies a ROOT job in a unique way in time and space.
// The TProcessID title consists of a TUUID object which provides a globally
// unique identifier (for more see TUUID.h).
//
// A TProcessID is automatically created by the TROOT constructor.
// When a TFile contains referenced objects (see TRef), the TProcessID
// object is written to the file.
// If a file has been written in multiple sessions (same machine or not),
// a TProcessID is written for each session.
// These objects are used by the class TRef to uniquely identified
// any TObject pointed by a TRef.
//
// When a referenced object is read from a file (its bit kIsReferenced is set),
// this object is entered into the objects table of the corresponding TProcessID.
// Each TFile has a list of TProcessIDs (see TFile::fProcessIDs) also
// accessible via TProcessID::fgPIDs (for all files).
// When this object is deleted, it is removed from the table via the cleanup
// mechanism invoked by the TObject destructor.
//
// Each TProcessID has a table (TObjArray *fObjects) that keeps track
// of all referenced objects. If a referenced object has a fUniqueID set,
// a pointer to this unique object may be found via fObjects->At(fUniqueID).
// In the same way, when a TRef::GetObject is called, GetObject uses
// its own fUniqueID to find the pointer to the referenced object.
// See TProcessID::GetObjectWithID and PutObjectWithID.
//
// When a referenced object is deleted, its slot in fObjects is set to null.
//
// See also TProcessUUID: a specialized TProcessID to manage the single list
// of TUUIDs.
//
//////////////////////////////////////////////////////////////////////////

#include "TProcessID.h"
#include "TROOT.h"
#include "TObjArray.h"
#include "TExMap.h"
#include "TVirtualMutex.h"

TObjArray  *TProcessID::fgPIDs   = 0; //pointer to the list of TProcessID
TProcessID *TProcessID::fgPID    = 0; //pointer to the TProcessID of the current session
UInt_t      TProcessID::fgNumber = 0; //Current referenced object instance count
TExMap     *TProcessID::fgObjPIDs= 0; //Table (pointer,pids)
ClassImp(TProcessID)

//______________________________________________________________________________
static inline ULong_t Void_Hash(const void *ptr)
{
   // Return hash value for this object.

   return TString::Hash(&ptr, sizeof(void*));
}

//______________________________________________________________________________
TProcessID::TProcessID()
{
   // Default constructor.

   fCount = 0;
   fObjects = 0;
}

//______________________________________________________________________________
TProcessID::~TProcessID()
{
   // Destructor.

   delete fObjects;
   fObjects = 0;
   R__LOCKGUARD2(gROOTMutex);
   fgPIDs->Remove(this);
}

//______________________________________________________________________________
TProcessID *TProcessID::AddProcessID()
{
   // Static function to add a new TProcessID to the list of PIDs.

   R__LOCKGUARD2(gROOTMutex);

   TProcessID *pid = new TProcessID();

   if (!fgPIDs) {
      fgPID  = pid;
      fgPIDs = new TObjArray(10);
      gROOT->GetListOfCleanups()->Add(fgPIDs);
   }
   UShort_t apid = fgPIDs->GetEntriesFast();
   pid->IncrementCount();

   fgPIDs->Add(pid);
   char name[20];
   snprintf(name,20,"ProcessID%d",apid);
   pid->SetName(name);
   TUUID u;
   apid = fgPIDs->GetEntriesFast();
   pid->SetTitle(u.AsString());
   return pid;
}

//______________________________________________________________________________
UInt_t TProcessID::AssignID(TObject *obj)
{
   // static function returning the ID assigned to obj
   // If the object is not yet referenced, its kIsReferenced bit is set
   // and its fUniqueID set to the current number of referenced objects so far.

   R__LOCKGUARD2(gROOTMutex);

   UInt_t uid = obj->GetUniqueID() & 0xffffff;
   if (obj == fgPID->GetObjectWithID(uid)) return uid;
   if (obj->TestBit(kIsReferenced)) {
      fgPID->PutObjectWithID(obj,uid);
      return uid;
   }
   fgNumber++;
   obj->SetBit(kIsReferenced);
   uid = fgNumber;
   obj->SetUniqueID(uid);
   fgPID->PutObjectWithID(obj,uid);
   return uid;
}

//______________________________________________________________________________
void TProcessID::CheckInit()
{
   // Initialize fObjects.
   if (!fObjects) fObjects = new TObjArray(100);
}

//______________________________________________________________________________
void TProcessID::Cleanup()
{
   // static function (called by TROOT destructor) to delete all TProcessIDs

   R__LOCKGUARD2(gROOTMutex);

   fgPIDs->Delete();
   gROOT->GetListOfCleanups()->Remove(fgPIDs);
   delete fgPIDs;
   fgPIDs = 0;
}

//______________________________________________________________________________
void TProcessID::Clear(Option_t *)
{
   // delete the TObjArray pointing to referenced objects
   // this function is called by TFile::Close("R")

   delete fObjects; fObjects = 0;
}

//______________________________________________________________________________
Int_t TProcessID::DecrementCount()
{

   // the reference fCount is used to delete the TProcessID
   // in the TFile destructor when fCount = 0

   fCount--;
   if (fCount < 0) fCount = 0;
   return fCount;
}

//______________________________________________________________________________
TProcessID *TProcessID::GetProcessID(UShort_t pid)
{
   // static function returning a pointer to TProcessID number pid in fgPIDs

   return (TProcessID*)fgPIDs->At(pid);
}

//______________________________________________________________________________
UInt_t TProcessID::GetNProcessIDs()
{
   // Return the (static) number of process IDs.
   return fgPIDs ? fgPIDs->GetLast()+1 : 0;
}

//______________________________________________________________________________
TProcessID *TProcessID::GetProcessWithUID(UInt_t uid, const void *obj)
{
   // static function returning a pointer to TProcessID with its pid
   // encoded in the highest byte of uid

   R__LOCKGUARD2(gROOTMutex);

   Int_t pid = (uid>>24)&0xff;
   if (pid==0xff) {
      // Look up the pid in the table (pointer,pid)
      if (fgObjPIDs==0) return 0;
      ULong_t hash = Void_Hash(obj);
      pid = fgObjPIDs->GetValue(hash,(Long_t)obj);
   }
   return (TProcessID*)fgPIDs->At(pid);
}

//______________________________________________________________________________
TProcessID *TProcessID::GetProcessWithUID(const TObject *obj)
{
   // static function returning a pointer to TProcessID with its pid
   // encoded in the highest byte of obj->GetUniqueID()

   return GetProcessWithUID(obj->GetUniqueID(),obj);
}

//______________________________________________________________________________
TProcessID *TProcessID::GetSessionProcessID()
{
   // static function returning the pointer to the session TProcessID

   return fgPID;
}

//______________________________________________________________________________
Int_t TProcessID::IncrementCount()
{
   // Increase the reference count to this object.

   if (!fObjects) fObjects = new TObjArray(100);
   fCount++;
   return fCount;
}

//______________________________________________________________________________
UInt_t TProcessID::GetObjectCount()
{
   // Return the current referenced object count
   // fgNumber is incremented everytime a new object is referenced

   return fgNumber;
}

//______________________________________________________________________________
TObject *TProcessID::GetObjectWithID(UInt_t uidd)
{
   //returns the TObject with unique identifier uid in the table of objects

   Int_t uid = uidd & 0xffffff;  //take only the 24 lower bits

   if (fObjects==0 || uid >= fObjects->GetSize()) return 0;
   return fObjects->UncheckedAt(uid);
}

//______________________________________________________________________________
TProcessID *TProcessID::GetPID()
{
   //static: returns pointer to current TProcessID
   
   return fgPID;
}

//______________________________________________________________________________
TObjArray *TProcessID::GetPIDs()
{
   //static: returns array of TProcessIDs
   
   return fgPIDs;
}


//______________________________________________________________________________
Bool_t TProcessID::IsValid(TProcessID *pid)
{
   // static function. return kTRUE if pid is a valid TProcessID

   R__LOCKGUARD2(gROOTMutex);

   if (fgPIDs==0) return kFALSE;
   if (fgPIDs->IndexOf(pid) >= 0) return kTRUE;
   if (pid == (TProcessID*)gROOT->GetUUIDs())  return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
void TProcessID::PutObjectWithID(TObject *obj, UInt_t uid)
{
   //stores the object at the uid th slot in the table of objects
   //The object uniqueid is set as well as its kMustCleanup bit

   if (uid == 0) uid = obj->GetUniqueID() & 0xffffff;

   if (!fObjects) fObjects = new TObjArray(100);
   fObjects->AddAtAndExpand(obj,uid);

   obj->SetBit(kMustCleanup);
   if ( (obj->GetUniqueID()&0xff000000)==0xff000000 ) {
      // We have more than 255 pids we need to store this
      // pointer in the table(pointer,pid) since there is no
      // more space in fUniqueID
      if (fgObjPIDs==0) fgObjPIDs = new TExMap;
      ULong_t hash = Void_Hash(obj);

      // We use operator() rather than Add() because
      // if the address has already been registered, we want to
      // update it's uniqueID (this can easily happen when the
      // referenced object have been stored in a TClonesArray.
      (*fgObjPIDs)(hash, (Long_t)obj) = GetUniqueID();
   }
}

//______________________________________________________________________________
void TProcessID::RecursiveRemove(TObject *obj)
{
   // called by the object destructor
   // remove reference to obj from the current table if it is referenced

   if (!fObjects) return;
   if (!obj->TestBit(kIsReferenced)) return;
   UInt_t uid = obj->GetUniqueID() & 0xffffff;
   if (obj == GetObjectWithID(uid)) fObjects->RemoveAt(uid);
}


//______________________________________________________________________________
void TProcessID::SetObjectCount(UInt_t number)
{
   // static function to set the current referenced object count
   // fgNumber is incremented everytime a new object is referenced

   fgNumber = number;
}
