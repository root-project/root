// @(#)root/cont:$Name:  $:$Id: TProcessID.cxx,v 1.14 2002/05/18 08:43:29 brun Exp $
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
#include "TFile.h"
#include "TObjArray.h"

TObjArray  *TProcessID::fgPIDs   = 0; //pointer to the list of TProcessID
TProcessID *TProcessID::fgPID    = 0; //pointer to the TProcessID of the current session
UInt_t      TProcessID::fgNumber = 0; //Current referenced object instance count

ClassImp(TProcessID)


//______________________________________________________________________________
TProcessID::TProcessID()
{
   fCount = 0;
   fObjects = 0;
}

//______________________________________________________________________________
TProcessID::~TProcessID()
{

   delete fObjects;
   fObjects = 0;
   fgPIDs->Remove(this);
}

//______________________________________________________________________________
TProcessID::TProcessID(const TProcessID &ref) : TNamed(ref)
{
   // TProcessID copy ctor.
}

//______________________________________________________________________________
TProcessID *TProcessID::AddProcessID()
{
// static function to add a new TProcessID to the list of PIDs

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
   sprintf(name,"ProcessID%d",apid);
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
void TProcessID::Cleanup()
{
   // static function (called by TROOT destructor) to delete all TProcessIDs

   fgPIDs->Delete();
   delete fgPIDs;
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
TProcessID *TProcessID::GetProcessWithUID(UInt_t uid)
{
// static function returning a pointer to TProcessID with its pid
// encoded in the highest byte of uid

   Int_t pid = (uid>>24)&0xff;
   return (TProcessID*)fgPIDs->At(pid);
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
   //if (!fObjects) fObjects = new TObjArray(100);
   
   Int_t uid = uidd & 0xffffff;  //take only the 24 lower bits
   
   if (uid > fObjects->GetSize()) return 0;
   return fObjects->UncheckedAt(uid);
}

//______________________________________________________________________________
void TProcessID::PutObjectWithID(TObject *obj, UInt_t uid)
{

   //stores the object at the uid th slot in the table of objects
   //The object uniqueid is set as well as its kMustCleanup bit
   //if (!fObjects) fObjects = new TObjArray(100);
   if (uid == 0) uid = obj->GetUniqueID() & 0xffffff;
   fObjects->AddAtAndExpand(obj,uid);
   obj->SetBit(kMustCleanup);
}

//______________________________________________________________________________
TProcessID  *TProcessID::ReadProcessID(UShort_t pidf, TFile *file)
{
// static function

   //The TProcessID with number pidf is read from file.
   //If the object is not already entered in the gROOT list, it is added.

   if (!file) return 0;
   TObjArray *pids = file->GetListOfProcessIDs();
   TProcessID *pid = 0;
   if (pidf < pids->GetSize()) pid = (TProcessID *)pids->UncheckedAt(pidf);
   if (pid) return pid;

   //check if fProcessIDs[uid] is set in file
   //if not set, read the process uid from file
   char pidname[32];
   sprintf(pidname,"ProcessID%d",pidf);
   TDirectory *dirsav = gDirectory;
   file->cd();
   pid = (TProcessID *)file->Get(pidname);
   if (dirsav) dirsav->cd();
   if (gDebug > 0) {
      printf("ReadProcessID, name=%s, file=%s, pid=%lx\n",pidname,file->GetName(),(Long_t)pid);
   }
   if (!pid) {
      //file->Error("ReadProcessID","Cannot find %s in file %s",pidname,file->GetName());
      return 0;
   }
      //check that a similar pid is not already registered in fgPIDs
   TIter next(fgPIDs);
   TProcessID *p;
   while ((p = (TProcessID*)next())) {
      if (!strcmp(p->GetTitle(),pid->GetTitle())) {
         delete pid;
         pids->AddAtAndExpand(p,pidf);
         p->IncrementCount();
         return p;
      }
   }
   pids->AddAtAndExpand(pid,pidf);
   pid->IncrementCount();
   fgPIDs->Add(pid);
   Int_t ind = fgPIDs->IndexOf(pid);
   pid->SetUniqueID((UInt_t)ind);
   return pid;
}

//______________________________________________________________________________
void TProcessID::RecursiveRemove(TObject *obj)
{
   // called by the object destructor
   // remove reference to obj from the current table if it is referenced

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

//______________________________________________________________________________
UShort_t TProcessID::WriteProcessID(TProcessID *pidd, TFile *file)
{
// static function
// Check if the ProcessID pid is already in the file.
// if not, add it and return the index  number in the local file list

   if (!file) return 0;
   TProcessID *pid = pidd;
   if (!pid) pid = fgPID;
   TObjArray *pids = file->GetListOfProcessIDs();
   Int_t npids = file->GetNProcessIDs();
   for (Int_t i=0;i<npids;i++) {
      if (pids->At(i) == pid) return (UShort_t)i;
   }
    
   TDirectory *dirsav = gDirectory;
   file->cd();
   file->SetBit(TFile::kHasReferences);
   pids->Add(pid);
   char name[32];
   sprintf(name,"ProcessID%d",npids);
   pid->Write(name);
   file->IncrementProcessIDs();
   if (gDebug > 0) {
      printf("WriteProcessID, name=%s, file=%s\n",name,file->GetName());   
   }
   if (dirsav) dirsav->cd();
   return (UShort_t)npids;
}
