// @(#)root/cont:$Name:  $:$Id: TProcessID.cxx,v 1.6 2001/11/20 09:32:54 brun Exp $
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
// The TProcessID title is made of :
//    - The process pid
//    - The system hostname
//    - The system time in clock units
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
// accessible via gROOT->GetListOfProcessIDs() (for all files).
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
//////////////////////////////////////////////////////////////////////////

#include "TProcessID.h"
#include "TROOT.h"
#include "TFile.h"
#include "TSystem.h"
#include "TUUID.h"

ClassImp(TProcessID)


//______________________________________________________________________________
TProcessID::TProcessID()
{
   fCount = 0;
   fObjects = 0;
   gROOT->GetListOfCleanups()->Add(this);
}

//______________________________________________________________________________
TProcessID::TProcessID(UShort_t pid)
{
   char name[20];
   sprintf(name,"ProcessID%d",pid);
   SetName(name);
   TUUID u;
   SetTitle(u.AsString());
   fObjects = 0;
   fCount = 0;
   gROOT->GetListOfCleanups()->Add(this);
}

//______________________________________________________________________________
TProcessID::~TProcessID()
{

   delete fObjects;
   fObjects = 0;
   gROOT->GetListOfProcessIDs()->Remove(this);
   gROOT->GetListOfCleanups()->Remove(this);
}
//______________________________________________________________________________
TProcessID::TProcessID(const TProcessID &ref)
{
   // TProcessID copy ctor.
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
Int_t TProcessID::IncrementCount() 
{

   if (!fObjects) fObjects = new TObjArray(100);
   fCount++;
   return fCount;
}

//______________________________________________________________________________
TObject *TProcessID::GetObjectWithID(UInt_t uid) 
{
   //returns the TObject with unique identifier uid in the table of objects
   if ((Int_t)uid > fObjects->GetSize()) return 0;
   return fObjects->UncheckedAt(uid);
}

//______________________________________________________________________________
void TProcessID::PutObjectWithID(TObject *obj, UInt_t uid) 
{

   //stores the object at the uid th slot in the table of objects
   //The object uniqueid is set as well as its kMustCleanup bit
   if (uid == 0) uid = obj->GetUniqueID();
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
   TProcessID *pid = (TProcessID *)pids->UncheckedAt(pidf);
   if (pid) return pid;
   
   //check if fProcessIDs[uid] is set in file
   //if not set, read the process uid from file
   char pidname[32];
   sprintf(pidname,"ProcessID%d",pidf);
   pid = (TProcessID *)file->Get(pidname);
   if (!pid) {
      file->Error("ReadProcessID","Cannot find %s in file %s",pidname,file->GetName());
      return 0;
   }
      //check that a similar pid is not already registered in gROOT
   TIter next(gROOT->GetListOfProcessIDs());
   TProcessID *apid;
   while ((apid=(TProcessID *)next())) {
      if (strcmp(apid->GetTitle(),pid->GetTitle())) continue;
      gROOT->GetListOfProcessIDs()->Add(pid);
      break;
   }
   pids->AddAt(pid,pidf);
   pid->IncrementCount();
   return pid;
}

//______________________________________________________________________________
void TProcessID::RecursiveRemove(TObject *obj)
{
   // called by the object destructor
   // remove reference to obj from the current table if it is referenced

   if (!obj->TestBit(kIsReferenced)) return;
   UInt_t uid = obj->GetUniqueID();
   if (obj == GetObjectWithID(uid)) fObjects->RemoveAt(uid);
}
