// @(#)root/cont:$Name:  $:$Id: TProcessID.cxx,v 1.1 2001/10/01 10:29:08 brun Exp $
// Author: Rene Brun   28/09/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProcessID                                                           //
//                                                                      //
// ROOT session descriptor (unique in time and space)                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProcessID.h"
#include "TROOT.h"
#include "TFile.h"
#include "TObjArray.h"
#include "TSystem.h"
#include "TDatime.h"

ClassImp(TProcessID)


// A TProcessID identifies a ROOT job in a unique way in time and space.
// The TProcessID title is made of :
//    - The process pid
//    - The system hostname
//    - The system time in clock units
//
// A TProcessID is automatically created by the TROOT constructor.
// When a TFile contains referenced objects (see TObjectRef), the TProcessID
// object is written to the file.
// If a file has been written in multiple sessions (same machine or not),
// a TProcessID is written for each session.
// These objects are used by the class TObjectRef to uniquely identified
// any TObject pointed by a TObjectRef.
// 
// When a referenced object is read from a file (its bit kIsReferenced is set),
// this object is entered into the TExmap *fMap of the corresponding TProcessID.
// Each TFile has a list of TProcessIDs (see TFile::fProcessIDs) also
// accessible via gROOT->GetListOfProcessIDs() (for all files).
// When this object is deleted, it is removed from the map via the cleanup
// mechanism invoked by the TObject destructor.
//______________________________________________________________________________
TProcessID::TProcessID()
{
   fMap = 0;
   gROOT->GetListOfCleanups()->Add(this);
}

//______________________________________________________________________________
TProcessID::TProcessID(Int_t pid)
{
   char name[20];
   char title[100];
   TDatime d;
   sprintf(name,"ProcessID%d",pid);
   sprintf(title,"pid%d_%s_%s",gSystem->GetPid(),gSystem->HostName(),d.AsString());
   SetName(name);
   SetTitle(title);
   fMap = 0;
   fCount = 0;
   gROOT->GetListOfCleanups()->Add(this);
}

//______________________________________________________________________________
TProcessID::~TProcessID()
{

   delete fMap;
   fMap = 0;
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

   fCount++;
   return fCount;
}

//______________________________________________________________________________
TObject *TProcessID::GetObjectWithID(Long_t uid) 
{
   //returns the TObject with unique identifier uid in the TExMap
   if (!fMap) return 0;
   Long_t id = fMap->GetValue(uid);
   return (TObject *)id;
}

//______________________________________________________________________________
void TProcessID::PutObjectWithID(Long_t uid, TObject *obj) 
{

   //stores the object with its key uid in the TExmap.
   //The object uniqueid is set as well as its kMustCleanup bit
   if (!fMap) fMap = new TExMap(100);
   Long_t id = (Long_t)obj;
   obj->SetBit(kMustCleanup);
   obj->SetUniqueID(UInt_t(uid));
   if (!fMap->GetValue(uid)) fMap->Add(uid,id);
}

//______________________________________________________________________________
TProcessID  *TProcessID::ReadProcessID(Int_t pidf, TFile *file)
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
   if (!fMap) return;
   Long_t uid = (Long_t)obj->GetUniqueID();
   fMap->Remove(uid);
}
