// @(#)root/cont:$Name:  $:$Id: TRef.cxx,v 1.4 2001/11/22 18:10:00 brun Exp $
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
// TRef
//
// Persistent Reference link to a TObject 
// A TRef is a lightweight object pointing to any TObject.
// This object can be used instead of normal C++ pointers in case
//  - the referenced object R and the pointer P are not written to the same file
//  - P is read before R
//  - R and P are written to different Tree branches
//
// When a top level object (eg Event *event) is a tree/graph of many objects,
// the normal ROOT Streaming mechanism ensures that only one copy of each object
// in the tree/graph is written to the output buffer to avoid circular
// dependencies.
// However if the object event is split into several files or into several
// branches of one or more Trees, normal C++ pointers cannot be used because
// each I/O operation will write the referenced objects.
// When a TRef is used to point to a TObject *robj. 
//For example in a class with
//     TRef  fRef;
// one can do:
//     fRef = robj;  //to set the pointer
// this TRef and robj can be written with two different I/O calls
// in the same or different files, in the same or different branches of a Tree.
// If the TRef is read and the referenced object has not yet been read,
// the TRef will return a null pointer. As soon as the referenced object
// will be read, the TRef will point to it.
//
// TRef also supports the complex situation where a TFile is updated
// multiple times on the same machine or a different machine.
//
// How does it work
// ----------------
// A TRef is itself a TObject with an additional transient pointer fPID.
// When the statement fRef = robj is executed, the fRef::fUniqueID is set
// to the value "obj-gSystem". This uid is in general a small integer, even
// on a 64 bit system.
// After having set fRef, one can immediatly return the value of robj
// with "gSystem + uid" using fRef.GetObject().
//
// When the TRef is written, the process id number pidf  (see TProcessID)
// is written as well as the uid.
// When the TRef is read, its pointer fPID is set to the value
// stored in the TObjArray of TFile::fProcessIDs (fProcessIDs[pidf]).
//
// When a referenced object robj is written, TObject::Streamer writes
// in addition to the standard (fBits,fUniqueID) the pair uid,pidf.
// When this robj is read by TObject::Streamer, the pair uid,pidf is read.
// At this point, robj is entered into the TExmap of the TProcessID
// corresponding to pidf. 
//
// Once the referenced object robj is in memory, TRef::GetObject will 
// store the object pointer robj-gSystem  into the fUniqueID such that
// the next access to the pointer will be fast (no need to search in
// the TExMap of the TProcessID anymore).
//
// WARNING: If MyClass is the class of the referenced object, The TObject
//          part of MyClass must be Streamed. One should not
//          call MyClass::Class()->IgnoreTObjectStreamer()
//
// Action on Demand
// ----------------
// The normal behaviour of a TRef has been described above. In addition,
// TRef supports also "Actions on Demand". It may happen that the object
// referenced is not yet in memory, on a separate file or not yet computed.
// In this case TRef is able to automatically execute an action:
//   - call to a compiled function (static function of member function)
//   - call to an interpreted function
//   - execution of a CINT script
//
// How to select this option?
// In the definition of the TRef data member in the original class, do:
//   TRef  fRef;   //EXEC:execName. points to something
// When the special keyword "EXEC:" is found in the comment field of the member,
// the next string is assumed to be the name of a TExec object.
// When a file is connected, the dictionary of the classes on the file
// is read in memory (see TFile::ReadStreamerInfo). When the TStreamerElement
// object is read, a TExec object is automatically created with the name
// specified after the keywork "EXEC:" in case a TExec with a same name does
// not already exist.
// The action to be executed via this TExec can be specified with:
//    - a call to the TExec constructor, if the constructor is called before
//      opening the file.
//    - a call to TExec::SetAction at any time.
// One can compute a pointer to an existing TExec with a name with:
//    TObjArray *listExecs = (TObjArray*)gROOT->FindObjectAny("Execs");
//    TExec *myExec = (TExec*)listExecs->FindObject(execName);
//    myExec->SetAction(actionCommand); where 
//      - listExecs points to to the array of TExecs. Up to 256 TExecs
//        can be specified in a process.
//      - actionCommand is a string containing a CINT instruction. Examples:
//          myExec->SetAction("LoadHits()");
//          myExec->SetAction(".x script.C");
//
//  When a TRef is dereferenced via TRef::GetObject, its TExec will be
// automatically executed. In the function/script being executed, one or more 
// of the following actions can be executed:
//  - load a file containing the referenced object. This function typically
//    looks in the file catalog (GRID).
//  - compute a pointer to the referenced object and communicate this pointer
//    back to the calling function TRef::GetObject via:
//      gROOT->SetSelectedPrimitive(theObject);
// As soon as an object is returned to GetObject, a bit is set in the TRef
// and the address of the referenced object stored. At the next call to GetRef,
// the address of the referenced object will be returned immediatly.
//
// Array of TRef
// -------------
// The special class TRefArray should be used to store multiple references.
//
// Example:
// Suppose a TObjArray *mytracks containing a list of Track objects
// Suppose a TRefArray *pions containing pointers to the pion tracks in mytracks.
//   This list is created with statements like: pions->Add(track);
// Suppose a TRefArray *muons containing pointers to the muon tracks in mytracks.
// The 3 arrays mytracks,pions and muons may be written separately.
//
//////////////////////////////////////////////////////////////////////////

#include "TRef.h"
#include "TROOT.h"
#include "TProcessID.h"
#include "TFile.h"
#include "TObjArray.h"
#include "TExec.h"
#include "TSystem.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"

ClassImp(TRef)

//______________________________________________________________________________
TRef::TRef(TObject *obj)
{
   // TRef copy ctor.

   *this = obj;
   fPID = 0;
}

//______________________________________________________________________________
TRef::TRef(const TRef &ref)
{
}

//______________________________________________________________________________
void TRef::operator=(TObject *obj)
{
   // TRef assignment operator.

   Long_t uid;
   if (obj) {
      if (obj->IsA()->CanIgnoreTObjectStreamer()) {
         Error("operator= ","Class: %s IgnoreTObjectStreamer. Cannot reference object",obj->ClassName());
         return;
      }
      obj->SetBit(kIsReferenced);
      uid = (Long_t)obj - (Long_t)gSystem;
   } else {
      uid = 0;
   }
   SetUniqueID((UInt_t)uid);
}

//______________________________________________________________________________
TObject *TRef::GetObject() const
{
   // Return a pointer to the referenced object.

   TObject *obj = 0;
   Long_t uid = (Long_t)GetUniqueID();
   if (uid == 0) return obj;
   if (!TestBit(1)) return (TObject*)(uid + (Long_t)gSystem);
   
   //Try to find the object from the map of the corresponding PID
   if (fPID) obj = fPID->GetObjectWithID(uid);
   
   //if object not found, then exec action if an action has been defined
   if (!obj) {
      //execid starts at bit 8 on 8 bits
      Int_t execid = TestBits(0xff00) >> 8;
      if (execid > 0) {
         TObjArray *lexecs = (TObjArray*)gROOT->FindObjectAny("Execs");
         if (lexecs) {
            TExec *exec = (TExec*)lexecs->At(execid-1);
            if (exec) {
               //store this object in gROOT just in case it is needed by the Exec
               TObject *ref = (TObject*)this;
               gROOT->SetSelectedPrimitive(ref);
               exec->Exec();
               //if the Exec has redefined the selectedPrimitive, we assume
               //that it contains the pointer to our requested object
               if (gROOT->GetSelectedPrimitive() != ref) {
                  obj = gROOT->GetSelectedPrimitive();
                  if (obj) obj->SetBit(kIsReferenced);
               } else {
                  //well may be the Exec has loaded the object
                  if (fPID) obj = fPID->GetObjectWithID(uid);
               }  
            }  
         }  
      }
   }
   
   //if object is found, mark the bit to speed up next calls
   if (obj) {
      ((TRef*)this)->ResetBit(1);
      uid = (Long_t)obj - (Long_t)gSystem;
      ((TRef*)this)->SetUniqueID((UInt_t)uid);
   }
   return obj;
}

//______________________________________________________________________________
void TRef::ReadRef(TObject *obj, TBuffer &R__b, TFile *file)
{
// static function

   Long_t uid;
   Int_t pidf = 0;
   R__b >> uid; 
   R__b >> pidf;
   
   TProcessID *pid = TProcessID::ReadProcessID(pidf,file);
   
   if (pid) pid->PutObjectWithID(uid,obj);
}

//______________________________________________________________________________
void TRef::SaveRef(TObject *obj, TBuffer &R__b, TFile *file)
{
// static function

   Long_t uid = (Long_t)obj - (Long_t)gSystem;
   Int_t pidf = 0;
   if (file) {
      pidf = file->GetProcessCount();
      file->SetBit(TFile::kHasReferences);
   }
   R__b << uid;
   R__b << pidf;
}

//______________________________________________________________________________
void TRef::Streamer(TBuffer &R__b)
{
   // Stream an object of class TRef.

   Long_t uid;
   Int_t pidf;
   if (R__b.IsReading()) {
      R__b >> pidf;
      R__b >> uid;
      SetBit(1,1);
      SetUniqueID((UInt_t)uid);
      fPID = TProcessID::ReadProcessID(pidf,gFile);
      Int_t execid = TStreamerInfo::GetCurrentElement()->GetUniqueID();
      if (execid) SetBit(execid << 8);
   } else {
      pidf = 0;
      uid = (Long_t)GetUniqueID();
      if (gFile) {
         pidf = gFile->GetProcessCount();
         gFile->SetBit(TFile::kHasReferences);
      }
      R__b << pidf;
      R__b << uid;
   }
}
