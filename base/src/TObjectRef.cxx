// @(#)root/cont:$Name:  $:$Id: TObjectRef.cxx,v 1.1 2001/10/01 10:29:08 brun Exp $
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
// TObjectRef                                                           //
//                                                                      //
// Persistent Reference link to a TObject                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObjectRef.h"
#include "TROOT.h"
#include "TProcessID.h"
#include "TFile.h"
#include "TObjArray.h"
#include "TSystem.h"

ClassImp(TObjectRef)

// A TObjectRef is a lightweight object pointing to any TObject.
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
// When a TObjectRef is used to point to a TObject *robj. 
//For example in a class with
//     TObjectRef  fRef;
// one can do:
//   fRef = robj;  //to set the pointer
// this TObjectRef and robj can be written with two different I/O calls
// in the same or different files, in the same or different branches of a Tree.
// If the TObjectRef is read and the referenced object has not yet been read,
// the TObjectRef will return a null pointer. As soon as the referenced object
// will be read, the TObjectRef will point to it.
//
// TObjectRef also supports the complex situation where a TFile is updated
// multiple times on the same machine or a different machine.
//
// How does it work
// ----------------
// A TObjectRef is itself a TObject with an additional transient pointer fPID.
// When the statement fRef = robj is executed, the fRef::fUniqueID is set
// to the value "obj-gSystem". This uid is in general a small integer, even
// on a 64 bit system.
// After having set fRef, one can immediatly return the value of robj
// with "gSystem + uid" using fRef.GetObject() or the dereferencing operator ->.
//
// When the TObjectRef is written, the process id number pidf  (see TProcessID)
// is written as well as the uid.
// When the TObjectRef is read, its pointer fPID is set to the value
// stored in the TObjArray of TFile::fProcessIDs (fProcessIDs[pidf]).
//
// When a referenced object robj is written, TObject::Streamer writes
// in addition to the standard (fBits,fUniqueID) the pair uid,pidf.
// When this robj is read by TObject::Streamer, the pair uid,pidf is read.
// At this point, robj is entered into the TExmap of the TProcessID
// corresponding to pidf. 
//
// Once the referenced object robj is in memory, TObjectRef::GetObject will 
// store the object pointer robj-gSystem  into the fUniqueID such that
// the next access to the pointer will be fast (no need to search in
// the TExMap of the TProcessID anymore).
//
// Implicit use of TObjectRef in ROOT collections
// ----------------------------------------------
// The TSeqCollection (TList, TObjArray, etc) have been modified with
// additional member functions AddRef, AddRefAt, etc to create automatically
// a TObjectRef when doing, eg:
//    myArray->AddRef(robj);
//
// Example:
// Suppose a TObjArray *mytracks containing a list of Track objects
// Suppose a TObjArray *pions containing pointers to the pion tracks in mytracks.
//   This list is created with statements like: pions->AddRef(track);
// Suppose a TObjArray *muons containing pointers to the muon tracks in mytracks.
// The 3 arrays mytracks,pions and muons may be written separately.
//

//______________________________________________________________________________
TObjectRef::TObjectRef(TObject *obj)
{
   // TObjectRef copy ctor.

   *this = obj;
   fPID = 0;
}

//______________________________________________________________________________
TObjectRef::TObjectRef(const TObjectRef &ref)
{
}

//______________________________________________________________________________
void TObjectRef::operator=(TObject *obj)
{
   // TObjectRef assignment operator.

   UInt_t uid;
   if (obj) {
      obj->SetBit(kIsReferenced);
      uid = (char*)obj - (char*)gSystem;
   } else {
      uid = 0;
   }
   SetUniqueID(uid);
}

//______________________________________________________________________________
TObject *TObjectRef::GetObject() 
{
   // Return a pointer to the referenced object.

   TObject *obj = 0;
   Long_t uid = (Long_t)GetUniqueID();
   if (uid == 0) return obj;
   if (!TestBit(1)) return (TObject*)(uid + (char*)gSystem);
   if (!fPID) return 0;
   obj = fPID->GetObjectWithID(uid);
   if (obj) {
      ResetBit(1);
      uid = (char*)obj - (char*)gSystem;
      SetUniqueID((UInt_t)uid);
   }
   return obj;
}

//______________________________________________________________________________
void TObjectRef::ReadRef(TObject *obj, TBuffer &R__b, TFile *file)
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
void TObjectRef::SaveRef(TObject *obj, TBuffer &R__b, TFile *file)
{
// static function

   Long_t uid = (char*)obj - (char*)gSystem;
   Int_t pidf = 0;
   if (file) {
      pidf = file->GetProcessCount();
   }
   R__b << uid;
   R__b << pidf;
}

//______________________________________________________________________________
void TObjectRef::Streamer(TBuffer &R__b)
{
   // Stream an object of class TObjectRef.

   Long_t uid;
   Int_t pidf;
   if (R__b.IsReading()) {
      R__b >> pidf;
      R__b >> uid;
      SetBit(1,1);
      SetUniqueID((UInt_t)uid);
      fPID = TProcessID::ReadProcessID(pidf,gFile);
   } else {
      uid = (Long_t)GetUniqueID();
      if (gFile) pidf = gFile->GetProcessCount();
      else       pidf = 0;
      R__b << pidf;
      R__b << uid;
   }
}
