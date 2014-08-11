// @(#)root/cont:$Id$
// Author: Rene Brun   28/09/2001

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// A TRefTable maintains the association between a referenced object    //
// and the parent object supporting this referenced object.             //
//                                                                      //
// The parent object is typically a branch of a TTree. For each object  //
// referenced in a TTree entry, the corresponding entry in the TTree's  //
// TBranchRef::fRefTable contains the index of the branch that          //
// needs to be loaded to bring the object into memory.                  //
//                                                                      //
// Persistency of a TRefTable is split into two parts:                  //
// * entry specific information is stored (read) by FillBuffer          //
//   (ReadBuffer). For each referenced object the object's fUniqueID    //
//   and the referencing TRef::fPID is stored (to allow the TRefTable   //
//   to autoload references created by different processes).            //
// * non-entry specific, i.e. global information is stored (read) by    //
//   the Streamer function. This comprises all members marked as        //
//   persistent.                                                        //
//                                                                      //
// As TObject::fUniqueID is only unique for a given TProcessID, a table //
// of unique IDs is kept for each used TProcessID. There is no natural  //
// order of TProcessIDs, so TRefTable stores a vector of the TGUID of   //
// all known TProcessIDs in fProcessGUIDs; the index of a TProcessID in //
// this vector defines the index of the auto-loading info in fParentIDs //
// for that TProcessID. The mapping of TProcessID* to index is cached   //
// for quick non-persistent lookup.                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRefTable.h"
#include "TObjArray.h"
#include "TProcessID.h"
#include <algorithm>

TRefTable *TRefTable::fgRefTable = 0;

ClassImp(TRefTable)
//______________________________________________________________________________
TRefTable::TRefTable() : fNumPIDs(0), fAllocSize(0), fN(0), fParentIDs(0), fParentID(-1),
                         fDefaultSize(10), fUID(0), fUIDContext(0), fSize(0), fParents(0), fOwner(0)
{
   // Default constructor for I/O.

   fgRefTable   = this;
}

//______________________________________________________________________________
TRefTable::TRefTable(TObject *owner, Int_t size) :
     fNumPIDs(0), fAllocSize(0), fN(0), fParentIDs(0), fParentID(-1),
     fDefaultSize(size<10 ? 10 : size), fUID(0), fUIDContext(0), fSize(0), fParents(new TObjArray(1)), fOwner(owner)
{
   // Create a TRefTable with initial size.

   fgRefTable   = this;
}

//______________________________________________________________________________
TRefTable::~TRefTable()
{
   // Destructor.

   delete [] fAllocSize;
   delete [] fN;
   for (Int_t pid = 0; pid < fNumPIDs; ++pid) {
      delete [] fParentIDs[pid];
   }
   delete [] fParentIDs;
   delete fParents;
   if (fgRefTable == this) fgRefTable = 0;
}

//______________________________________________________________________________
Int_t TRefTable::Add(Int_t uid, TProcessID *context)
{
   // Add a new uid to the table.
   // we add a new pair (uid,fparent) to the map
   // This function is called by TObject::Streamer or TStreamerInfo::WriteBuffer

   if (!context)
      context = TProcessID::GetSessionProcessID();
   Int_t iid = GetInternalIdxForPID(context);

   Int_t newsize = 0;
   uid = uid & 0xffffff;
   if (uid >= fAllocSize[iid]) {
      newsize = uid + uid / 2;
      if (newsize < fDefaultSize)
         newsize = fDefaultSize;
      newsize = ExpandForIID(iid, newsize);
   }
   if (newsize < 0) {
      Error("Add", "Cannot allocate space to store uid=%d", uid);
      return -1;
   }
   if (fParentID < 0) {
      Error("Add", "SetParent must be called before adding uid=%d", uid);
      return -1;
   }
   fParentIDs[iid][uid] = fParentID + 1;
   if (uid >= fN[iid]) fN[iid] = uid + 1;
   return uid;
}


//______________________________________________________________________________
Int_t TRefTable::AddInternalIdxForPID(TProcessID *procid)
{
   // Add the internal index for fProcessIDs, fAllocSize, etc given a PID.

   if (!procid)
      procid = TProcessID::GetSessionProcessID();
   Int_t pid = procid->GetUniqueID();
   if (fMapPIDtoInternal.size() <= (size_t) pid)
      fMapPIDtoInternal.resize(TProcessID::GetNProcessIDs(), -1);

   Int_t iid = fMapPIDtoInternal[pid];
   if (iid == -1) {
      // need to update
      iid = FindPIDGUID(procid->GetTitle());
      if (iid == -1) {
         fProcessGUIDs.push_back(procid->GetTitle());
         iid = fProcessGUIDs.size() - 1;
      }
      fMapPIDtoInternal[pid] = iid;
   }

   ExpandPIDs(iid + 1);
   return iid;
}

//______________________________________________________________________________
void TRefTable::Clear(Option_t * /*option*/ )
{
   // Clear all entries in the table.

   for (Int_t iid = 0; iid < fNumPIDs; ++iid) {
      memset(fParentIDs[iid], 0, sizeof(Int_t) * fN[iid]);
   }
   memset(fN, 0, sizeof(Int_t) * fNumPIDs);
   fParentID = -1;
}

//______________________________________________________________________________
Int_t TRefTable::Expand(Int_t pid, Int_t newsize)
{
   // Expand fParentIDs to newsize for ProcessID pid.

   Int_t iid = GetInternalIdxForPID(pid);
   if (iid < 0) return -1;
   return ExpandForIID(iid, newsize);
}

//______________________________________________________________________________
Int_t TRefTable::ExpandForIID(Int_t iid, Int_t newsize)
{
   // Expand fParentIDs to newsize for internel ProcessID index iid.

   if (newsize < 0)  return newsize;
   if (newsize != fAllocSize[iid]) {
      Int_t *temp = fParentIDs[iid];
      if (newsize != 0) {
         fParentIDs[iid] = new Int_t[newsize];
         if (newsize < fAllocSize[iid])
            memcpy(fParentIDs[iid], temp, newsize * sizeof(Int_t));
         else {
            memcpy(fParentIDs[iid], temp, fAllocSize[iid] * sizeof(Int_t));
            memset(&fParentIDs[iid][fAllocSize[iid]], 0,
                   (newsize - fAllocSize[iid]) * sizeof(Int_t));
         }
      } else {
         fParentIDs[iid] = 0;
      }
      if (fAllocSize[iid]) delete [] temp;
      fAllocSize[iid] = newsize;
   }
   return newsize;
}

//______________________________________________________________________________
void TRefTable::ExpandPIDs(Int_t numpids)
{
   // Expand the arrays of managed PIDs

   if (numpids <= fNumPIDs) return;

   // else add to internal tables
   Int_t oldNumPIDs = fNumPIDs;
   fNumPIDs  = numpids;

   Int_t *temp = fAllocSize;
   fAllocSize = new Int_t[fNumPIDs];
   if (temp) memcpy(fAllocSize, temp, oldNumPIDs * sizeof(Int_t));
   memset(&fAllocSize[oldNumPIDs], 0,
          (fNumPIDs - oldNumPIDs) * sizeof(Int_t));
   delete [] temp;

   temp = fN;
   fN = new Int_t[fNumPIDs];
   if (temp) memcpy(fN, temp, oldNumPIDs * sizeof(Int_t));
   memset(&fN[oldNumPIDs], 0, (fNumPIDs - oldNumPIDs) * sizeof(Int_t));
   delete [] temp;

   Int_t **temp2 = fParentIDs;
   fParentIDs = new Int_t *[fNumPIDs];
   if (temp2) memcpy(fParentIDs, temp2, oldNumPIDs * sizeof(Int_t *));
   memset(&fParentIDs[oldNumPIDs], 0,
          (fNumPIDs - oldNumPIDs) * sizeof(Int_t*));
}

//______________________________________________________________________________
void TRefTable::FillBuffer(TBuffer & b)
{
   // Fill buffer b with the fN elements in fParentdIDs.
   // This function is called by TBranchRef::FillLeaves.

   b << -fNumPIDs; // write out "-" to signal new TRefTable buffer format using PID table
   for (Int_t iid = 0; iid < fNumPIDs; ++iid) {
      b << fN[iid];
      b.WriteFastArray(fParentIDs[iid], fN[iid]);
   }
}


//______________________________________________________________________________
Int_t TRefTable::FindPIDGUID(const char *guid) const
{
   // Get fProcessGUIDs' index of the TProcessID with GUID guid
   std::vector<std::string>::const_iterator posPID
      = std::find(fProcessGUIDs.begin(), fProcessGUIDs.end(), guid);
   if (posPID == fProcessGUIDs.end()) return -1;
   return posPID - fProcessGUIDs.begin();
}

//______________________________________________________________________________
TObject *TRefTable::GetParent(Int_t uid, TProcessID *context /* =0 */ ) const
{
   // Return object corresponding to uid.
   if (!fParents) return 0;

   Int_t iid = -1;
   if (!context) context = TProcessID::GetSessionProcessID();
   iid = GetInternalIdxForPID(context);

   uid = uid & 0xFFFFFF;
   if (uid < 0 || uid >= fN[iid]) return 0;
   Int_t pnumber = fParentIDs[iid][uid] - 1;
   Int_t nparents = fParents->GetEntriesFast();
   if (pnumber < 0 || pnumber >= nparents) return 0;
   return fParents->UncheckedAt(pnumber);
}

//______________________________________________________________________________
Int_t TRefTable::GetInternalIdxForPID(TProcessID *procid) const
{
   // Get the index for fProcessIDs, fAllocSize, etc given a PID.
   // Uses fMapPIDtoInternal and the pid's GUID / fProcessGUID

   return const_cast <TRefTable*>(this)->AddInternalIdxForPID(procid);
}

//______________________________________________________________________________
Int_t TRefTable::GetInternalIdxForPID(Int_t pid) const
{
   // Get the index for fProcessIDs, fAllocSize, etc given a PID.
   // Uses fMapPIDtoInternal and the pid's GUID / fProcessGUID

   return GetInternalIdxForPID(TProcessID::GetProcessID(pid));
}


//______________________________________________________________________________
TRefTable *TRefTable::GetRefTable()
{
   // Static function returning the current TRefTable.

   return fgRefTable;
}

//______________________________________________________________________________
Bool_t TRefTable::Notify()
{
   // This function is called by TRef::Streamer or TStreamerInfo::ReadBuffer
   // when reading a reference.
   // This function, in turns, notifies the TRefTable owner for action.
   // eg, when the owner is a TBranchRef, TBranchRef::Notify is called
   // to read the branch containing the referenced object.

   return fOwner->Notify();
}

//______________________________________________________________________________
void TRefTable::ReadBuffer(TBuffer &b)
{
   // Fill buffer b with the fN elements in fParentdIDs.
   // This function is called by TBranchRef::ReadLeaves

   Int_t firstInt = 0;          // we don't know yet what it means
   b >> firstInt;

   Int_t numIids = -1;
   Int_t startIid = 0;
   if (firstInt < 0) numIids = -firstInt; // new format
   else {
      // old format, only one PID
      numIids = 1;

      TProcessID *fileProcessID = b.GetLastProcessID(this);

      startIid = GetInternalIdxForPID(fileProcessID);
      if (startIid == -1) {
         fProcessGUIDs.push_back(fileProcessID->GetTitle());
         startIid = fProcessGUIDs.size() - 1;
      }
      numIids += startIid;
   }

   ExpandPIDs(numIids);
   for (Int_t iid = startIid; iid < numIids; ++iid) {
      Int_t newN = 0;
      if (firstInt < 0) b >> newN;
      else newN = firstInt;
      if (newN > fAllocSize[iid])
         ExpandForIID(iid, newN + newN / 2);
      fN[iid] = newN;
      b.ReadFastArray(fParentIDs[iid], fN[iid]);
   }
}

//______________________________________________________________________________
void TRefTable::Reset(Option_t * /*option*/ )
{
   // Clear all entries in the table.
   Clear();
   if (fParents) fParents->Clear();
}

//______________________________________________________________________________
Int_t TRefTable::SetParent(const TObject* parent, Int_t branchID)
{
   // -- Set current parent object, typically a branch of a tree.
   //
   // This function is called by TBranchElement::Fill() and by
   // TBranchElement::GetEntry().
   //
   if (!fParents) {
      return -1;
   }
   Int_t nparents = fParents->GetEntriesFast();
   if (branchID != -1) {
      // -- The branch already has an id cached, just use it.
      fParentID = branchID;
   }
   else {
      // -- The branch does *not* have an id cached, find it or generate one.
      // Lookup the branch.
      fParentID = fParents->IndexOf(parent);
      if (fParentID < 0) {
         // -- The branch is not known, generate an id number.
         fParents->AddAtAndExpand(const_cast<TObject*>(parent), nparents);
         fParentID = nparents;
      }
   }
   return fParentID;
}

//______________________________________________________________________________
void TRefTable::SetRefTable(TRefTable *table)
{
   // Static function setting the current TRefTable.

   fgRefTable = table;
}

//______________________________________________________________________________
void TRefTable::Streamer(TBuffer &R__b)
{
   // Stream an object of class TRefTable.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(TRefTable::Class(),this);
   } else {
      R__b.WriteClassBuffer(TRefTable::Class(),this);
      //make sure that all TProcessIDs referenced in the Tree are put to the buffer
      //this is important in case the buffer is a TMessage to be sent through a TSocket
#if 0
      TObjArray *pids = TProcessID::GetPIDs();
      Int_t npids = pids->GetEntries();
      Int_t npid2 = fProcessGUIDs.size();
      for (Int_t i = 0; i < npid2; i++) {
         TProcessID *pid;
         for (Int_t ipid = 0;ipid<npids;ipid++) {
            pid = (TProcessID*)pids->At(ipid);
            if (!pid) continue;
            if (!strcmp(pid->GetTitle(),fProcessGUIDs[i].c_str()))
               R__b.WriteProcessID(pid);
         }
      }
#endif
   }
}
