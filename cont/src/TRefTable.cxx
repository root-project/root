// @(#)root/cont:$Name:  $:$Id: TRefTable.cxx,v 1.25 2004/01/30 13:58:48 brun Exp $
// Author: Rene Brun   28/09/2001

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//
// TRefTable
//
// A TRefTable maintains the association between a referenced object    //
// and the parent object supporting this referenced object.             //
// The parent object is typically a branch of a TTree.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRefTable.h"

ClassImp(TRefTable)

//______________________________________________________________________________
TRefTable::TRefTable(Int_t size)
{
   // Create a TRefTable with initial size

   if (size < 10) size = 10;
   fSize = size;
   fN = 0;
   fParentID  = -1;
   fParentIDs = new Int_t[fSize];
   for (Int_t i=0;i<fSize;i++) {
      fParentIDs[i] = 0;
   }
   fParents = new TObjArray(1);
}

//______________________________________________________________________________
Int_t TRefTable::Add(Int_t uid)
{
   // add a new uid to the table
   // we add a new pair (uid,fparent) to the map
   // This function is called by TObject::Streamer or TStreamerInfo::WriteBuffer
   
   if (uid <= 0) {
      Error("Add","Attempt to add an invalid uid=%d",uid);
      return uid;
   }
   Int_t newsize = 0;
   if (uid >= fSize) newsize = Expand(uid+uid/2);
   if (newsize < 0) {
      Error("Add","Cannot allocate space to store uid=%d",uid);
      return -1;
   }
   if (fParentID < 0) {
      Error("Add","SetParent must be called before adding uid=%d",uid);
      return -1;
   }
   fParentIDs[uid] = fParentID+1;
   if (uid >= fN) fN = uid+1;
   return uid;
}

//______________________________________________________________________________
void TRefTable::Clear(Option_t * /*option*/)
{
   // clear all entries in the table
   for (Int_t i=0;i<fN;i++) {
      fParentIDs[i] = 0;
   }
   fN = 0;
   fParentID = -1;
}

//______________________________________________________________________________
Int_t TRefTable::Expand(Int_t newsize)
{
   // expand fParentID to newsize
   
   if (newsize < 0) return newsize;
   if (newsize != fSize) {
      Int_t *temp = fParentIDs;
      if (newsize != 0) {
         fParentIDs = new Int_t[newsize];
         if (newsize < fSize) memcpy(fParentIDs,temp, newsize*sizeof(Int_t));
         else {
            memcpy(fParentIDs,temp,fSize*sizeof(Int_t));
            memset(&fParentIDs[fSize],0,(newsize-fSize)*sizeof(Int_t));
         }
      } else {
        fParentIDs = 0;
      }
      if (fSize) delete [] temp;
      fSize = newsize;
   }
   return newsize;
}

//______________________________________________________________________________
void TRefTable::FillBuffer(TBuffer &b)
{
   // fill buffer b with the fN elements in fParentdIDs
   // This function is called by TBranchRef::FillLeaves
   
   b << fN;
   b.WriteFastArray(fParentIDs,fN);
}

//______________________________________________________________________________
TObject *TRefTable::GetParent(Int_t uid) const
{
   // return object corresponding to uid
   if (uid < 0 || uid >= fN) return 0;
   Int_t pnumber = fParentIDs[uid]-1;
   Int_t nparents = fParents->GetEntriesFast();
   if (pnumber < 0 || pnumber >= nparents) return 0;
   return fParents->UncheckedAt(pnumber);
}

//______________________________________________________________________________
void TRefTable::ReadBuffer(TBuffer &b)
{
   // fill buffer b with the fN elements in fParentdIDs
   // This function is called by TBranchRef::ReadLeaves
   
   b >> fN;
   if (fN > fSize) fSize = Expand(fN +fN/2);
   b.ReadFastArray(fParentIDs,fN);
}
 
//______________________________________________________________________________
Int_t TRefTable::SetParent(const TObject *parent)
{
   // Set Current parent object
   // The parent object is typically a branch of a Tree.
   // This function is called by TBranchElement::Fill
   
   Int_t nparents = fParents->GetEntriesFast();
   Int_t ind = fParents->IndexOf(parent);
   if (ind >= 0) {
      fParentID = ind;
   } else {
      fParents->AddAtAndExpand((TObject*)parent,nparents);
      fParentID = nparents;
   }
   return fParentID;
}
