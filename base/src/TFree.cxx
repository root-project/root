// @(#)root/base:$Name:  $:$Id: TFree.cxx,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $
// Author: Rene Brun   28/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TFree.h"
#include "TList.h"
#include "Bytes.h"

ClassImp(TFree)

//______________________________________________________________________________
//
// Service class for TFile.
// Each file has a linked list of free segments. Each free segment
// is described by its firts and last address.
// When an object is written to a file via TObject::Write, a new Key (see TKey)
// is created. The first free segment big enough to accomodate the object
// is used.
// If the object size has a length corresponding to the size of the free segment,
// the free segment is deleted from the list of free segments.
// When an object is deleted from a file, a new TFree object is generated.
// If the deleted object is contiguous to an already deleted object, the free
// segments are merged in one single segment.
//

//______________________________________________________________________________
TFree::TFree()
{
//*-*-*-*-*-*-*-*-*-*-*TFree default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================

}

//______________________________________________________________________________
TFree::TFree(TList *lfree, Seek_t first, Seek_t last)
{
//*-*-*-*-*-*-*-*-*-*-*Constructor for a FREE segment*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============================
   fFirst = first;
   fLast  = last;
   lfree->Add(this);
}

//______________________________________________________________________________
TFree *TFree::AddFree(TList *lfree, Seek_t first, Seek_t last)
{
//*-*-*-*-*-*-*-*-*-*Add a new free segment to the list of free segments*-*-*
//*-*                ===================================================
//  If last just preceedes an existing free segment, then first becomes
//     the new starting location of the free segment.
//  if first just follows an existing free segment, then last becomes
//     the new ending location of the free segment.
//  if first just follows an existing free segment AND last just preceedes
//     an existing free segment, these two segments are merged into
//     one single segment.
//
   TFree *idcur = this;
   while (idcur) {
      Seek_t curfirst = idcur->GetFirst();
      Seek_t curlast  = idcur->GetLast();
      if (curlast == first-1) {
         idcur->SetLast(last);
         TFree *idnext = (TFree*)lfree->After(idcur);
         if (idnext == 0) return idcur;
         if (idnext->GetFirst() > last+1) return idcur;
         idcur->SetLast( idnext->GetLast() );
         lfree->Remove(idnext);
         return idcur;
      }
      if (curfirst == last+1) {
         idcur->SetFirst(first);
         return idcur;
      }
      if (first < curfirst) {
         TFree * newfree = new TFree();
         newfree->SetFirst(first);
         newfree->SetLast(last);
         lfree->AddBefore(idcur, newfree);
         return newfree;
      }
      idcur = (TFree*)lfree->After(idcur);
   }
   return 0;
}

//______________________________________________________________________________
TFree::~TFree()
{
//*-*-*-*-*-*-*-*-*-*-*-*TFree Destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ================

}

//______________________________________________________________________________
void TFree::FillBuffer(char *&buffer)
{
//*-*-*-*-*-*-*-*-*-*-*Encode FREE structure into output buffer*-*-*-*-*-*-*
//*-*                  ========================================
   Version_t version = TFree::Class_Version();
   tobuf(buffer, version);
   tobuf(buffer, fFirst);
   tobuf(buffer, fLast);
}

//______________________________________________________________________________
TFree *TFree::GetBestFree(TList *lfree, Int_t nbytes)
{
//*-*-*-*-*-*-*-*-*-*Return the best free segment where to store nbytes*-*-*-*
//*-*                ==================================================
   TFree *idcur = this;
   if (idcur == 0) return 0;
   TFree *idcur1 = 0;
   do {
      Int_t nleft = Int_t(idcur->fLast - idcur->fFirst +1);
      if (nleft == nbytes) return idcur;             //*-* found an exact match
      if(nleft > nbytes+3) if (idcur1 == 0) idcur1=idcur;
      idcur = (TFree*)lfree->After(idcur);
   } while (idcur !=0);
   return idcur1;                                    //*-* return first segment >nbytes
}

//______________________________________________________________________________
void TFree::ReadBuffer(char *&buffer)
{
//*-*-*-*-*-*-*-*-*-*-*-*Decode one FREE structure from input buffer*-*-*-*-*
//*-*                    ===========================================
   Version_t version;
   frombuf(buffer, &version);
   frombuf(buffer, &fFirst);
   frombuf(buffer, &fLast);
}
