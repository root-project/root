// @(#)root/io:$Id$
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
#include "TFile.h"
#include "Bytes.h"
#include "Riostream.h"

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
   // TFree default constructor.

   fFirst = fLast = 0;
}

//______________________________________________________________________________
TFree::TFree(TList *lfree, Long64_t first, Long64_t last)
{
   // Constructor for a FREE segment.

   fFirst = first;
   fLast  = last;
   lfree->Add(this);
}

//______________________________________________________________________________
TFree *TFree::AddFree(TList *lfree, Long64_t first, Long64_t last)
{
// Add a new free segment to the list of free segments.
//
//  If last just precedes an existing free segment, then first becomes
//     the new starting location of the free segment.
//  if first just follows an existing free segment, then last becomes
//     the new ending location of the free segment.
//  if first just follows an existing free segment AND last just precedes
//     an existing free segment, these two segments are merged into
//     one single segment.
//

   TFree *idcur = this;
   while (idcur) {
      Long64_t curfirst = idcur->GetFirst();
      Long64_t curlast  = idcur->GetLast();
      if (curlast == first-1) {
         idcur->SetLast(last);
         TFree *idnext = (TFree*)lfree->After(idcur);
         if (idnext == 0) return idcur;
         if (idnext->GetFirst() > last+1) return idcur;
         idcur->SetLast( idnext->GetLast() );
         lfree->Remove(idnext);
         delete idnext;
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
   // TFree Destructor.

}

//______________________________________________________________________________
void TFree::FillBuffer(char *&buffer)
{
   // Encode FREE structure into output buffer.

   Version_t version = TFree::Class_Version();
   if (fLast > TFile::kStartBigFile) version += 1000;
   tobuf(buffer, version);
//printf("TFree::fillBuffer, fFirst=%lld, fLast=%lld, version=%d\n",fFirst,fLast,version);
   if (version > 1000) {
      tobuf(buffer, fFirst);
      tobuf(buffer, fLast);
   } else {
      tobuf(buffer, (Int_t)fFirst);
      tobuf(buffer, (Int_t)fLast);
   }
}

//______________________________________________________________________________
TFree *TFree::GetBestFree(TList *lfree, Int_t nbytes)
{
   // Return the best free segment where to store nbytes.

   TFree *idcur = this;
   if (idcur == 0) return 0;
   TFree *idcur1 = 0;
   do {
      Long64_t nleft = Long64_t(idcur->fLast - idcur->fFirst +1);
      if (nleft == nbytes) {
         // Found an exact match
         return idcur;
      }
      if(nleft > (Long64_t)(nbytes+3)) {
         if (idcur1 == 0) {
            idcur1=idcur;
         }
      }
      idcur = (TFree*)lfree->After(idcur);
   } while (idcur !=0);

   // return first segment >nbytes
   if (idcur1) return idcur1;

   //try big file
   idcur = (TFree*)lfree->Last();
   Long64_t last = idcur->fLast+1000000000;
   idcur->SetLast(last);
   return idcur;
}

//______________________________________________________________________________
void TFree::ls(Option_t *) const
{
   // List free segment contents.

   std::cout <<"Free Segment: "<<fFirst<<"\t"<<fLast<<std::endl;
}

//______________________________________________________________________________
void TFree::ReadBuffer(char *&buffer)
{
   // Decode one FREE structure from input buffer

   Version_t version;
   frombuf(buffer, &version);
   if (version > 1000) {
      frombuf(buffer, &fFirst);
      frombuf(buffer, &fLast);
   } else {
      Int_t first,last;
      frombuf(buffer, &first);  fFirst = (Long64_t)first;
      frombuf(buffer, &last);   fLast  = (Long64_t)last;
   }
}

//______________________________________________________________________________
Int_t TFree::Sizeof() const
{
   // return number of bytes occupied by this TFree on permanent storage

   if (fLast > TFile::kStartBigFile) return 18;
   else                              return 10;
}

