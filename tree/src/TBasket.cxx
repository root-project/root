// @(#)root/tree:$Name:  $:$Id: TBasket.cxx,v 1.3 2000/12/20 15:45:37 brun Exp $
// Author: Rene Brun   19/01/96
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBasket.h"
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include "TMath.h"

R__EXTERN  TBranch *gBranch;

extern "C" void R__zip (Int_t cxlevel, Int_t *nin, char *bufin, Int_t *lout, char *bufout, Int_t *nout);
extern "C" void R__unzip(Int_t *nin, char *bufin, Int_t *lout, char *bufout, Int_t *nout);

const UInt_t kDisplacementMask = 0xFF000000;  // In the streamer the two highest bytes of
                                             // the fEntryOffset are used to stored displacement.

ClassImp(TBasket)

//_______________________________________________________________________
//
//  Manages buffers for branches of a Tree.
//  See picture in TTree.
//

//_______________________________________________________________________
TBasket::TBasket()
{
//               Default contructor
   fDisplacement  = 0;
   fEntryOffset   = 0;
   fZipBuffer     = 0;
   fBufferRef     = 0;
   fBuffer        = 0;
   fHeaderOnly    = kFALSE;
}

//_______________________________________________________________________
TBasket::TBasket(const char *name, const char *title, TBranch *branch)
{
//            Basket normal constructor
//
//
   SetName(name);
   SetTitle(title);
   fClassName   = "TBasket";
   fBufferSize  = branch->GetBasketSize();
   fNevBufSize  = branch->GetEntryOffsetLen();
   fNevBuf      = 0;
   fEntryOffset = 0;  //Must be set to 0 before calling Sizeof
   fDisplacement= 0;  //Must be set to 0 before calling Sizeof
   fBuffer      = 0;  //Must be set to 0 before calling Sizeof
   fBufferRef   = new TBuffer(TBuffer::kWrite, fBufferSize);
   fHeaderOnly  = kTRUE;
   fLast        = 0; // RDK: Must initialize before calling Streamer()
   Streamer(*fBufferRef);
   fKeylen      = fBufferRef->Length();
   fObjlen      = fBufferSize - fKeylen;
   fLast        = fKeylen;
   fBuffer      = 0;
   fBranch      = branch;
   fZipBuffer   = 0;
   fHeaderOnly  = kFALSE;
   if (fNevBufSize) fEntryOffset = new Int_t[fNevBufSize];
   branch->GetTree()->IncrementTotalBuffers(fBufferSize);
}

//_______________________________________________________________________
TBasket::~TBasket()
{
//          Basket destructor
//

   if (fDisplacement) delete [] fDisplacement;
   if (fEntryOffset)  delete [] fEntryOffset;
   if (fZipBuffer)    delete [] fZipBuffer;
//   if (fBufferRef)   delete fBufferRef; //this is done in the TKey destructor
//   fBufferRef   = 0;
   fDisplacement= 0;
   fEntryOffset = 0;
   fZipBuffer   = 0;
}

//_______________________________________________________________________
void TBasket::AdjustSize(Int_t newsize)
{
//      Increase the size of the current fBuffer up to newsize

   char *newbuf = (char*)TStorage::ReAlloc(fBuffer,newsize,fBufferSize);
   fBufferSize  = newsize;
   fBuffer      = newbuf;
}


//_______________________________________________________________________
Int_t TBasket::DropBuffers()
{
//      Drop buffers of this basket if it is not the current basket

   if (!fBuffer && !fBufferRef) return 0;
//   delete [] fBuffer;
   if (fDisplacement) delete [] fDisplacement;
   if (fEntryOffset)  delete [] fEntryOffset;
   if (fBufferRef)    delete fBufferRef;
   fBufferRef   = 0;
   fBuffer      = 0;
   fDisplacement= 0;
   fEntryOffset = 0;
   fBranch->GetTree()->IncrementTotalBuffers(-fBufferSize);
   return fBufferSize;
}

//_______________________________________________________________________
Int_t TBasket::GetEntryPointer(Int_t entry)
{
//*-*-*-*-*-*-*Get pointer to buffer for internal entry*-*-*-*-*-*
//*-*          ========================================

   Int_t offset;
   if (fEntryOffset) offset = fEntryOffset[entry];
   else              offset = fKeylen + entry*fNevBufSize;
   fBufferRef->SetBufferOffset(offset);
   return offset;
}

//_______________________________________________________________________
Int_t TBasket::ReadBasketBuffers(Seek_t pos, Int_t len, TFile *file)
{
//*-*-*-*-*-*-*-*-*Read basket buffers in memory and cleanup*-*-*-*-*-*-*
//*-*              =========================================
//
//       Read a basket buffer. Check if buffers of previous ReadBasket
//       should not be dropped. Remember, we keep buffers
//       in memory up to fMaxVirtualSize.

   Int_t badread= 0;
   TDirectory *cursav = gDirectory;
   gBranch->GetDirectory()->cd();

   if (gBranch->GetTree()->MemoryFull(fBufferSize)) gBranch->DropBaskets();

   fBufferRef = new TBuffer(TBuffer::kRead, len);
   char *buffer = fBufferRef->Buffer();
   file->Seek(pos);
   file->ReadBuffer(buffer,len);
   Streamer(*fBufferRef);
   if (fObjlen > fNbytes-fKeylen) {
      fBuffer = new char[fObjlen+fKeylen];
      memcpy(fBuffer,buffer,fKeylen);
      char *objbuf = fBuffer + fKeylen;
      Int_t nin = fNbytes-fKeylen;
      Int_t nout;
      R__unzip(&nin, &buffer[fKeylen], &fObjlen, objbuf, &nout);
      if (nout != fObjlen) {
         Error("ReadBasketBuffers", "fObjlen = %d, nout = %d", fObjlen, nout);
         badread = 1;
      }
      delete [] buffer;
      fBufferRef->SetBuffer(fBuffer, fObjlen+fKeylen );
   } else {
      fBuffer = fBufferRef->Buffer();
   }
   cursav->cd();

   fBranch->GetTree()->IncrementTotalBuffers(fBufferSize);

//        read offsets table
   if (!fBranch->GetEntryOffsetLen()) return badread;
   delete [] fEntryOffset;
   fEntryOffset = 0;
   fBufferRef->SetBufferOffset(fLast);
   fBufferRef->ReadArray(fEntryOffset);
   delete [] fDisplacement;
   fDisplacement = 0;
   if (fBufferRef->Length() != fBufferRef->BufferSize()) {
     // There is more data in the buffer!  It is the diplacement
     // array.
     fBufferRef->ReadArray(fDisplacement);
   }
   return badread;
}

//_______________________________________________________________________
Int_t TBasket::ReadBasketBytes(Seek_t pos, TFile *file)
{
//*-*-*-*-*-*-*-*-*Read basket buffers in memory and cleanup*-*-*-*-*-*-*
//*-*              =========================================
//
//       Read first bytes of a logical record starting at position pos
//       return record length (first 4 bytes of record)

   const Int_t len = 128;
   char buffer[len];
   Int_t keylen;
   file->GetRecordHeader(buffer, pos,len, fNbytes, fObjlen, keylen);
   fKeylen = keylen;
   return fNbytes;
}

//_______________________________________________________________________
void TBasket::SetReadMode()
{
   // Set read mode of basket.

   fLast = fBufferRef->Length();
   fBufferRef->SetReadMode();
}

//_______________________________________________________________________
void TBasket::SetWriteMode()
{
   // Set write mode of basket.

   fBufferRef->SetWriteMode();
   fBufferRef->SetBufferOffset(fLast);
}

//_______________________________________________________________________
void TBasket::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================

   char flag;
   if (b.IsReading()) {
      fBranch = gBranch;
      TKey::Streamer(b); //this must be first
      Version_t v = b.ReadVersion();
      b >> fBufferSize;
      b >> fNevBufSize;
      b >> fNevBuf;
      b >> fLast;
      b >> flag;
      if (fLast > fBufferSize) fBufferSize = fLast;
      if (!flag) return;
      if (flag%10 != 2) {
         delete [] fEntryOffset;
         fEntryOffset = new Int_t[fNevBufSize];
         if (fNevBuf) b.ReadArray(fEntryOffset);
         if (20<flag && flag<40) {
           for(int i=0; i<fNevBuf; i++){
              fEntryOffset[i] &= ~kDisplacementMask;
           }
         }
         if (flag>40) {
           fDisplacement = new Int_t[fNevBufSize];
           b.ReadArray(fDisplacement);
         }
      }
      if (flag == 1 || flag > 10) {
         fBufferRef = new TBuffer(TBuffer::kRead,fBufferSize);
         char *buf  = fBufferRef->Buffer();
         if (v > 1) b.ReadFastArray(buf,fLast);
         else       b.ReadArray(buf);
         fBufferRef->SetBufferOffset(fLast);
         fBranch->GetTree()->IncrementTotalBuffers(fBufferSize);
      }
   } else {
      TKey::Streamer(b);   //this must be first
      b.WriteVersion(TBasket::IsA());
      if (fBufferRef && !fHeaderOnly) fLast = fBufferRef->Length();
      if (fLast > fBufferSize) fBufferSize = fLast;
      b << fBufferSize;
      b << fNevBufSize;
      b << fNevBuf;
      b << fLast;
      flag = 1;
      if (!fEntryOffset)  flag  = 2;
      if (fBufferRef)     flag += 10;
      if (fDisplacement)  flag += 40;
      if (fHeaderOnly)    flag  = 0;
      b << flag;
      if (fHeaderOnly) return;
      if (fEntryOffset && fNevBuf) {
         b.WriteArray(fEntryOffset, fNevBuf);
         if (fDisplacement) b.WriteArray(fDisplacement, fNevBuf);
      }
      if (fBufferRef) {
         char *buf  = fBufferRef->Buffer();
         b.WriteFastArray(buf, fLast);
      }
   }
}

//_______________________________________________________________________
void TBasket::Update(Int_t offset, Int_t skipped)
{
//    Update basket header and EntryOffset table

   if (fEntryOffset) {
      if (fNevBuf+1 >= fNevBufSize) {
         Int_t newsize = TMath::Max(10,2*fNevBufSize);
         Int_t *newoff = (Int_t*)TStorage::ReAlloc(fEntryOffset,
                               newsize*sizeof(Int_t),fNevBufSize*sizeof(Int_t));
         if (fDisplacement) {
            Int_t *newdisp = (Int_t*)TStorage::ReAlloc(fDisplacement,
                               newsize*sizeof(Int_t),fNevBufSize*sizeof(Int_t));
            fDisplacement = newdisp;
         }
         fEntryOffset  = newoff;
         fNevBufSize   = newsize;
         //Update branch only for the first 10 baskets
         if (fBranch->GetWriteBasket() < 10) fBranch->SetEntryOffsetLen(newsize);

      }
      fEntryOffset[fNevBuf] = offset;

      if (skipped!=offset && !fDisplacement){
         fDisplacement = new Int_t[fNevBufSize];
         for (Int_t i = 0; i<fNevBufSize; i++) fDisplacement[i] = fEntryOffset[i];
      }
      if (fDisplacement) {
         fDisplacement[fNevBuf] = skipped;
         fBufferRef->SetBufferDisplacement(skipped);
      }
   }

   fNevBuf++;
}

//_______________________________________________________________________
Int_t TBasket::WriteBuffer()
{
//    Write buffer of this basket on the current file

   const Int_t kWrite = 1;
   TDirectory *cursav = gDirectory;
   TFile *file = fBranch->GetFile(kWrite);
   if (!file) return 0;

   fBranch->GetDirectory()->cd();
   if (gFile ? !gFile->IsWritable() : 1) { cursav->cd(); return 0;}
//*-*- Transfer fEntryOffset table at the end of fBuffer. Offsets to fBuffer
//     are transformed in entry length to optimize compression algorithm.
   fLast      = fBufferRef->Length();
   if (fEntryOffset) {
      fBufferRef->WriteArray(fEntryOffset,fNevBuf+1);
      delete [] fEntryOffset; fEntryOffset = 0;
      if (fDisplacement) {
        fBufferRef->WriteArray(fDisplacement,fNevBuf+1);
        delete [] fDisplacement; fDisplacement = 0;
      }
   }

   Int_t lbuf, nout;
   lbuf       = fBufferRef->Length();
   fObjlen    = lbuf - fKeylen;

   fHeaderOnly = kTRUE;
   fCycle = fBranch->GetWriteBasket();
   Int_t cxlevel = fBranch->GetCompressionLevel();
   if (cxlevel) {
      if (cxlevel == 2) cxlevel--;
      Int_t buflen = fKeylen + fObjlen + 28; //add 28 bytes in case object is placed in a deleted gap
      fBuffer = new char[buflen];
      char *objbuf = fBufferRef->Buffer() + fKeylen;
      Int_t bufmax = buflen-fKeylen;
      R__zip(cxlevel, &fObjlen, objbuf, &bufmax, &fBuffer[fKeylen], &nout);
      if (nout >= fObjlen) {
         delete [] fBuffer;
         fBuffer = fBufferRef->Buffer();
         Create(fObjlen);
         fBufferRef->SetBufferOffset(0);
         Streamer(*fBufferRef);         //write key itself again
         nout = fObjlen;
      } else {
         Create(nout);
         fBufferRef->SetBufferOffset(0);
         Streamer(*fBufferRef);         //write key itself again
         memcpy(fBuffer,fBufferRef->Buffer(),fKeylen);
         delete fBufferRef; fBufferRef = 0;
      }
   } else {
      fBuffer = fBufferRef->Buffer();
      Create(fObjlen);
      fBufferRef->SetBufferOffset(0);
      Streamer(*fBufferRef);         //write key itself again
      nout = fObjlen;
   }

//  TKey::WriteFile calls FillBuffer. TBasket inherits from TKey, hence
//  TBasket::FillBuffer is called.
   TKey::WriteFile(0);
   fHeaderOnly = kFALSE;

   cursav->cd();
   return fKeylen+nout;
}
