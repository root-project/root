// @(#)root/tree:$Name:  $:$Id: TBasket.cxx,v 1.44 2007/01/19 16:48:00 brun Exp $
// Author: Rene Brun   19/01/96
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBasket.h"
#include "TBufferFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include "TBufferFile.h"
#include "TMath.h"

R__EXTERN  TBranch *gBranch;

extern "C" void R__zip (Int_t cxlevel, Int_t *nin, char *bufin, Int_t *lout, char *bufout, Int_t *nout);
extern "C" void R__unzip(Int_t *nin, UChar_t *bufin, Int_t *lout, char *bufout, Int_t *nout);

const Int_t  kMAXBUF = 0xFFFFFF;
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
   // Default contructor.

   fDisplacement  = 0;
   fEntryOffset   = 0;
   fBufferRef     = 0;
   fBuffer        = 0;
   fHeaderOnly    = kFALSE;
   fBufferSize    = 0;
   fNevBufSize    = 0;
   fNevBuf        = 0;
   fLast          = 0;
   fBranch        = 0;
}

//_______________________________________________________________________
TBasket::TBasket(TDirectory *motherDir) : TKey(motherDir)
{
   // Simple Constructor.

   fDisplacement  = 0;
   fEntryOffset   = 0;
   fBufferRef     = 0;
   fBuffer        = 0;
   fHeaderOnly    = kFALSE;
   fBufferSize    = 0;
   fNevBufSize    = 0;
   fNevBuf        = 0;
   fLast          = 0;
   fBranch        = 0;
}

//_______________________________________________________________________
TBasket::TBasket(const char *name, const char *title, TBranch *branch) : 
   TKey(branch->GetDirectory())
{
   // Basket normal constructor.

   SetName(name);
   SetTitle(title);
   fClassName   = "TBasket";
   fBufferSize  = branch->GetBasketSize();
   fNevBufSize  = branch->GetEntryOffsetLen();
   fNevBuf      = 0;
   fEntryOffset = 0;  //Must be set to 0 before calling Sizeof
   fDisplacement= 0;  //Must be set to 0 before calling Sizeof
   fBuffer      = 0;  //Must be set to 0 before calling Sizeof
   fBufferRef   = new TBufferFile(TBuffer::kWrite, fBufferSize);
   fVersion    += 1000;
   if (branch->GetDirectory()) {
      TFile *file = branch->GetFile();
      fBufferRef->SetParent(file);
   }
   fHeaderOnly  = kTRUE;
   fLast        = 0; // RDK: Must initialize before calling Streamer()
   
   Streamer(*fBufferRef);
   fKeylen      = fBufferRef->Length();
   fObjlen      = fBufferSize - fKeylen;
   fLast        = fKeylen;
   fBuffer      = 0;
   fBranch      = branch;
   fHeaderOnly  = kFALSE;
   if (fNevBufSize) {
      fEntryOffset = new Int_t[fNevBufSize];
      for (Int_t i=0;i<fNevBufSize;i++) fEntryOffset[i] = 0;
   }
   branch->GetTree()->IncrementTotalBuffers(fBufferSize);
}

//_______________________________________________________________________
TBasket::~TBasket()
{
   // Basket destructor.

   if (fDisplacement) delete [] fDisplacement;
   if (fEntryOffset)  delete [] fEntryOffset;
   fDisplacement= 0;
   fEntryOffset = 0;
}

//_______________________________________________________________________
void TBasket::AdjustSize(Int_t newsize)
{
   // Increase the size of the current fBuffer up to newsize.

   char *newbuf = TStorage::ReAllocChar(fBuffer,newsize,fBufferSize);
   fBufferSize  = newsize;
   fBuffer      = newbuf;
}

//_______________________________________________________________________
Long64_t TBasket::CopyTo(TFile *to) 
{
   // Copy the basket of this branch onto the file to.

//   Global variables no longer required by key store   
//   TDirectory::TContext c(gDirectory,to);

   fBufferRef->SetWriteMode();
   Int_t nout = fNbytes - fKeylen;
   fBuffer = fBufferRef->Buffer();
   Create(nout, to);
   fBufferRef->SetBufferOffset(0);
   fHeaderOnly = kTRUE;
   Streamer(*fBufferRef);
   fHeaderOnly = kFALSE;
   Int_t nBytes = WriteFile(0, to);

   return nBytes>0 ? nBytes : -1;
}

//_______________________________________________________________________
void TBasket::DeleteEntryOffset()
{
   //  Delete fEntryOffset array.

   if (fEntryOffset) delete [] fEntryOffset;
   fEntryOffset = 0;
   fNevBufSize  = 0;
}


//_______________________________________________________________________
Int_t TBasket::DropBuffers()
{
   // Drop buffers of this basket if it is not the current basket.

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
   // Get pointer to buffer for internal entry.

   Int_t offset;
   if (fEntryOffset) offset = fEntryOffset[entry];
   else              offset = fKeylen + entry*fNevBufSize;
   fBufferRef->SetBufferOffset(offset);
   return offset;
}

//_______________________________________________________________________
Int_t TBasket::LoadBasketBuffers(Long64_t pos, Int_t len, TFile *file)
{ 
   // Load basket buffers in memory without unziping.
   // This function is called by TTreeCloner.
   // The function returns 0 in case of success, 1 in case of error.

   fBufferRef = new TBufferFile(TBuffer::kRead, len);
   fBufferRef->SetParent(file);
   char *buffer = fBufferRef->Buffer();
   file->Seek(pos);
   if (file->ReadBuffer(buffer,len)) {
      return 1; //error while reading
   }

   fBufferRef->SetReadMode();
   fBufferRef->SetBufferOffset(0);
   Streamer(*fBufferRef);

   return 0;
}

//_______________________________________________________________________
void TBasket::MoveEntries(Int_t dentries)
{
   // Remove the first dentries of this basket, moving entries at
   // dentries to the start of the buffer.
   
   Int_t i;

   if (dentries >= fNevBuf) return;
   Int_t bufbegin;
   Int_t moved;

   if (fEntryOffset) {
      bufbegin = fEntryOffset[dentries];
      moved = bufbegin-GetKeylen();

      // First store the original location in the fDisplacement array
      // and record the new start offset

      if (!fDisplacement) {
         fDisplacement = new Int_t[fNevBufSize];
      }
      for (i = 0; i<(fNevBufSize-dentries); ++i) {
         fDisplacement[i] = fEntryOffset[i+dentries];
         fEntryOffset[i]  = fEntryOffset[i+dentries] - moved;
      }
      for (i = fNevBufSize-dentries; i<fNevBufSize; ++i) {
         fDisplacement[i] = 0;      
         fEntryOffset[i]  = 0;
      }

   } else {
      // If there is no EntryOffset array, this means
      // that each entry has the same size and that 
      // it does not point to other objects (hence there
      // is no need for a displacement array).
      bufbegin = GetKeylen() + dentries*fNevBufSize;
      moved = bufbegin-GetKeylen();
   }
   TBuffer *buf = GetBufferRef();
   char *buffer = buf->Buffer();
   memmove(buffer+GetKeylen(),buffer+bufbegin,buf->Length()-bufbegin);
   buf->SetBufferOffset(buf->Length()-moved);
   fNevBuf -= dentries;
}

//_______________________________________________________________________
Int_t TBasket::ReadBasketBuffers(Long64_t pos, Int_t len, TFile *file)
{
   // Read basket buffers in memory and cleanup.
   //
   // Read a basket buffer. Check if buffers of previous ReadBasket
   // should not be dropped. Remember, we keep buffers in memory up to
   // fMaxVirtualSize.
   // The function returns 0 in case of success, 1 in case of error

   Int_t badread= 0;
   TDirectory *cursav = gDirectory;
   fBranch->GetDirectory()->cd();

   if (fBranch->GetTree()->MemoryFull(fBufferSize)) fBranch->DropBaskets();

   fBufferRef = new TBufferFile(TBuffer::kRead, len);
   fBufferRef->SetParent(file);
   
   char *buffer = fBufferRef->Buffer();
   file->Seek(pos);
   if (file->ReadBuffer(buffer,len)) {
      badread = 1;
      return badread;
   }

   Streamer(*fBufferRef);

   Bool_t oldCase = fObjlen==fNbytes-fKeylen 
        && GetBranch()->GetCompressionLevel()!=0
        && file->GetVersion()<=30401;
   if (fObjlen > fNbytes-fKeylen || oldCase) {
      if (TestBit(TBufferFile::kNotDecompressed) && (fNevBuf==1)) {
         // By-passing buffer unzipping has been requested and is
         // possible (only 1 entry in this basket).
         fBuffer = fBufferRef->Buffer();
         
         // Make sure that the buffer is set at the END of the data
         fBufferRef->SetBufferOffset(fNbytes);
        
         // Indicate that this buffer is weird.
         fBufferRef->SetBit(TBufferFile::kNotDecompressed); 
        
         // Usage of this mode assume the existance of only ONE 
         // entry in this basket.
         delete [] fEntryOffset; fEntryOffset = 0;
         delete [] fDisplacement; fDisplacement = 0;
         
         fBranch->GetTree()->IncrementTotalBuffers(fBufferSize);

         return badread;
      }
      fBuffer = new char[fObjlen+fKeylen];
      memcpy(fBuffer,buffer,fKeylen);
      char *objbuf = fBuffer + fKeylen;
      UChar_t *bufcur = (UChar_t *)&buffer[fKeylen];
      Int_t nin, nout, nbuf;
      Int_t noutot = 0;
      while (1) {
         nin  = 9 + ((Int_t)bufcur[3] | ((Int_t)bufcur[4] << 8) | ((Int_t)bufcur[5] << 16));
         nbuf = (Int_t)bufcur[6] | ((Int_t)bufcur[7] << 8) | ((Int_t)bufcur[8] << 16);
         if (oldCase && (nin > fObjlen || nbuf > fObjlen)) {
            //buffer was very likely not compressed in an old version
            delete [] fBuffer;
            fBuffer = fBufferRef->Buffer();
            goto AfterBuffer;
         }
         R__unzip(&nin, bufcur, &nbuf, objbuf, &nout);
         if (!nout) break;
         noutot += nout;
         if (noutot >= fObjlen) break;
         bufcur += nin;
         objbuf += nout;
      }
      if (noutot != fObjlen) {
         Error("ReadBasketBuffers", "fNbytes = %d, fKeylen = %d, fObjlen = %d, noutot = %d, nout=%d, nin=%d, nbuf=%d", fNbytes,fKeylen,fObjlen, noutot,nout,nin,nbuf);
         badread = 1;
      }
      fBufferRef->SetBuffer(fBuffer, fObjlen+fKeylen );
      len = fObjlen+fKeylen;
   } else {
      fBuffer = fBufferRef->Buffer();
   }
AfterBuffer:
   cursav->cd();

   fBranch->GetTree()->IncrementTotalBuffers(fBufferSize);

//        read offsets table
   if (!fBranch->GetEntryOffsetLen()) return badread;
   delete [] fEntryOffset;
   fEntryOffset = 0;
   fBufferRef->SetBufferOffset(fLast);
   fBufferRef->ReadArray(fEntryOffset);
   if (!fEntryOffset) {
      fEntryOffset = new Int_t[fNevBuf+1];
      fEntryOffset[0] = fKeylen;
      Warning("ReadBasketBuffers","basket:%s has fNevBuf=%d but fEntryOffset=0, pos=%lld, len=%d, fNbytes=%d, fObjlen=%d, trying to repair",GetName(),fNevBuf,pos,len,fNbytes,fObjlen);
      return badread;
   }
   delete [] fDisplacement;
   fDisplacement = 0;
   if (fBufferRef->Length() != len) { 
      // There is more data in the buffer!  It is the displacement
      // array.  If len is less than TBuffer::kMinimalSize the actual
      // size of the buffer is too large, so we can not use the
      // fBufferRef->BufferSize()
      fBufferRef->ReadArray(fDisplacement);
   }

   return badread;
}

//_______________________________________________________________________
Int_t TBasket::ReadBasketBytes(Long64_t pos, TFile *file)
{
   // Read basket buffers in memory and cleanup
   //
   // Read first bytes of a logical record starting at position pos
   // return record length (first 4 bytes of record).

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
   // Stream a class object.

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
         fBufferRef = new TBufferFile(TBuffer::kRead,fBufferSize);
         fBufferRef->SetParent(b.GetParent());
         char *buf  = fBufferRef->Buffer();
         if (v > 1) b.ReadFastArray(buf,fLast);
         else       b.ReadArray(buf);
         fBufferRef->SetBufferOffset(fLast);
         fBranch->GetTree()->IncrementTotalBuffers(fBufferSize);
      }
   } else {
      TKey::Streamer(b);   //this must be first
      b.WriteVersion(TBasket::IsA());
      Int_t curLast = fBufferRef->Length();
      if (fBufferRef && !fHeaderOnly && !fSeekKey && curLast > fLast) fLast = curLast;
      if (fLast > fBufferSize) fBufferSize = fLast;

//   static TStopwatch timer;
//   timer.Start(kFALSE);

//       //  Check may be fEntryOffset is equidistant
//       //  This attempts by Victor fails :(
//       int equidist = 0;
//       if (1 && fEntryOffset && fNevBuf>=3) {
//          equidist = 1;
//          int dist = fEntryOffset[1]-fEntryOffset[0];
//          int curr = fEntryOffset[1];
//          for (int i=1;i<fNevBuf;i++,curr+=dist) {
//             if (fEntryOffset[i]==curr) continue;
//             equidist = 0;
//             break;
//          }
//          if (equidist) {
//             fNevBufSize=dist;
//             delete [] fEntryOffset; fEntryOffset = 0;
//          }
//           if (equidist) {
//              fprintf(stderr,"detected an equidistant case fNbytes==%d fLast==%d\n",fNbytes,fLast);
//           }
//       }
//  also he add (a little further
//       if (!fEntryOffset || equidist)  flag  = 2;
    
//   timer.Stop();
//   Double_t rt1 = timer.RealTime();
//   Double_t cp1 = timer.CpuTime();
//   fprintf(stderr,"equidist cost :  RT=%6.2f s  Cpu=%6.2f s\n",rt1,cp1);

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
   // Update basket header and EntryOffset table.

   if (fEntryOffset) {
      if (fNevBuf+1 >= fNevBufSize) {
         Int_t newsize = TMath::Max(10,2*fNevBufSize);
         Int_t *newoff = TStorage::ReAllocInt(fEntryOffset, newsize,
                                              fNevBufSize);
         if (fDisplacement) {
            Int_t *newdisp = TStorage::ReAllocInt(fDisplacement, newsize,
                                                  fNevBufSize);
            fDisplacement = newdisp;
         }
         fEntryOffset  = newoff;
         fNevBufSize   = newsize;
         //Update branch only for the first 10 baskets
         if (fBranch->GetWriteBasket() < 10) fBranch->SetEntryOffsetLen(newsize);
//          fprintf(stderr,"fNevBuf+1==%d fNevBufSize==%d\n",fNevBuf+1,fNevBufSize);
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
   // Write buffer of this basket on the current file.
   //
   // The function returns the number of bytes committed to the memory.
   // If a write error occurs, the number of bytes returned is -1.
   // If no data are written, the number of bytes returned is 0.
   //

   const Int_t kWrite = 1;
   TDirectory *cursav = gDirectory;
   TFile *file = fBranch->GetFile(kWrite);
   if (!file) return 0;

   fBranch->GetDirectory()->cd();
   if (!file->IsWritable()) { 
      cursav->cd(); 
      return -1;
   }
   
   fMotherDir = fBranch->GetDirectory();
   
   if (fBufferRef->TestBit(TBufferFile::kNotDecompressed)) {
      // Read the basket information that was saved inside the buffer.
      Bool_t writing = fBufferRef->IsWriting();
      fBufferRef->SetReadMode();
      fBufferRef->SetBufferOffset(0);

      Streamer(*fBufferRef);
      if (writing) fBufferRef->SetWriteMode();
      Int_t nout = fNbytes - fKeylen;

      fBuffer = fBufferRef->Buffer();

      Create(nout);
      fBufferRef->SetBufferOffset(0);
      fHeaderOnly = kTRUE;

      Streamer(*fBufferRef);         //write key itself again
      int nBytes = WriteFile(0);
      fHeaderOnly = kFALSE;
      cursav->cd();
      return nBytes>0 ? fKeylen+nout : -1;
   }

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

   Int_t lbuf, nout, noutot, bufmax, nzip;
   lbuf       = fBufferRef->Length();
   fObjlen    = lbuf - fKeylen;

   fHeaderOnly = kTRUE;
   fCycle = fBranch->GetWriteBasket();
   Int_t cxlevel = fBranch->GetCompressionLevel();
   if (cxlevel > 0) {
      //if (cxlevel == 2) cxlevel--; RB: I cannot remember why we had this!
      Int_t nbuffers = fObjlen/kMAXBUF;
      Int_t buflen = fKeylen + fObjlen + 28; //add 28 bytes in case object is placed in a deleted gap
      fBuffer = new char[buflen];
      char *objbuf = fBufferRef->Buffer() + fKeylen;
      char *bufcur = &fBuffer[fKeylen];
      noutot = 0;
      nzip   = 0;
      for (Int_t i=0;i<=nbuffers;i++) {
         if (i == nbuffers) bufmax = fObjlen -nzip;
         else               bufmax = kMAXBUF;
         //compress the buffer
         R__zip(cxlevel, &bufmax, objbuf, &bufmax, bufcur, &nout);
         
         // test if buffer has really been compressed. In case of small buffers 
         // when the buffer contains random data, it may happen that the compressed
         // buffer is larger than the input. In this case, we write the original uncompressed buffer
         if (nout == 0 || nout >= fObjlen) {
            nout = fObjlen;
            fBuffer = fBufferRef->Buffer();
            Create(fObjlen);
            fBufferRef->SetBufferOffset(0);

            Streamer(*fBufferRef);         //write key itself again
            if ((nout+fKeylen)>buflen) {
               Warning("WriteBuffer","Possible memory corruption due to compression algorithm, wrote %d bytes past the end of a block of %d bytes. fNbytes=%d, fObjLen=%d, fKeylen=%d",
                  (nout+fKeylen-buflen),buflen,fNbytes,fObjlen,fKeylen);
            }
            goto WriteFile;
         }
         bufcur += nout;
         noutot += nout;
         objbuf += kMAXBUF;
         nzip   += kMAXBUF;
      }
      nout = noutot;
      Create(noutot);
      fBufferRef->SetBufferOffset(0);

      Streamer(*fBufferRef);         //write key itself again
      memcpy(fBuffer,fBufferRef->Buffer(),fKeylen);
      delete fBufferRef; fBufferRef = 0;
   } else {
      fBuffer = fBufferRef->Buffer();
      Create(fObjlen);
      fBufferRef->SetBufferOffset(0);

      Streamer(*fBufferRef);         //write key itself again
      nout = fObjlen;
   }

WriteFile:
   Int_t nBytes = WriteFile(0);
   fHeaderOnly = kFALSE;
   cursav->cd();
   return nBytes>0 ? fKeylen+nout : -1;
}

