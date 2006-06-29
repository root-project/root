// @(#)root/base:$Name:  $:$Id: TFileCacheRead.cxx,v 1.1 2006/06/27 14:36:27 brun Exp $
// Author: Rene Brun   18/05/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileCacheRead : a cache when reading files on the network           //
//                                                                      //
// A caching system to speed up network I/O, i.e. when there is         //
// no operating system caching support (like the buffer cache for       //
// local disk I/O). The cache makes sure that every I/O is done with    //
// a (large) fixed length buffer thereby avoiding many small I/O's.     //
// Currently the read cache system is used by the classes TNetFile,     //
// TRFIOFile and TWebFile.                                              //
// One creates a read cache via  TFile::SetCacheRead.                   //
//                                                                      //
// When processing TTree, TChain, a specialized class TTreeCache that   //
// derives from this class is automatically created.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TFile.h"
#include "TFileCacheRead.h"
#include "TFileCacheWrite.h"

ClassImp(TFileCacheRead)

//______________________________________________________________________________
TFileCacheRead::TFileCacheRead() : TObject()
{
   // Default Constructor.

   fBufferSize  = 0;
   fBufferLen   = 0;
   fNseek       = 0;
   fNtot        = 0;
   fNb          = 0;
   fSeekSize    = 0;
   fSeek        = 0;
   fSeekIndex   = 0;
   fSeekSort    = 0;
   fPos         = 0;
   fSeekLen     = 0;
   fSeekSortLen = 0;
   fSeekPos     = 0;
   fLen         = 0;
   fFile        = 0;
   fBuffer      = 0;
   fIsSorted    = kFALSE;
}

//_____________________________________________________________________________
TFileCacheRead::TFileCacheRead(TFile *file, Int_t buffersize)
           : TObject()
{
   // Creates a TFileCacheRead data structure.

   if (buffersize <=10000) fBufferSize = 100000;
   fBufferSize  = buffersize;
   fBufferLen   = 0;
   fNseek       = 0;
   fNtot        = 0;
   fNb          = 0;
   fSeekSize    = 10000;
   fSeek        = new Long64_t[fSeekSize];
   fSeekIndex   = new Long64_t[fSeekSize];
   fSeekSort    = new Long64_t[fSeekSize];
   fPos         = new Long64_t[fSeekSize];
   fSeekLen     = new Int_t[fSeekSize];
   fSeekSortLen = new Int_t[fSeekSize];
   fSeekPos     = new Int_t[fSeekSize];
   fLen         = new Int_t[fSeekSize];
   fFile        = file;
   fBuffer      = new char[fBufferSize];
   fIsSorted    = kFALSE;
   if (file) file->SetCacheRead(this);
}

//______________________________________________________________________________
TFileCacheRead::TFileCacheRead(const TFileCacheRead &pf) : TObject(pf)
{
   // Copy Constructor.
}

//______________________________________________________________________________
TFileCacheRead& TFileCacheRead::operator=(const TFileCacheRead& pf)
{
   // Assignment.

   if (this != &pf) TObject::operator=(pf);
   return *this;
}

//_____________________________________________________________________________
TFileCacheRead::~TFileCacheRead()
{
   // Destructor.

   delete [] fSeek;
   delete [] fSeekIndex;
   delete [] fSeekSort;
   delete [] fSeekLen;
   delete [] fSeekSortLen;
   delete [] fSeekPos;
   delete [] fBuffer;
}

//_____________________________________________________________________________
void TFileCacheRead::Prefetch(Long64_t pos, Int_t len)
{
   // Add block of length len at position pos in the list of blocks to
   // be prefetched. If pos <= 0 the current blocks (if any) are reset.

   fIsSorted = kFALSE;
   if (pos <= 0) {
      fNseek = 0;
      fNtot  = 0;
      return;
   }
   if (fNseek >= fSeekSize) {
      //reallocate buffers
      fSeekSize *= 2;
      Long64_t *aSeek        = new Long64_t[fSeekSize];
      Long64_t *aSeekIndex   = new Long64_t[fSeekSize];
      Long64_t *aSeekSort    = new Long64_t[fSeekSize];
      Long64_t *aPos         = new Long64_t[fSeekSize];
      Int_t    *aSeekLen     = new Int_t[fSeekSize];
      Int_t    *aSeekSortLen = new Int_t[fSeekSize];
      Int_t    *aSeekPos     = new Int_t[fSeekSize];
      Int_t    *aLen         = new Int_t[fSeekSize];
      for (Int_t i=0;i<fNseek;i++) {
         aSeek[i]        = fSeek[i];
         aSeekIndex[i]   = fSeekIndex[i];
         aSeekSort[i]    = fSeekSort[i];
         aPos[i]         = fPos[i];
         aSeekLen[i]     = fSeekLen[i];
         aSeekSortLen[i] = fSeekSortLen[i];
         aSeekPos[i]     = fSeekPos[i];
         aLen[i]         = fLen[i];
      }
      delete [] fSeek;
      delete [] fSeekIndex;
      delete [] fSeekSort;
      delete [] fPos;
      delete [] fSeekLen;
      delete [] fSeekSortLen;
      delete [] fSeekPos;
      delete [] fLen;
      fSeek        = aSeek;
      fSeekIndex   = aSeekIndex;
      fSeekSort    = aSeekSort;
      fPos         = aPos;
      fSeekLen     = aSeekLen;
      fSeekSortLen = aSeekSortLen;
      fSeekPos     = aSeekPos;
      fLen         = aLen;
   }

   fSeek[fNseek] = pos;
   fSeekLen[fNseek] = len;
   fNseek++;
   fNtot += len;
}

//_____________________________________________________________________________
void TFileCacheRead::Print(Option_t *option) const
{
   // Print class internal structure.

   TString opt = option;
   opt.ToLower();
   printf("Number of blocks: %d, total size : %d\n",fNseek,fNtot);
   if (!opt.Contains("a")) return;
   for (Int_t i=0;i<fNseek;i++) {
      if (fIsSorted && !opt.Contains("s")) {
         printf("block: %5d, from: %lld to %lld, len=%d bytes\n",i,fSeekSort[i],fSeekSort[i]+fSeekSortLen[i],fSeekSortLen[i]);
      } else {
         printf("block: %5d, from: %lld to %lld, len=%d bytes\n",i,fSeek[i],fSeek[i]+fSeekLen[i],fSeekLen[i]);
      }
   }
   printf ("Number of long buffers = %d\n",fNb);
   for (Int_t j=0;j<fNb;j++) {
      printf("fPos[%d]=%lld, fLen=%d\n",j,fPos[j],fLen[j]);
   }
}

//_____________________________________________________________________________
Int_t TFileCacheRead::ReadBuffer(char *buf, Long64_t pos, Int_t len)
{
   // Read buffer at position pos.
   // If pos is in the list of prefetched blocks read from fBuffer,
   // otherwise need to make a normal read from file. Returns -1 in case of
   // read error, 0 in case not in cache, 1 in case read from cache.

   if (fNseek > 0 && !fIsSorted) {
      Sort();
      if (fFile->ReadBuffers(fBuffer,fPos,fLen,fNb))
         return -1;
   }

   // in case we are writing and reading to/from this file, we much check
   // if this buffer is in the write cache (not yet written to the file)
   if (TFileCacheWrite *cachew = fFile->GetCacheWrite()) {
      if (cachew->ReadBuffer(buf,pos,len) == 0) {
         fFile->Seek(pos+len);
         return 1;
      }
   }

   Int_t loc = (Int_t)TMath::BinarySearch(fNseek,fSeekSort,pos);
   if (loc >= 0 && loc <fNseek && pos == fSeekSort[loc]) {
      memcpy(buf,&fBuffer[fSeekPos[loc]],len);
      fFile->Seek(pos+len);
      //printf("TFileCacheRead::ReadBuffer, pos=%lld, len=%d, slen=%d, loc=%d\n",pos,len,fSeekSortLen[loc],loc);
      return 1;
   }

   return 0;
}

//_____________________________________________________________________________
void TFileCacheRead::SetFile(TFile *file)
{
   //set the file using this cache

   fFile = file;
}

//_____________________________________________________________________________
void TFileCacheRead::Sort()
{
   // Sort buffers to be prefetched in increasing order of positions.
   // Merge consecutive blocks if necessary.

   if (!fNseek) return;
   TMath::Sort(fNseek,fSeek,fSeekIndex,kFALSE);
   Int_t i;
   Int_t nb = 0;
   for (i=0;i<fNseek;i++) {
      Long64_t ind = fSeekIndex[i];
      fSeekSort[i] = fSeek[ind];
      fSeekSortLen[i] = fSeekLen[ind];
   }
   if (fNtot > fBufferSize) {
      fBufferSize = fNtot + 100;
      delete [] fBuffer;
      fBuffer = new char[fBufferSize];
     // printf("CHANGING fBufferSize=%d, fNseek=%d, fNtot=%d\n",fBufferSize, fNseek,fNtot);
   }
   fPos[0]  = fSeekSort[0];
   fLen[0]  = fSeekSortLen[0];
   fSeekPos[0] = 0;
   for (i=1;i<fNseek;i++) {
      fSeekPos[i] = fSeekPos[i-1] + fSeekSortLen[i-1];
      if (fSeekSort[i] != fSeekSort[i-1]+fSeekSortLen[i-1]) {
         nb++;
         fPos[nb] = fSeekSort[i];
         fLen[nb] = fSeekSortLen[i];
      } else {
         fLen[nb] += fSeekSortLen[i];
      }
   }
   fNb = nb+1;
   fIsSorted = kTRUE;
}
