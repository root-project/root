// @(#)root/io:$Id$
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
// TFileCacheWrite : a cache when writing files over the network        //
//                                                                      //
// A caching system to speed up network I/O, i.e. when there is         //
// no operating system caching support (like the buffer cache for       //
// local disk I/O). The cache makes sure that every I/O is done with    //
// a (large) fixed length buffer thereby avoiding many small I/O's.     //
// Currently the write cache system is used by the classes TNetFile,    //
// TXNetFile and TWebFile (via TFile::WriteBuffers()).                  //
//                                                                      //
// The write cache is automatically created when writing a remote file  //
// (created in TFile::Open()).                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TFile.h"
#include "TFileCacheWrite.h"

ClassImp(TFileCacheWrite)

//______________________________________________________________________________
TFileCacheWrite::TFileCacheWrite() : TObject()
{
   // Default Constructor.

   fBufferSize  = 0;
   fNtot        = 0;
   fSeekStart   = 0;
   fFile        = 0;
   fBuffer      = 0;
   fRecursive   = kFALSE;
}

//_____________________________________________________________________________
TFileCacheWrite::TFileCacheWrite(TFile *file, Int_t buffersize)
           : TObject()
{
   // Creates a TFileCacheWrite data structure.
   // The write cache will be connected to file.
   // The size of the cache will be buffersize,
   // if buffersize < 10000 a default size of 512 Kbytes is used

   if (buffersize < 10000) buffersize = 512000;
   fBufferSize  = buffersize;
   fSeekStart   = 0;
   fNtot        = 0;
   fFile        = file;
   fRecursive   = kFALSE;
   fBuffer      = new char[fBufferSize];
   if (file) file->SetCacheWrite(this);
   if (gDebug > 0) Info("TFileCacheWrite","Creating a write cache with buffersize=%d bytes",buffersize);
}

//_____________________________________________________________________________
TFileCacheWrite::~TFileCacheWrite()
{
   // Destructor.

   delete [] fBuffer;
}

//_____________________________________________________________________________
Bool_t TFileCacheWrite::Flush()
{
   // Flush the current write buffer to the file.
   // Returns kTRUE in case of error.

   if (!fNtot) return kFALSE;
   fFile->Seek(fSeekStart);
   //printf("Flushing buffer at fSeekStart=%lld, fNtot=%d\n",fSeekStart,fNtot);
   fRecursive = kTRUE;
   Bool_t status = fFile->WriteBuffer(fBuffer, fNtot);
   fRecursive = kFALSE;
   fNtot = 0;
   return status;
}

//_____________________________________________________________________________
void TFileCacheWrite::Print(Option_t *option) const
{
   // Print class internal structure.

   TString opt = option;
   printf("Write cache for file %s\n",fFile->GetName());
   printf("Size of write cache: %d bytes to be written at %lld\n",fNtot,fSeekStart);
   opt.ToLower();
}

//_____________________________________________________________________________
Int_t TFileCacheWrite::ReadBuffer(char *buf, Long64_t pos, Int_t len)
{
   // Called by the read cache to check if the requested data is not
   // in the write cache buffer.
   //        Returns -1 if data not in write cache,
   //        0 otherwise.

   if (pos < fSeekStart || pos+len > fSeekStart+fNtot) return -1;
   memcpy(buf,fBuffer+pos-fSeekStart,len);
   return 0;
}

//_____________________________________________________________________________
Int_t TFileCacheWrite::WriteBuffer(const char *buf, Long64_t pos, Int_t len)
{
   // Write buffer at position pos in the write buffer.
   // The function returns 1 if the buffer has been successfully entered into the write buffer.
   // The function returns 0 in case WriteBuffer() was recusively called via Flush().
   // The function returns -1 in case of error.

   if (fRecursive) return 0;

   //printf("TFileCacheWrite::WriteBuffer, pos=%lld, len=%d, fSeekStart=%lld, fNtot=%d\n",pos,len,fSeekStart,fNtot);

   if (fSeekStart + fNtot != pos) {
      //we must flush the current cache
      if (Flush()) return -1; //failure
   }
   if (fNtot + len >= fBufferSize) {
      if (Flush()) return -1; //failure
      if (len >= fBufferSize) {
         //buffer larger than the cache itself: direct write to file
         fRecursive = kTRUE;
         if (fFile->WriteBuffer(buf,len)) return -1;  // failure
         fRecursive = kFALSE;
         return 1;
      }
   }
   if (!fNtot) fSeekStart = pos;
   memcpy(fBuffer+fNtot,buf,len);
   fNtot += len;

   return 1;
}

//_____________________________________________________________________________
void TFileCacheWrite::SetFile(TFile *file)
{
   // Set the file using this cache.
   // Any write not yet flushed will be lost.

   fFile = file;
}
