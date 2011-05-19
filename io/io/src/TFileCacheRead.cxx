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
// TFileCacheRead : a cache when reading files over the network         //
//                                                                      //
// A caching system to speed up network I/O, i.e. when there is         //
// no operating system caching support (like the buffer cache for       //
// local disk I/O). The cache makes sure that every I/O is done with    //
// a (large) fixed length buffer thereby avoiding many small I/O's.     //
// Currently the read cache system is used by the classes TNetFile,     //
// TXNetFile and TWebFile (via TFile::ReadBuffers()).                   //
//                                                                      //
// When processing TTree, TChain, a specialized class TTreeCache that   //
// derives from this class is automatically created.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TEnv.h"
#include "TFile.h"
#include "TFileCacheRead.h"
#include "TFileCacheWrite.h"
#include "TMath.h"

ClassImp(TFileCacheRead)

//______________________________________________________________________________
TFileCacheRead::TFileCacheRead() : TObject()
{
   // Default Constructor.

   fBufferSizeMin = 0;
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
   fIsTransferred = kFALSE;

   //values for the second prefetched block
   fBNseek       = 0;
   fBNtot        = 0;
   fBNb          = 0;
   fBSeekSize    = 0;
   fBSeek        = 0;
   fBSeekSort    = 0;
   fBSeekIndex   = 0;
   fBPos         = 0;
   fBSeekLen     = 0;
   fBSeekSortLen = 0;
   fBSeekPos     = 0;
   fBLen         = 0;

   fAsyncReading = kFALSE;
   fPrefetch        = 0;
   fPrefetchBlocks  = 0;
}

//_____________________________________________________________________________
TFileCacheRead::TFileCacheRead(TFile *file, Int_t buffersize)
           : TObject()
{
   // Creates a TFileCacheRead data structure.

   if (buffersize <=10000) fBufferSize = 100000;
   else fBufferSize = buffersize;

   fBufferSizeMin = fBufferSize;
   fBufferLen   = 0;
   fNseek       = 0;
   fNtot        = 0;
   fNb          = 0;
   fSeekSize    = 10000;
   fSeek        = new Long64_t[fSeekSize];
   fSeekIndex   = new Int_t[fSeekSize];
   fSeekSort    = new Long64_t[fSeekSize];
   fPos         = new Long64_t[fSeekSize];
   fSeekLen     = new Int_t[fSeekSize];
   fSeekSortLen = new Int_t[fSeekSize];
   fSeekPos     = new Int_t[fSeekSize];
   fLen         = new Int_t[fSeekSize];
   fFile        = file;

   //initialisation for the second block    
   fBNseek       = 0;
   fBNtot        = 0;
   fBNb          = 0;
   fBSeekSize    = 10000;
   fBSeek        = new Long64_t[fBSeekSize];
   fBSeekIndex   = new Int_t[fBSeekSize];
   fBSeekSort    = new Long64_t[fBSeekSize];
   fBPos         = new Long64_t[fBSeekSize];
   fBSeekLen     = new Int_t[fBSeekSize];
   fBSeekSortLen = new Int_t[fBSeekSize];
   fBSeekPos     = new Int_t[fBSeekSize];
   fBLen         = new Int_t[fBSeekSize];

   fBuffer = 0;
   fPrefetch = 0;
   fPrefetchBlocks = 0;

   //initialise the prefetch object and set the cache directory
   // start the thread only if the file is not local  
   fEnablePrefetching = gEnv->GetValue("TFile.AsyncPrefetching", 0);
   if (fEnablePrefetching && strcmp(file->GetEndpointUrl()->GetProtocol(), "file")){
      fPrefetch = new TFilePrefetch(file);
      const char* cacheDir = gEnv->GetValue("Cache.Directory", "");
      if (strcmp(cacheDir, ""))
         if (!fPrefetch->SetCache((char*) cacheDir))
            fprintf(stderr, "Error while trying to set the cache directory.\n");
      fPrefetch->ThreadStart();
   }
   else //disable the async pref for local files
      fEnablePrefetching = 0;

   //environment variable used to switch to the new method of reading asynchronously
   if (fEnablePrefetching){
      fAsyncReading = kFALSE;
   }
   else {
      fAsyncReading = gEnv->GetValue("TFile.AsyncReading", 0);
   if (fAsyncReading) {
      // Check if asynchronous reading is supported by this TFile specialization
      fAsyncReading = kFALSE;
      if (file && !(file->ReadBufferAsync(0, 0)))
         fAsyncReading = kTRUE;
   }
   if (!fAsyncReading) {
      // we use sync primitives, hence we need the local buffer
      fBuffer = new char[fBufferSize];
   }
   }  

   fIsSorted    = kFALSE;
   fIsTransferred = kFALSE;
   fBIsSorted = kFALSE;

   if (file) file->SetCacheRead(this);
}

//_____________________________________________________________________________
TFileCacheRead::~TFileCacheRead()
{
   // Destructor.

   delete [] fSeek;
   delete [] fSeekIndex;
   delete [] fSeekSort;
   delete [] fPos;
   delete [] fSeekLen;
   delete [] fSeekSortLen;
   delete [] fSeekPos;
   delete [] fLen;
   delete [] fBuffer;
   delete [] fBSeek;
   delete [] fBSeekIndex;
   delete [] fBSeekSort;
   delete [] fBPos;
   delete [] fBSeekLen;
   delete [] fBSeekSortLen;
   delete [] fBSeekPos;
   delete [] fBLen;
   SafeDelete(fPrefetch);
}

//_____________________________________________________________________________
void TFileCacheRead::Prefetch(Long64_t pos, Int_t len)
{
   // Add block of length len at position pos in the list of blocks to
   // be prefetched. If pos <= 0 the current blocks (if any) are reset.

   fIsSorted = kFALSE;
   fIsTransferred = kFALSE;
   if (pos <= 0) {
      fNseek = 0;
      fNtot  = 0;
      return;
   }
   if (fNseek >= fSeekSize) {
      //reallocate buffers
      fSeekSize *= 2;
      Long64_t *aSeek        = new Long64_t[fSeekSize];
      Int_t    *aSeekIndex   = new Int_t[fSeekSize];
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


//____________________________________________________________________________
void TFileCacheRead::SecondPrefetch(Long64_t pos, Int_t len){
 
   //add a new element and increase the size if necessary                                                                                            
   fBIsSorted = kFALSE;
   if (pos <= 0) {
      fBNseek = 0;
      fBNtot  = 0;
      return;
   }
   if (fBNseek >= fBSeekSize) {
      //reallocate buffers                                                                                                                                     
      fBSeekSize *= 2;
      Long64_t *aSeek        = new Long64_t[fBSeekSize];
      Int_t    *aSeekIndex   = new Int_t[fBSeekSize];
      Long64_t *aSeekSort    = new Long64_t[fBSeekSize];
      Long64_t *aPos         = new Long64_t[fBSeekSize];
      Int_t    *aSeekLen     = new Int_t[fBSeekSize];
      Int_t    *aSeekSortLen = new Int_t[fBSeekSize];
      Int_t    *aSeekPos     = new Int_t[fBSeekSize];
      Int_t    *aLen         = new Int_t[fBSeekSize];
      for (Int_t i=0;i<fBNseek;i++) {
         aSeek[i]        = fBSeek[i];
         aSeekIndex[i]   = fBSeekIndex[i];
         aSeekSort[i]    = fBSeekSort[i];
         aPos[i]         = fBPos[i];
         aSeekLen[i]     = fBSeekLen[i];
         aSeekSortLen[i] = fBSeekSortLen[i];
         aSeekPos[i]     = fBSeekPos[i];
         aLen[i]         = fBLen[i];
      }
      delete [] fBSeek;
      delete [] fBSeekIndex;
      delete [] fBSeekSort;
      delete [] fBPos;
      delete [] fBSeekLen;
      delete [] fBSeekSortLen;
      delete [] fBSeekPos;
      delete [] fBLen;
      fBSeek        = aSeek;
      fBSeekIndex   = aSeekIndex;
      fBSeekSort    = aSeekSort;
      fBPos         = aPos;
      fBSeekLen     = aSeekLen;
      fBSeekSortLen = aSeekSortLen;
      fBSeekPos     = aSeekPos;
      fBLen         = aLen;
   }

   fBSeek[fBNseek] = pos;
   fBSeekLen[fBNseek] = len;
   fBNseek++;
   fBNtot += len;
}


//_____________________________________________________________________________
void TFileCacheRead::Print(Option_t *option) const
{
   // Print cache statistics, like
   //   ******TreeCache statistics for file: cms2.root ******
   //   Reading............................: 72761843 bytes in 7 transactions
   //   Readahead..........................: 256000 bytes with overhead = 0 bytes
   //   Average transaction................: 10394.549000 Kbytes
   //   Number of blocks in current cache..: 210, total size: 6280352
   //
   // if option = "a" the list of blocks in the cache is printed
   // NB: this function is automatically called by TTreeCache::Print
      
   TString opt = option;
   opt.ToLower();
   printf("Reading............................: %lld bytes in %d transactions\n",fFile->GetBytesRead(),  fFile->GetReadCalls());
   printf("Readahead..........................: %d bytes with overhead = %lld bytes\n",TFile::GetReadaheadSize(),fFile->GetBytesReadExtra());
   printf("Average transaction................: %f Kbytes\n",0.001*Double_t(fFile->GetBytesRead())/Double_t(fFile->GetReadCalls()));
   printf("Number of blocks in current cache..: %d, total size: %d\n",fNseek,fNtot);
   if (fPrefetch){
     printf("Prefetching .......................: %lli blocks\n", fPrefetchBlocks);
     printf("Prefetching Wait Time..............: %f seconds\n", fPrefetch->GetWaitTime() / 1e+6);
   }

   if (!opt.Contains("a")) return;
   for (Int_t i=0;i<fNseek;i++) {
      if (fIsSorted && !opt.Contains("s")) {
         printf("block: %5d, from: %lld to %lld, len = %d bytes\n",i,fSeekSort[i],fSeekSort[i]+fSeekSortLen[i],fSeekSortLen[i]);
      } else {
         printf("block: %5d, from: %lld to %lld, len = %d bytes\n",i,fSeek[i],fSeek[i]+fSeekLen[i],fSeekLen[i]);
      }
   }
   printf ("Number of long buffers = %d\n",fNb);
   for (Int_t j=0;j<fNb;j++) {
      printf("fPos[%d] = %lld, fLen = %d\n",j,fPos[j],fLen[j]);
   }
}

//_____________________________________________________________________________
Int_t TFileCacheRead::ReadBuffer(char *buf, Long64_t pos, Int_t len)
{
   // Read buffer at position pos.
   // If pos is in the list of prefetched blocks read from fBuffer,
   // otherwise need to make a normal read from file. Returns -1 in case of
   // read error, 0 in case not in cache, 1 in case read from cache.

   Int_t loc = -1;
   return ReadBufferExt(buf, pos, len, loc);
}

//_____________________________________________________________________________
Int_t TFileCacheRead::ReadBufferExt(char *buf, Long64_t pos, Int_t len, Int_t &loc)
{
   if (fEnablePrefetching)
      return ReadBufferExtPrefetch(buf, pos, len, loc);
   else
      return ReadBufferExtNormal(buf, pos, len, loc);
}


//_____________________________________________________________________________
Int_t TFileCacheRead::ReadBufferExtPrefetch(char *buf, Long64_t pos, Int_t len, Int_t &loc)
{
   //prefetch the first block                                                                                                                                
   if (fNseek > 0 && !fIsSorted) {
      Sort();
      loc = -1;
      fPrefetch->ReadBlock(fPos, fLen, fNb);
      fPrefetchBlocks++;
      fIsTransferred = kTRUE;
   }

   //try to prefetch the second block                                                                                                                        
   if (fBNseek > 0 && !fBIsSorted) {
      SecondSort();
      loc = -1;
      fPrefetch->ReadBlock(fBPos, fBLen, fBNb);
      fPrefetchBlocks++;
   }

   // in case we are writing and reading to/from this file, we must check                                                                                    
   // if this buffer is in the write cache (not yet written to the file)
   if (TFileCacheWrite *cachew = fFile->GetCacheWrite()) {
      if (cachew->ReadBuffer(buf,pos,len) == 0) {
         fFile->SetOffset(pos+len);
         return 1;
      }
   }

   //try to prefetch from the first block                                                                                                                  
   if (loc < 0)
      loc = (Int_t)TMath::BinarySearch(fNseek,fSeekSort,pos);

   if (loc >= 0 && loc < fNseek && pos == fSeekSort[loc]) {
      if (buf && fPrefetch){
         //prefetch with the new method                  
         fPrefetch->ReadBuffer(buf, pos, len);
      }
      return 1;
   }
   else if (buf && fPrefetch){
      //try to preferch from the second block                                                                                                               
      loc = (Int_t)TMath::BinarySearch(fBNseek, fBSeekSort, pos);
 
      if (loc >= 0 && loc < fBNseek && pos == fBSeekSort[loc]){
         if (fPrefetch->ReadBuffer(buf, pos, len))
             return 1;
      }
   }   
   
   return 0;
}


//_____________________________________________________________________________
Int_t TFileCacheRead::ReadBufferExtNormal(char *buf, Long64_t pos, Int_t len, Int_t &loc)
{
   // Base function for ReadBuffer. Also gives out the position
   // of the block in the internal buffer. This helps TTreeCacheUnzip to avoid
   // doing twice the binary search

   if (fNseek > 0 && !fIsSorted) {
      Sort();
      loc = -1;

      // If ReadBufferAsync is not supported by this implementation...
      if (!fAsyncReading) {
         // Then we use the vectored read to read everything now
         if (fFile->ReadBuffers(fBuffer,fPos,fLen,fNb)) {
            return -1;
         }
         fIsTransferred = kTRUE;
      } else {
         // In any case, we'll start to request the chunks.
         // This implementation simply reads all the chunks in advance
         // in the async way.

         // Use the async readv instead of single reads
         fFile->ReadBuffers(0, 0, 0, 0); //Clear the XrdClient cache
         if (fFile->ReadBuffers(0,fPos,fLen,fNb)) {
            return -1;
         }
         fIsTransferred = kTRUE;
      }
   }

   // in case we are writing and reading to/from this file, we much check
   // if this buffer is in the write cache (not yet written to the file)
   if (TFileCacheWrite *cachew = fFile->GetCacheWrite()) {
      if (cachew->ReadBuffer(buf,pos,len) == 0) {
         fFile->SetOffset(pos+len);
         return 1;
      }
   }

   // If asynchronous reading is supported by this implementation...
   if (fAsyncReading) {

         // Now we dont have to look for it in the local buffer
         // if it's async, we expect that the communication library
         // will handle it more efficiently than we can do here

      Int_t retval;
      if (loc < 0)
         loc = (Int_t)TMath::BinarySearch(fNseek,fSeekSort,pos);
      
      // We use the internal list just to notify if the list is to be reconstructed
      if (loc >= 0 && loc < fNseek && pos == fSeekSort[loc]) {
         // Block found, the caller will get it
         
         if (buf) {
            // disable cache to avoid infinite recursion
            fFile->SetCacheRead(0);
            if (fFile->ReadBuffer(buf, pos, len)) {
               return -1;
            }
            fFile->SetOffset(pos+len);
            fFile->SetCacheRead(this);
         }
         
         retval = 1;
      } else {
         // Block not found in the list, we report it as a miss
         retval = 0;
      }

      if (gDebug > 0)
         Info("ReadBuffer","pos=%lld, len=%d, retval=%d, loc=%d, fseekSort[loc]=%lld, fSeekLen[loc]=%d", pos, len, retval, loc, fSeekSort[loc], fSeekLen[loc]);
      
      return retval;
   } else {

      if (loc < 0)
         loc = (Int_t)TMath::BinarySearch(fNseek,fSeekSort,pos);

      if (loc >= 0 && loc <fNseek && pos == fSeekSort[loc]) {
         if (buf) {
            memcpy(buf,&fBuffer[fSeekPos[loc]],len);
            fFile->SetOffset(pos+len);
         }
         return 1;
      }
   }

   return 0;
}

//_____________________________________________________________________________
void TFileCacheRead::SetFile(TFile *file)
{
   // Set the file using this cache and reset the current blocks (if any).

   fFile = file;

   if (fAsyncReading) {
      // If asynchronous reading is not supported by this TFile specialization
      // we use sync primitives, hence we need the local buffer
      if (file && file->ReadBufferAsync(0, 0)) {
         fAsyncReading = kFALSE;
         fBuffer       = new char[fBufferSize];
      }
   }

   Prefetch(0,0);
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
      Int_t ind = fSeekIndex[i];
      fSeekSort[i] = fSeek[ind];
      fSeekSortLen[i] = fSeekLen[ind];
   }
   if (fNtot > fBufferSizeMin) {
      fBufferSize = fNtot + 100;
      delete [] fBuffer;
      fBuffer = 0;
      // If ReadBufferAsync is not supported by this implementation
      // it means that we are using sync primitives, hence we need the local buffer
      if (!fAsyncReading)
         fBuffer = new char[fBufferSize];
   }
   fPos[0]  = fSeekSort[0];
   fLen[0]  = fSeekSortLen[0];
   fSeekPos[0] = 0;
   for (i=1;i<fNseek;i++) {
      fSeekPos[i] = fSeekPos[i-1] + fSeekSortLen[i-1];
      //in the test below 16 MBytes is pure empirirical and may depend on the file system.
      //increasing this number must be done with care, as it may increase
      //the job real time (mismatch with OS buffers)
      if ((fSeekSort[i] != fSeekSort[i-1]+fSeekSortLen[i-1]) ||
          (fLen[nb] > 16000000)) {
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


//_____________________________________________________________________________                                                                             
void TFileCacheRead::SecondSort()
{
   // Sort buffers to be prefetched in increasing order of positions.                                                                                      
   // Merge consecutive blocks if necessary.        
   // Sort buffers to be prefetched in increasing order of positions. 
   // Merge consecutive blocks if necessary.                                                                                                               

   if (!fBNseek) return;
   TMath::Sort(fBNseek,fBSeek,fBSeekIndex,kFALSE);
   Int_t i;
   Int_t nb = 0;
   for (i=0;i<fBNseek;i++) {
      Int_t ind = fBSeekIndex[i];
      fBSeekSort[i] = fBSeek[ind];
      fBSeekSortLen[i] = fBSeekLen[ind];
   }
   if (fBNtot > fBufferSizeMin) {
      fBufferSize = fBNtot + 100;
      delete [] fBuffer;
      fBuffer = 0;
      // If ReadBufferAsync is not supported by this implementation                                                                                           
      // it means that we are using sync primitives, hence we need the local buffer                                                                           
      if (!fAsyncReading)
         fBuffer = new char[fBufferSize];
   }
   fBPos[0]  = fBSeekSort[0];
   fBLen[0]  = fBSeekSortLen[0];
   fBSeekPos[0] = 0;
   for (i=1;i<fBNseek;i++) {
      fBSeekPos[i] = fBSeekPos[i-1] + fBSeekSortLen[i-1];
      //in the test below 16 MBytes is pure empirirical and may depend on the file system.                                                                    
      //increasing this number must be done with care, as it may increase                                                                                     
      //the job real time (mismatch with OS buffers)                                                                                                          
      if ((fBSeekSort[i] != fBSeekSort[i-1]+fBSeekSortLen[i-1]) ||
         (fBLen[nb] > 16000000)) {
         nb++;
         fBPos[nb] = fBSeekSort[i];
         fBLen[nb] = fBSeekSortLen[i];
      } else {
         fBLen[nb] += fBSeekSortLen[i];
      }
   }
   fBNb = nb+1;
   fBIsSorted = kTRUE;
}

//______________________________________________________________________________
TFilePrefetch* TFileCacheRead::GetPrefetchObj(){
  
   return this->fPrefetch;
}
