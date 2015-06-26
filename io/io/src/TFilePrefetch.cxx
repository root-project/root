// @(#)root/io:$Id$
// Author: Elvin Sindrilaru   19/05/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TFilePrefetch.h"
#include "TTimeStamp.h"
#include "TVirtualPerfStats.h"
#include "TVirtualMonitoring.h"

#include <iostream>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cctype>

static const int kMAX_READ_SIZE    = 2;   //maximum size of the read list of blocks

inline int xtod(char c) { return (c>='0' && c<='9') ? c-'0' : ((c>='A' && c<='F') ? c-'A'+10 : ((c>='a' && c<='f') ? c-'a'+10 : 0)); }

using namespace std;

ClassImp(TFilePrefetch)

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TFilePrefetch::TFilePrefetch(TFile* file) :
  fFile(file),
  fConsumer(0),
  fThreadJoined(kTRUE)
{
   fPendingBlocks    = new TList();
   fReadBlocks       = new TList();

   fPendingBlocks->SetOwner();
   fReadBlocks->SetOwner();

   fMutexReadList    = new TMutex();
   fMutexPendingList = new TMutex();
   fNewBlockAdded    = new TCondition(0);
   fReadBlockAdded   = new TCondition(0);
   fSemMasterWorker  = new TSemaphore(0);
   fSemWorkerMaster  = new TSemaphore(0);
   fSemChangeFile    = new TSemaphore(0);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TFilePrefetch::~TFilePrefetch()
{
   if (!fThreadJoined) {
     WaitFinishPrefetch();
   }

   SafeDelete(fConsumer);
   SafeDelete(fPendingBlocks);
   SafeDelete(fReadBlocks);
   SafeDelete(fMutexReadList);
   SafeDelete(fMutexPendingList);
   SafeDelete(fNewBlockAdded);
   SafeDelete(fReadBlockAdded);
   SafeDelete(fSemMasterWorker);
   SafeDelete(fSemWorkerMaster);
   SafeDelete(fSemChangeFile);
}


////////////////////////////////////////////////////////////////////////////////
/// Killing the async prefetching thread

void TFilePrefetch::WaitFinishPrefetch()
{
   fSemMasterWorker->Post();

   TMutex *mutexCond = fNewBlockAdded->GetMutex();
   while ( fSemWorkerMaster->Wait(10) != 0 ) {
      mutexCond->Lock();
      fNewBlockAdded->Signal();
      mutexCond->UnLock();
   }

   fConsumer->Join();
   fThreadJoined=kTRUE;
}


////////////////////////////////////////////////////////////////////////////////
/// Read one block and insert it in prefetchBuffers list.

void TFilePrefetch::ReadAsync(TFPBlock* block, Bool_t &inCache)
{
   char* path = 0;

   if (CheckBlockInCache(path, block)){
      block->SetBuffer(GetBlockFromCache(path, block->GetDataSize()));
      inCache = kTRUE;
   }
   else{
      fFile->ReadBuffers(block->GetBuffer(), block->GetPos(), block->GetLen(), block->GetNoElem());
      if (fFile->GetArchive()) {
         for (Int_t i = 0; i < block->GetNoElem(); i++)
            block->SetPos(i, block->GetPos(i) - fFile->GetArchiveOffset());
      }
      inCache =kFALSE;
   }
   delete[] path;
}

////////////////////////////////////////////////////////////////////////////////
/// Get blocks specified in prefetchBlocks.

void TFilePrefetch::ReadListOfBlocks()
{
   Bool_t inCache = kFALSE;
   TFPBlock*  block = 0;

   while((block = GetPendingBlock())){
      ReadAsync(block, inCache);
      AddReadBlock(block);
      if (!inCache)
         SaveBlockInCache(block);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Search for a requested element in a block and return the index.

Bool_t TFilePrefetch::BinarySearchReadList(TFPBlock* blockObj, Long64_t offset, Int_t len, Int_t* index)
{
   Int_t first = 0, last = -1, mid = -1;
   last = (Int_t) blockObj->GetNoElem()-1;

   while (first <= last){
      mid = first + (last - first) / 2;
      if ((offset >= blockObj->GetPos(mid) && offset <= (blockObj->GetPos(mid) + blockObj->GetLen(mid))
           && ( (offset + len) <= blockObj->GetPos(mid) + blockObj->GetLen(mid)))){

         *index = mid;
         return true;
      }
      else if (blockObj->GetPos(mid) < offset){
         first = mid + 1;
      }
      else{
         last = mid - 1;
      }
   }
   return false;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the time spent wating for buffer to be read in microseconds.

Long64_t TFilePrefetch::GetWaitTime()
{
   return Long64_t(fWaitTime.RealTime()*1.e+6);
}

////////////////////////////////////////////////////////////////////////////////
/// Return a prefetched element.

Bool_t TFilePrefetch::ReadBuffer(char* buf, Long64_t offset, Int_t len)
{
   Bool_t found = false;
   TFPBlock* blockObj = 0;
   TMutex *mutexBlocks = fMutexReadList;
   Int_t index = -1;

   while (1){
      mutexBlocks->Lock();
      TIter iter(fReadBlocks);
      while ((blockObj = (TFPBlock*) iter.Next())){
         index = -1;
         if (BinarySearchReadList(blockObj, offset, len, &index)){
            found = true;
            break;
         }
      }
      if (found)
         break;
      else{
         mutexBlocks->UnLock();

         fWaitTime.Start(kFALSE);
         fReadBlockAdded->Wait(); //wait for a new block to be added
         fWaitTime.Stop();
      }
   }

   if (found){
      char *pBuff = blockObj->GetPtrToPiece(index);
      pBuff += (offset - blockObj->GetPos(index));
      memcpy(buf, pBuff, len);
   }
   mutexBlocks->UnLock();
   return found;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a TFPBlock object or recycle one and add it to the prefetchBlocks list.

void TFilePrefetch::ReadBlock(Long64_t* offset, Int_t* len, Int_t nblock)
{
   TFPBlock* block = CreateBlockObj(offset, len, nblock);
   AddPendingBlock(block);
}

////////////////////////////////////////////////////////////////////////////////
/// Safe method to add a block to the pendingList.

void TFilePrefetch::AddPendingBlock(TFPBlock* block)
{
   TMutex *mutexBlocks = fMutexPendingList;
   TMutex *mutexCond = fNewBlockAdded->GetMutex();

   mutexBlocks->Lock();
   fPendingBlocks->Add(block);
   mutexBlocks->UnLock();

   mutexCond->Lock();
   fNewBlockAdded->Signal();
   mutexCond->UnLock();
}

////////////////////////////////////////////////////////////////////////////////
/// Safe method to remove a block from the pendingList.

TFPBlock* TFilePrefetch::GetPendingBlock()
{
   TFPBlock* block = 0;
   TMutex *mutex = fMutexPendingList;
   mutex->Lock();

   if (fPendingBlocks->GetSize()){
      block = (TFPBlock*)fPendingBlocks->First();
      block = (TFPBlock*)fPendingBlocks->Remove(block);
   }
   mutex->UnLock();
   return block;
}

////////////////////////////////////////////////////////////////////////////////
/// Safe method to add a block to the readList.

void TFilePrefetch::AddReadBlock(TFPBlock* block)
{
   TMutex *mutexCond = fReadBlockAdded->GetMutex();
   TMutex *mutex = fMutexReadList;
   mutex->Lock();

   if (fReadBlocks->GetSize() >= kMAX_READ_SIZE){
      TFPBlock* movedBlock = (TFPBlock*) fReadBlocks->First();
      movedBlock = (TFPBlock*)fReadBlocks->Remove(movedBlock);
      delete movedBlock;
      movedBlock = 0;
   }

   fReadBlocks->Add(block);
   mutex->UnLock();

   //signal the addition of a new block
   mutexCond->Lock();
   fReadBlockAdded->Signal();
   mutexCond->UnLock();
}


////////////////////////////////////////////////////////////////////////////////
/// Create a new block or recycle an old one.

TFPBlock* TFilePrefetch::CreateBlockObj(Long64_t* offset, Int_t* len, Int_t noblock)
{
   TFPBlock* blockObj = 0;
   TMutex *mutex = fMutexReadList;

   mutex->Lock();

   if (fReadBlocks->GetSize() >= kMAX_READ_SIZE){
      blockObj = static_cast<TFPBlock*>(fReadBlocks->First());
      fReadBlocks->Remove(blockObj);
      mutex->UnLock();
      blockObj->ReallocBlock(offset, len, noblock);
   }
   else{
      mutex->UnLock();
      blockObj = new TFPBlock(offset, len, noblock);
   }
   return blockObj;
}

////////////////////////////////////////////////////////////////////////////////
/// Return reference to the consumer thread.

TThread* TFilePrefetch::GetThread() const
{
   return fConsumer;
}


////////////////////////////////////////////////////////////////////////////////
/// Change the file
/// When prefetching is enabled we also need to:
/// - make sure the async thread is not doing any work
/// - clear all blocks from prefetching and read list
/// - reset the file pointer

void TFilePrefetch::SetFile(TFile *file)
{
   if (!fThreadJoined) {
     fSemChangeFile->Wait();
   }

   if (fFile) {
     // Remove all pending and read blocks
     fMutexPendingList->Lock();
     fPendingBlocks->Clear();
     fMutexPendingList->UnLock();

     fMutexReadList->Lock();
     fReadBlocks->Clear();
     fMutexReadList->UnLock();
   }

   fFile = file;
   if (!fThreadJoined) {
     fSemChangeFile->Post();
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Used to start the consumer thread.

Int_t TFilePrefetch::ThreadStart()
{
   int rc;

   fConsumer = new TThread((TThread::VoidRtnFunc_t) ThreadProc, (void*) this);
   rc = fConsumer->Run();
   if ( !rc ) {
      fThreadJoined = kFALSE;
   }
   return rc;
}


////////////////////////////////////////////////////////////////////////////////
/// Execution loop of the consumer thread.

TThread::VoidRtnFunc_t TFilePrefetch::ThreadProc(void* arg)
{
   TFilePrefetch* pClass = (TFilePrefetch*) arg;
   TSemaphore* semChangeFile = pClass->fSemChangeFile;
   semChangeFile->Post();
   pClass->fNewBlockAdded->Wait();
   semChangeFile->Wait();

   while( pClass->fSemMasterWorker->TryWait() != 0 ) {
      pClass->ReadListOfBlocks();

      // Use the semaphore to deal with the case when the file pointer
      // is changed on the fly by TChain
      semChangeFile->Post();
      pClass->fNewBlockAdded->Wait();
      semChangeFile->Wait();
   }

   pClass->fSemWorkerMaster->Post();
   return (TThread::VoidRtnFunc_t) 1;
}

//############################# CACHING PART ###################################

////////////////////////////////////////////////////////////////////////////////
/// Sum up individual hex values to obtain a decimal value.

Int_t TFilePrefetch::SumHex(const char *hex)
{
   Int_t result = 0;
   const char* ptr = hex;

   for(Int_t i=0; i < (Int_t)strlen(hex); i++)
      result += xtod(ptr[i]);

   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Test if the block is in cache.

Bool_t TFilePrefetch::CheckBlockInCache(char*& path, TFPBlock* block)
{
   if (fPathCache == "")
      return false;

   Bool_t found = false;
   TString fullPath(fPathCache); // path of the cached files.

   Int_t value = 0;

   if (!gSystem->OpenDirectory(fullPath))
      gSystem->mkdir(fullPath);

   //dir is SHA1 value modulo 16; filename is the value of the SHA1(offset+len)
   TMD5* md = new TMD5();

   TString concatStr;
   for (Int_t i=0; i < block->GetNoElem(); i++){
      concatStr.Form("%lld", block->GetPos(i));
      md->Update((UChar_t*)concatStr.Data(), concatStr.Length());
   }

   md->Final();
   TString fileName( md->AsString() );
   value = SumHex(fileName);
   value = value % 16;
   TString dirName;
   dirName.Form("%i", value);

   fullPath += "/" + dirName + "/" + fileName;

   FileStat_t stat;
   if (gSystem->GetPathInfo(fullPath, stat) == 0) {
      path = new char[fullPath.Length() + 1];
      strlcpy(path, fullPath,fullPath.Length() + 1);
      found = true;
   } else
      found = false;

   delete md;
   return found;
}

////////////////////////////////////////////////////////////////////////////////
/// Return a buffer from cache.

char* TFilePrefetch::GetBlockFromCache(const char* path, Int_t length)
{
   char *buffer = 0;
   TString strPath = path;

   strPath += "?filetype=raw";
   TFile* file = new TFile(strPath);

   Double_t start = 0;
   if (gPerfStats != 0) start = TTimeStamp();

   buffer = (char*) calloc(length, sizeof(char));
   file->ReadBuffer(buffer, 0, length);

   fFile->fBytesRead  += length;
   fFile->fgBytesRead += length;
   fFile->SetReadCalls(fFile->GetReadCalls() + 1);
   fFile->fgReadCalls++;

   if (gMonitoringWriter)
      gMonitoringWriter->SendFileReadProgress(fFile);
   if (gPerfStats != 0) {
      gPerfStats->FileReadEvent(fFile, length, start);
   }

   file->Close();
   delete file;
   return buffer;
}

////////////////////////////////////////////////////////////////////////////////
/// Save the block content in cache.

void TFilePrefetch::SaveBlockInCache(TFPBlock* block)
{
   if (fPathCache == "")
      return;

   //dir is SHA1 value modulo 16; filename is the value of the SHA1
   TMD5* md = new TMD5();

   TString concatStr;
   for(Int_t i=0; i< block->GetNoElem(); i++){
      concatStr.Form("%lld", block->GetPos(i));
      md->Update((UChar_t*)concatStr.Data(), concatStr.Length());
   }
   md->Final();

   TString fileName( md->AsString() );
   Int_t value = SumHex(fileName);
   value = value % 16;

   TString fullPath( fPathCache );
   TString dirName;
   dirName.Form("%i", value);
   fullPath += ("/" + dirName);

   if (!gSystem->OpenDirectory(fullPath))
      gSystem->mkdir(fullPath);

   TFile* file = 0;
   fullPath += ("/" + fileName);
   FileStat_t stat;
   if (gSystem->GetPathInfo(fullPath, stat) == 0) {
      fullPath += "?filetype=raw";
      file = TFile::Open(fullPath, "update");
   } else{
      fullPath += "?filetype=raw";
      file = TFile::Open(fullPath, "new");
   }

   if (file) {
      // coverity[unchecked_value] We do not print error message, have not error
      // return code and close the file anyway, not need to check the return value.
      file->WriteBuffer(block->GetBuffer(), block->GetDataSize());
      file->Close();
      delete file;
   }
   delete md;
}


////////////////////////////////////////////////////////////////////////////////
/// Set the path of the cache directory.

Bool_t TFilePrefetch::SetCache(const char* path)
{
  fPathCache = path;

  if (!gSystem->OpenDirectory(path)){
    return (!gSystem->mkdir(path) ? true : false);
  }

  // Directory already exists
  return true;
}

