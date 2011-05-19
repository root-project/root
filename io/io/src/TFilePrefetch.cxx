#ifndef ROOT_TFilePrefetch
#include "TFilePrefetch.h"
#endif
#ifndef ROOT_TTimeStamp
#include "TTimeStamp.h"
#endif
#ifndef ROOT_TVirtualPerfStats
#include "TVirtualPerfStats.h"
#endif
#ifndef ROOT_TVirtualMonitoring
#include "TVirtualMonitoring.h"
#endif

#include <iostream>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <sys/time.h>

#define MAX_READ_SIZE  4                       //maximum size of the read list of blocks
#define MAX_RECYCLE_SIZE 2                      //maximum size of the recycle list of blocks

#define xtod(c) ((c>='0' && c<='9') ? c-'0' : ((c>='A' && c<='F') ? c-'A'+10 : ((c>='a' && c<='f') ? c-'a'+10 : 0)))

using namespace std;

ClassImp(TFilePrefetch)

//____________________________________________________________________________________________
//constructor 
TFilePrefetch::TFilePrefetch(TFile* file)
{

   fConsumer = 0;  
   fFile = file;
   fPathCache= NULL;
   fPendingBlocks = new TList();
   fReadBlocks = new TList();
   fRecycleBlocks = new TList();
   fMutexReadList = new TMutex();
   fMutexPendingList = new TMutex();
   fMutexRecycleList = new TMutex();
   fNewBlockAdded = new TCondition(0);
   fReadBlockAdded = new TCondition(0);
   fSem = new TSemaphore(0);
   fWaitTime = 0;

}


//____________________________________________________________________________________________
//class destructor
TFilePrefetch::~TFilePrefetch()
{
  
   //killing consumer thread
   fSem->Post();
   fNewBlockAdded->Signal();

   delete fPendingBlocks;
   delete fReadBlocks;
   delete fRecycleBlocks;
   delete fMutexReadList;
   delete fMutexPendingList;
   delete fMutexRecycleList;
   delete fNewBlockAdded;
   delete fReadBlockAdded;
   delete fSem;
}

//____________________________________________________________________________________________
void TFilePrefetch::ReadAsync(TFPBlock* block, Bool_t &inCache)
{
   //Read one block and insert it in prefetchBuffers list.
  
   char* path = NULL;
 
   if (CheckBlockInCache(path, block)){        
      block->SetBuffer(GetBlockFromCache(path, block->GetFullSize()));
      inCache = kTRUE;
   }
   else{
     fFile->ReadBuffers(block->GetBuffer(), block->GetPos(), block->GetLen(), block->GetNoElem());
     inCache =kFALSE;
   }
   delete[] path;
}


//____________________________________________________________________________________________
void TFilePrefetch::ReadListOfBlocks()
{
   //Get blocks specified in prefetchBlocks.

   Bool_t inCache = kFALSE;
   TFPBlock*  block = NULL;

   while((block = GetPendingBlock())){  
     ReadAsync(block, inCache);
     AddReadBlock(block);
     if (!inCache)
        SaveBlockInCache(block);
   }
}

//____________________________________________________________________________________________
Bool_t TFilePrefetch::BinarySearchReadList(TFPBlock* blockObj, Long64_t offset, Int_t len, Int_t* index)
{
   //Search for a requested element in a block and return the index.

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


//____________________________________________________________________________________________
Bool_t TFilePrefetch::ReadBuffer(char* buf, Long64_t offset, Int_t len)
{
   //Return a prefetched element.

   struct timeval tv;
   Bool_t found = false;
   TFPBlock* blockObj = NULL;
   TMutex *mutexBlocks = fMutexReadList;
   Int_t index = -1;
   Long64_t time;

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

         gettimeofday(&tv, NULL);
         time = ((Long64_t)1e+6)*tv.tv_sec + tv.tv_usec;
         fReadBlockAdded->Wait(); //wait for a new block to be added
         gettimeofday(&tv, NULL);
         fWaitTime += ((tv.tv_sec*((Long64_t)1e+6) + tv.tv_usec) - time);
      }
   }

   if (found){
      Int_t auxInt = 0;       
      char* ptrInt = NULL;    

      for(Int_t i=0; i < blockObj->GetNoElem(); i++){

         ptrInt = blockObj->GetBuffer();
         ptrInt += auxInt;

         if (index == i){
            ptrInt+= (offset - blockObj->GetPos(i));
            memcpy(buf, ptrInt, len);
            break;
         }
         auxInt += blockObj->GetLen(i);
      }
   }
   mutexBlocks->UnLock();
   return found;
}


//____________________________________________________________________________________________
void TFilePrefetch::ReadBlock(Long64_t* offset, Int_t* len, Int_t nblock)
{
   //Create a TFPBlock object or recycle one and add it to the prefetchBlocks list.

   TFPBlock* block = CreateBlockObj(offset, len, nblock);
   AddPendingBlock(block);
}


//____________________________________________________________________________________________
void TFilePrefetch::AddPendingBlock(TFPBlock* block)
{
   //Safe method to add a block to the pendingList.

   TMutex *mutexBlocks = fMutexPendingList;

   mutexBlocks->Lock();
   fPendingBlocks->Add(block);
   mutexBlocks->UnLock();   
   fNewBlockAdded->Signal();
}


//____________________________________________________________________________________________
TFPBlock* TFilePrefetch::GetPendingBlock()
{
   //Safe method to remove a block from the pendingList.

   TFPBlock* block = NULL;
   TMutex *mutexBlocks = fMutexPendingList;
   mutexBlocks->Lock();

   if (fPendingBlocks->GetSize()){
      block = (TFPBlock*)fPendingBlocks->First();
     block = (TFPBlock*)fPendingBlocks->Remove(block);
   }
   mutexBlocks->UnLock();   
   return block;
}


//____________________________________________________________________________________________
void TFilePrefetch::AddReadBlock(TFPBlock* block)
{
   //Safe method to add a block to the readList.

   TMutex *mutexBlocks = fMutexReadList;
   mutexBlocks->Lock();

   if (fReadBlocks->GetSize() >= MAX_READ_SIZE){
      TFPBlock* movedBlock = (TFPBlock*) fReadBlocks->First();
      movedBlock = (TFPBlock*)fReadBlocks->Remove(movedBlock);
      AddRecycleBlock(movedBlock);
   }

   fReadBlocks->Add(block);
   mutexBlocks->UnLock();  
   fReadBlockAdded->Signal();
}


//____________________________________________________________________________________________
void TFilePrefetch::AddRecycleBlock(TFPBlock* block)
{
   //Safe method to add a block to the recycleList.

   TMutex *mutexBlocks = fMutexRecycleList;
   mutexBlocks->Lock(); 

   if (fRecycleBlocks->GetSize() >= MAX_RECYCLE_SIZE){
      delete block;
   }
   else{
      fRecycleBlocks->Add(block);
   }
   mutexBlocks->UnLock(); 
}


//____________________________________________________________________________________________
TFPBlock* TFilePrefetch::CreateBlockObj(Long64_t* offset, Int_t* len, Int_t noblock)
{
   //Create a new block or recycle an old one.

   TFPBlock* blockObj = NULL;
   TMutex *mutexRecycle = fMutexRecycleList;
   
   mutexRecycle->Lock();

   if (fRecycleBlocks->GetSize()){
      blockObj = static_cast<TFPBlock*>(fRecycleBlocks->First());
      fRecycleBlocks->Remove(blockObj);
      blockObj->ReallocBlock(offset, len, noblock);
      mutexRecycle->UnLock();   
   }
   else{
      mutexRecycle->UnLock();  
      blockObj = new TFPBlock(offset, len, noblock);
   }
   return blockObj;
}

//____________________________________________________________________________________________
TThread* TFilePrefetch::GetThread()
{
   //Return reference to the consumer thread.

   return fConsumer;
}


//____________________________________________________________________________________________
Int_t TFilePrefetch::ThreadStart()
{
   //Used to start the consumer thread.
     
   fConsumer= new TThread("consumerThread",
                             (void(*) (void *))ThreadProc,
                              (void*) this);
   fConsumer->Run();
   return 1;
}

//____________________________________________________________________________________________
void TFilePrefetch::ThreadProc(void* arg)
{
   //Execution loop of the consumer thread.

   TFilePrefetch* tmp = (TFilePrefetch*) arg;

   while(tmp->fSem->TryWait() !=0){
      tmp->ReadListOfBlocks();
      tmp->fNewBlockAdded->Wait();
   } 
}

//########################################### CACHING PART ###############################################################

//____________________________________________________________________________________________
Int_t TFilePrefetch::SumHex(char *hex)
{
   //Sum up individual hex values to obtain a decimal value.

   Int_t result = 0;
   char* ptr = hex;

   for(Int_t i=0; i < (Int_t)strlen(hex); i++)
      result += xtod(ptr[i]);

   return result;
}


//____________________________________________________________________________________________
Bool_t TFilePrefetch::CheckBlockInCache(char*& path, TFPBlock* block)
{
   //Test if the block is in cache.

   Bool_t found = false;
   TString fullPath;    //path of the cached files
   Int_t value = 0;
   FileStat_t* stat = new FileStat_t();

   if (!fPathCache){
      delete stat;
      return false;
   }
   else
      fullPath = *(fPathCache);

   if (gSystem->OpenDirectory(fullPath.Data()) == 0)
      gSystem->mkdir(fullPath.Data()); 

   char* concatStr = new char[100];
   TString fileName, dirName;

   //dir is SHA1 value modulo 16; filename is the value of the SHA1(offset+len)
   TMD5* md = new TMD5();

   for (Int_t i=0; i < block->GetNoElem(); i++){
      sprintf(concatStr, "%lld", block->GetPos(i));
      md->Update((UChar_t*)concatStr, strlen(concatStr));
   }

   md->Final();
   fileName = md->AsString();
   value = SumHex((char *)fileName.Data());
   value = value % 16;
   sprintf(concatStr, "%i", value);
   dirName = concatStr;

   fullPath += "/" + dirName + "/" + fileName;

   if (gSystem->GetPathInfo(fullPath.Data(), *stat) == 0){    
      path = new char[fullPath.Sizeof() + 1];
      strcpy(path, fullPath.Data());
      found = true;
   }
   else
      found = false;

   delete[] concatStr;
   delete stat;
   delete md;
   return found;
}


//____________________________________________________________________________________________
char* TFilePrefetch::GetBlockFromCache(char* path, Int_t length)
{
   //Return a buffer from cache.

   char*buffer = NULL;
   TString strPath = path;

   strPath += "?filetype=raw";
   TFile* file = new TFile(strPath.Data());

   Double_t start = 0;
   if (gPerfStats != 0) start = TTimeStamp();

   buffer = (char*) calloc(length+1, sizeof(char));
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

   delete file;
   return buffer;
}


//____________________________________________________________________________________________
void TFilePrefetch::SaveBlockInCache(TFPBlock* block)
{
   //Save the block content in cache.
  
   TString fullPath;
   TString fileName, dirName;
   char* concatStr = new char[100];
   Int_t value = 0;
   TFile* file = NULL;

   if (!fPathCache){
      delete[] concatStr;
      return;
   }
   else
      fullPath = *(fPathCache);

   //dir is SHA1 value modulo 16; filename is the value of the SHA1
   TMD5* md = new TMD5();
   FileStat_t* stat = new FileStat_t();

   for(Int_t i=0; i< block->GetNoElem(); i++){
      sprintf(concatStr, "%lld", block->GetPos(i));
      md->Update((UChar_t*)concatStr, strlen(concatStr));
   }
   md->Final();

   fileName  = md->AsString();
   value = SumHex((char*)fileName.Data());
   value = value % 16;

   sprintf(concatStr, "%i", value);
   dirName = concatStr;
   fullPath += ("/" + dirName);
  
   if (gSystem->OpenDirectory(fullPath.Data()) == false)
      gSystem->mkdir(fullPath.Data());
   
   fullPath += ("/" + fileName);
   if (gSystem->GetPathInfo(fullPath.Data(), *stat) == 0){  
      fullPath += "?filetype=raw";
      file = TFile::Open(fullPath.Data(), "update");
   }
   else{
      fullPath += "?filetype=raw";
      file = TFile::Open(fullPath.Data(), "new");
   }

   file->WriteBuffer(block->GetBuffer(), block->GetFullSize());
   file->Close();

   file = 0;
   delete file;
   delete stat;
   delete[] concatStr;
   delete md;
}


//____________________________________________________________________________________________
Bool_t TFilePrefetch::CheckCachePath(char* locationCache)
{
   //Validate the input file cache path.

   Bool_t found = true;
   TString path = locationCache;
   Ssiz_t pos = path.Index(":/");

   if (pos > 0){
      TSubString prot   = path(0, pos);
      TSubString dir  = path(pos + 2, path.Length());
      TString protocol(prot);
      TString directory(dir);

      for(Int_t i=0; i < directory.Sizeof()-1; i++)
        if (!isdigit(directory[i]) && !isalpha(directory[i]) && directory[i] !='/' && directory[i] != ':'){
           found = false;
           break;
        }
   }
   else
      found = false;
 
   return found;
}


//____________________________________________________________________________________________
Bool_t TFilePrefetch::SetCache(char* path)
{
   //Set the path of the cache directory.

   if (CheckCachePath(path)){
      fPathCache = new TString(path);
  
      if (!gSystem->OpenDirectory(path)){        
        gSystem->mkdir(path);
      }      
   }
   else{
      return false;
   }
   return true;
}

