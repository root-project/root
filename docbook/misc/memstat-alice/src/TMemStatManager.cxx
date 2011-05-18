// @(#)root/new:$Name$:$Id$
// Author: D.Bertini and M.Ivanov   10/08/2000  -- Anar Manafov (A.Manafov@gsi.de) 28/04/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//****************************************************************************//
/*
 TMemStatManager - manager class

 The current memory statistic is written to the file

 Important information  used for visualization
 std::vector<TMemStatStackInfo>       fStackVector;    // vector with stack symbols
 std::vector<TMemStatInfoStamp>       fStampVector;    // vector of stamp information
 std::vector<TTimeStamp>       fStampTime;      // vector of stamp information
 std::vector<TMemStatCodeInfo>        fCodeInfoArray;  // vector with code info
*/
//****************************************************************************//

// STD
#include <cstdio>
#include <string>

// ROOT
#include "TSystem.h"
#include "TEnv.h"
#include "TError.h"
#include "Riostream.h"
#include "TObject.h"
#include "TFile.h"
#include "TTree.h"
#include "TObjString.h"
// Memstat
#include "TMemStatDepend.h"
#include "TMemStatInfo.h"
#include "TMemStatManager.h"

const char * const g_cszFileName("memstat.root");
const Int_t g_STHashSize(65536);   //!current size of the hash table

ClassImp(TMemStatManager)

TMemStatManager * TMemStatManager::fgInstance = NULL;


//****************************************************************************//
//                                   Global Stack Table
//****************************************************************************//

TMemStatManager::TMemStatManager():
      TObject(),
      fSTHashTable(g_STHashSize, -1),           //!pointer to the hash table
      fCount(0),                 //!number of entries in table
      fStampNumber(0),           //current stamp number
      fStackVector(),            // vector with stack symbols
      fStampVector(),            // vector of stamp information
      fStampTime(),              // vector of stamp information
      fCodeInfoArray()        ,  // vector with code info
      fCodeInfoMap(),            //! map of code information
      fDebugLevel(0),            //!debug level
      fStampCallBack(0),         //! call back function to register stamp
      fPreviousMallocHook(TMemStatDepend::GetMallocHook()),    //!
      fPreviousFreeHook(TMemStatDepend::GetFreeHook()),      //!
      fLastStamp(),              //last written stamp
      fCurrentStamp(),           //current stamp
      fAutoStampSize(2000000),   //change of size invoking STAMP
      fAutoStampN(200000),       //change of number of allocation  STAMP
      fAutoStampDumpSize(50000), //change of number of allocation  STAMP
      fMinStampSize(100),        //minimal cut what will be dumped  to the tree
      fSize(65536),              //!size of hash table
      fLeak(NULL),               //!pointer to the hash table
      fAllocCount(0),            //!number of memory allocation blocks
      fMultDeleteTable(),        //!pointer to the table
      fDumpTree(0),              //!tree to dump information
      fDumpSysTree(0),           //!tree to dump information
      fUseGNUBuildinBacktrace(kFALSE)
{
   // Default constructor

   SetBit(kUserDisable, kTRUE);
   fStampCallBack = TMemStatManager::SAddStamps;         //! call back function

}

//______________________________________________________________________________
void TMemStatManager::Init()
{
   //Initialize MemStat manager - used only for instance

   SetBit(kUserDisable, kTRUE);

   fStampNumber = 0;
   fAllocCount = 0;
   FreeHashtable();
   fLeak = (TMemTable_t **) malloc(sizeof(void *) * fSize);
   fMultDeleteTable.fLeaks = 0;
   fMultDeleteTable.fAllocCount = 0;
   fMultDeleteTable.fTableSize = 0;
   fStackVector.reserve(fSize);             // vector with stack symbols
   fStampVector.reserve(fSize*10);          // vector of stamp information
   fCodeInfoArray.reserve(fSize);           // vector with code info
   fStampTime.reserve(fSize);
   fStampTime[0] = TTimeStamp();
   for (int i = 0; i < fSize; ++i) {
      fLeak[i] = (TMemTable_t *) malloc(sizeof(TMemTable_t));
      fLeak[i]->fAllocCount = 0;
      fLeak[i]->fMemSize = 0;
      fLeak[i]->fFirstFreeSpot = 0;
      fLeak[i]->fTableSize = 0;
      fLeak[i]->fLeaks = 0;
   }
   //Initialize ST table.
   fCount = 0;

   SetBit(kUserDisable, kTRUE);
}

//______________________________________________________________________________
TMemStatManager* TMemStatManager::GetInstance()
{
   // GetInstance of MemStatManager
   // Only instance catch the alloc and free hook

   if (!fgInstance) {
      fgInstance = new TMemStatManager;
      fgInstance->Init();
   }
   return fgInstance;
}

//______________________________________________________________________________
void TMemStatManager::Close()
{
   // to be documented

   delete fgInstance;
   fgInstance = NULL;
}

//______________________________________________________________________________
TMemStatManager::~TMemStatManager()
{
   //   Destructor
   //   if instance is destructed - the hooks are reseted to old hooks

   if (this != TMemStatManager::GetInstance())
      return;
   SetBit(kStatDisable);
   Disable();
   AddStamps("End");
   DumpTo(kTree, kTRUE, "End");
   DumpTo(kSysTree, kTRUE, "End");
   Disable();

   FreeHashtable();
}

//______________________________________________________________________________
void TMemStatManager::Enable()
{
   // Enable hooks

   if (this != GetInstance())
      return;

   // set hook to our functions
   TMemStatDepend::SetMallocHook(AllocHook);
   TMemStatDepend::SetFreeHook(FreeHook);
   SetBit(kUserDisable, kFALSE);
}

//______________________________________________________________________________
void TMemStatManager::Disable()
{
   // disble MemStatManager

   if (this != GetInstance())
      return;

   // set hook to our functions
   TMemStatDepend::SetMallocHook(fPreviousMallocHook);
   TMemStatDepend::SetFreeHook(fPreviousFreeHook);
   SetBit(kUserDisable, kTRUE);
}

//______________________________________________________________________________
void *TMemStatManager::AllocHook(size_t size, const void* /*caller*/)
{
   // AllocHook

   TMemStatManager* instance = TMemStatManager::GetInstance();
   TMemStatDepend::SetMallocHook(instance->fPreviousMallocHook);
   void *p = instance->AddPointer(size);
   TMemStatDepend::SetMallocHook(AllocHook);
   return p;
}

//______________________________________________________________________________
void TMemStatManager::FreeHook(void* ptr, const void* /*caller*/)
{
   // FreeHook

   TMemStatManager* instance = TMemStatManager::GetInstance();
   TMemStatDepend::SetFreeHook(instance->fPreviousFreeHook);
   instance->FreePointer(ptr);
   TMemStatDepend::SetFreeHook(FreeHook);
}

//______________________________________________________________________________
TMemStatStackInfo *TMemStatManager::STAddInfo(int size, void **stackptrs)
{
   // Add stack information to table.
   // add next stack to table

   const UInt_t currentSize = fStackVector.size();
   if (currentSize >= fStackVector.capacity())
      fStackVector.reserve(2*currentSize + 1);

   fStackVector.push_back(TMemStatStackInfo());
   TMemStatStackInfo *info = &(fStackVector[currentSize]);
   info->Init(size, stackptrs, this, currentSize);
   info->fStackID = currentSize;

   //add info to hash table
   const int hash = int(info->Hash() % g_STHashSize);
   Int_t hashIndex = fSTHashTable[hash];
   TMemStatStackInfo *info2 = NULL;

   if (-1 == hashIndex) {
      fSTHashTable[hash] = info->fStackID;
   } else {
      info2 = &fStackVector[hashIndex];
      while (hashIndex >= 0) {
         hashIndex = info2->fNextHash;
         if (hashIndex >= 0)
            info2 = &fStackVector[hashIndex];
      }
      info2->fNextHash = info->fStackID;
   }

   ++fCount;
   fStackVector.push_back(*info);
   return info;
}

//______________________________________________________________________________
TMemStatStackInfo *TMemStatManager::STFindInfo(int size, void **stackptrs)
{
   // Try to find stack info in hash table if doesn't find it will add it.

   const int hash = int(TMemStatStackInfo::HashStack(size, (void **)stackptrs) % g_STHashSize);

   if (fSTHashTable[hash] < 0)
      return STAddInfo(size, stackptrs); // hash value not in hash table

   Int_t hashIndex = fSTHashTable[hash];
   TMemStatStackInfo *info = NULL;

   info = &fStackVector[hashIndex];
   while (hashIndex >= 0) {
      if (info->Equal(size, stackptrs) == 1)
         return info;   // info found
      hashIndex = info->fNextHash;
      if (hashIndex >= 0)
         info = &fStackVector[hashIndex];
   }
   return STAddInfo(size, stackptrs);  // not found info - create new
}

//______________________________________________________________________________
void TMemStatManager::SAddStamps(const char * stampname)
{
   //
   // static version add  stamps to the list of stamps for changed stacks
   //
   TMemStatManager *man = GetInstance();
   man->AddStamps(stampname);
}

//______________________________________________________________________________
void  TMemStatManager::AddStamps(const char * stampname)
{
   // add the stamp to the list of stamps

   const UInt_t ssize = fStackVector.size();
   for (UInt_t i = 0; i < ssize; ++i) {
      if (fStackVector[i].fCurrentStamp.fAllocSize > fMinStampSize)
         fStackVector[i].MakeStamp(fStampNumber);
   }
   const UInt_t csize = fCodeInfoArray.size();
   for (UInt_t i = 0; i < csize; ++i) {
      if (fCodeInfoArray[i].fCurrentStamp.fAllocSize > fMinStampSize)
         fCodeInfoArray[i].MakeStamp(fStampNumber);
   }

   fCurrentStamp.fID = -1;
   fCurrentStamp.fStampNumber = fStampNumber;
   AddStamp() = fCurrentStamp;

   fStampTime[fStampNumber] = TTimeStamp();
   if (fStampVector.size() >= fAutoStampDumpSize || stampname) {
      DumpTo(kTree, kTRUE, stampname);
      DumpTo(kSysTree, kTRUE, stampname);
   }
   ++fStampNumber;
}

//______________________________________________________________________________
TMemStatInfoStamp &TMemStatManager::AddStamp()
{
   // add one stamp to the list of stamps

   const UInt_t size = fStampVector.size();
   fStampVector.push_back(TMemStatInfoStamp());
   TMemStatInfoStamp &stamp =  fStampVector[size];
   stamp.fStampNumber = fStampNumber;
   return stamp;
}

//______________________________________________________________________________
TMemStatCodeInfo &TMemStatManager::GetCodeInfo(void *address)
{
   //  to be documented

   TMemStatCodeInfo *info(NULL);
   const UInt_t index = fCodeInfoMap[address];
   if (index > 0) {
      info = &(fCodeInfoArray[fCodeInfoMap[address]]);
   } else {
      const UInt_t size = fCodeInfoArray.size();
      fCodeInfoArray.push_back(TMemStatCodeInfo());
      info = &(fCodeInfoArray[size]);
      fCodeInfoMap[address] = size;
      info->fCodeID = size;
      info->fCurrentStamp.fID = info->fCodeID;
      info->fLastStamp.fID = info->fCodeID;
   }
   return *info;
}

//______________________________________________________________________________
void TMemStatManager::RehashLeak(int newSize)
{
   // Rehash leak pointers.

   if (newSize <= fSize)
      return;
   TMemTable_t **newLeak = (TMemTable_t **) malloc(sizeof(void *) * newSize);
   for (int i = 0; i < newSize; ++i) {
      //build new branches
      newLeak[i] = (TMemTable_t *) malloc(sizeof(TMemTable_t));
      newLeak[i]->fAllocCount = 0;
      newLeak[i]->fMemSize = 0;
      newLeak[i]->fFirstFreeSpot = 0;
      newLeak[i]->fTableSize = 0;
      newLeak[i]->fLeaks = 0;
   }
   for (int ib = 0; ib < fSize; ++ib) {
      TMemTable_t *branch = fLeak[ib];
      for (int i = 0; i < branch->fTableSize; i++)
         if (branch->fLeaks[i].fAddress != 0) {
            int hash = int(TString::Hash(&branch->fLeaks[i].fAddress, sizeof(void*)) % newSize);
            TMemTable_t *newbranch = newLeak[hash];
            if (newbranch->fAllocCount >= newbranch->fTableSize) {
               int newTableSize =
                  newbranch->fTableSize ==
                  0 ? 16 : newbranch->fTableSize * 2;
               newbranch->fLeaks =
                  (TMemInfo_t *) realloc(newbranch->fLeaks,
                                         sizeof(TMemInfo_t) * newTableSize);
               if (!newbranch->fLeaks) {
                  Error("TMemStatManager::AddPointer", "realloc failure");
                  _exit(1);
               }
               memset(newbranch->fLeaks + newbranch->fTableSize, 0,
                      sizeof(TMemInfo_t) * (newTableSize -
                                            newbranch->fTableSize));
               newbranch->fTableSize = newTableSize;
            }
            memcpy(&newbranch->fLeaks[newbranch->fAllocCount],
                   &branch->fLeaks[i], sizeof(TMemInfo_t));
            newbranch->fAllocCount++;
            newbranch->fMemSize += branch->fLeaks[i].fSize;
         }
      free(branch->fLeaks);
      free(branch);
   }                 //loop over all old branches and rehash information
   free(fLeak);
   fLeak = newLeak;
   fSize = newSize;
}

//______________________________________________________________________________
void *TMemStatManager::AddPointer(size_t  size, void *ptr)
{
   // Add pointer to table.

   if (TestBit(kUserDisable) || TestBit(kStatDisable)) {
      return malloc(size);
   }

   Bool_t status = TestBit(kStatRoutine);
   SetBit(kStatRoutine, kTRUE);

   void *p = NULL;

   if (ptr == 0) {
      p = malloc(size);
      if (!p) {
         Error("TMemStatManager::AddPointer", "malloc failure");
         TMemStatManager::GetInstance()->Disable();
         TMemStatManager::GetInstance()->Close();
         _exit(1);
      }
   } else {
      p = realloc((char *) ptr, size);
      if (!p) {
         Error("TMemStatManager::AddPointer", "realloc failure");
         TMemStatManager::GetInstance()->Disable();
         TMemStatManager::GetInstance()->Close();
         _exit(1);
      }
      SetBit(kStatRoutine, status);
      return p;
   }
   if (status) {
      SetBit(kStatRoutine, status);
      return p;
   }

   if (!fSize)
      Init();
   ++fAllocCount;
   if ((fAllocCount / fSize) > 128)
      RehashLeak(fSize * 2);
   int hash = int(TString::Hash(&p, sizeof(void*)) % fSize);
   TMemTable_t *branch = fLeak[hash];
   branch->fAllocCount++;
   branch->fMemSize += size;

   fCurrentStamp.Inc(size);
   if ((fCurrentStamp.fTotalAllocCount - fLastStamp.fTotalAllocCount) > fAutoStampN ||
         (fCurrentStamp.fAllocCount - fLastStamp.fAllocCount) > Int_t(fAutoStampN) ||
         (fCurrentStamp.fTotalAllocSize - fLastStamp.fTotalAllocSize) > fAutoStampSize ||
         (fCurrentStamp.fAllocSize - fLastStamp.fAllocSize) > Int_t(fAutoStampSize)) {
      AddStamps();
      fLastStamp = fCurrentStamp;
      if (fAutoStampN < 0.001*fLastStamp.fTotalAllocCount) fAutoStampN = 1 + UInt_t(0.001 * fLastStamp.fTotalAllocCount);
      if (fAutoStampSize < 0.001*fLastStamp.fTotalAllocSize) fAutoStampSize = 1 + UInt_t(0.001 * fLastStamp.fTotalAllocSize);
   }

   for (;;) {
      for (int i = branch->fFirstFreeSpot; i < branch->fTableSize; ++i)
         if (branch->fLeaks[i].fAddress == 0) {
            branch->fLeaks[i].fAddress = p;
            branch->fLeaks[i].fSize = size;
            void *stptr[TMemStatStackInfo::kStackHistorySize + 1];
            int stackentries = TMemStatDepend::Backtrace(stptr, TMemStatStackInfo::kStackHistorySize, fUseGNUBuildinBacktrace);
            TMemStatStackInfo *info = STFindInfo(stackentries, stptr);
            info->Inc(size, this);
            if (info->fCurrentStamp.fStampNumber == 0) {
               info->MakeStamp(fStampNumber);  // add stamp after each addition
            }
            branch->fLeaks[i].fStackIndex = info->fStackID;
            branch->fFirstFreeSpot = i + 1;
            SetBit(kStatRoutine, status);
            return p;
         }

      int newTableSize =
         branch->fTableSize == 0 ? 16 : branch->fTableSize * 2;
      branch->fLeaks =
         (TMemInfo_t *) realloc(branch->fLeaks,
                                sizeof(TMemInfo_t) * newTableSize);
      if (!branch->fLeaks) {
         Error("TMemStatManager::AddPointer", "realloc failure (2)");
         _exit(1);
      }
      memset(branch->fLeaks + branch->fTableSize, 0, sizeof(TMemInfo_t) *
             (newTableSize - branch->fTableSize));
      branch->fTableSize = newTableSize;
   }
}

//______________________________________________________________________________
void TMemStatManager::FreePointer(void *p)
{
   // Free pointer.

   if (p == 0)
      return;
   if (TestBit(kUserDisable) || TestBit(kStatDisable)) {
      free(p);
      return;
   }

   const Bool_t status = TestBit(kStatRoutine);
   SetBit(kStatRoutine, kTRUE);

   if (status) {
      SetBit(kStatRoutine, status);
      return;
   }

   const int hash = static_cast<int>(TString::Hash(&p, sizeof(void*)) % fSize);
   --fAllocCount;
   TMemTable_t *branch = fLeak[hash];
   for (int i = 0; i < branch->fTableSize; i++) {
      if (branch->fLeaks[i].fAddress == p) {
         branch->fLeaks[i].fAddress = 0;
         branch->fMemSize -= branch->fLeaks[i].fSize;
         if (i < branch->fFirstFreeSpot)
            branch->fFirstFreeSpot = i;
         free(p);
         TMemStatStackInfo *info =
            &(fStackVector[branch->fLeaks[i].fStackIndex]);
         info->Dec(branch->fLeaks[i].fSize, this);
         fCurrentStamp.Dec(branch->fLeaks[i].fSize);
         branch->fAllocCount--;
         SetBit(kStatRoutine, status);
         return;
      }
   }
   //
   //if try to delete non existing pointer
   //printf("***TMemStatManager::FreePointer: Multiple deletion %8p ** ?  \n",p);
   //  printf("-+-+%8p  \n",p);
   //free(p);
   if (fMultDeleteTable.fTableSize + 1 > fMultDeleteTable.fAllocCount) {
      int newTableSize =
         fMultDeleteTable.fTableSize ==
         0 ? 16 : fMultDeleteTable.fTableSize * 2;
      fMultDeleteTable.fLeaks =
         (TMemInfo_t *) realloc(fMultDeleteTable.fLeaks,
                                sizeof(TMemInfo_t) * newTableSize);
      fMultDeleteTable.fAllocCount = newTableSize;
   }

   fMultDeleteTable.fLeaks[fMultDeleteTable.fTableSize].fAddress = 0;
   //void *sp = 0;
   void *stptr[TMemStatStackInfo::kStackHistorySize + 1];
   int stackentries = TMemStatDepend::Backtrace(stptr, TMemStatStackInfo::kStackHistorySize, fUseGNUBuildinBacktrace);
   TMemStatStackInfo *info = STFindInfo(stackentries/*j*/, stptr);
   info->Dec(0, this);
   fMultDeleteTable.fLeaks[fMultDeleteTable.fTableSize].fStackIndex =
      info->fStackID;
   fMultDeleteTable.fTableSize++;
   SetBit(kStatRoutine, status);
}

//______________________________________________________________________________
void TMemStatManager::DumpTo(EDumpTo _DumpTo, Bool_t _clearStamps, const char *_stampName)
{
   //write current status to file
   const Bool_t status = TestBit(TMemStatManager::kStatDisable);
   SetBit(TMemStatManager::kStatDisable, kTRUE);
   if (!fDumpFile.get())
      fDumpFile.reset( TFile::Open(g_cszFileName, "recreate") );
   //
   TTimeStamp stamp;
   MemInfo_t  memInfo;
   ProcInfo_t procInfo;
   gSystem->GetMemInfo(&memInfo);
   gSystem->GetProcInfo(&procInfo);
   Float_t memUsage[4] = { memInfo.fMemUsed, memInfo.fSwapUsed,
                           procInfo.fMemResident*0.001, procInfo.fMemVirtual*0.001};
   // No need to delete this pointer
   TTimeStamp *ptimeStamp(new TTimeStamp);
   // pass ownership to an auto_ptr
   auto_ptr<TTimeStamp> ptimeStamp_(ptimeStamp);

   TObjString *pnameStamp =
      (_stampName != 0) ? new TObjString(_stampName) : new TObjString(Form("autoStamp%d", fStampNumber));
   // pass ownership to an auto_ptr
   auto_ptr<TObjString> pnameStamp_(pnameStamp);

   const TMemStatManager * pmanager = this;
   Int_t stampNumber = fStampNumber;

   // pass ownership to an auto_ptr
   TMemStatInfoStamp *currentStamp(new TMemStatInfoStamp(fCurrentStamp));
   // pass ownership to an auto_ptr
   auto_ptr<TMemStatInfoStamp> currentStamp_(currentStamp);

   TTree *pDumpTo(NULL);
   bool bNewTree = false;
   switch (_DumpTo) {
   case kTree:
      if (!fDumpTree) {
         fDumpTree = new TTree("MemStat", "MemStat");
         bNewTree = true;
      }
      pDumpTo = fDumpTree;
      break;
   case kSysTree:
      if (!fDumpSysTree) {
         fDumpSysTree = new TTree("MemSys", "MemSys");
         bNewTree = true;
      }
      pDumpTo = fDumpSysTree;
      break;
   default: // TODO: Log me!
      return;
   }

   if (bNewTree) {
      if (kTree == _DumpTo)
         pDumpTo->Branch("Manager", "TMemStatManager", &pmanager);
      pDumpTo->Branch("StampTime.", "TTimeStamp", &ptimeStamp);
      pDumpTo->Branch("StampName.", "TObjString", &pnameStamp);
      pDumpTo->Branch("StampNumber", &stampNumber, "StampNumber/I");
      pDumpTo->Branch("CurrentStamp", "TMemStatInfoStamp", &currentStamp);
      pDumpTo->Branch("Mem0", &memUsage[0], "Mem0/F");
      pDumpTo->Branch("Mem1", &memUsage[1], "Mem1/F");
      pDumpTo->Branch("Mem2", &memUsage[2], "Mem2/F");
      pDumpTo->Branch("Mem3", &memUsage[3], "Mem3/F");
   } else {
      if (kTree == _DumpTo)
         pDumpTo->SetBranchAddress("Manager", &pmanager);
      pDumpTo->SetBranchAddress("StampTime.", &ptimeStamp);
      pDumpTo->SetBranchAddress("StampName.", &pnameStamp);
      pDumpTo->SetBranchAddress("StampNumber", &stampNumber);
      pDumpTo->SetBranchAddress("CurrentStamp", &currentStamp);
      pDumpTo->SetBranchAddress("Mem0", &memUsage[0]);
      pDumpTo->SetBranchAddress("Mem1", &memUsage[1]);
      pDumpTo->SetBranchAddress("Mem2", &memUsage[2]);
      pDumpTo->SetBranchAddress("Mem3", &memUsage[3]);
   }

   pDumpTo->Fill();
   pDumpTo->AutoSave("Stat");
   if (_clearStamps)
      fStampVector.clear();
   SetBit(TMemStatManager::kStatDisable, status);
}
