// @(#)root/memstat:$Name$:$Id$
// Author: D.Bertini and M.Ivanov   18/06/2007 -- Anar Manafov (A.Manafov@gsi.de) 28/04/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMemStatManager
#define ROOT_TMemStatManager

//****************************************************************************//
//
//  TMemStatManager
//  Memory statistic manager class
//
//****************************************************************************//
// STD
#include <map>
#include <vector>
#include <memory>
#include <cstdlib>
// ROOT
#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TTimeStamp
#include "TTimeStamp.h"
#endif
// Memstat
#ifndef ROOT_TMemStatDepend
#include "TMemStatDepend.h"
#endif
#ifndef ROOT_TmemStatInfo
#include "TMemStatInfo.h"
#endif


class TTree;
class TMemStatStackInfo;

typedef std::vector<Int_t> IntVector_t;
typedef std::auto_ptr<TFile> TFilePtr_t;

class TMemStatManager: public TObject
{
   struct TMemInfo_t {
      void   *fAddress;    //mem address
      size_t  fSize;       //size of the allocated memory
      Int_t   fStackIndex; //index of the stack info
   };

   struct TMemTable_t {
      Int_t     fAllocCount;    //number of memory allocation blocks
      Int_t     fMemSize;       //total memory allocated size
      Int_t     fTableSize;     //amount of entries in the below array
      Int_t     fFirstFreeSpot; //where is the first free spot in the leaks array?
      TMemInfo_t *fLeaks;         //leak table
   };

   struct TDeleteTable_t {
      Int_t     fAllocCount;    //how many memory blocks do we have
      Int_t     fTableSize;     //amount of entries in the below array
      TMemInfo_t *fLeaks;         //leak table
   };

public:
   typedef std::vector<TMemStatCodeInfo> CodeInfoContainer_t;

   enum EStatusBits {
      kUserDisable = BIT(18),       // user disable-enable switch  switch
      kStatDisable = BIT(16),       // true if disable statistic
      kStatRoutine = BIT(17)        // indicator inside of stat routine  (AddPointer or FreePointer)
   };
   enum EDumpTo { kTree, kSysTree };

   TMemStatManager();
   virtual ~TMemStatManager();

   void Enable();                              //enable memory statistic
   void Disable();                             //Disable memory statistic
   void SetAutoStamp(UInt_t sizeMem, UInt_t n, UInt_t max) {
      fAutoStampSize = sizeMem;
      fAutoStampN = n;
      fAutoStampDumpSize = max;
   }
   void AddStamps(const char * stampname = 0);           //add  stamps to the list of stamps for changed stacks
   static void SAddStamps(const Char_t * stampname);             // static version add  stamps to the list of stamps for changed stacks

   static TMemStatManager* GetInstance();       //get instance of class - ONLY ONE INSTANCE
   static void Close();                         //close MemStatManager
   TMemStatInfoStamp &AddStamp();                   //add one stamp to the list of stamps
   TMemStatCodeInfo &GetCodeInfo(void *address);
   UInt_t GetCodeInfoIndex(void *address) {
      return fCodeInfoMap[address];
   }
   void DumpTo(EDumpTo _DumpTo, Bool_t _clearStamps = kTRUE, const char * _stampName = 0);  //write current status to file

public:
   typedef void (*StampCallback_t)(const Char_t * desription);
   //stack data members
   IntVector_t fSTHashTable; //!pointer to the hash table
   Int_t fCount;        //!number of entries in table
   Int_t fStampNumber;  //current stamp number
   std::vector<TMemStatStackInfo> fStackVector;            // vector with stack symbols
   std::vector<TMemStatInfoStamp> fStampVector;            // vector of stamp information
   std::vector<TTimeStamp> fStampTime;              // vector of stamp information
   CodeInfoContainer_t  fCodeInfoArray;          // vector with code info
   std::map<const void*, UInt_t> fCodeInfoMap;      //! map of code information
   Int_t fDebugLevel;                               //!debug level
   TMemStatManager::StampCallback_t fStampCallBack; //!call back function
   void SetUseGNUBuildinBacktrace(Bool_t _NewVal) {
      fUseGNUBuildinBacktrace = _NewVal;
   }

protected:
   TMemStatDepend::MallocHookFunc_t fPreviousMallocHook;    //!old malloc function
   TMemStatDepend::FreeHookFunc_t fPreviousFreeHook;        //!old free function
   void Init();
   TMemStatStackInfo *STAddInfo(Int_t size, void **stackptrs);
   TMemStatStackInfo *STFindInfo(Int_t size, void **stackptrs);
   void RehashLeak(Int_t newSize);                  //rehash leak pointers
   void *AddPointer(size_t size, void *ptr = 0);    //add pointer to the table
   void FreePointer(void *p);                       //free pointer
   static void *AllocHook(size_t size, const void* /*caller*/);
   static void FreeHook(void* ptr, const void* /*caller*/);
   TMemStatInfoStamp fLastStamp;           //last written stamp
   TMemStatInfoStamp fCurrentStamp;        //current stamp
   UInt_t fAutoStampSize;           //change of size invoking STAMP
   UInt_t fAutoStampN;              //change of number of allocation  STAMP
   UInt_t fAutoStampDumpSize;       //
   Int_t fMinStampSize;             // the minimal size to be dumped to tree
   //  memory information
   Int_t fSize;                     //!size of hash table
   TMemTable_t **fLeak;               //!pointer to the hash table
   Int_t fAllocCount;               //!number of memory allocation blocks
   TDeleteTable_t fMultDeleteTable;   //!pointer to the table
   TFilePtr_t fDumpFile;              //!file to dump current information
   TTree *fDumpTree;                //!tree to dump information
   TTree *fDumpSysTree;             //!tree to dump information
   static TMemStatManager *fgInstance; // pointer to instance
   static void *fgStackTop;             // stack top pointer

   void FreeHashtable() {
      if (!fLeak)
         return;

      for (Int_t i = 0; i < fSize; ++i)
         free(fLeak[i]);
      free(fLeak);
   }

   Bool_t fUseGNUBuildinBacktrace;

   ClassDef(TMemStatManager, 1) // a manager of memstat sessions.
};

#endif
