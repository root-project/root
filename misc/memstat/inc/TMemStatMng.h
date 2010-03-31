// @(#)root/memstat:$Id$
// Author: Anar Manafov (A.Manafov@gsi.de) 2008-03-02

/*************************************************************************
* Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/
#ifndef ROOT_TMemStatMng
#define ROOT_TMemStatMng

// STD
#include <map>
// ROOT
#include "TTimeStamp.h"
// Memstat
#include "TMemStatHook.h"
#include "TMemStatDef.h"


class TTree;
class TFile;
class TH1I;
class TObjArray;

namespace memstat {

   class TMemStatFAddrContainer {
      typedef std::map<pointer_t, Int_t> Container_t;
      typedef Container_t::iterator pos_type;
      typedef Container_t::value_type value_type;

   public:
      bool add(pointer_t addr, Int_t idx) {
         std::pair<pos_type, bool> ret = fContainer.insert(value_type(addr, idx));
         return (ret.second);
      }

      Int_t find(pointer_t addr) {
         pos_type iter = fContainer.find(addr);
         if (fContainer.end() == iter)
            return -1;

         return iter->second;
      }

   private:
      Container_t fContainer;
   };

   class TMemStatMng: public TObject {
      typedef std::map<unsigned long, Int_t> CRCSet_t;

   private:
      TMemStatMng();
      virtual ~TMemStatMng();

   public:
      void Enable();                       //enable memory statistic
      void Disable();                      //Disable memory statistic
      static TMemStatMng* GetInstance();   //get instance of class - ONLY ONE INSTANCE
      static void Close();                 //close MemStatManager

   public:
      //stack data members
      void SetUseGNUBuiltinBacktrace(Bool_t newVal) {
         fUseGNUBuiltinBacktrace = newVal;
      }

   protected:
#if !defined(__APPLE__)
      TMemStatHook::MallocHookFunc_t fPreviousMallocHook;    //!old malloc function
      TMemStatHook::FreeHookFunc_t fPreviousFreeHook;        //!old free function
#endif
      void Init();
      void AddPointer(void *ptr, Int_t size);    //add pointer to the table
      static void *AllocHook(size_t size, const void* /*caller*/);
      static void FreeHook(void* ptr, const void* /*caller*/);
      static void MacAllocHook(void *ptr, size_t size);
      static void MacFreeHook(void *ptr);


      //  memory information
      TFile* fDumpFile;               //!file to dump current information
      TTree *fDumpTree;               //!tree to dump information
      static TMemStatMng *fgInstance; // pointer to instance
      static void *fgStackTop;        // stack top pointer

      Bool_t fUseGNUBuiltinBacktrace;
      TTimeStamp fTimeStamp;
      Double_t fBeginTime;    //time when monitoring starts
      pointer_t fPos;         //position in memory where alloc/free happens
      Int_t    fTimems;       //10000*(current time - begin time)
      Int_t    fNBytes;       //number of bytes allocated/freed
      UInt_t   fN;
      Int_t    fBtID;         //back trace identifier

   private:
      TMemStatFAddrContainer fFAddrs;
      TObjArray *fFAddrsList;
      TH1I *fHbtids;
      CRCSet_t fBTChecksums;
      Int_t fBTCount;
      UInt_t  fBTIDCount;

      ClassDef(TMemStatMng, 0)   // a manager of memstat sessions.
   };

}

#endif
