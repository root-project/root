// @(#)root/memstat:$Id$
// Author: Anar Manafov (A.Manafov@gsi.de) 2008-03-02

/*************************************************************************
* Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/
// ROOT
#include "TSystem.h"
#include "TEnv.h"
#include "TError.h"
#include "Riostream.h"
#include "TObject.h"
#include "TFile.h"
#include "TTree.h"
#include "TArrayL64.h"
#include "TH1.h"
// Memstat
#include "TMemStatBacktrace.h"
#include "TMemStatMng.h"
#include "TMemStatChecksum.h"

using namespace memstat;

ClassImp(TMemStatMng)

TMemStatMng* TMemStatMng::fgInstance = NULL;

//****************************************************************************//
//
//****************************************************************************//

TMemStatMng::TMemStatMng():
      TObject(),
      fPreviousMallocHook(TMemStatHook::GetMallocHook()),
      fPreviousFreeHook(TMemStatHook::GetFreeHook()),
      fDumpTree(NULL),
      fUseGNUBuiltinBacktrace(kFALSE),
      fBeginTime(0),
      fPos(0),
      fTimems(0),
      fNBytes(0),
      fN(0),
      fBtID(0),
      fBTCount(0)
{
   // Default constructor
}

//______________________________________________________________________________
void TMemStatMng::Init()
{
   //Initialize MemStat manager - used only for instance
   fBeginTime = fTimeStamp.AsDouble();

   //fDumpFile = new TFile(Form("yams_%d.root", gSystem->GetPid()), "recreate");
   fDumpFile = new TFile(g_cszFileName, "recreate");
   Int_t opt = 200000;
   if (!fDumpTree) {
      fDumpTree = new TTree("T", "Memory Statistics");
      fDumpTree->Branch("pos",   &fPos,   "pos/l", opt);
      fDumpTree->Branch("time",  &fTimems, "time/I", opt);
      fDumpTree->Branch("nbytes", &fNBytes, "nbytes/I", opt);
      fDumpTree->Branch("btid",  &fBtID,  "btid/I", opt);
   }

   fBTCount = 0;

   fBTIDCount = 0;

   fFAddrsList = new TObjArray();
   fFAddrsList->SetOwner(kTRUE);

   fHbtids  = new TH1I("btids", "table of btids", 10000, 0, 1);   //where fHbtids is a member of the manager class
   fHbtids->SetDirectory(0);
   fDumpTree->GetUserInfo()->Add(fHbtids);
}

//______________________________________________________________________________
TMemStatMng* TMemStatMng::GetInstance()
{
   // GetInstance - a static function
   // Only instance catch the alloc and free hook

   if (!fgInstance) {
      fgInstance = new TMemStatMng;
      fgInstance->Init();
   }
   return fgInstance;
}

//______________________________________________________________________________
void TMemStatMng::Close()
{
   // Close - a static function

   // TODO: This is a temporary solution until we find a properalgorithm for SaveData
   fgInstance->fDumpFile->WriteObject(fgInstance->fFAddrsList, "FAddrsList");

   // to be documented
   fgInstance->fDumpTree->AutoSave();

   delete fgInstance->fFAddrsList;

   delete fgInstance;
   fgInstance = NULL;
}

//______________________________________________________________________________
TMemStatMng::~TMemStatMng()
{
   //   Destructor
   //   if instance is destructed - the hooks are reseted to old hooks

   if (this != TMemStatMng::GetInstance())
      return;

   cout << ">>> All free/malloc calls count: " << fBTIDCount << endl;
   cout << ">>> Unique BTIDs count: " << fBTChecksums.size() << endl;

   Disable();
}

//______________________________________________________________________________
void TMemStatMng::Enable()
{
   // Enable hooks

   if (this != GetInstance())
      return;
#if defined(__APPLE__)
   TMemStatHook::trackZoneMalloc(MacAllocHook, MacFreeHook);
#else
   // set hook to our functions
   TMemStatHook::SetMallocHook(AllocHook);
   TMemStatHook::SetFreeHook(FreeHook);
#endif
}

//______________________________________________________________________________
void TMemStatMng::Disable()
{
   // disble MemStatManager

   if (this != GetInstance())
      return;
#if defined(__APPLE__)
   TMemStatHook::untrackZoneMalloc();
#else
   // set hook to our functions
   TMemStatHook::SetMallocHook(fPreviousMallocHook);
   TMemStatHook::SetFreeHook(fPreviousFreeHook);
#endif
}

//______________________________________________________________________________
void TMemStatMng::MacAllocHook(void *ptr, size_t _size)
{
   // AllocHook - a static function
   TMemStatMng* instance = TMemStatMng::GetInstance();
   // Restore all old hooks
   instance->Disable();

   // Call our routine
   instance->AddPointer(ptr, Int_t(_size));

   // Restore our own hooks
   instance->Enable();
}

//______________________________________________________________________________
void TMemStatMng::MacFreeHook(void *ptr)
{
   // AllocHook - a static function
   TMemStatMng* instance = TMemStatMng::GetInstance();
   // Restore all old hooks
   instance->Disable();

   // Call our routine
   instance->AddPointer(ptr, -1);

   // Restore our own hooks
   instance->Enable();
}

//______________________________________________________________________________
void *TMemStatMng::AllocHook(size_t size, const void* /*caller*/)
{
   // AllocHook - a static function
   TMemStatMng* instance = TMemStatMng::GetInstance();
   // Restore all old hooks
   instance->Disable();

   // Call recursively
   void *result = malloc(size);
   // Call our routine
   instance->AddPointer(result, Int_t(size));
   //  TTimer::SingleShot(0, "TYamsMemMng", instance, "SaveData()");

   // Restore our own hooks
   instance->Enable();

   return result;
}

//______________________________________________________________________________
void TMemStatMng::FreeHook(void* ptr, const void* /*caller*/)
{
   // FreeHook - a static function
   TMemStatMng* instance = TMemStatMng::GetInstance();
   // Restore all old hooks
   instance->Disable();

   // Call recursively
   free(ptr);

   // Call our routine
   instance->AddPointer(ptr, -1);

   // Restore our own hooks
   instance->Enable();
}

//______________________________________________________________________________
void TMemStatMng::AddPointer(void *ptr, Int_t size)
{
   // Add pointer to table.

   void *stptr[g_BTStackLevel + 1];
   const int stackentries = getBacktrace(stptr, g_BTStackLevel, fUseGNUBuiltinBacktrace);

   // save only unique BTs
   const unsigned long crc = CCRC::CalculateDigest(reinterpret_cast<void*>(stptr), sizeof(void*) * stackentries);

   ++fBTIDCount;

   CRCSet_t::const_iterator found = fBTChecksums.find(crc);
   // TODO: define a proper default value
   Int_t btid = -1;
   if (fBTChecksums.end() == found) {

      // check the size of the BT array container
      int nbins = fHbtids->GetNbinsX();
      //check that the current allocation in fHbtids is enough, otherwise expend it with
      if (fBTCount + stackentries + 1 >= nbins) {
         fHbtids->SetBins(nbins*2, 0, 1);
      }

      int *btids = fHbtids->GetArray();
      // A first value is a number of entries in a given stack
      btids[fBTCount++] = stackentries;
      btid = fBTCount;

      // add new BT's CRC value
      pair<CRCSet_t::iterator, bool> res = fBTChecksums.insert(CRCSet_t::value_type(crc, btid));
      if (!res.second)
         Error("AddPointer", "Can't added new BTID to the container.");

      for (int i = 0; i < stackentries; ++i) {
         pointer_t func_addr = reinterpret_cast<pointer_t>(stptr[i]);


         // save all functions of this BT
         if (fFAddrs.find(func_addr) < 0) {
            TString strFuncAddr;
            strFuncAddr += func_addr;
            TString strSymbolInfo;
            getSymbolFullInfo(stptr[i], &strSymbolInfo);
            TNamed *nm = new TNamed(strFuncAddr, strSymbolInfo);
            fFAddrsList->Add(nm);
            fFAddrs.add(func_addr, fFAddrsList->GetSize() - 1);
         }

         // add BT to the container
         Int_t idx = fFAddrs.find(reinterpret_cast<pointer_t>(stptr[i]));
         //TODO: in the error code prtint the address.
         // Acutally there must not be a case, when we can't find an index
         if (idx < 0)
            Error("AddPointer", "There is no index for a given BT function return address.");
         // even if we have -1 as an index we add it to the container
         btids[fBTCount++] = idx;
      }

   } else {
      // reuse existing BT
      btid = found->second;
   }

   if (btid < 0)
      Error("AddPointer", "negative BT id");

   fTimeStamp.Set();
   Double_t CurTime = fTimeStamp.AsDouble();
   fTimems = Int_t(10000.*(CurTime - fBeginTime));
   fPos    =  reinterpret_cast<pointer_t>(ptr);
   fNBytes = size;
   fN      = 0;
   fBtID   = btid;
   fDumpTree->Fill();
   //SaveToTree();
}
