// @(#)root/proofplayer:$Id$
// Author: Jan Iwaszkiewicz   11/12/06

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPacketizerAdaptive                                                  //
//                                                                      //
// This packetizer is based on TPacketizer but uses different           //
// load-balancing algorithms and data structures.                       //
// Two main improvements in the load-balancing strategy:                //
// - First one was to change the order in which the files are assigned  //
//   to the computing nodes in such a way that network transfers are    //
//   evenly distributed in the query time. Transfer of the remote files //
//   was often becoming a bottleneck at the end of a query.             //
// - The other improvement is the use of time-based packet size. We     //
//   measure the processing rate of all the nodes and calculate the     //
//   packet size, so that it takes certain amount of time. In this way  //
//   packetizer prevents the situation where the query can't finish     //
//   because of one slow node.                                          //
//                                                                      //
// The data structures: TFileStat, TFileNode and TSlaveStat are         //
// enriched + changed and TFileNode::Compare method is changed.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TPacketizerAdaptive.h"

#include "Riostream.h"
#include "TDSet.h"
#include "TError.h"
#include "TEnv.h"
#include "TEntryList.h"
#include "TEventList.h"
#include "TMap.h"
#include "TMessage.h"
#include "TMonitor.h"
#include "TNtupleD.h"
#include "TObject.h"
#include "TParameter.h"
#include "TPerfStats.h"
#include "TProofDebug.h"
#include "TProof.h"
#include "TProofServ.h"
#include "TSlave.h"
#include "TSocket.h"
#include "TSortedList.h"
#include "TUrl.h"
#include "TClass.h"
#include "TRandom.h"
#include "TMath.h"
#include "TObjString.h"
#include "TList.h"

//
// The following three utility classes manage the state of the
// work to be performed and the slaves involved in the process.
// A list of TFileNode(s) describes the hosts with files, each
// has a list of TFileStat(s) keeping the state for each TDSet
// element (file).
//
// The list of TSlaveStat(s) keep track of the work (being) done
// by each slave
//


//------------------------------------------------------------------------------

class TPacketizerAdaptive::TFileStat : public TObject {

private:
   Bool_t         fIsDone;       // is this element processed
   TFileNode     *fNode;         // my FileNode
   TDSetElement  *fElement;      // location of the file and its range
   Long64_t       fNextEntry;    // cursor in the range, -1 when done // needs changing

public:
   TFileStat(TFileNode *node, TDSetElement *elem, TList *file);

   Bool_t         IsDone() const {return fIsDone;}
   Bool_t         IsSortable() const { return kTRUE; }
   void           SetDone() {fIsDone = kTRUE;}
   TFileNode     *GetNode() const {return fNode;}
   TDSetElement  *GetElement() const {return fElement;}
   Long64_t       GetNextEntry() const {return fNextEntry;}
   void           MoveNextEntry(Long64_t step) {fNextEntry += step;}

   // This method is used to keep a sorted list of remaining files to be processed
   Int_t          Compare(const TObject* obj) const
   {
      // Return -1 if elem.entries < obj.elem.entries, 0 if elem.entries equal
      // and 1 if elem.entries < obj.elem.entries.
      const TFileStat *fst = dynamic_cast<const TFileStat*>(obj);
      if (fst && GetElement() && fst->GetElement()) {
         Long64_t ent = GetElement()->GetNum();
         Long64_t entfst = fst->GetElement()->GetNum();
         if (ent > 0 && entfst > 0) {
            if (ent > entfst) {
               return 1;
            } else if (ent < entfst) {
               return -1;
            } else {
               return 0;
            }
         }
      }
      // No info: assume equal (no change in order)
      return 0;
   }
   void Print(Option_t * = 0) const
   {  // Notify file name and entries
      Printf("TFileStat: %s %lld", fElement ? fElement->GetName() : "---",
                                   fElement ? fElement->GetNum() : -1);
   }
};

TPacketizerAdaptive::TFileStat::TFileStat(TFileNode *node, TDSetElement *elem, TList *files)
   : fIsDone(kFALSE), fNode(node), fElement(elem), fNextEntry(elem->GetFirst())
{
   // Constructor: add to the global list
   if (files) files->Add(this);
}

//------------------------------------------------------------------------------

// a class describing a file node as a part of a session
class TPacketizerAdaptive::TFileNode : public TObject {

private:
   TString        fNodeName;        // FQDN of the node
   TList         *fFiles;           // TDSetElements (files) stored on this node
   TObject       *fUnAllocFileNext; // cursor in fFiles
   TList         *fActFiles;        // files with work remaining
   TObject       *fActFileNext;     // cursor in fActFiles
   Int_t          fMySlaveCnt;      // number of slaves running on this node
                                    // (which can process remote files)
   Int_t          fExtSlaveCnt;     // number of external slaves processing
                                    // files on this node
   Int_t          fRunSlaveCnt;     // total number of slaves processing files
                                    // on this node
   Long64_t       fProcessed;       // number of events processed on this node
   Long64_t       fEvents;          // number of entries in files on this node

   Int_t          fStrategy;        // 0 means the classic and 1 (default) - the adaptive strategy

   TSortedList   *fFilesToProcess;  // Global list of files (TFileStat) to be processed (owned by TPacketizer)

public:
   TFileNode(const char *name, Int_t strategy, TSortedList *files);
   ~TFileNode() { delete fFiles; delete fActFiles; }

   void        IncMySlaveCnt() { fMySlaveCnt++; }
   Int_t       GetMySlaveCnt() const { return fMySlaveCnt; }
   void        IncExtSlaveCnt(const char *slave) { if (fNodeName != slave) fExtSlaveCnt++; }
   void        DecExtSlaveCnt(const char *slave) { if (fNodeName != slave) fExtSlaveCnt--; R__ASSERT(fExtSlaveCnt >= 0); }
   Int_t       GetSlaveCnt() const { return fMySlaveCnt + fExtSlaveCnt; }
   void        IncRunSlaveCnt() { fRunSlaveCnt++; }
   void        DecRunSlaveCnt() { fRunSlaveCnt--; R__ASSERT(fRunSlaveCnt >= 0); }
   Int_t       GetRunSlaveCnt() const { return fRunSlaveCnt; }
   Int_t       GetExtSlaveCnt() const { return fExtSlaveCnt; }
   Int_t       GetNumberOfActiveFiles() const { return fActFiles->GetSize(); }
   Bool_t      IsSortable() const { return kTRUE; }
   Int_t       GetNumberOfFiles() { return fFiles->GetSize(); }
   void        IncProcessed(Long64_t nEvents)
                  { fProcessed += nEvents; }
   Long64_t    GetProcessed() const { return fProcessed; }
   void        DecreaseProcessed(Long64_t nEvents) { fProcessed -= nEvents; }
   // this method is used by Compare() it adds 1, so it returns a number that
   // would be true if one more slave is added.
   Long64_t    GetEventsLeftPerSlave() const
      { return ((fEvents - fProcessed)/(fRunSlaveCnt + 1)); }
   void        IncEvents(Long64_t nEvents) { fEvents += nEvents; }
   const char *GetName() const { return fNodeName.Data(); }
   Long64_t    GetNEvents() const { return fEvents; }

   void Print(Option_t * = 0) const
   {
      TFileStat *fs = 0;
      TDSetElement *e = 0;
      Int_t nn = 0;
      Printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
      Printf("+++ TFileNode: %s +++", fNodeName.Data());
      Printf("+++ Evts: %lld (total: %lld) ", fProcessed, fEvents);
      Printf("+++ Worker count: int:%d, ext: %d, tot:%d ", fMySlaveCnt, fExtSlaveCnt, fRunSlaveCnt);
      Printf("+++ Files: %d ", fFiles ? fFiles->GetSize() : 0);
      if (fFiles && fFiles->GetSize() > 0) {
         TIter nxf(fFiles);
         while ((fs = (TFileStat *) nxf())) {
            if ((e = fs->GetElement())) {
               Printf("+++  #%d: %s  %lld - %lld (%lld) - next: %lld ", ++nn, e->GetName(),
                     e->GetFirst(), e->GetFirst() + e->GetNum() - 1, e->GetNum(), fs->GetNextEntry());
            } else {
               Printf("+++  #%d: no element! ", ++nn);
            }
         }
      }
      Printf("+++ Active files: %d ", fActFiles ? fActFiles->GetSize() : 0);
      if (fActFiles && fActFiles->GetSize() > 0) {
         TIter nxaf(fActFiles);
         while ((fs = (TFileStat *) nxaf())) {
            if ((e = fs->GetElement())) {
               Printf("+++  #%d: %s  %lld - %lld (%lld) - next: %lld", ++nn, e->GetName(),
                      e->GetFirst(), e->GetFirst() + e->GetNum() - 1, e->GetNum(), fs->GetNextEntry());
            } else {
               Printf("+++  #%d: no element! ", ++nn);
            }
         }
      }
      Printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
   }

   void Add(TDSetElement *elem, Bool_t tolist)
   {
      TList *files = tolist ? (TList *)fFilesToProcess : (TList *)0;
      TFileStat *f = new TFileStat(this, elem, files);
      fFiles->Add(f);
      if (fUnAllocFileNext == 0) fUnAllocFileNext = fFiles->First();
   }

   TFileStat *GetNextUnAlloc()
   {
      TObject *next = fUnAllocFileNext;

      if (next != 0) {
         // make file active
         fActFiles->Add(next);
         if (fActFileNext == 0) fActFileNext = fActFiles->First();

         // move cursor
         fUnAllocFileNext = fFiles->After(fUnAllocFileNext);
      }
      return (TFileStat *) next;
   }

   TFileStat *GetNextActive()
   {
      TObject *next = fActFileNext;

      if (fActFileNext != 0) {
         fActFileNext = fActFiles->After(fActFileNext);
         if (fActFileNext == 0) fActFileNext = fActFiles->First();
      }

      return (TFileStat *) next;
   }

   void RemoveActive(TFileStat *file)
   {
      if (fActFileNext == file) fActFileNext = fActFiles->After(file);
      fActFiles->Remove(file);
      if (fFilesToProcess) fFilesToProcess->Remove(file);
      if (fActFileNext == 0) fActFileNext = fActFiles->First();
   }

   Int_t Compare(const TObject *other) const
   {
      // Must return -1 if this is smaller than obj, 0 if objects are equal
      // and 1 if this is larger than obj.
      // smaller means more needing a new worker.
      // Two cases are considered depending on
      // relation between harddrive speed and network bandwidth.

      const TFileNode *obj = dynamic_cast<const TFileNode*>(other);
      if (!obj) {
         Error("Compare", "input is not a TPacketizer::TFileNode object");
         return 0;
      }
      
      // how many more events it has than obj

      if (fStrategy == 1) {
         // The default adaptive strategy.
         Int_t myVal = GetRunSlaveCnt();
         Int_t otherVal = obj->GetRunSlaveCnt();
         if (myVal < otherVal) {
            return -1;
         } else if (myVal > otherVal) {
            return 1;
         } else {
            // if this has more events to process than obj
            if ((fEvents - fProcessed) >
                (obj->GetNEvents() - obj->GetProcessed())) {
               return -1;
            } else {
               return 1;
            }
         }
      } else {
         Int_t myVal = GetSlaveCnt();
         Int_t otherVal = obj->GetSlaveCnt();
         if (myVal < otherVal) {
            return -1;
         } else if (myVal > otherVal) {
            return 1;
         } else {
            return 0;
         }
      }
   }

   void Reset()
   {
      fUnAllocFileNext = fFiles->First();
      fActFiles->Clear();
      fActFileNext = 0;
      fExtSlaveCnt = 0;
      fMySlaveCnt = 0;
      fRunSlaveCnt = 0;
   }
};


TPacketizerAdaptive::TFileNode::TFileNode(const char *name, Int_t strategy, TSortedList *files)
   : fNodeName(name), fFiles(new TList), fUnAllocFileNext(0),
     fActFiles(new TList), fActFileNext(0), fMySlaveCnt(0),
     fExtSlaveCnt(0), fRunSlaveCnt(0), fProcessed(0), fEvents(0),
     fStrategy(strategy), fFilesToProcess(files)
{
   // Constructor

   fFiles->SetOwner();
   fActFiles->SetOwner(kFALSE);
}

//------------------------------------------------------------------------------

class TPacketizerAdaptive::TSlaveStat : public TVirtualPacketizer::TVirtualSlaveStat {

friend class TPacketizerAdaptive;

private:
   TFileNode     *fFileNode;     // corresponding node or 0
   TFileStat     *fCurFile;      // file currently being processed
   TDSetElement  *fCurElem;      // TDSetElement currently being processed
   Long64_t       fCurProcessed; // events processed in the current file
   Float_t        fCurProcTime;  // proc time spent on the current file
   TList         *fDSubSet;      // packets processed by this worker

public:
   TSlaveStat(TSlave *slave);
   ~TSlaveStat();
   TFileNode  *GetFileNode() const { return fFileNode; }
   Long64_t    GetEntriesProcessed() const { return fStatus?fStatus->GetEntries():-1; }
   Double_t    GetProcTime() const { return fStatus?fStatus->GetProcTime():-1; }
   TFileStat  *GetCurFile() { return fCurFile; }
   void        SetFileNode(TFileNode *node) { fFileNode = node; }
   void        UpdateRates(TProofProgressStatus *st);
   Float_t     GetAvgRate() { return fStatus->GetRate(); }
   Float_t     GetCurRate() {
      return (fCurProcTime?fCurProcessed/fCurProcTime:0); }
   Int_t       GetLocalEventsLeft() {
      return fFileNode?(fFileNode->GetEventsLeftPerSlave()):0; }
   TList      *GetProcessedSubSet() { return fDSubSet; }
   TProofProgressStatus *GetProgressStatus() { return fStatus; }
   TProofProgressStatus *AddProcessed(TProofProgressStatus *st = 0);
};

//______________________________________________________________________________
TPacketizerAdaptive::TSlaveStat::TSlaveStat(TSlave *slave)
   : fFileNode(0), fCurFile(0), fCurElem(0),
     fCurProcessed(0), fCurProcTime(0)
{
   // Constructor

   fDSubSet = new TList();
   fDSubSet->SetOwner();
   fSlave = slave;
   fStatus = new TProofProgressStatus();
   // The slave name is a special one in PROOF-Lite: avoid blocking on the DNS
   // for non existing names
   fWrkFQDN = slave->GetName();
   if (strcmp(slave->ClassName(), "TSlaveLite")) {
      fWrkFQDN = TUrl(fWrkFQDN).GetHostFQDN();
      // Get full name for local hosts
      if (fWrkFQDN.Contains("localhost") || fWrkFQDN == "127.0.0.1")
         fWrkFQDN = TUrl(gSystem->HostName()).GetHostFQDN();
   }
   PDB(kPacketizer, 2)
      Info("TSlaveStat", "wrk FQDN: %s", fWrkFQDN.Data());
}

//______________________________________________________________________________
TPacketizerAdaptive::TSlaveStat::~TSlaveStat()
{
   // Cleanup

   SafeDelete(fDSubSet);
   SafeDelete(fStatus);
}

//______________________________________________________________________________
void TPacketizerAdaptive::TSlaveStat::UpdateRates(TProofProgressStatus *st)
{
   // Update packetizer rates

   if (!st) {
      Error("UpdateRates", "no status object!");
      return;
   }
   if (fCurFile->IsDone()) {
      fCurProcTime = 0;
      fCurProcessed = 0;
   } else {
      fCurProcTime += st->GetProcTime() - GetProcTime();
      fCurProcessed += st->GetEntries() - GetEntriesProcessed();
   }
   fCurFile->GetNode()->IncProcessed(st->GetEntries() - GetEntriesProcessed());
   st->SetLastEntries(st->GetEntries() - fStatus->GetEntries());
   SafeDelete(fStatus);
   fStatus = st;
}

//______________________________________________________________________________
TProofProgressStatus *TPacketizerAdaptive::TSlaveStat::AddProcessed(TProofProgressStatus *st)
{
   // Add the current element to the fDSubSet (subset processed by this worker)
   // and if the status arg is given, then change the size of the packet.
   // return the difference (*st - *fStatus)

   if (st && fDSubSet && fCurElem) {
      if (fCurElem->GetNum() != st->GetEntries() - GetEntriesProcessed())
         fCurElem->SetNum(st->GetEntries() - GetEntriesProcessed());
      fDSubSet->Add(fCurElem);
      TProofProgressStatus *diff = new TProofProgressStatus(*st - *fStatus);
      return diff;
   } else {
      Error("AddProcessed", "processed subset of current elem undefined");
      return 0;
   }
}

//------------------------------------------------------------------------------

ClassImp(TPacketizerAdaptive)

//______________________________________________________________________________
TPacketizerAdaptive::TPacketizerAdaptive(TDSet *dset, TList *slaves,
                                         Long64_t first, Long64_t num,
                                         TList *input, TProofProgressStatus *st)
                    : TVirtualPacketizer(input, st)
{
   // Constructor

   PDB(kPacketizer,1) Info("TPacketizerAdaptive",
                           "enter (first %lld, num %lld)", first, num);

   // Init pointer members
   fSlaveStats = 0;
   fUnAllocated = 0;
   fActive = 0;
   fFileNodes = 0;
   fMaxPerfIdx = 1;
   fCachePacketSync = kTRUE;
   fMaxEntriesRatio = 2.;

   fMaxSlaveCnt = -1;
   fPacketAsAFraction = 4;
   fStrategy = 1;
   fFilesToProcess = new TSortedList;
   fFilesToProcess->SetOwner(kFALSE);

   if (!fProgressStatus) {
      Error("TPacketizerAdaptive", "No progress status");
      return;
   }

   // Attempt to synchronize the packet size with the tree cache size
   Int_t cpsync = -1;
   if (TProof::GetParameter(input, "PROOF_PacketizerCachePacketSync", cpsync) != 0) {
      // Check if there is a global cache-packet sync setting
      cpsync = gEnv->GetValue("Packetizer.CachePacketSync", 1);
   }
   if (cpsync >= 0) fCachePacketSync = (cpsync > 0) ? kTRUE : kFALSE;

   // Max file entries to avg allowed ratio for cache-to-packet synchronization
   // (applies only if fCachePacketSync is true; -1. disables the bound)
   if (TProof::GetParameter(input, "PROOF_PacketizerMaxEntriesRatio", fMaxEntriesRatio) != 0) {
      // Check if there is a global ratio setting
      fMaxEntriesRatio = gEnv->GetValue("Packetizer.MaxEntriesRatio", 2.);
   }

   // The possibility to change packetizer strategy to the basic TPacketizer's
   // one (in which workers always process their local data first).
   Int_t strategy = -1;
   if (TProof::GetParameter(input, "PROOF_PacketizerStrategy", strategy) != 0) {
      // Check if there is a global strategy setting
      strategy = gEnv->GetValue("Packetizer.Strategy", 1);
   }
   if (strategy == 0) {
      fStrategy = 0;
      Info("TPacketizerAdaptive", "using the basic strategy of TPacketizer");
   } else if (strategy != 1) {
      Warning("TPacketizerAdaptive", "unsupported strategy index (%d): ignore", strategy);
   }

   Long_t maxSlaveCnt = 0;
   if (TProof::GetParameter(input, "PROOF_MaxSlavesPerNode", maxSlaveCnt) == 0) {
      if (maxSlaveCnt < 1) {
         Info("TPacketizerAdaptive",
              "The value of PROOF_MaxSlavesPerNode must be grater than 0");
         maxSlaveCnt = 0;
      }
   } else {
      // Try also with Int_t (recently supported in TProof::SetParameter)
      Int_t mxslcnt = -1;
      if (TProof::GetParameter(input, "PROOF_MaxSlavesPerNode", mxslcnt) == 0) {
         if (mxslcnt < 1) {
            Info("TPacketizerAdaptive",
                 "The value of PROOF_MaxSlavesPerNode must be grater than 0");
            mxslcnt = 0;
         }
         maxSlaveCnt = (Long_t) mxslcnt;
      }
   }

   if (!maxSlaveCnt)
      maxSlaveCnt = gEnv->GetValue("Packetizer.MaxWorkersPerNode", 0);
   if (maxSlaveCnt > 0) {
      fMaxSlaveCnt = maxSlaveCnt;
      Info("TPacketizerAdaptive", "Setting max number of workers per node to %ld",
           fMaxSlaveCnt);
   }

   // if forceLocal parameter is set to 1 then eliminate the cross-worker
   // processing;
   // This minimizes the network usage on the PROOF cluser at the expense of
   // longer jobs processing times.
   // To process successfully the session must have workers with all the data!
   fForceLocal = kFALSE;
   Int_t forceLocal = 0;
   if (TProof::GetParameter(input, "PROOF_ForceLocal", forceLocal) == 0) {
      if (forceLocal == 1)
         fForceLocal = kTRUE;
      else
         Info("TPacketizerAdaptive",
            "The only accepted value of PROOF_ForceLocal parameter is 1 !");
   }

   // Below we provide a possibility to change the way packet size is
   // calculated or define the packet time directly.
   // fPacketAsAFraction can be interpreted as follows:
   // packet time is (expected job proc. time) / fPacketSizeAsAFraction.
   // It substitutes 20 in the old formula to calculate the fPacketSize:
   // fPacketSize = fTotalEntries / (20 * nslaves)
   Int_t packetAsAFraction = 0;
   if (TProof::GetParameter(input, "PROOF_PacketAsAFraction", packetAsAFraction) == 0) {
      if (packetAsAFraction > 0) {
         fPacketAsAFraction = packetAsAFraction;
         Info("TPacketizerAdaptive",
              "using alternate fraction of query time as a packet size: %d",
              packetAsAFraction);
      } else
         Info("TPacketizerAdaptive", "packetAsAFraction parameter must be higher than 0");
   }

   // Packet re-assignement
   fTryReassign = 0;
   Int_t tryReassign = 0;
   if (TProof::GetParameter(input, "PROOF_TryReassign", tryReassign) != 0)
      tryReassign = gEnv->GetValue("Packetizer.TryReassign", 0);
   fTryReassign = tryReassign;
   if (fTryReassign != 0)
      Info("TPacketizerAdaptive", "failed packets will be re-assigned");

   // Save the config parameters in the dedicated list so that they will be saved
   // in the outputlist and therefore in the relevant TQueryResult
   fConfigParams->Add(new TParameter<Int_t>("PROOF_PacketizerCachePacketSync", (Int_t)fCachePacketSync));
   fConfigParams->Add(new TParameter<Double_t>("PROOF_PacketizerMaxEntriesRatio", fMaxEntriesRatio));
   fConfigParams->Add(new TParameter<Int_t>("PROOF_PacketizerStrategy", fStrategy));
   fConfigParams->Add(new TParameter<Int_t>("PROOF_MaxWorkersPerNode", (Int_t)fMaxSlaveCnt));
   fConfigParams->Add(new TParameter<Int_t>("PROOF_ForceLocal", (Int_t)fForceLocal));
   fConfigParams->Add(new TParameter<Int_t>("PROOF_PacketAsAFraction", fPacketAsAFraction));

   Double_t baseLocalPreference = 1.2;
   fBaseLocalPreference = (Float_t)baseLocalPreference;
   if (TProof::GetParameter(input, "PROOF_BaseLocalPreference", baseLocalPreference) == 0)
      fBaseLocalPreference = (Float_t)baseLocalPreference;

   fFileNodes = new TList;
   fFileNodes->SetOwner();
   fUnAllocated = new TList;
   fUnAllocated->SetOwner(kFALSE);
   fActive = new TList;
   fActive->SetOwner(kFALSE);

   fValid = kTRUE;

   // Resolve end-point urls to optmize distribution
   // dset->Lookup(); // moved to TProofPlayerRemote::Process

   // Read list of mounted disks
   TObjArray *partitions = 0;
   TString partitionsStr;
   if (TProof::GetParameter(input, "PROOF_PacketizerPartitions", partitionsStr) != 0)
      partitionsStr = gEnv->GetValue("Packetizer.Partitions", "");
   if (!partitionsStr.IsNull()) {
      Info("TPacketizerAdaptive", "Partitions: %s", partitionsStr.Data());
      partitions = partitionsStr.Tokenize(",");
   }

   // Split into per host and disk entries
   dset->Reset();
   TDSetElement *e;
   while ((e = (TDSetElement*)dset->Next())) {

      if (e->GetValid()) continue;

      // The dataset name, if any
      if (fDataSet.IsNull() && e->GetDataSet() && strlen(e->GetDataSet()))
         fDataSet = e->GetDataSet();

      TUrl url = e->GetFileName();
      PDB(kPacketizer,2)
         Info("TPacketizerAdaptive", "element name: %s (url: %s)", e->GetFileName(), url.GetUrl());

      // Map non URL filenames to dummy host
      TString host;
      if ( !url.IsValid() ||
          (strncmp(url.GetProtocol(),"root", 4) &&
           strncmp(url.GetProtocol(),"rfio", 4) &&
           strncmp(url.GetProtocol(),"file", 4)) ) {
         host = "no-host";
      } else if ( url.IsValid() && !strncmp(url.GetProtocol(),"file", 4)) {
         host = "localhost";
         url.SetProtocol("root");
      } else {
         host = url.GetHostFQDN();
      }
      // Get full name for local hosts
      if (host.Contains("localhost") || host == "127.0.0.1") {
         url.SetHost(gSystem->HostName());
         host = url.GetHostFQDN();
      }

      // Find on which disk is the file, if any
      TString disk;
      if (partitions) {
         TIter iString(partitions);
         TObjString* os = 0;
         while ((os = (TObjString *)iString())) {
            // Compare begining of the url with disk mountpoint
            if (strncmp(url.GetFile(), os->GetName(), os->GetString().Length()) == 0) {
               disk = os->GetName();
               break;
            }
         }
      }
      // Node's url
      TString nodeStr;
      if (disk.IsNull())
         nodeStr.Form("%s://%s", url.GetProtocol(), host.Data());
      else
         nodeStr.Form("%s://%s/%s", url.GetProtocol(), host.Data(), disk.Data());
      TFileNode *node = (TFileNode *) fFileNodes->FindObject(nodeStr);

      if (node == 0) {
         node = new TFileNode(nodeStr, fStrategy, fFilesToProcess);
         fFileNodes->Add(node);
         PDB(kPacketizer,2)
            Info("TPacketizerAdaptive", "creating new node '%s' or the element", nodeStr.Data());
      } else {
         PDB(kPacketizer,2)
            Info("TPacketizerAdaptive", "adding element to existing node '%s'", nodeStr.Data());
      }

      node->Add(e, kFALSE);
   }

   fSlaveStats = new TMap;
   fSlaveStats->SetOwner(kFALSE);

   TSlave *slave;
   TIter si(slaves);
   while ((slave = (TSlave*) si.Next())) {
      fSlaveStats->Add( slave, new TSlaveStat(slave) );
      fMaxPerfIdx = slave->GetPerfIdx() > fMaxPerfIdx ?
         slave->GetPerfIdx() : fMaxPerfIdx;
   }

   // Setup file & filenode structure
   Reset();
   // Optimize the number of files to be open when running on subsample
   Int_t validateMode = 0;
   Int_t gprc = TProof::GetParameter(input, "PROOF_ValidateByFile", validateMode);
   Bool_t byfile = (gprc == 0 && validateMode > 0 && num > -1) ? kTRUE : kFALSE;
   if (num > -1)
      PDB(kPacketizer,2)
         Info("TPacketizerAdaptive",
              "processing subset of entries: validating by file? %s", byfile ? "yes": "no");
   ValidateFiles(dset, slaves, num, byfile);


   if (!fValid) return;

   // apply global range (first,num) to dset and rebuild structure
   // ommitting TDSet elements that are not needed

   Int_t files = 0;
   fTotalEntries = 0;
   fUnAllocated->Clear();  // avoid dangling pointers
   fActive->Clear();
   fFileNodes->Clear();    // then delete all objects
   PDB(kPacketizer,2)
      Info("TPacketizerAdaptive",
           "processing range: first %lld, num %lld", first, num);

   dset->Reset();
   Long64_t cur = 0;
   while (( e = (TDSetElement*)dset->Next())) {

      // Skip invalid or missing file; It will be moved
      // from the dset to the 'MissingFiles' list in the player.
      if (!e->GetValid()) continue;

      TUrl url = e->GetFileName();
      Long64_t eFirst = e->GetFirst();
      Long64_t eNum = e->GetNum();
      PDB(kPacketizer,2)
         Info("TPacketizerAdaptive", "processing element '%s'", e->GetFileName());
      PDB(kPacketizer,2)
         Info("TPacketizerAdaptive",
              " --> first %lld, elenum %lld (cur %lld) (entrylist: %p)", eFirst, eNum, cur, e->GetEntryList());

      if (!e->GetEntryList()) {
         // This element is before the start of the global range, skip it
         if (cur + eNum < first) {
            cur += eNum;
            PDB(kPacketizer,2)
               Info("TPacketizerAdaptive", " --> skip element cur %lld", cur);
            continue;
         }

         // This element is after the end of the global range, skip it
         if (num != -1 && (first+num <= cur)) {
            cur += eNum;
            PDB(kPacketizer,2)
               Info("TPacketizerAdaptive", " --> drop element cur %lld", cur);
            continue; // break ??
         }

         Bool_t inRange = kFALSE;
         if (cur <= first || (num != -1 && (first+num <= cur+eNum))) {

            if (cur <= first) {
               // If this element contains the start of the global range
               // adjust its start and number of entries
               e->SetFirst( eFirst + (first - cur) );
               e->SetNum( e->GetNum() - (first - cur) );
               PDB(kPacketizer,2)
                  Info("TPacketizerAdaptive", " --> adjust start %lld and end %lld",
                       eFirst + (first - cur), first + num - cur);
               inRange = kTRUE;
            }
            if (num != -1 && (first+num <= cur+eNum)) {
               // If this element contains the end of the global range
               // adjust its number of entries
               e->SetNum( first + num - e->GetFirst() - cur );
               PDB(kPacketizer,2)
                  Info("TPacketizerAdaptive", " --> adjust end %lld", first + num - cur);
               inRange = kTRUE;
            }

         } else {
            // Increment the counter ...
            PDB(kPacketizer,2)
               Info("TPacketizerAdaptive", " --> increment 'cur' by %lld", eNum);
            cur += eNum;
         }
         // Re-adjust eNum and cur, if needed
         if (inRange) {
            cur += eNum;
            eNum = e->GetNum();
         }

      } else {
         TEntryList *enl = dynamic_cast<TEntryList *>(e->GetEntryList());
         if (enl) {
            eNum = enl->GetN();
            PDB(kPacketizer,2)
               Info("TPacketizerAdaptive", " --> entry-list element: %lld entries", eNum);
         } else {
            TEventList *evl = dynamic_cast<TEventList *>(e->GetEntryList());
            eNum = evl ? evl->GetN() : eNum;
            PDB(kPacketizer,2)
               Info("TPacketizerAdaptive", " --> event-list element: %lld entries (evl:%p)", eNum, evl);
         }
         if (!eNum) {
            PDB(kPacketizer,2)
               Info("TPacketizerAdaptive", " --> empty entry- or event-list element!");
            continue;
         }
      }
      PDB(kPacketizer,2)
         Info("TPacketizerAdaptive", " --> next cur %lld", cur);

      // Map non URL filenames to dummy host
      TString host;
      if ( !url.IsValid() ||
          (strncmp(url.GetProtocol(),"root", 4) &&
           strncmp(url.GetProtocol(),"rfio", 4) &&
           strncmp(url.GetProtocol(),"file", 4)) ) {
         host = "no-host";
      } else if ( url.IsValid() && !strncmp(url.GetProtocol(),"file", 4)) {
         host = "localhost";
         url.SetProtocol("root");
      } else {
         host = url.GetHostFQDN();
      }
      // Get full name for local hosts
      if (host.Contains("localhost") || host == "127.0.0.1") {
         url.SetHost(gSystem->HostName());
         host = url.GetHostFQDN();
      }

      // Find, on which disk is the file
      TString disk;
      if (partitions) {
         TIter iString(partitions);
         TObjString* os = 0;
         while ((os = (TObjString *)iString())) {
            // Compare begining of the url with disk mountpoint
            if (strncmp(url.GetFile(), os->GetName(), os->GetString().Length()) == 0) {
               disk = os->GetName();
               break;
            }
         }
      }
      // Node's url
      TString nodeStr;
      if (disk.IsNull())
         nodeStr.Form("%s://%s", url.GetProtocol(), host.Data());
      else
         nodeStr.Form("%s://%s/%s", url.GetProtocol(), host.Data(), disk.Data());
      TFileNode *node = (TFileNode*) fFileNodes->FindObject(nodeStr);


      if (node == 0) {
         node = new TFileNode(nodeStr, fStrategy, fFilesToProcess);
         fFileNodes->Add( node );
         PDB(kPacketizer, 2)
            Info("TPacketizerAdaptive", " --> creating new node '%s' for element", nodeStr.Data());
      } else {
         PDB(kPacketizer, 2)
            Info("TPacketizerAdaptive", " --> adding element to exiting node '%s'", nodeStr.Data());
      }

      ++files;
      fTotalEntries += eNum;
      node->Add(e, kTRUE);
      node->IncEvents(eNum);
      PDB(kPacketizer,2) e->Print("a");
   }
   PDB(kPacketizer,1)
      Info("TPacketizerAdaptive", "processing %lld entries in %d files on %d hosts",
                                  fTotalEntries, files, fFileNodes->GetSize());

   // Set the total number for monitoring
   if (gPerfStats)
      gPerfStats->SetNumEvents(fTotalEntries);

   Reset();

   InitStats();

   if (!fValid)
      SafeDelete(fProgress);

   PDB(kPacketizer,1) Info("TPacketizerAdaptive", "return");
}

//______________________________________________________________________________
TPacketizerAdaptive::~TPacketizerAdaptive()
{
   // Destructor.

   if (fSlaveStats) {
      fSlaveStats->DeleteValues();
   }

   SafeDelete(fSlaveStats);
   SafeDelete(fUnAllocated);
   SafeDelete(fActive);
   SafeDelete(fFileNodes);
   SafeDelete(fFilesToProcess);
}

//______________________________________________________________________________
void TPacketizerAdaptive::InitStats()
{
   // (re)initialise the statistics
   // called at the begining or after a worker dies.

   // calculating how many files from TDSet are not cached on
   // any slave
   Int_t noRemoteFiles = 0;
   fNEventsOnRemLoc = 0;
   Int_t totalNumberOfFiles = 0;
   TIter next(fFileNodes);
   while (TFileNode *fn = (TFileNode*)next()) {
      totalNumberOfFiles += fn->GetNumberOfFiles();
      if (fn->GetMySlaveCnt() == 0) {
         noRemoteFiles += fn->GetNumberOfFiles();
         fNEventsOnRemLoc += (fn->GetNEvents() - fn->GetProcessed());
      }
   }

   if (totalNumberOfFiles == 0) {
      Info("InitStats", "no valid or non-empty file found: setting invalid");
      // No valid files: set invalid and return
      fValid = kFALSE;
      return;
   }

   fFractionOfRemoteFiles = (1.0 * noRemoteFiles) / totalNumberOfFiles;
   Info("InitStats",
        "fraction of remote files %f", fFractionOfRemoteFiles);

   if (!fValid)
      SafeDelete(fProgress);

   PDB(kPacketizer,1) Info("InitStats", "return");
}

//______________________________________________________________________________
TPacketizerAdaptive::TFileStat *TPacketizerAdaptive::GetNextUnAlloc(TFileNode *node, const char *nodeHostName)
{
   // Get next unallocated file from 'node' or other nodes:
   // First try 'node'. If there is no more files, keep trying to
   // find an unallocated file on other nodes.

   TFileStat *file = 0;

   if (node != 0) {
      PDB(kPacketizer, 2)
         Info("GetNextUnAlloc", "looking for file on node %s", node->GetName());
      file = node->GetNextUnAlloc();
      if (file == 0) RemoveUnAllocNode(node);
   } else {
      if (nodeHostName && strlen(nodeHostName) > 0) {

         TFileNode *fn;
         // Make sure that they are in the corrected order
         fUnAllocated->Sort();
         PDB(kPacketizer,2) fUnAllocated->Print();

         // Loop over unallocated fileNode list
         for (int i = 0; i < fUnAllocated->GetSize(); i++) {

            if ((fn = (TFileNode *) fUnAllocated->At(i))) {
               TUrl uu(fn->GetName());
               PDB(kPacketizer, 2)
                  Info("GetNextUnAlloc", "comparing %s with %s...", nodeHostName, uu.GetHost());

               // Check, whether node's hostname is matching with current fileNode (fn)
               if (!strcmp(nodeHostName, uu.GetHost())) {
                  node = fn;

                  // Fetch next unallocated file from this node
                  if ((file = node->GetNextUnAlloc()) == 0) {
                     RemoveUnAllocNode(node);
                     node = 0;
                  } else {
                     PDB(kPacketizer, 2)
                        Info("GetNextUnAlloc", "found! (host: %s)", uu.GetHost());
                     break;
                  }
               }
            } else {
               Warning("GetNextUnAlloc", "unallocate entry %d is empty!", i);
            }
         }

         if (node != 0 && fMaxSlaveCnt > 0 && node->GetExtSlaveCnt() >= fMaxSlaveCnt) {
            // Unlike in TPacketizer we look at the number of ext slaves only.
            PDB(kPacketizer,1)
               Info("GetNextUnAlloc", "reached Workers-per-Node Limit (%ld)", fMaxSlaveCnt);
            node = 0;
         }
      }

      if (node == 0) {
         while (file == 0 && ((node = NextNode()) != 0)) {
            PDB(kPacketizer, 2)
               Info("GetNextUnAlloc", "looking for file on node %s", node->GetName());
            if ((file = node->GetNextUnAlloc()) == 0) RemoveUnAllocNode(node);
         }
      }
   }

   if (file != 0) {
      // if needed make node active
      if (fActive->FindObject(node) == 0) {
         fActive->Add(node);
      }
   }

   PDB(kPacketizer, 2) {
      if (!file) {
         Info("GetNextUnAlloc", "no file found!");
      } else {
         file->Print();
      }
   }

   return file;
}

//______________________________________________________________________________
TPacketizerAdaptive::TFileNode *TPacketizerAdaptive::NextNode()
{
   // Get next node which has unallocated files.
   // the order is determined by TFileNode::Compare

   fUnAllocated->Sort();
   PDB(kPacketizer,2) {
      fUnAllocated->Print();
   }

   TFileNode *fn = (TFileNode*) fUnAllocated->First();
   if (fn != 0 && fMaxSlaveCnt > 0 && fn->GetExtSlaveCnt() >= fMaxSlaveCnt) {
      // unlike in TPacketizer we look at the number of ext slaves only.
      PDB(kPacketizer,1)
         Info("NextNode", "reached Workers-per-Node Limit (%ld)", fMaxSlaveCnt);
      fn = 0;
   }

   return fn;
}

//______________________________________________________________________________
void TPacketizerAdaptive::RemoveUnAllocNode(TFileNode * node)
{
   // Remove unallocated node.

   fUnAllocated->Remove(node);
}

//______________________________________________________________________________
TPacketizerAdaptive::TFileStat *TPacketizerAdaptive::GetNextActive()
{
   // Get next active file.

   TFileNode *node;
   TFileStat *file = 0;

   while (file == 0 && ((node = NextActiveNode()) != 0)) {
         file = node->GetNextActive();
         if (file == 0) RemoveActiveNode(node);
   }

   return file;
}


//______________________________________________________________________________
TPacketizerAdaptive::TFileNode *TPacketizerAdaptive::NextActiveNode()
{
   // Get next active node.

   fActive->Sort();
   PDB(kPacketizer,2) {
      Info("NextActiveNode", "enter");
      fActive->Print();
   }

   TFileNode *fn = (TFileNode*) fActive->First();
   // look at only ext slaves
   if (fn != 0 && fMaxSlaveCnt > 0 && fn->GetExtSlaveCnt() >= fMaxSlaveCnt) {
      PDB(kPacketizer,1)
         Info("NextActiveNode","reached Workers-per-Node limit (%ld)", fMaxSlaveCnt);
      fn = 0;
   }

   return fn;
}

//______________________________________________________________________________
void TPacketizerAdaptive::RemoveActive(TFileStat *file)
{
   // Remove file from the list of actives.

   TFileNode *node = file->GetNode();

   node->RemoveActive(file);
   if (node->GetNumberOfActiveFiles() == 0) RemoveActiveNode(node);
}

//______________________________________________________________________________
void TPacketizerAdaptive::RemoveActiveNode(TFileNode *node)
{
   // Remove node from the list of actives.

   fActive->Remove(node);
}

//______________________________________________________________________________
void TPacketizerAdaptive::Reset()
{
   // Reset the internal data structure for packet distribution.

   fUnAllocated->Clear();
   fUnAllocated->AddAll(fFileNodes);

   fActive->Clear();

   TIter files(fFileNodes);
   TFileNode *fn;
   while ((fn = (TFileNode*) files.Next()) != 0) {
      fn->Reset();
   }

   TIter slaves(fSlaveStats);
   TObject *key;
   while ((key = slaves.Next()) != 0) {
      TSlaveStat *slstat = (TSlaveStat*) fSlaveStats->GetValue(key);
      if (!slstat) {
         Warning("Reset", "TSlaveStat associated to key '%s' is NULL", key->GetName());
         continue;
      }
      // Find out which file nodes are on the worker machine and assign the
      // one with less workers assigned
      TFileNode *fnmin = 0;
      Int_t fncnt = fSlaveStats->GetSize();
      files.Reset();
      while ((fn = (TFileNode*) files.Next()) != 0) {
         if (!strcmp(slstat->GetName(), TUrl(fn->GetName()).GetHost())) {
            if (fn->GetMySlaveCnt() < fncnt) {
               fnmin = fn;
               fncnt = fn->GetMySlaveCnt();
            }
         }
      }
      if (fnmin != 0 ) {
         slstat->SetFileNode(fnmin);
         fnmin->IncMySlaveCnt();
         PDB(kPacketizer, 2)
            Info("Reset","assigning node '%s' to '%s' (cnt: %d)",
                         fnmin->GetName(), slstat->GetName(), fnmin->GetMySlaveCnt());
      }
      slstat->fCurFile = 0;
   }
}

//______________________________________________________________________________
void TPacketizerAdaptive::ValidateFiles(TDSet *dset, TList *slaves,
                                        Long64_t maxent, Bool_t byfile)
{
   // Check existence of file/dir/tree an get number of entries.
   // Assumes the files have been setup.

   TMap     slaves_by_sock;
   TMonitor mon;
   TList    workers;


   // Setup the communication infrastructure

   workers.AddAll(slaves);
   TIter    si(slaves);
   TSlave   *slm;
   while ((slm = (TSlave*)si.Next()) != 0) {
      PDB(kPacketizer,3)
      Info("ValidateFiles","socket added to monitor: %p (%s)",
          slm->GetSocket(), slm->GetName());
      mon.Add(slm->GetSocket());
      slaves_by_sock.Add(slm->GetSocket(), slm);
   }

   mon.DeActivateAll();

   ((TProof*)gProof)->DeActivateAsyncInput();

   // Some monitoring systems (TXSocketHandler) need to know this
   ((TProof*)gProof)->fCurrentMonitor = &mon;

   // Identify the type
   if (!strcmp(dset->GetType(), "TTree")) SetBit(TVirtualPacketizer::kIsTree);

   // Preparing for client notification
   TString msg("Validating files");
   UInt_t n = 0;
   UInt_t tot = dset->GetListOfElements()->GetSize();
   Bool_t st = kTRUE;

   Long64_t totent = 0, nopenf = 0;
   while (kTRUE) {

      // send work
      while (TSlave *s = (TSlave *)workers.First()) {

         workers.Remove(s);

         // find a file

         TSlaveStat *slstat = (TSlaveStat*)fSlaveStats->GetValue(s);
         if (!slstat) {
            Error("ValidateFiles", "TSlaveStat associated to slave '%s' is NULL", s->GetName());
            continue;
         }

         TFileNode *node = 0;
         TFileStat *file = 0;

         // try its own node first
         if ((node = slstat->GetFileNode()) != 0) {
            PDB(kPacketizer,3) node->Print();
            file = GetNextUnAlloc(node);
            if (file == 0)
               slstat->SetFileNode(0);
         }

         // look for a file on any other node if necessary
         if (file == 0)
            file = GetNextUnAlloc();

         if (file != 0) {
            // files are done right away
            RemoveActive(file);

            slstat->fCurFile = file;
            TDSetElement *elem = file->GetElement();
            Long64_t entries = elem->GetEntries(kTRUE, kFALSE);
            if (entries < 0 || strlen(elem->GetTitle()) <= 0) {
               // This is decremented when we get the reply
               file->GetNode()->IncExtSlaveCnt(slstat->GetName());
               TMessage m(kPROOF_GETENTRIES);
               m << dset->IsTree()
               << TString(elem->GetFileName())
               << TString(elem->GetDirectory())
               << TString(elem->GetObjName());

               s->GetSocket()->Send( m );
               mon.Activate(s->GetSocket());
               PDB(kPacketizer,2)
                  Info("ValidateFiles",
                       "sent to worker-%s (%s) via %p GETENTRIES on %s %s %s %s",
                       s->GetOrdinal(), s->GetName(), s->GetSocket(),
                       dset->IsTree() ? "tree" : "objects", elem->GetFileName(),
                       elem->GetDirectory(), elem->GetObjName());
            } else {
               // Fill the info
               elem->SetTDSetOffset(entries);
               if (entries > 0) {
                  // Most likely valid
                  elem->SetValid();
                  if (!elem->GetEntryList()) {
                     if (elem->GetFirst() > entries) {
                        Error("ValidateFiles",
                              "first (%lld) higher then number of entries (%lld) in %s",
                               elem->GetFirst(), entries, elem->GetFileName());
                        // disable element
                        slstat->fCurFile->SetDone();
                        elem->Invalidate();
                        dset->SetBit(TDSet::kSomeInvalid);
                     }
                     if (elem->GetNum() == -1) {
                        elem->SetNum(entries - elem->GetFirst());
                     } else if (elem->GetFirst() + elem->GetNum() > entries) {
                        Warning("ValidateFiles", "num (%lld) + first (%lld) larger then number of"
                                 " keys/entries (%lld) in %s", elem->GetNum(), elem->GetFirst(),
                                 entries, elem->GetFileName());
                        elem->SetNum(entries - elem->GetFirst());
                     }
                     PDB(kPacketizer,2)
                        Info("ValidateFiles",
                             "found elem '%s' with %lld entries", elem->GetFileName(), entries);
                  }
               }
               // Count
               totent += entries;
               nopenf++;
               // Notify the client
               n++;
               gProof->SendDataSetStatus(msg, n, tot, st);

               // This worker is ready for the next validation
               workers.Add(s);
            }
         }
      }

      // Check if there is anything to wait for
      if (mon.GetActive() == 0) {
         if (byfile && maxent > 0) {
            // How many files do we still need ?
            Long64_t nrestf = (maxent - totent) * nopenf / totent ;
            if (nrestf <= 0 && maxent > totent) nrestf = 1;
            if (nrestf > 0) {
               PDB(kPacketizer,3)
                  Info("ValidateFiles", "{%lld, %lld, %lld}: needs to validate %lld more files",
                                         maxent, totent, nopenf, nrestf);
               si.Reset();
               while ((slm = (TSlave *) si.Next()) && nrestf--) {
                  workers.Add(slm);
               }
               continue;
            } else {
               PDB(kPacketizer,3)
                  Info("ValidateFiles", "no need to validate more files");
               break;
            }
         } else {
            break;
         }
      }

      PDB(kPacketizer,3) {
         Info("ValidateFiles", "waiting for %d slaves:", mon.GetActive());
         TList *act = mon.GetListOfActives();
         TIter next(act);
         while (TSocket *s = (TSocket*) next()) {
            TSlave *sl = (TSlave *) slaves_by_sock.GetValue(s);
            if (sl)
               Info("ValidateFiles", "   worker-%s (%s)",
                    sl->GetOrdinal(), sl->GetName());
         }
         delete act;
      }

      TSocket *sock = mon.Select();
      // If we have been interrupted break
      if (!sock) {
         Error("ValidateFiles", "selection has been interrupted - STOP");
         mon.DeActivateAll();
         fValid = kFALSE;
         break;
      }
      mon.DeActivate(sock);

      PDB(kPacketizer,3) Info("ValidateFiles", "select returned: %p", sock);

      TSlave *slave = (TSlave *) slaves_by_sock.GetValue( sock );
      if (!sock->IsValid()) {
         // A socket got invalid during validation
         Error("ValidateFiles", "worker-%s (%s) got invalid - STOP",
               slave->GetOrdinal(), slave->GetName());
         ((TProof*)gProof)->MarkBad(slave, "socket got invalid during validation");
         fValid = kFALSE;
         break;
      }

      TMessage *reply;

      if (sock->Recv(reply) <= 0) {
         // Notify
         Error("ValidateFiles", "Recv failed! for worker-%s (%s)",
                                slave->GetOrdinal(), slave->GetName());
         // Help! lost a slave? ('slave' is deleted inside here ...)
         ((TProof*)gProof)->MarkBad(slave, "receive failed during validation");
         fValid = kFALSE;
         continue;
      }

      if (reply->What() != kPROOF_GETENTRIES) {
         // Not what we want: handover processing to the central machinery
         Int_t what = reply->What();
         ((TProof*)gProof)->HandleInputMessage(slave, reply);
         if (what == kPROOF_FATAL) {
             Error("ValidateFiles", "kPROOF_FATAL from worker-%s (%s)",
                                    slave->GetOrdinal(), slave->GetName());
             fValid = kFALSE;
         } else {
            // Reactivate the socket
            mon.Activate(sock);
         }
         // Get next message
         continue;
      }

      TSlaveStat *slavestat = (TSlaveStat*) fSlaveStats->GetValue( slave );
      TDSetElement *e = slavestat->fCurFile->GetElement();
      slavestat->fCurFile->GetNode()->DecExtSlaveCnt(slavestat->GetName());
      Long64_t entries;

      (*reply) >> entries;

      // Extract object name, if there
      if ((reply->BufferSize() > reply->Length())) {
         TString objname;
         (*reply) >> objname;
         e->SetTitle(objname);
      }

      e->SetTDSetOffset(entries);
      if (entries > 0) {

         // This dataset element is most likely valid
         e->SetValid();

         if (!e->GetEntryList()) {
            if (e->GetFirst() > entries) {
               Error("ValidateFiles",
                     "first (%lld) higher then number of entries (%lld) in %s",
                      e->GetFirst(), entries, e->GetFileName());

               // Invalidate the element
               slavestat->fCurFile->SetDone();
               e->Invalidate();
               dset->SetBit(TDSet::kSomeInvalid);
            }

            if (e->GetNum() == -1) {
               e->SetNum(entries - e->GetFirst());
            } else if (e->GetFirst() + e->GetNum() > entries) {
               Error("ValidateFiles",
                     "num (%lld) + first (%lld) larger then number of keys/entries (%lld) in %s",
                      e->GetNum(), e->GetFirst(), entries, e->GetFileName());
               e->SetNum(entries - e->GetFirst());
            }
         }

         // Count
         totent += entries;
         nopenf++;

         // Notify the client
         n++;
         gProof->SendDataSetStatus(msg, n, tot, st);

      } else {

         Error("ValidateFiles", "cannot get entries for file: %s - skipping", e->GetFileName() );
         //
         // Need to fix this with a user option to allow incomplete file sets (rdm)
         //
         //fValid = kFALSE; // all element must be readable!
         if (gProofServ) {
            TMessage m(kPROOF_MESSAGE);
            m << TString(Form("Cannot get entries for file: %s - skipping",
                              e->GetFileName()));
            gProofServ->GetSocket()->Send(m);
         }

         // invalidate element
         e->Invalidate();
         dset->SetBit(TDSet::kSomeInvalid);
      }
      PDB(kPacketizer,3) Info("ValidateFiles", " %lld events validated", totent);

      // Ready for the next job, unless we have enough files
      if (maxent < 0 || ((totent < maxent) && !byfile))
         workers.Add(slave);
   }

   // report std. output from slaves??

   ((TProof*)gProof)->ActivateAsyncInput();

   // This needs to be reset
   ((TProof*)gProof)->fCurrentMonitor = 0;

   // No reason to continue if invalid
   if (!fValid)
      return;

   // compute the offset for each file element
   Long64_t offset = 0;
   Long64_t newOffset = 0;
   TIter next(dset->GetListOfElements());
   TDSetElement *el;
   while ( (el = dynamic_cast<TDSetElement*> (next())) ) {
      if (el->GetValid()) {
         newOffset = offset + el->GetTDSetOffset();
         el->SetTDSetOffset(offset);
         offset = newOffset;
      }
   }
}

//______________________________________________________________________________
Int_t TPacketizerAdaptive::CalculatePacketSize(TObject *slStatPtr, Long64_t cachesz, Int_t learnent)
{
   // The result depends on the fStrategy

   Long64_t num;
   if (fStrategy == 0) {
      // TPacketizer's heuristic for starting packet size
      // Constant packet size;
      Int_t nslaves = fSlaveStats->GetSize();
      if (nslaves > 0) {
         num = fTotalEntries / (fPacketAsAFraction * nslaves);
      } else {
         num = 1;
      }
   } else {
      // The dynamic heuristic for setting the packet size (default)
      // Calculates the packet size based on performance of this slave
      // and estimated time left until the end of the query.
      TSlaveStat* slstat = (TSlaveStat*)slStatPtr;
      Float_t rate = slstat->GetCurRate();
      if (!rate)
         rate = slstat->GetAvgRate();
      if (rate) {

         // Global average rate
         Float_t avgProcRate = (GetEntriesProcessed()/(GetCumProcTime() / fSlaveStats->GetSize()));
         Float_t packetTime = ((fTotalEntries - GetEntriesProcessed())/avgProcRate)/fPacketAsAFraction;

         // Bytes-to-Event conversion
         Float_t bevt = (GetEntriesProcessed() > 0) ? GetBytesRead() / GetEntriesProcessed() : -1.;

         // Make sure it is not smaller then the cache, if the info is available and the size
         // synchronization is required. But apply the cache-packet size synchronization only if there
         // are enough left files to process and the files are all of similar sizes. Otherwise we risk
         // to not exploit optimally all potentially active workers.
         Bool_t cpsync = fCachePacketSync;
         if (fMaxEntriesRatio > 0. && cpsync) {
            if (fFilesToProcess && fFilesToProcess->GetSize() <= fSlaveStats->GetSize()) {
               Long64_t remEntries = fTotalEntries - GetEntriesProcessed();
               Long64_t maxEntries = -1;
               if (fFilesToProcess->Last()) {
                  TDSetElement *elem = (TDSetElement *) ((TPacketizerAdaptive::TFileStat *) fFilesToProcess->Last())->GetElement();
                  if (elem) maxEntries = elem->GetNum();
               }
               if (maxEntries > remEntries / fSlaveStats->GetSize() * fMaxEntriesRatio) {
                  PDB(kPacketizer,3) {
                     Info("CalculatePacketSize", "%s: switching off synchronization of packet and cache sizes:", slstat->GetOrdinal());
                     Info("CalculatePacketSize", "%s: few files (%d) remaining of very different sizes (max/avg = %.2f > %.2f)",
                                                 slstat->GetOrdinal(), fFilesToProcess->GetSize(),
                                                (Double_t)maxEntries / remEntries * fSlaveStats->GetSize(), fMaxEntriesRatio);
                  }
                  cpsync = kFALSE;
               }
            }
         }
         if (bevt > 0. && cachesz > 0 && cpsync) {
            if ((Long64_t)(rate * packetTime * bevt) < cachesz)
               packetTime = cachesz / bevt / rate;
         }

         // Apply min-max again, if required
         if (fMaxPacketTime > 0. && packetTime > fMaxPacketTime) packetTime = fMaxPacketTime;
         if (fMinPacketTime > 0. && packetTime < fMinPacketTime) packetTime = fMinPacketTime;

         // Translate the packet length in number of entries
         num = (Long64_t)(rate * packetTime);

         // Notify
         PDB(kPacketizer,2)
            Info("CalculatePacketSize","%s: avgr: %f, rate: %f, left: %lld, pacT: %f, sz: %f (csz: %f), num: %lld",
                 slstat->GetOrdinal(), avgProcRate, rate, fTotalEntries - GetEntriesProcessed(),
                 packetTime, ((bevt > 0) ? num*bevt/1048576. : -1.), cachesz/1048576., num);

      } else {
         // First packet for this worker in this query
         // Twice the learning phase
         num = (learnent > 0) ? 5 * learnent : 1000;

         // Notify
         PDB(kPacketizer,2)
            Info("CalculatePacketSize","%s: num: %lld", slstat->GetOrdinal(),  num);
      }
   }
   if (num < 1) num = 1;
   return num;
}

//______________________________________________________________________________
Int_t TPacketizerAdaptive::AddProcessed(TSlave *sl,
                                        TProofProgressStatus *status,
                                        Double_t latency,
                                        TList **listOfMissingFiles)
{
   // To be used by GetNextPacket but also in reaction to kPROOF_STOPPROCESS
   // message (when the worker was asked to stop processing during a packet).
   // returns the #entries intended in the last packet - #processed entries

   // find slave
   TSlaveStat *slstat = (TSlaveStat*) fSlaveStats->GetValue( sl );
   if (!slstat) {
      Error("AddProcessed", "%s: TSlaveStat instance for worker %s not found!",
                            (sl ? sl->GetOrdinal() : "x.x"),
                            (sl ? sl->GetName() : "**undef**"));
      return -1;
   }

   // update stats & free old element

   if ( slstat->fCurElem != 0 ) {
      Long64_t expectedNumEv = slstat->fCurElem->GetNum();
      // Calculate the number of events processed in the last packet
      Long64_t numev;
      if (status && status->GetEntries() > 0)
         numev = status->GetEntries() - slstat->GetEntriesProcessed();
      else
         numev = 0;

      // Calculate the progress made in the last packet
      TProofProgressStatus *progress = 0;
      if (numev > 0) {
         // This also moves the pointer in the corrsponding TFileInfo
         progress = slstat->AddProcessed(status);
         if (progress) {
            (*fProgressStatus) += *progress;
            // update processing rate
            slstat->UpdateRates(status);
         }
      } else {
          progress = new TProofProgressStatus();
      }
      if (progress) {
         PDB(kPacketizer,2)
            Info("AddProcessed", "%s: %s: %lld %7.3lf %7.3lf %7.3lf %lld",
               sl->GetOrdinal(), sl->GetName(), progress->GetEntries(), latency,
               progress->GetProcTime(), progress->GetCPUTime(), progress->GetBytesRead());

         if (gPerfStats)
            gPerfStats->PacketEvent(sl->GetOrdinal(), sl->GetName(),
                                    slstat->fCurElem->GetFileName(),
                                    progress->GetEntries(),
                                    latency,
                                    progress->GetProcTime(),
                                    progress->GetCPUTime(),
                                    progress->GetBytesRead());
         delete progress;
      }
      if (numev != expectedNumEv) {
         // The last packet was not fully processed
         // and will be split in two:
         // - The completed part was marked as done.
         // - Create a new packet with the part to be resubmitted.
         TDSetElement *newPacket = new TDSetElement(*(slstat->fCurElem));
         if (newPacket && numev < expectedNumEv) {
            Long64_t first = newPacket->GetFirst();
            newPacket->SetFirst(first + numev);
            if (ReassignPacket(newPacket, listOfMissingFiles) == -1)
               SafeDelete(newPacket);
         } else
            Error("AddProcessed", "%s: processed too much? (%lld, %lld)",
                                  sl->GetOrdinal(), numev, expectedNumEv);

         // TODO: a signal handler which will send info from the worker
         // after a packet fails.
         /* Add it to the failed packets list.
         if (!fFailedPackets) {
            fFailedPackets = new TList();
         }
         fFailedPackets->Add(slstat->fCurElem);
         */
      }

      slstat->fCurElem = 0;
      return (expectedNumEv - numev);
   } else {
      // the kPROOF_STOPPRPOCESS message is send after the worker receives zero
      // as the reply to kPROOF_GETNEXTPACKET
      return -1;
   }
}

//______________________________________________________________________________
TDSetElement *TPacketizerAdaptive::GetNextPacket(TSlave *sl, TMessage *r)
{
   // Get next packet;
   // A meaningfull difference to TPacketizer is the fact that this
   // packetizer, for each worker, tries to predict whether the worker
   // will finish processing it's local files before the end of the query.
   // If yes, it allocates, to those workers, files from non-slave filenodes
   // or from slaves that are overloaded. The check is done every time a new
   // file needs to be assigned.

   if ( !fValid ) {
      return 0;
   }

   // find slave

   TSlaveStat *slstat = (TSlaveStat*) fSlaveStats->GetValue( sl );
   if (!slstat) {
      Error("GetNextPacket", "TSlaveStat instance for worker %s not found!",
                            (sl ? sl->GetName() : "**undef**"));
      return 0;
   }

   // Attach to current file
   TFileStat *file = slstat->fCurFile;

   // Update stats & free old element

   Bool_t firstPacket = kFALSE;
   Long64_t cachesz = -1;
   Int_t learnent = -1;
   if ( slstat->fCurElem != 0 ) {

      Long64_t restEntries = 0;
      Double_t latency, proctime, proccpu;
      TProofProgressStatus *status = 0;
      Bool_t fileNotOpen = kFALSE, fileCorrupted = kFALSE;

      if (sl->GetProtocol() > 18) {

         (*r) >> latency;
         (*r) >> status;

         if (sl->GetProtocol() > 25) {
            (*r) >> cachesz >> learnent;
            if (r->BufferSize() > r->Length()) (*r) >> restEntries;
         }
         fileNotOpen = status->TestBit(TProofProgressStatus::kFileNotOpen) ? kTRUE : kFALSE;
         fileCorrupted = status->TestBit(TProofProgressStatus::kFileCorrupted) ? kTRUE : kFALSE;

      } else {

         Long64_t bytesRead = -1;

         (*r) >> latency >> proctime >> proccpu;
         // only read new info if available
         if (r->BufferSize() > r->Length()) (*r) >> bytesRead;
         if (r->BufferSize() > r->Length()) (*r) >> restEntries;
         Long64_t totev = 0;
         if (r->BufferSize() > r->Length()) (*r) >> totev;

         status = new TProofProgressStatus(totev, bytesRead, -1, proctime, proccpu);
         fileNotOpen = (restEntries < 0) ? kTRUE : kFALSE;
      }

      if (!fileNotOpen && !fileCorrupted) {
         if (AddProcessed(sl, status, latency) != 0)
            Error("GetNextPacket", "%s: the worker processed a different # of entries", sl->GetOrdinal());
         if (fProgressStatus->GetEntries() >= fTotalEntries) {
            if (fProgressStatus->GetEntries() > fTotalEntries)
               Error("GetNextPacket", "%s: processed too many entries! (%lld, %lld)",
                                      sl->GetOrdinal(), fProgressStatus->GetEntries(), fTotalEntries);
            // Send last timer message and stop the timer
            HandleTimer(0);
            SafeDelete(fProgress);
         }
      } else {
         if (file) {
            if (file->GetElement()) {
               if (fileCorrupted) {
                  Info("GetNextPacket", "%s: file '%s' turned corrupted: invalidating file (%lld)",
                                       sl->GetOrdinal(), file->GetElement()->GetName(), restEntries); 
                  Int_t nunproc = AddProcessed(sl, status, latency);
                  PDB(kPacketizer,1)
                     Info("GetNextPacket", "%s: %d entries un-processed", sl->GetOrdinal(), nunproc);
                  // Remaining to be processed
                  Long64_t num = 0;
                  if (file->GetElement()->TestBit(TDSetElement::kCorrupted)) {
                     // Add the remainign entries in the packet to the ones already registered
                     num = file->GetElement()->GetEntries() + restEntries;
                  } else {
                     // First call: add the remaining entries in the packet and those of the file
                     // not yet assigned
                     Long64_t rest = file->GetElement()->GetEntries() - file->GetNextEntry();
                     num = restEntries + rest;
                  }
                  file->GetElement()->SetEntries(num);
                  PDB(kPacketizer,1)
                     Info("GetNextPacket", "%s: removed file: %s, entries left: %lld", sl->GetOrdinal(),
                                           file->GetElement()->GetName(), file->GetElement()->GetEntries());
                  // Flag as corrupted
                  file->GetElement()->SetBit(TDSetElement::kCorrupted);
               } else {
                  Info("GetNextPacket", "%s: file '%s' could not be open: invalidating related element",
                                        sl->GetOrdinal(), file->GetElement()->GetName()); 
               }
               // Invalidate the element
               file->GetElement()->Invalidate();
               // Add it to the failed packets list
               if (!fFailedPackets) fFailedPackets = new TList();
               if (!fFailedPackets->FindObject(file->GetElement()))
                  fFailedPackets->Add(file->GetElement());
            }
            // Deactivate this TFileStat
            file->SetDone();
            RemoveActive(file);
         } else {
            Info("GetNextPacket", "%s: error raised by worker, but TFileStat object invalid:"
                                  " protocol error?", sl->GetOrdinal()); 
         }
      }
   } else {
      firstPacket = kTRUE;
   }

   if ( fStop ) {
      HandleTimer(0);
      return 0;
   }

   TString nodeName;
   if (file != 0) nodeName = file->GetNode()->GetName();
   TString nodeHostName(slstat->GetName());

   PDB(kPacketizer,3)
      Info("GetNextPacket", "%s: entries processed: %lld - looking for a packet from node '%s'",
                            sl->GetOrdinal(), fProgressStatus->GetEntries(), nodeName.Data());

   // If current file is just finished
   if ( file != 0 && file->IsDone() ) {
      file->GetNode()->DecExtSlaveCnt(slstat->GetName());
      file->GetNode()->DecRunSlaveCnt();
      if (gPerfStats)
         gPerfStats->FileEvent(sl->GetOrdinal(), sl->GetName(), file->GetNode()->GetName(),
                               file->GetElement()->GetFileName(), kFALSE);
      file = 0;
   }
   // Reset the current file field
   slstat->fCurFile = file;

   Long64_t avgEventsLeftPerSlave =
      (fTotalEntries - fProgressStatus->GetEntries()) / fSlaveStats->GetSize();
   if (fTotalEntries == fProgressStatus->GetEntries())
      return 0;
   // Get a file if needed
   if ( file == 0) {
      // Needs a new file
      Bool_t openLocal;
      // Aiming for localPreference == 1 when #local == #remote events left
      Float_t localPreference = fBaseLocalPreference - (fNEventsOnRemLoc /
                                (0.4 *(fTotalEntries - fProgressStatus->GetEntries())));
      if ( slstat->GetFileNode() != 0 ) {
         // Local file node exists and has more events to process.
         fUnAllocated->Sort();
         TFileNode* firstNonLocalNode = (TFileNode*)fUnAllocated->First();
         Bool_t nonLocalNodePossible;
         if (fForceLocal)
            nonLocalNodePossible = 0;
         else
            nonLocalNodePossible = firstNonLocalNode ?
                    (fMaxSlaveCnt < 0 || (fMaxSlaveCnt > 0 && firstNonLocalNode->GetExtSlaveCnt() < fMaxSlaveCnt))
                                                     : 0;
         openLocal = !nonLocalNodePossible;
         Float_t slaveRate = slstat->GetAvgRate();
         if ( nonLocalNodePossible && fStrategy == 1) {
            // OpenLocal is set to kFALSE
            if ( slstat->GetFileNode()->GetRunSlaveCnt() >
                 slstat->GetFileNode()->GetMySlaveCnt() - 1 )
                // External slaves help slstat -> don't open nonlocal files
                // -1 because, at this point slstat is not running
                  openLocal = kTRUE;
            else if ( slaveRate == 0 ) { // first file for this slave
               // GetLocalEventsLeft() counts the potential slave
               // as running on its fileNode.
               if ( slstat->GetLocalEventsLeft() * localPreference
                   > (avgEventsLeftPerSlave))
                  openLocal = kTRUE;
               else if ( (firstNonLocalNode->GetEventsLeftPerSlave())
                     < slstat->GetLocalEventsLeft() * localPreference )
                  openLocal = kTRUE;
               else if ( firstNonLocalNode->GetExtSlaveCnt() > 1 )
                  openLocal = kTRUE;
               else if ( firstNonLocalNode->GetRunSlaveCnt() == 0 )
                  openLocal = kTRUE;
            } else {
               // At this point slstat has a non zero avg rate > 0
               Float_t slaveTime = slstat->GetLocalEventsLeft()/slaveRate;
               // And thus fCumProcTime, fProcessed > 0
               Float_t avgTime = avgEventsLeftPerSlave
                                 /(fProgressStatus->GetEntries()/GetCumProcTime());
               if (slaveTime * localPreference > avgTime)
                  openLocal = kTRUE;
               else if ((firstNonLocalNode->GetEventsLeftPerSlave())
                        < slstat->GetLocalEventsLeft() * localPreference)
                  openLocal = kTRUE;
            }
         }
         if (openLocal || fStrategy == 0) {
            // Try its own node
            file = slstat->GetFileNode()->GetNextUnAlloc();
            if (!file)
               file = slstat->GetFileNode()->GetNextActive();
            if ( file == 0 ) {
               // No more files on this worker
               slstat->SetFileNode(0);
            }
         }
      }

      // Try to find an unused filenode first
      if(file == 0 && !fForceLocal)
         file = GetNextUnAlloc(0, nodeHostName);

      // Then look at the active filenodes
      if(file == 0 && !fForceLocal)
         file = GetNextActive();

      if (file == 0) return 0;

      PDB(kPacketizer,3) if (fFilesToProcess) fFilesToProcess->Print();

      slstat->fCurFile = file;
      // if remote and unallocated file
      if (file->GetNode()->GetMySlaveCnt() == 0 &&
         file->GetElement()->GetFirst() == file->GetNextEntry()) {
         fNEventsOnRemLoc -= file->GetElement()->GetNum();
         if (fNEventsOnRemLoc < 0) {
            Error("GetNextPacket",
                  "inconsistent value for fNEventsOnRemLoc (%lld): stop delivering packets!",
                   fNEventsOnRemLoc);
            return 0;
         }
      }
      file->GetNode()->IncExtSlaveCnt(slstat->GetName());
      file->GetNode()->IncRunSlaveCnt();
      if (gPerfStats)
         gPerfStats->FileEvent(sl->GetOrdinal(), sl->GetName(),
                               file->GetNode()->GetName(),
                               file->GetElement()->GetFileName(), kTRUE);
   }

   Long64_t num = CalculatePacketSize(slstat, cachesz, learnent);

   // Get a packet

   TDSetElement *base = file->GetElement();
   Long64_t first = file->GetNextEntry();
   Long64_t last = base->GetFirst() + base->GetNum();

   // If the remaining part is smaller than the (packetsize * 1.5)
   // then increase the packetsize

   if ( first + num * 1.5 >= last ) {
      num = last - first;
      file->SetDone(); // done
      // Delete file from active list (unalloc list is single pass, no delete needed)
      RemoveActive(file);
   }

   // Update NextEntry in the file object
   file->MoveNextEntry(num);

   slstat->fCurElem = CreateNewPacket(base, first, num);
   if (base->GetEntryList())
      slstat->fCurElem->SetEntryList(base->GetEntryList(), first, num);

   // Flag the first packet of a new run (dataset)
   if (firstPacket)
      slstat->fCurElem->SetBit(TDSetElement::kNewRun);
   else
      slstat->fCurElem->ResetBit(TDSetElement::kNewRun);

   PDB(kPacketizer,2)
      Info("GetNextPacket","%s: %s %lld %lld (%lld)", sl->GetOrdinal(), base->GetFileName(), first, first + num - 1, num);

   return slstat->fCurElem;
}

//______________________________________________________________________________
Int_t TPacketizerAdaptive::GetActiveWorkers()
{
   // Return the number of workers still processing

   Int_t actw = 0;
   TIter nxw(fSlaveStats);
   TObject *key;
   while ((key = nxw())) {
      TSlaveStat *wrkstat = (TSlaveStat *) fSlaveStats->GetValue(key);
      if (wrkstat && wrkstat->fCurFile) actw++;
   }
   // Done
   return actw;
}

//______________________________________________________________________________
Float_t TPacketizerAdaptive::GetCurrentRate(Bool_t &all)
{
   // Get Estimation of the current rate; just summing the current rates of
   // the active workers

   all = kTRUE;
   // Loop over the workers
   Float_t currate = 0.;
   if (fSlaveStats && fSlaveStats->GetSize() > 0) {
      TIter nxw(fSlaveStats);
      TObject *key;
      while ((key = nxw()) != 0) {
         TSlaveStat *slstat = (TSlaveStat *) fSlaveStats->GetValue(key);
         if (slstat && slstat->GetProgressStatus() && slstat->GetEntriesProcessed() > 0) {
            // Sum-up the current rates
            currate += slstat->GetProgressStatus()->GetCurrentRate();
         } else {
            all = kFALSE;
         }
      }
   }
   // Done
   return currate;
}

//______________________________________________________________________________
Int_t TPacketizerAdaptive::GetEstEntriesProcessed(Float_t t, Long64_t &ent,
                                                  Long64_t &bytes, Long64_t &calls)
{
   // Get estimation for the number of processed entries and bytes read at time t,
   // based on the numbers already processed and the latests worker measured speeds.
   // If t <= 0 the current time is used.
   // Only the estimation for the entries is currently implemented.
   // This is needed to smooth the instantaneous rate plot.

   // Default value
   ent = GetEntriesProcessed();
   bytes = GetBytesRead();
   calls = GetReadCalls();

   // Parse option
   if (fUseEstOpt == kEstOff)
      // Do not use estimation
      return 0;
   Bool_t current = (fUseEstOpt == kEstCurrent) ? kTRUE : kFALSE;

   TTime tnow = gSystem->Now();
   Double_t now = (t > 0) ? (Double_t)t : Long64_t(tnow) / (Double_t)1000.;
   Double_t dt = -1;

   // Loop over the workers
   Bool_t all = kTRUE;
   Float_t trate = 0.;
   if (fSlaveStats && fSlaveStats->GetSize() > 0) {
      ent = 0;
      TIter nxw(fSlaveStats);
      TObject *key;
      while ((key = nxw()) != 0) {
         TSlaveStat *slstat = (TSlaveStat *) fSlaveStats->GetValue(key);
         if (slstat) {
            // Those surely processed
            Long64_t e = slstat->GetEntriesProcessed();
            if (e <= 0) all = kFALSE;
            // Time elapsed since last update
            dt = now - slstat->GetProgressStatus()->GetLastUpdate();
            // Add estimated entries processed since last update
            Float_t rate = (current && slstat->GetCurRate() > 0) ? slstat->GetCurRate()
                                                                 : slstat->GetAvgRate();
            trate += rate;
            // Add estimated entries processed since last update
            e += (Long64_t) (dt * rate);
            // Add to the total
            ent += e;
            // Notify
            PDB(kPacketizer,3)
               Info("GetEstEntriesProcessed","%s: e:%lld rate:%f dt:%f e:%lld",
                                          slstat->fSlave->GetOrdinal(),
                                          slstat->GetEntriesProcessed(), rate, dt, e);
         }
      }
   }
   // Notify
   dt = now - fProgressStatus->GetLastUpdate();
   PDB(kPacketizer,2)
      Info("GetEstEntriesProcessed",
           "dt: %f, estimated entries: %lld (%lld), bytes read: %lld rate: %f (all: %d)",
                               dt, ent, GetEntriesProcessed(), bytes, trate, all);

   // Check values
   ent = (ent > 0) ? ent : fProgressStatus->GetEntries();
   ent = (ent <= fTotalEntries) ? ent : fTotalEntries;
   bytes = (bytes > 0) ? bytes : fProgressStatus->GetBytesRead();

   // Done
   return ((all) ? 0 : 1);
}

//______________________________________________________________________________
void TPacketizerAdaptive::MarkBad(TSlave *s, TProofProgressStatus *status,
                                  TList **listOfMissingFiles)
{
   // This method can be called at any time during processing
   // as an effect of handling kPROOF_STOPPROCESS
   // If the output list from this worker is going to be sent back to the master,
   // the 'status' includes the number of entries processed by the slave.
   // From this we calculate the remaining part of the packet.
   // 0 indicates that the results from that worker were lost completely.
   // Assume that the filenodes for which we have a TFileNode object
   // are still up and running.

   TSlaveStat *slaveStat = (TSlaveStat *)(fSlaveStats->GetValue(s));
   if (!slaveStat) {
      Error("MarkBad", "Worker does not exist");
      return;
   }
   // Update worker counters
   if (slaveStat->fCurFile && slaveStat->fCurFile->GetNode()) {
      slaveStat->fCurFile->GetNode()->DecExtSlaveCnt(slaveStat->GetName());
      slaveStat->fCurFile->GetNode()->DecRunSlaveCnt();
   }

   // If status is defined, the remaining part of the last packet is
   // reassigned in AddProcessed called from handling kPROOF_STOPPROCESS
   if (!status) {
      // Get the subset processed by the bad worker.
      TList *subSet = slaveStat->GetProcessedSubSet();
      if (subSet) {
         // Take care of the current packet
         if (slaveStat->fCurElem) {
            subSet->Add(slaveStat->fCurElem);
         }
         // Merge overlapping or subsequent elements
         Int_t nmg = 0, ntries = 100;
         TDSetElement *e = 0, *enxt = 0;
         do {
            nmg = 0;
            e = (TDSetElement *) subSet->First();
            while ((enxt = (TDSetElement *) subSet->After(e))) {
               if (e->MergeElement(enxt) >= 0) {
                  nmg++;
                  subSet->Remove(enxt);
                  delete enxt;
               } else {
                  e = enxt;
               }
            }
         } while (nmg > 0 && --ntries > 0);
         // reassign the packets assigned to the bad slave and save the size;
         SplitPerHost(subSet, listOfMissingFiles);
         // the elements were reassigned so should not be deleted
         subSet->SetOwner(0);
      } else {
         Warning("MarkBad", "subset processed by bad worker not found!");
      }
      (*fProgressStatus) -= *(slaveStat->GetProgressStatus());
   }
   // remove slavestat from the map
   fSlaveStats->Remove(s);
   delete slaveStat;
   // recalculate fNEventsOnRemLoc and others
   InitStats();
}

//______________________________________________________________________________
Int_t TPacketizerAdaptive::ReassignPacket(TDSetElement *e,
                                          TList **listOfMissingFiles)
{
   // The file in the listOfMissingFiles can appear several times;
   // in order to fix that, a TDSetElement::Merge method is needed.

   if (!e) {
      Error("ReassignPacket", "empty packet!");
      return -1;
   }
   // Check the old filenode
   TUrl url = e->GetFileName();
   // Check the host from which 'e' was previously read.
   // Map non URL filenames to dummy host
   TString host;
   if ( !url.IsValid() ||
       (strncmp(url.GetProtocol(),"root", 4) &&
        strncmp(url.GetProtocol(),"rfio", 4))) {
      host = "no-host";
   } else {
      host = url.GetHost();
   }

   // If accessible add it back to the old node
   // and do DecProcessed
   TFileNode *node = (TFileNode*) fFileNodes->FindObject( host );
   if (node && fTryReassign) {
      // The packet 'e' was processing data from this node.
      node->DecreaseProcessed(e->GetNum());
      // The file should be already in fFilesToProcess ...
      node->Add(e, kFALSE);
      if (!fUnAllocated->FindObject(node))
         fUnAllocated->Add(node);
      return 0;
   } else {
      // Add to the list of missing files
      TFileInfo *fi = e->GetFileInfo();
      if (listOfMissingFiles && *listOfMissingFiles)
         (*listOfMissingFiles)->Add((TObject *)fi);
      return -1;
   }
}

//______________________________________________________________________________
void TPacketizerAdaptive::SplitPerHost(TList *elements,
                                       TList **listOfMissingFiles)
{
   // Split into per host entries
   // The files in the listOfMissingFiles can appear several times;
   // in order to fix that, a TDSetElement::Merge method is needed.

   if (!elements) {
      Error("SplitPerHost", "Empty list of packets!");
      return;
   }
   if (elements->GetSize() <= 0) {
      Error("SplitPerHost", "The input list contains no elements");
      return;
   }
   TIter subSetIter(elements);
   TDSetElement *e;
   while ((e = (TDSetElement*) subSetIter.Next())) {
      if (ReassignPacket(e, listOfMissingFiles) == -1) {
         // Remove from the list in order to delete it.
         if (elements->Remove(e))
            Error("SplitPerHost", "Error removing a missing file");
         delete e;
      }

   }
}
