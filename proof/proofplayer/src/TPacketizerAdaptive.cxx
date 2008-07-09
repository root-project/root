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
#include "TTimer.h"
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
   TFileStat(TFileNode *node, TDSetElement *elem);

   Bool_t         IsDone() const {return fIsDone;}
   void           SetDone() {fIsDone = kTRUE;}
   TFileNode     *GetNode() const {return fNode;}
   TDSetElement  *GetElement() const {return fElement;}
   Long64_t       GetNextEntry() const {return fNextEntry;}
   void           MoveNextEntry(Long64_t step) {fNextEntry += step;}
};


TPacketizerAdaptive::TFileStat::TFileStat(TFileNode *node, TDSetElement *elem)
   : fIsDone(kFALSE), fNode(node), fElement(elem), fNextEntry(elem->GetFirst())
{
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

public:
   TFileNode(const char *name);
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
   // this method is used by Compare() it adds 1, so it returns a number that
   // would be true if one more slave is added.
   Long64_t    GetEventsLeftPerSlave() const
      { return ((fEvents - fProcessed)/(fRunSlaveCnt + 1)); }
   void        IncEvents(Long64_t nEvents) { fEvents += nEvents; }
   const char *GetName() const { return fNodeName.Data(); }
   Long64_t    GetNEvents() const { return fEvents; }

   void Add(TDSetElement *elem)
   {
      TFileStat *f = new TFileStat(this, elem);
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
      R__ASSERT(obj != 0);

      // how many more events it has than obj

      if (TPacketizerAdaptive::fgStrategy == 1) {
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

   void Print(Option_t *) const
   {
      cout << "OBJ: " << IsA()->GetName() << "\t" << fNodeName
           << "\tMySlaveCount " << fMySlaveCnt
           << "\tSlaveCount " << fExtSlaveCnt << endl;
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


TPacketizerAdaptive::TFileNode::TFileNode(const char *name)
   : fNodeName(name), fFiles(new TList), fUnAllocFileNext(0),
     fActFiles(new TList), fActFileNext(0), fMySlaveCnt(0),
     fExtSlaveCnt(0), fProcessed(0), fEvents(0)
{
   // Constructor

   fFiles->SetOwner();
   fActFiles->SetOwner(kFALSE);
}

//------------------------------------------------------------------------------

class TPacketizerAdaptive::TSlaveStat : public TObject {

friend class TPacketizerAdaptive;

private:
   TSlave        *fSlave;        // corresponding TSlave record
   TFileNode     *fFileNode;     // corresponding node or 0
   TFileStat     *fCurFile;      // file currently being processed
   TDSetElement  *fCurElem;      // TDSetElement currently being processed
   Long64_t       fProcessed;    // number of entries processed
   Float_t        fProcTime;     // sum of processing time of packets
   Long64_t       fCurProcessed; // events processed in the current file
   Float_t        fCurProcTime;  // proc time spent on the current file
   TList         *fDSubSet;      // packets processed by this worker

public:
   TSlaveStat(TSlave *slave);
   ~TSlaveStat();
   TFileNode  *GetFileNode() const { return fFileNode; }
   const char *GetName() const { return fSlave->GetName(); }
   Long64_t    GetEntriesProcessed() const { return fProcessed; }
   TFileStat  *GetCurFile() { return fCurFile; }
   void        SetFileNode(TFileNode *node) { fFileNode = node; }
   void        UpdateRates(Long64_t nEvents, Float_t time);
   Float_t     GetAvgRate() { return (fProcTime?fProcessed/fProcTime:0); }
   Float_t     GetCurRate() {
      return (fCurProcTime?fCurProcessed/fCurProcTime:0); }
   Int_t       GetLocalEventsLeft() {
      return fFileNode?(fFileNode->GetEventsLeftPerSlave()):0; }
   TList      *GetProcessedSubSet() { return fDSubSet; }
   Int_t       AddProcessed();   // Add curent packet to the list of processed.
};

//______________________________________________________________________________
TPacketizerAdaptive::TSlaveStat::TSlaveStat(TSlave *slave)
   : fSlave(slave), fFileNode(0), fCurFile(0), fCurElem(0), fProcessed(0),
     fProcTime(0), fCurProcessed(0), fCurProcTime(0)
{
   // Constructor

   fDSubSet = new TList();
   fDSubSet->SetOwner();
}

//______________________________________________________________________________
TPacketizerAdaptive::TSlaveStat::~TSlaveStat()
{
   // Cleanup

   SafeDelete(fDSubSet);
}

//______________________________________________________________________________
void TPacketizerAdaptive::TSlaveStat::UpdateRates(Long64_t nEvents,
                                                  Float_t time)
{
   // Update packetizer rates

   if (fCurFile->IsDone()) {
      fCurProcTime = 0;
      fCurProcessed = 0;
   } else {
      fCurProcTime += time;
      fCurProcessed += nEvents;
   }
   fProcTime += time;
   fProcessed += nEvents;
   fCurFile->GetNode()->IncProcessed(nEvents);
}

//______________________________________________________________________________
Int_t TPacketizerAdaptive::TSlaveStat::AddProcessed()
{
   // Add the current element to the fDSubSet (subset processed by this worker)

   if (fDSubSet && fCurElem) {
      fDSubSet->Add(fCurElem);
      return 0;
   } else
      return -1;
}

//------------------------------------------------------------------------------

ClassImp(TPacketizerAdaptive)

Long_t   TPacketizerAdaptive::fgMaxSlaveCnt = 2;
Int_t    TPacketizerAdaptive::fgPacketAsAFraction = 4;
Double_t TPacketizerAdaptive::fgMinPacketTime = 3;
Int_t    TPacketizerAdaptive::fgStrategy = 1;

//______________________________________________________________________________
TPacketizerAdaptive::TPacketizerAdaptive(TDSet *dset, TList *slaves,
                          Long64_t first, Long64_t num, TList *input)
                    : TVirtualPacketizer(input)
{
   // Constructor

   PDB(kPacketizer,1) Info("TPacketizerAdaptive",
                           "enter (first %lld, num %lld)", first, num);

   // Init pointer members
   fSlaveStats = 0;
   fUnAllocated = 0;
   fActive = 0;
   fFileNodes = 0;
   fCumProcTime = 0;
   fMaxPerfIdx = 1;

   // The possibility to change packetizer strategy to the basic TPacketizer's
   // one (in which workers always process their local data first).
   Int_t strategy = -1;
   if (TProof::GetParameter(input, "PROOF_PacketizerStrategy", strategy) == 0) {
      if (strategy == 0) {
         fgStrategy = 0;
         fgMaxSlaveCnt = 4; // can be overwritten by PROOF_MaxSlavesPerNode
         Info("TPacketizerAdaptive", "using the basic strategy of TPacketizer");
      } else
         Info("TPacketizerAdaptive", "The only accepted value for the"
              " PROOF_PacketizerStrategy parameter is 0!");
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
      fgMaxSlaveCnt = maxSlaveCnt;
      Info("TPacketizerAdaptive", "Setting max number of workers per node to %ld",
           fgMaxSlaveCnt);
   } else {
      // Use number of CPUs (or minimum 2) as default
      SysInfo_t si;
      gSystem->GetSysInfo(&si);
      fgMaxSlaveCnt =  (si.fCpus > 2) ? si.fCpus : 2;
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
   if (TProof::GetParameter(input, "PROOF_PacketAsAFraction",
                            packetAsAFraction) == 0) {
      if (packetAsAFraction > 0) {
         fgPacketAsAFraction = packetAsAFraction;
         Info("TPacketizerAdaptive",
              "using alternate fraction of query time as a packet size: %ld",
              packetAsAFraction);
      } else
         Info("TPacketizerAdaptive", "packetAsAFraction parameter must be higher than 0");
   }
   Double_t minPacketTime = 0;
   if (TProof::GetParameter(input, "PROOF_MinPacketTime",
                            minPacketTime) == 0) {
      Info("TPacketizerAdaptive", "using alternate minimum time of a packet: %f",
           minPacketTime);
      fgMinPacketTime = (Int_t) minPacketTime;
   }

   Double_t baseLocalPreference = 1.2;
   TProof::GetParameter(input, "PROOF_BaseLocalPreference", baseLocalPreference);
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

   // Split into per host entries
   dset->Reset();
   TDSetElement *e;
   while ((e = (TDSetElement*)dset->Next())) {
      if (e->GetValid()) continue;

      TUrl url = e->GetFileName();

      // Map non URL filenames to dummy host
      TString host;
      if ( !url.IsValid() ||
          (strncmp(url.GetProtocol(),"root", 4) &&
           strncmp(url.GetProtocol(),"rfio", 4)) ) {
         host = "no-host";
      } else {
         host = url.GetHost();
      }

      TFileNode *node = (TFileNode*) fFileNodes->FindObject( host );

      if (node == 0) {
         node = new TFileNode(host);
         fFileNodes->Add(node);
      }

      node->Add( e );
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

   Reset();                // setup file & filenode structure
   ValidateFiles(dset, slaves);


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
           "processing Range: First %lld, Num %lld", first, num);

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
         Info("TPacketizerAdaptive",
              "processing element: First %lld, Num %lld (cur %lld)", eFirst, eNum, cur);

      if (!e->GetEntryList()) {
         // this element is before the start of the global range, skip it
         if (cur + eNum < first) {
            cur += eNum;
            PDB(kPacketizer,2)
               Info("TPacketizerAdaptive",
                    "processing element: skip element cur %lld", cur);
            continue;
         }

         // this element is after the end of the global range, skip it
         if (num != -1 && (first+num <= cur)) {
            cur += eNum;
            PDB(kPacketizer,2)
               Info("TPacketizerAdaptive",
                    "processing element: drop element cur %lld", cur);
            continue; // break ??
         }

         // If this element contains the end of the global range
         // adjust its number of entries
         if (num != -1 && (first+num < cur+eNum)) {
            e->SetNum( first + num - cur );
            PDB(kPacketizer,2)
               Info("TPacketizerAdaptive",
                    "processing element: Adjust end %lld", first + num - cur);
         }

         // If this element contains the start of the global range
         // adjust its start and number of entries
         if (cur < first) {
            e->SetFirst( eFirst + (first - cur) );
            e->SetNum( e->GetNum() - (first - cur) );
            PDB(kPacketizer,2)
               Info("TPacketizerAdaptive",
                    "processing element: Adjust start %lld and end %lld",
                    eFirst + (first - cur), first + num - cur);
         }

         cur += eNum;
      } else {
         Long64_t n = 0;
         TEntryList *enl = dynamic_cast<TEntryList *>(e->GetEntryList());
         if (enl) {
            n = enl->GetN();
         } else {
            TEventList *evl = dynamic_cast<TEventList *>(e->GetEntryList());
            n = evl ? evl->GetN() : n;
         }
         if (!n)
            continue;
      }
      PDB(kPacketizer,2)
         Info("TPacketizerAdaptive",
              "processing element: next cur %lld", cur);

      // Map non URL filenames to dummy host
      TString host;
      if ( !url.IsValid() ||
          (strncmp(url.GetProtocol(),"root", 4) &&
           strncmp(url.GetProtocol(),"rfio", 4)) ) {
         host = "no-host";
      } else {
         host = url.GetHost();
      }

      TFileNode *node = (TFileNode*) fFileNodes->FindObject( host );

      if ( node == 0 ) {
         node = new TFileNode( host );
         fFileNodes->Add( node );
      }

      ++files;
      fTotalEntries += e->GetNum();
      node->Add( e );
      node->IncEvents(e->GetNum());
      PDB(kPacketizer,2) e->Print("a");
   }
   // it overwrites previous value!!
   TEntryList *enl = dynamic_cast<TEntryList *>(dset->GetEntryList());
   if (enl) {
      fTotalEntries = enl->GetN();
   } else {
      TEventList *evl = dynamic_cast<TEventList *>(dset->GetEntryList());
      fTotalEntries = evl ? evl->GetN() : fTotalEntries;
   }
   PDB(kGlobal,1)
      Info("TPacketizerAdaptive", "processing %lld entries in %d files on %d hosts",
                                  fTotalEntries, files, fFileNodes->GetSize());
   Reset();

   // calculating how many files from TDSet are not cached on
   // any slave
   Int_t noRemoteFiles = 0;
   fNEventsOnRemLoc = 0;
   Int_t totalNumberOfFiles = 0;
   TIter next(fFileNodes);
   while (TFileNode *fn = (TFileNode*)next()) {
      totalNumberOfFiles += fn->GetNumberOfFiles();
      if (fn->GetSlaveCnt() == 0) {
         noRemoteFiles += fn->GetNumberOfFiles();
         fNEventsOnRemLoc += (fn->GetNEvents() - fn->GetProcessed());
      }
   }

   if (totalNumberOfFiles == 0) {
      Info("TPacketizerAdaptive", "no valid or non-empty file found: setting invalid");
      // No valid files: set invalid and return
      fValid = kFALSE;
      return;
   }

   fFractionOfRemoteFiles = noRemoteFiles / totalNumberOfFiles;
   Info("TPacketizerAdaptive",
        "fraction of remote files %f", fFractionOfRemoteFiles);

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
}

//______________________________________________________________________________
TPacketizerAdaptive::TFileStat *TPacketizerAdaptive::GetNextUnAlloc(TFileNode *node)
{
   // Get next unallocated file from 'node' or other nodes:
   // First try 'node'. If there is no more files, keep trying to
   // find an unallocated file on other nodes.

   TFileStat *file = 0;

   if (node != 0) {
      file = node->GetNextUnAlloc();
      if (file == 0) RemoveUnAllocNode(node);
   } else {
      while (file == 0 && ((node = NextNode()) != 0)) {
         file = node->GetNextUnAlloc();
         if (file == 0) RemoveUnAllocNode(node);
      }
   }

   if (file != 0) {
      // if needed make node active
      if (fActive->FindObject(node) == 0) {
         fActive->Add(node);
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
   if (fn != 0 && fn->GetExtSlaveCnt() >= fgMaxSlaveCnt) {
      // unlike in TPacketizer we look at the number of ext slaves only.
      PDB(kPacketizer,1) Info("NextNode",
                              "Reached Slaves per Node Limit (%ld)", fgMaxSlaveCnt);
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
   if (fn != 0 && fn->GetExtSlaveCnt() >= fgMaxSlaveCnt) {
      PDB(kPacketizer,1)
         Info("NextActiveNode","reached Workers-per-Node limit (%ld)", fgMaxSlaveCnt);
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
   // Reset the internal datastructure for packet distribution.

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
      fn = (TFileNode*) fFileNodes->FindObject(slstat->GetName());
      if (fn != 0 ) {
         slstat->SetFileNode(fn);
         fn->IncMySlaveCnt();
      }
      slstat->fCurFile = 0;
   }
}

//______________________________________________________________________________
void TPacketizerAdaptive::ValidateFiles(TDSet *dset, TList *slaves)
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
      PDB(kPacketizer,3) Info("ValidateFiles","socket added to monitor: %p (%s)",
          slm->GetSocket(), slm->GetName());
      mon.Add(slm->GetSocket());
      slaves_by_sock.Add(slm->GetSocket(), slm);
   }

   mon.DeActivateAll();

   ((TProof*)gProof)->DeActivateAsyncInput();

   // Some monitoring systems (TXSocketHandler) need to know this
   ((TProof*)gProof)->fCurrentMonitor = &mon;

   // Preparing for client notification
   TString msg("Validating files");
   UInt_t n = 0;
   UInt_t tot = dset->GetListOfElements()->GetSize();
   Bool_t st = kTRUE;

   while (kTRUE) {

      // send work
      while (TSlave *s = (TSlave *)workers.First()) {

         workers.Remove(s);

         // find a file

         TSlaveStat *slstat = (TSlaveStat*)fSlaveStats->GetValue(s);
         TFileNode *node = 0;
         TFileStat *file = 0;

         // try its own node first
         if ((node = slstat->GetFileNode()) != 0) {
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
                  if (!elem->GetEntryList()) {
                     if (elem->GetFirst() > entries) {
                        Error("ValidateFiles",
                              "first (%d) higher then number of entries (%d) in %d",
                              elem->GetFirst(), entries, elem->GetFileName() );
                        // disable element
                        slstat->fCurFile->SetDone();
                        elem->Invalidate();
                        dset->SetBit(TDSet::kSomeInvalid);
                     }
                     if (elem->GetNum() == -1) {
                        elem->SetNum(entries - elem->GetFirst());
                     } else if (elem->GetFirst() + elem->GetNum() > entries) {
                        Warning("ValidateFiles", "Num (%lld) + First (%lld) larger then number of"
                                 " keys/entries (%lld) in %s", elem->GetNum(), elem->GetFirst(),
                                 entries, elem->GetFileName());
                        elem->SetNum(entries - elem->GetFirst());
                     }
                     PDB(kPacketizer,2)
                        Info("ValidateFiles",
                             "found elem '%s' with %lld entries", elem->GetFileName(), entries);
                     elem->SetValid();
                  }
               }
               // Notify the client
               n++;
               gProof->SendDataSetStatus(msg, n, tot, st);

               // This worker is ready for the next validation
               workers.Add(s);
            }
         }
      }

      // Check if there is anything to wait for
      if (mon.GetActive() == 0) break;

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
         // Help! lost a slave?
         ((TProof*)gProof)->MarkBad(slave, "receive failed during validation");
         fValid = kFALSE;
         Error("ValidateFiles", "Recv failed! for worker-%s (%s)",
               slave->GetOrdinal(), slave->GetName());
         continue;
         }

      if (reply->What() == kPROOF_FATAL) {
         Error("ValidateFiles", "kPROOF_FATAL from worker-%s (%s)",
               slave->GetOrdinal(), slave->GetName());
         ((TProof*)gProof)->MarkBad(slave, "received kPROOF_FATAL during validation");
         fValid = kFALSE;
         continue;
      } else if (reply->What() == kPROOF_LOGFILE) {
         PDB(kPacketizer,3) Info("ValidateFiles", "got logfile");
         Int_t size;
         (*reply) >> size;
         ((TProof*)gProof)->RecvLogFile(sock, size);
         mon.Activate(sock);
         continue;
      } else if (reply->What() == kPROOF_LOGDONE) {
         PDB(kPacketizer,3) Info("ValidateFiles", "got logdone");
         mon.Activate(sock);
         continue;
      } else if (reply->What() == kPROOF_TOUCH) {
         PDB(kPacketizer,3) Info("ValidateFiles", "got logdone");
         slave->Touch();
         mon.Activate(sock);
         continue;
      } else if ( reply->What() == kPROOF_MESSAGE ) {
         // Send one level up
         TString s;
         (*reply) >> s;
         Bool_t lfeed = kTRUE;
         if ((reply->BufferSize() > reply->Length()))
            (*reply) >> lfeed;
         TMessage m(kPROOF_MESSAGE);
         m << s << lfeed;
         gProofServ->GetSocket()->Send(m);
         mon.Activate(sock);
         continue;
      } else if (reply->What() != kPROOF_GETENTRIES) {
         // Help! unexpected message type
         Error("ValidateFiles",
               "unexpected message type (%d) from worker-%s (%s)",
               reply->What(), slave->GetOrdinal(), slave->GetName());
         ((TProof*)gProof)->MarkBad(slave, "unexpected message type during validation");
         fValid = kFALSE;
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

         if (!e->GetEntryList()) {
            if (e->GetFirst() > entries) {
               Error("ValidateFiles",
                     "first (%d) higher then number of entries (%d) in %d",
                     e->GetFirst(), entries, e->GetFileName() );

               // Invalidate the element
               slavestat->fCurFile->SetDone();
               e->Invalidate();
               dset->SetBit(TDSet::kSomeInvalid);
            }

            if (e->GetNum() == -1) {
               e->SetNum(entries - e->GetFirst());
            } else if (e->GetFirst() + e->GetNum() > entries) {
               Error("ValidateFiles",
                     "Num (%d) + First (%d) larger then number of keys/entries (%d) in %s",
                     e->GetNum(), e->GetFirst(), entries, e->GetFileName() );
               e->SetNum(entries - e->GetFirst());
            }
            e->SetValid();
         }


         // Notify the client
         n++;
         gProof->SendDataSetStatus(msg, n, tot, st);

      } else {

         Error("ValidateFiles", "cannot get entries for %s (", e->GetFileName() );
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

      workers.Add(slave); // Ready for the next job
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
Long64_t TPacketizerAdaptive::GetEntriesProcessed(TSlave *slave) const
{
   // Get entries processed by the specified slave.

   if ( fSlaveStats == 0 ) return 0;

   TSlaveStat *slstat = (TSlaveStat*) fSlaveStats->GetValue( slave );

   if ( slstat == 0 ) return 0;

   return slstat->GetEntriesProcessed();
}

//______________________________________________________________________________
Int_t TPacketizerAdaptive::CalculatePacketSize(TObject *slStatPtr)
{
   // The result depends on the fgStrategy

   Long64_t num;
   if (fgStrategy == 0) {
      // TPacketizer's heuristic for starting packet size
      // Constant packet size;
      Int_t nslaves = fSlaveStats->GetSize();
      if (nslaves > 0) {
         num = fTotalEntries / (fgPacketAsAFraction * nslaves);
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
         Float_t avgProcRate = (fProcessed/(fCumProcTime / fSlaveStats->GetSize()));
         Float_t packetTime;
         packetTime = ((fTotalEntries - fProcessed)/avgProcRate)/fgPacketAsAFraction;
         if (packetTime < fgMinPacketTime)
            packetTime = fgMinPacketTime;
         // in case the worker has suddenly slowed down
         if (rate < 0.25 * slstat->GetAvgRate())
            rate = (rate + slstat->GetAvgRate()) / 2;
         num = (Long64_t)(rate * packetTime);
      } else { //first packet for this slave in this query
         Int_t packetSize = (fTotalEntries - fProcessed)
                            / (6 * fgPacketAsAFraction * fSlaveStats->GetSize());
         num = Long64_t(packetSize *
               ((Float_t)slstat->fSlave->GetPerfIdx() / fMaxPerfIdx));
      }
   }
   if (num < 1) num = 1;
   return num;
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
   // update stats & free old element

   if ( slstat->fCurElem != 0 ) {
      Double_t latency, proctime, proccpu;
      Long64_t bytesRead = -1;
      Long64_t totalEntries = -1;

      Long64_t expectedNumEv = slstat->fCurElem->GetNum();

      (*r) >> latency >> proctime >> proccpu;
      // only read new info if available
      if (r->BufferSize() > r->Length()) (*r) >> bytesRead;
      if (r->BufferSize() > r->Length()) (*r) >> totalEntries;
      Long64_t totev = 0;
      if (r->BufferSize() > r->Length()) (*r) >> totev;

      // Calculate the number of events processed in the last packet
      Long64_t numev;
      if (totev > 0)
         numev = totev - slstat->fProcessed;
      else
         numev = 0;

      if (numev == expectedNumEv) {
         // The last packet was sucessfully processed
         slstat->AddProcessed();
         fProcessed += ((numev > 0) ? numev : 0);
         fBytesRead += ((bytesRead > 0) ? bytesRead : 0);

         // update processing rate
         slstat->UpdateRates(numev, proctime);

         fCumProcTime += proctime;

      } else {
         // The last packet was not processed properly.
         // Add it to the failed packets list.

         if (!fFailedPackets) {
            fFailedPackets = new TList();
         }
         fFailedPackets->Add(slstat->fCurElem);
      }

      PDB(kPacketizer,2)
         Info("GetNextPacket","worker-%s (%s): %lld %7.3lf %7.3lf %7.3lf %lld",
              sl->GetOrdinal(), sl->GetName(),
              numev, latency, proctime, proccpu, bytesRead);

      if (gPerfStats != 0) {
         gPerfStats->PacketEvent(sl->GetOrdinal(), sl->GetName(), slstat->fCurElem->GetFileName(),
                                 numev, latency, proctime, proccpu, bytesRead);
      }

      slstat->fCurElem = 0;
      if ( fProcessed == fTotalEntries ) {
         HandleTimer(0);   // Send last timer message
         delete fProgress; fProgress = 0;
      }

   }

   if ( fStop ) {
      HandleTimer(0);
      return 0;
   }

   TFileStat *file = slstat->fCurFile;
   // if current file is just finished
   if ( file != 0 && file->IsDone() ) {
      file->GetNode()->DecExtSlaveCnt(slstat->GetName());
      file->GetNode()->DecRunSlaveCnt();
      if (gPerfStats != 0) {
         gPerfStats->FileEvent(sl->GetOrdinal(), sl->GetName(), file->GetNode()->GetName(),
                               file->GetElement()->GetFileName(), kFALSE);
      }
      file = 0;
   }

   Long64_t avgEventsLeftPerSlave =
      (fTotalEntries - fProcessed) / fSlaveStats->GetSize();
   if (fTotalEntries == fProcessed)
      return 0;
   // get a file if needed
   if ( file == 0) {
      // needs a new file
      Bool_t openLocal;
      // aiming for localPreference == 1 when #local == #remote events left
      Float_t localPreference = fBaseLocalPreference - (fNEventsOnRemLoc /
                                (0.4 *(fTotalEntries - fProcessed)));
      if ( slstat->GetFileNode() != 0 ) {
         // local file node exists and has more events to process.
         fUnAllocated->Sort();
         TFileNode* firstNonLocalNode = (TFileNode*)fUnAllocated->First();
         Bool_t nonLocalNodePossible;
         if (fForceLocal)
            nonLocalNodePossible = 0;
         else
            nonLocalNodePossible = firstNonLocalNode?
               (firstNonLocalNode->GetExtSlaveCnt() < fgMaxSlaveCnt):0;
         openLocal = !nonLocalNodePossible;
         Float_t slaveRate = slstat->GetAvgRate();
         if ( nonLocalNodePossible && fgStrategy == 1) {
            // openLocal is set to kFALSE
            if ( slstat->GetFileNode()->GetRunSlaveCnt() >
                 slstat->GetFileNode()->GetMySlaveCnt() - 1 )
                // external slaves help slstat -> don't open nonlocal files
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
               // at this point slstat has a non zero avg rate > 0
               Float_t slaveTime = slstat->GetLocalEventsLeft()/slaveRate;
               // and thus fCumProcTime, fProcessed > 0
               Float_t avgTime = avgEventsLeftPerSlave
                                 /(fProcessed/fCumProcTime);
               if (slaveTime * localPreference > avgTime)
                  openLocal = kTRUE;
               else if ((firstNonLocalNode->GetEventsLeftPerSlave())
                        < slstat->GetLocalEventsLeft() * localPreference)
                  openLocal = kTRUE;
            }
         }
         if (openLocal || fgStrategy == 0) {
            // Try its own node
            file = slstat->GetFileNode()->GetNextUnAlloc();
            if (!file)
               file = slstat->GetFileNode()->GetNextActive();
            if ( file == 0 ) {
               //no more files on this slave.
               slstat->SetFileNode(0);
            }
         }
      }

      // try to find an unused filenode first
      if(file == 0 && !fForceLocal) {
         file = GetNextUnAlloc();
      }

      // then look at the active filenodes
      if(file == 0 && !fForceLocal) {
         file = GetNextActive();
      }

      if ( file == 0 ) return 0;

      slstat->fCurFile = file;
      // if remote and unallocated file
      if (file->GetNode()->GetMySlaveCnt() == 0 &&
         file->GetElement()->GetFirst() == file->GetNextEntry()) {
         fNEventsOnRemLoc -= file->GetElement()->GetNum();
         if (fNEventsOnRemLoc < 0) {
            Error("GetNextPacket",
                  "inconsistent value for fNEventsOnRemLoc (%d): stop delivering packets!",
                  fNEventsOnRemLoc);
            return 0;
         }
      }
      file->GetNode()->IncExtSlaveCnt(slstat->GetName());
      file->GetNode()->IncRunSlaveCnt();
      if (gPerfStats != 0) {
         gPerfStats->FileEvent(sl->GetOrdinal(), sl->GetName(),
                               file->GetNode()->GetName(),
                               file->GetElement()->GetFileName(), kTRUE);
      }
   }

   Long64_t num = CalculatePacketSize(slstat);

   // get a packet

   TDSetElement *base = file->GetElement();
   Long64_t first = file->GetNextEntry();
   Long64_t last = base->GetFirst() + base->GetNum();

   // if the remaining part is smaller than the (packetsize * 1.5)
   // then increase the packetsize

   if ( first + num * 1.5 >= last ) {
      num = last - first;
      file->SetDone(); // done

      // delete file from active list (unalloc list is single pass, no delete needed)
      RemoveActive(file);

   } else {
      file->MoveNextEntry(num);
   }

   slstat->fCurElem = CreateNewPacket(base, first, num);

   if (base->GetEntryList())
      slstat->fCurElem->SetEntryList(base->GetEntryList(), first, num);

   return slstat->fCurElem;
}

//______________________________________________________________________________
Int_t TPacketizerAdaptive::GetEstEntriesProcessed(Float_t t,
                                                  Long64_t &ent, Long64_t &bytes)
{
   // Get estimation for the number of processed entries and bytes read at time t,
   // based on the numbers already processed and the latests worker measured speeds.
   // Only the estimation for the entries is currently implemented.
   // This is needed to smooth the instantaneous rate plot.

   // Default value
   ent = fProcessed;
   bytes = fBytesRead;

   // Parse option
   if (fUseEstOpt == kEstOff)
      // Do not use estimation
      return 0;
   Bool_t current = (fUseEstOpt == kEstCurrent) ? kTRUE : kFALSE;

   // Loop over the workers
   Float_t trate = 0.;
   if (fSlaveStats && fSlaveStats->GetSize() > 0) {
      ent = 0;
      TIter nxw(fSlaveStats);
      TObject *key;
      while ((key = nxw()) != 0) {
         TSlaveStat *slstat = (TSlaveStat *) fSlaveStats->GetValue(key);
         if (slstat) {
            // Those surely processed
            Long64_t e = slstat->fProcessed;
            // Time elapsed since last update
            Float_t dt = (t > slstat->fProcTime) ? t - slstat->fProcTime : 0;
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
                                          slstat->fProcessed, rate, dt, e);
         }
      }
   }
   // Notify
   PDB(kPacketizer,2)
      Info("GetEstEntriesProcessed",
           "estimated entries: %lld, bytes read: %lld rate: %f", ent, bytes, trate);

   // Check values
   ent = (ent > 0) ? ent : fProcessed;
   ent = (ent <= fTotalEntries) ? ent : fTotalEntries;
   bytes = (bytes > 0) ? bytes : fBytesRead;

   // Done
   return 0;
}
