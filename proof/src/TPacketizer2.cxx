// @(#)root/proof:$Name:  $:$Id: TPacketizer2.cxx,v 1.25 2004/06/13 16:26:35 rdm Exp $
// Author: Maarten Ballintijn    18/03/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPacketizer2                                                         //
//                                                                      //
// This class generates packets to be processed on PROOF slave servers. //
// A packet is an event range (begin entry and number of entries) or    //
// object range (first object and number of objects) in a TTree         //
// (entries) or a directory (objects) in a file.                        //
// Packets are generated taking into account the performance of the     //
// remote machine, the time it took to process a previous packet on     //
// the remote machine, the locality of the database files, etc.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TPacketizer2.h"

#include "TObject.h"
#include "TSlave.h"
#include "TMap.h"
#include "TMessage.h"
#include "TMonitor.h"
#include "TSocket.h"
#include "TDSet.h"
#include "TUrl.h"
#include "TError.h"
#include "TProof.h"
#include "TProofDebug.h"
#include "TTimer.h"
#include "TProofServ.h"
#include "TProofPlayer.h"
#include "TPerfStats.h"
#include "Riostream.h"


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

class TPacketizer2::TFileStat : public TObject {

private:
   Bool_t         fIsDone;       // is this element processed
   TFileNode     *fNode;         // my FileNode
   TDSetElement  *fElement;      // location of the file and its range
   Long64_t       fNextEntry;    // cursor in the range, -1 when done

public:
   TFileStat(TFileNode *node, TDSetElement *elem);

   Bool_t         IsDone() const {return fIsDone;}
   void           SetDone() {fIsDone = kTRUE;}
   TFileNode     *GetNode() const {return fNode;}
   TDSetElement  *GetElement() const {return fElement;}
   Long64_t       GetNextEntry() const {return fNextEntry;}
   void           MoveNextEntry(Long64_t step) {fNextEntry += step;}
};


TPacketizer2::TFileStat::TFileStat(TFileNode *node, TDSetElement *elem)
   : fIsDone(kFALSE), fNode(node), fElement(elem), fNextEntry(elem->GetFirst())
{
}


//------------------------------------------------------------------------------

class TPacketizer2::TFileNode : public TObject {

private:
   TString        fNodeName;        // FQDN of the node
   TList         *fFiles;           // TDSetElements (files) stored on this node
   TObject       *fUnAllocFileNext; // cursor in fFiles
   TList         *fActFiles;        // files with work remaining
   TObject       *fActFileNext;     // cursor in fActFiles
   Int_t          fMySlaveCnt;      // number of slaves running on this node
   Int_t          fSlaveCnt;        // number of external slaves processing files on this node

public:
   TFileNode(const char *name);
   ~TFileNode() { delete fFiles; delete fActFiles; }

   void        IncMySlaveCnt() { fSlaveCnt++; }
   void        IncSlaveCnt(const char *slave) { if (fNodeName != slave) fSlaveCnt++; }
   void        DecSlaveCnt(const char *slave) { if (fNodeName != slave) fSlaveCnt--; }
   Int_t       GetNumberOfActiveFiles() const { return fActFiles->GetSize(); }
   Bool_t      IsSortable() const { return kTRUE; }

   const char *GetName() const { return fNodeName.Data(); }

   void Add(TDSetElement *elem)
   {
      TFileStat *f = new TFileStat(this,elem);
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
      const TFileNode *obj = dynamic_cast<const TFileNode*>(other);
      Assert(obj != 0);

      Int_t myVal = fMySlaveCnt + fSlaveCnt;
      Int_t otherVal = obj->fMySlaveCnt + obj->fSlaveCnt;
      if (myVal < otherVal) {
         return -1;
      } else if (myVal > otherVal) {
         return 1;
      } else {
         return 0;
      }
   }

   void Print(Option_t *) const
   {
      cout << "OBJ: " << IsA()->GetName() << "\t" << fNodeName
           << "\tMySlaveCount " << fMySlaveCnt
           << "\tSlaveCount " << fSlaveCnt << endl;
   }

   void Reset()
   {
      fUnAllocFileNext = fFiles->First();
      fActFiles->Clear();
      fActFileNext = 0;
      fMySlaveCnt = 0;
   }
};


TPacketizer2::TFileNode::TFileNode(const char *name)
   : fNodeName(name), fFiles(new TList), fUnAllocFileNext(0),fActFiles(new TList),
     fActFileNext(0), fMySlaveCnt(0), fSlaveCnt(0)
{
   fFiles->SetOwner();
   fActFiles->SetOwner(kFALSE);
}


//------------------------------------------------------------------------------

class TPacketizer2::TSlaveStat : public TObject {

friend class TPacketizer2;

private:
   TSlave        *fSlave;        // corresponding TSlave record
   TFileNode     *fFileNode;     // corresponding node or 0
   TFileStat     *fCurFile;      // file currently being processed
   TDSetElement  *fCurElem;      // TDSetElement currently being processed
   Long64_t       fProcessed;    // number of entries processed

public:
   TSlaveStat(TSlave *slave);

   TFileNode  *GetFileNode() const { return fFileNode; }
   const char *GetName() const { return fSlave->GetName(); }
   Long64_t    GetEntriesProcessed() const { return fProcessed; }

   void        SetFileNode(TFileNode *node) { fFileNode = node; }
};


TPacketizer2::TSlaveStat::TSlaveStat(TSlave *slave)
   : fSlave(slave), fFileNode(0), fCurFile(0), fCurElem(0), fProcessed(0)
{
}


//------------------------------------------------------------------------------

ClassImp(TPacketizer2)


//______________________________________________________________________________
TPacketizer2::TPacketizer2(TDSet *dset, TList *slaves, Long64_t first, Long64_t num)
{
   PDB(kPacketizer,1) Info("TPacketizer2", "Enter");

   fProcessed = 0;
   fMaxPerfIdx = 1;

   fPackets = new TList;
   fPackets->SetOwner();

   fFileNodes = new TList;
   fFileNodes->SetOwner();
   fUnAllocated = new TList;
   fUnAllocated->SetOwner(kFALSE);
   fActive = new TList;
   fActive->SetOwner(kFALSE);


   fValid = kTRUE;

   // Split into per host entries
   dset->Reset();
   TDSetElement *e;
   while ((e = (TDSetElement*)dset->Next())) {
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


   // apply global range (first,num) to dset and rebuild structure
   // ommitting TDSet elements that are not needed

   Int_t files = 0;
   fTotalEntries = 0;
   fUnAllocated->Clear();  // avoid dangling pointers
   fActive->Clear();
   fFileNodes->Clear();    // then delete all objects

   dset->Reset();
   Long64_t cur = 0;
   while (( e = (TDSetElement*)dset->Next())) {
      TUrl url = e->GetFileName();

      // this element is before the start of the global range, skip it
      if (cur + e->GetNum() < first) {
         cur += e->GetNum();
         continue;
      }

      // this element contains the start of the global range
      // adjust its start and number of entries
      if (cur < first) {
         e->SetFirst( e->GetFirst() + (first - cur) );
         e->SetNum( e->GetNum() - (first + cur) );
      }

      // this element is after the end of the global range, skip it
      if (num != -1 && (first+num <= cur)) {
         cur += e->GetNum();
         continue;
      }

      // this element contains the end of the global range
      // adjust its number of entries
      if ( num != -1 && ( first+num < cur + e->GetNum() ) ) {
         e->SetNum( first + num - cur );
      }

      cur += e->GetNum();

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
   }

   PDB(kGlobal,1) Info("TPacketizer2","Processing %lld entries in %d files on %d hosts",
                       fTotalEntries, files, fFileNodes->GetSize());

   Reset();


   // Heuristic for starting packet size
   fPacketSize = fTotalEntries / (20 * fSlaveStats->GetSize());
   if ( fPacketSize < 1 ) fPacketSize = 1;
   PDB(kPacketizer,1) Info("TPacketizer2", "Base Packetsize = %lld", fPacketSize);

   if ( fValid ) {
      fProgress = new TTimer;
      fProgress->SetObject(this);
      fProgress->Start(500,kFALSE);
   }

   PDB(kPacketizer,1) Info("TPacketizer2", "Return");
}


//______________________________________________________________________________
TPacketizer2::~TPacketizer2()
{
   if (fSlaveStats) {
      fSlaveStats->DeleteValues();
   }

   delete fPackets;
   delete fSlaveStats;
   delete fUnAllocated;
   delete fActive;
   delete fFileNodes;
   delete fProgress;
}

//______________________________________________________________________________
TPacketizer2::TFileStat *TPacketizer2::GetNextUnAlloc(TFileNode *node)
{
   TFileStat *file = 0;

   if (node != 0) {
      file = node->GetNextUnAlloc();
      if (file == 0) RemoveUnAllocNode(node);
   } else {
      while (file == 0 && ((node = NextUnAllocNode()) != 0)) {
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
TPacketizer2::TFileNode *TPacketizer2::NextUnAllocNode()
{
   fUnAllocated->Sort();
   PDB(kPacketizer,2) {
      cout << "TPacketizer2::NextUnAllocNode()" << endl;
      fUnAllocated->Print();
   }
   return (TFileNode*) fUnAllocated->First();
}


//______________________________________________________________________________
void TPacketizer2::RemoveUnAllocNode(TFileNode * node)
{
   fUnAllocated->Remove(node);
}


//______________________________________________________________________________
TPacketizer2::TFileStat *TPacketizer2::GetNextActive()
{
   TFileNode *node;
   TFileStat *file = 0;

   while (file == 0 && ((node = NextActiveNode()) != 0)) {
         file = node->GetNextActive();
         if (file == 0) RemoveActiveNode(node);
   }

   return file;
}


//______________________________________________________________________________
TPacketizer2::TFileNode *TPacketizer2::NextActiveNode()
{
   fActive->Sort();
   PDB(kPacketizer,2) {
      cout << "TPacketizer2::NextActiveNode()" << endl;
      fActive->Print();
   }
   return (TFileNode*) fActive->First();
}


//______________________________________________________________________________
void TPacketizer2::RemoveActive(TFileStat *file)
{
   TFileNode *node = file->GetNode();

   node->RemoveActive(file);
   if (node->GetNumberOfActiveFiles() == 0) RemoveActiveNode(node);
}

//______________________________________________________________________________
void TPacketizer2::RemoveActiveNode(TFileNode *node)
{
   fActive->Remove(node);
}


//______________________________________________________________________________
void TPacketizer2::Reset()
{
   // Reset the internal datastructure for packet distribution

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
      TFileNode *fn = (TFileNode*) fFileNodes->FindObject(slstat->GetName());
      if (fn != 0 ) {
         slstat->SetFileNode(fn);
         fn->IncMySlaveCnt();
      }
      slstat->fCurFile = 0;
   }
}


//______________________________________________________________________________
void TPacketizer2::ValidateFiles(TDSet *dset, TList *slaves)
{
   // Check existence of file/dir/tree an get number of entries
   // Assumes the files have been setup

   TMap     slaves_by_sock;
   TMonitor mon;
   TList    workers;


   // Setup the communication infrastructure

   workers.AddAll(slaves);
   TIter    si(slaves);
   TSlave   *slave;
   while ((slave = (TSlave*)si.Next()) != 0) {
      PDB(kPacketizer,3) Info("TPacketizer2","Socket added to monitor: %p (%s)",
          slave->GetSocket(), slave->GetName());
      mon.Add(slave->GetSocket());
      slaves_by_sock.Add(slave->GetSocket(),slave);
   }

   mon.DeActivateAll();

   ((TProof*)gProof)->DeActivateAsyncInput();

   Bool_t done = kFALSE;
   while (kTRUE) {

      // send work
      while( TSlave *s = (TSlave*)workers.First() ) {

         workers.Remove(s);

         // find a file

         TSlaveStat *slstat = (TSlaveStat*)fSlaveStats->GetValue(s);
         TFileNode *node = 0;
         TFileStat *file = 0;

         // try its own node first
         if ( (node = slstat->GetFileNode()) != 0 ) {
            file = GetNextUnAlloc(node);
            if ( file == 0 ) {
               slstat->SetFileNode(0);
            }
         }

         // look for a file on any other node if necessary
         if (file == 0) {
            file = GetNextUnAlloc();
         }

         if ( file != 0 ) {
            // files are done right away
            RemoveActive(file);

            slstat->fCurFile = file;
            file->GetNode()->IncSlaveCnt(slstat->GetName());
            TMessage m(kPROOF_REPORTSIZE);
            TDSetElement *elem = file->GetElement();
            m << dset->IsTree()
              << TString(elem->GetFileName())
              << TString(elem->GetDirectory())
              << TString(elem->GetObjName());

            s->GetSocket()->Send( m );
            mon.Activate(s->GetSocket());
            PDB(kPacketizer,2) Info("TPacketizer2","sent to slave-%d (%s) via %p reportsize on %s %s %s %s",
                s->GetOrdinal(), s->GetName(), s->GetSocket(), dset->IsTree() ? "tree" : "objects",
                elem->GetFileName(), elem->GetDirectory(), elem->GetObjName());
         } else {
            // Done
            done = kTRUE;
            workers.Clear();
         }
      }

      if ( mon.GetActive() == 0 ) break; // nothing to wait for anymore

      PDB(kPacketizer,3) {
         Info("TPacketizer2", "waiting for %d slaves:", mon.GetActive());
         TList *act = mon.GetListOfActives();
         TIter next(act);
         while (TSocket *s = (TSocket*) next()) {
            TSlave *sl = (TSlave *) slaves_by_sock.GetValue(s);
            if (sl)
               Info("TPacketizer2", "   slave-%d (%s)", sl->GetOrdinal(), sl->GetName());
         }
         delete act;
      }

      TSocket *sock = mon.Select();
      mon.DeActivate(sock);

      PDB(kPacketizer,3) Info("TPacketizer2","Select returned: %p", sock);

      TSlave *slave = (TSlave *) slaves_by_sock.GetValue( sock );

      TMessage *reply;

      if ( sock->Recv(reply) <= 0 ) {
         // Help! lost a slave?
         ((TProof*)gProof)->MarkBad(slave);
         fValid = kFALSE;
         Error("TPacketizer2","Recv failed! for slave-%d (%s)",
               slave->GetOrdinal(), slave->GetName());
         continue;
      }

      if ( reply->What() == kPROOF_FATAL ) {
         Error("TPacketizer2","kPROOF_FATAL from slave-%d (%s)",
               slave->GetOrdinal(), slave->GetName());
         ((TProof*)gProof)->MarkBad(slave);
         fValid = kFALSE;
         continue;
      } else if ( reply->What() == kPROOF_LOGFILE ) {
         PDB(kPacketizer,3) Info("TPacketizer2","Got logfile");
         Int_t size;
         (*reply) >> size;
         ((TProof*)gProof)->RecvLogFile(sock, size);
         mon.Activate(sock);
         continue;
      } else if ( reply->What() == kPROOF_LOGDONE ) {
         PDB(kPacketizer,3) Info("TPacketizer2","Got logdone");
         mon.Activate(sock);
         continue;
      } else if ( reply->What() != kPROOF_REPORTSIZE ) {
         // Help! unexpected message type
         Error("TPacketizer2","unexpected message type (%d) from slave-%d (%s)", reply->What(),
               slave->GetOrdinal(), slave->GetName());
         ((TProof*)gProof)->MarkBad(slave);
         fValid = kFALSE;
         continue;
      }

      TSlaveStat *slavestat = (TSlaveStat*) fSlaveStats->GetValue( slave );
      TDSetElement *e = slavestat->fCurFile->GetElement();
      slavestat->fCurFile->GetNode()->DecSlaveCnt(slavestat->GetName());
      Long64_t entries;

      (*reply) >> entries;

      if ( entries > 0 ) {

         if ( e->GetFirst() > entries ) {
            Error("TPacketizer2","first (%d) higher then number of entries (%d) in %d",
                  e->GetFirst(), entries, e->GetFileName() );

            // disable element
            slavestat->fCurFile->SetDone();
            fValid = kFALSE; // ???
         }

         if ( e->GetNum() == -1 ) {
            e->SetNum( entries - e->GetFirst() );
         } else if ( e->GetFirst() + e->GetNum() > entries ) {
            Error("TPacketizer2",
                  "Num (%d) + First (%d) larger then number of keys/entries (%d) in %s",
                  e->GetNum(), e->GetFirst(), entries, e->GetFileName() );
            e->SetNum( entries - e->GetFirst() );
         }

      } else {

         Error("TPacketizer2", "cannot get entries for %s (", e->GetFileName() );

         // disable element
         slavestat->fCurFile->SetDone();
         fValid = kFALSE; // all element must be readable!

      }

      if ( !done ) {
         workers.Add(slave); // Ready for the next job
      }
   }

   // report std. output from slaves??

   ((TProof*)gProof)->ActivateAsyncInput();

   if (!done) {
      // we ran out of slaves ...
      fValid = kFALSE;
   }

}


//______________________________________________________________________________
Long64_t TPacketizer2::GetEntriesProcessed(TSlave *slave) const
{
   if ( fSlaveStats == 0 ) return 0;

   TSlaveStat *slstat = (TSlaveStat*) fSlaveStats->GetValue( slave );

   if ( slstat == 0 ) return 0;

   return slstat->GetEntriesProcessed();
}


//______________________________________________________________________________
TDSetElement *TPacketizer2::GetNextPacket(TSlave *sl, TMessage *r)
{
   if ( !fValid ) {
      return 0;
   }

   // find slave

   TSlaveStat *slstat = (TSlaveStat*) fSlaveStats->GetValue( sl );

   Assert( slstat != 0 );

   // update stats & free old element

   if ( slstat->fCurElem != 0 ) {
      Double_t latency, proctime, proccpu;
      Long64_t bytesRead = -1;

      Long64_t numev = slstat->fCurElem->GetNum();
      slstat->fProcessed += numev;
      fProcessed += numev;

      fPackets->Add(slstat->fCurElem);
      (*r) >> latency >> proctime >> proccpu;
      // only read new info if available
      if (r->BufferSize() > r->Length()) (*r) >> bytesRead;

      PDB(kPacketizer,2) Info("GetNextPacket","slave-%d (%s): %lld %7.3lf %7.3lf %7.3lf %lld",
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

   // get a file if needed

   TFileStat *file = slstat->fCurFile;

   if ( file != 0 && file->IsDone() ) {
      file->GetNode()->DecSlaveCnt(slstat->GetName());
      if (gPerfStats != 0) {
         gPerfStats->FileEvent(sl->GetOrdinal(), sl->GetName(), file->GetNode()->GetName(),
                          file->GetElement()->GetFileName(), kFALSE);
      }
      file = 0;
   }

   if ( file == 0 ) {

      // Try its own node first
      if (slstat->GetFileNode() != 0) {
         file = GetNextUnAlloc(slstat->GetFileNode());
         if ( file == 0 ) {
            slstat->SetFileNode(0);
         }
      }

      // try to find an unused filenode first
      if(file == 0) {
         file = GetNextUnAlloc();
      }

      // then look at the active filenodes
      if(file == 0) {
         file = GetNextActive();
      }

      if ( file == 0 ) return 0;

      slstat->fCurFile = file;
      file->GetNode()->IncSlaveCnt(slstat->GetName());
      if (gPerfStats != 0) {
         gPerfStats->FileEvent(sl->GetOrdinal(), sl->GetName(),
                          file->GetNode()->GetName(),
                          file->GetElement()->GetFileName(), kTRUE);
      }
   }

   // get a packet

   TDSetElement *base = file->GetElement();
   Long64_t num = Long64_t(fPacketSize*(Float_t)slstat->fSlave->GetPerfIdx()/fMaxPerfIdx);
   Long64_t first = file->GetNextEntry();
   Long64_t last = base->GetFirst() + base->GetNum();

   if ( first + num >= last ) {
      num = last - first;
      file->SetDone(); // done

      // delete file from active list (unalloc list is single pass, no delete needed)
      RemoveActive(file);

   } else {
      file->MoveNextEntry(num);
   }

   slstat->fCurElem = new TDSetElement(0,base->GetFileName(),base->GetObjName(),
                                       base->GetDirectory(),first,num);

   return slstat->fCurElem;
}


//______________________________________________________________________________
Bool_t TPacketizer2::HandleTimer(TTimer *)
{
   // Send progress message to client.

   if (fProgress == 0) return kFALSE; // timer stopped already

   TMessage m(kPROOF_PROGRESS);

   m << fTotalEntries << fProcessed;

   // send message to client;
   gProofServ->GetSocket()->Send(m);

   return kFALSE; // ignored?
}
