// @(#)root/proof:$Name:  $:$Id: TPacketizer2.cxx,v 1.21 2004/03/11 11:02:55 brun Exp $
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
#include "TProofStats.h"


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

class TFileNode;

class TFileStat : public TObject {

friend class TPacketizer2;

private:
   TFileNode     *fNode;         // my FileNode
   TDSetElement  *fElement;      // location of the file and its range
   Long64_t       fNextEntry;    // cursor in the range, -1 when done

public:
   TFileStat(TFileNode *node, TDSetElement *elem);

};


TFileStat::TFileStat(TFileNode *node, TDSetElement *elem)
   : fNode(node), fElement(elem), fNextEntry(elem->GetFirst())
{
}


//------------------------------------------------------------------------------

class TFileNode : public TObject {

friend class TPacketizer2;

private:
   TString        fNodeName;     // FQDN of the node
   TList         *fFiles;        // TDSetElements (files) stored on this node
   TIter         *fFileIter;     // iterator on fFiles
   TList         *fActive;       // files with work remaining
   TObject       *fActiveNext;   // cursor in fActive

public:

   TFileNode(const char *name);
   ~TFileNode() { delete fFiles; delete fFileIter; delete fActive; }

   const char *GetName() const { return fNodeName.Data(); }
   void Add(TDSetElement *elem)
   {
      TFileStat *f = new TFileStat(this,elem);
      fFiles->Add(f);
      fActive->Add(f);
   }
};


TFileNode::TFileNode(const char *name)
   : fNodeName(name), fFiles(new TList), fFileIter(0), fActive(new TList),
     fActiveNext(0)
{
   fFiles->SetOwner();
   fActive->SetOwner(kFALSE);
}


//------------------------------------------------------------------------------

class TSlaveStat : public TObject {

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


TSlaveStat::TSlaveStat(TSlave *slave)
   : fSlave(slave), fFileNode(0), fCurFile(0), fCurElem(0), fProcessed(0)
{
}


//------------------------------------------------------------------------------

ClassImp(TPacketizer2)


//______________________________________________________________________________
TPacketizer2::TPacketizer2(TDSet *dset, TList *slaves, Long64_t first, Long64_t num)
{
   PDB(kPacketizer,1) Info("TPacketizer2", "Enter");

   fValid = kTRUE;

   TProof* proof = dynamic_cast<TProof*>(gProof);
   TProofPlayerRemote* rplayer = dynamic_cast<TProofPlayerRemote*>(proof->GetPlayer());
   fStat = rplayer->GetProofStats();

   fProcessed = 0;
   fMaxPerfIdx = 1;

   fPackets = new TList;
   fPackets->SetOwner();

   fFileNodes = new TList;
   fFileNodes->SetOwner();

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

      if ( node == 0 ) {
         node = new TFileNode( host );
         fFileNodes->Add( node );
      }

      node->Add( e );
   }

   fUnAllocated = new TList;
   fUnAllocated->AddAll(fFileNodes);
   fUnAllocated->SetOwner(kFALSE);
   fUnAllocNext = fUnAllocated->First();

   fActive = new TList;
   fActive->AddAll(fFileNodes);
   fActive->SetOwner(kFALSE);
   fActiveNext = fActive->First();

   TIter nodes(fFileNodes);
   TFileNode *node;
   while ( (node = (TFileNode*) nodes.Next()) != 0 ) {
      node->fFileIter = new TIter(node->fFiles);
      node->fActiveNext = node->fActive->First();
   }

   fSlaveStats = new TMap;
   fSlaveStats->SetOwner(kFALSE);

   TSlave *slave;
   TIter si(slaves);
   while ((slave = (TSlave*) si.Next())) {
      TSlaveStat *slstat = new TSlaveStat(slave);
      fSlaveStats->Add( slave, slstat );
      slstat->SetFileNode((TFileNode*) fFileNodes->FindObject(slstat->GetName()));
      fMaxPerfIdx = slave->GetPerfIdx() > fMaxPerfIdx ?
         slave->GetPerfIdx() : fMaxPerfIdx;
   }

   // Check existence of file/dir/tree an get number of entries

   TMap     slaves_by_sock;
   TMonitor mon;
   TList    workers;
   workers.AddAll( slaves );

   // Setup the communication infrastructure
   si.Reset();
   while ((slave = (TSlave*) si.Next())) {
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
      while( TSlave *s = (TSlave*) workers.First() ) {

         workers.Remove(s);

         // find a file

         TSlaveStat *slstat = (TSlaveStat*) fSlaveStats->GetValue(s);
         TFileNode *node = 0;
         TFileStat *file = 0;

         // Try its own node first
         if ( (node = slstat->GetFileNode()) != 0 ) {
            file = (TFileStat*) node->fFileIter->Next();
            if ( file == 0 ) {
               slstat->SetFileNode(0);
               if ( fUnAllocNext == node ) {
                  fUnAllocNext = fUnAllocated->After(node);
               }
               fUnAllocated->Remove( node );
               if ( fUnAllocNext == 0 ) fUnAllocNext = fUnAllocated->First();
            }
         }

         while ( fUnAllocNext != 0 && file == 0 ) {
            node = (TFileNode*) fUnAllocNext;
            file = (TFileStat*) node->fFileIter->Next();
            if ( file == 0 ) {
               fUnAllocNext = fUnAllocated->After(node);
               fUnAllocated->Remove( node );
               if ( fUnAllocNext == 0 ) fUnAllocNext = fUnAllocated->First();
            }
         }

         if ( file != 0 ) {
            slstat->fCurFile = file;
            TMessage m(kPROOF_REPORTSIZE);
            TDSetElement *elem = file->fElement;
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
      e = slavestat->fCurFile->fElement;
      Long64_t entries;

      (*reply) >> entries;

      if ( entries > 0 ) {

         if ( e->GetFirst() > entries ) {
            Error("TPacketizer2","first (%d) higher then number of entries (%d) in %d",
                  e->GetFirst(), entries, e->GetFileName() );

            // disable element
            slavestat->fCurFile->fNextEntry = -1;
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
         slavestat->fCurFile->fNextEntry = -1;
         fValid = kFALSE; // all element must be readable!

      }

      if ( !done ) {
         workers.Add(slave); // Ready for the next job
      }
   }

   ((TProof*)gProof)->ActivateAsyncInput();

   if (!done) {
      // we ran out of slaves ...
      fValid = kFALSE;
   }

   // report output from slaves??

   // Apply overal range to list and rebuild structure

   Int_t files = 0;
   fTotalEntries = 0;
   fUnAllocated->Clear();
   fActive->Clear();
   fFileNodes->Clear(); // Delete all objects

   dset->Reset();
   Long64_t cur = 0;
   while (( e = (TDSetElement*)dset->Next())) {
      TUrl url = e->GetFileName();

      if ( cur + e->GetNum() < first ) {
         cur += e->GetNum();
         continue;
      }

      if ( cur < first ) {
         e->SetFirst( e->GetFirst() + (first - cur) );
         e->SetNum( e->GetNum() - (first + cur) );
      }

      if ( num != -1 && ( first+num < cur + e->GetNum() ) ) {
         e->SetNum( first + num - cur );
      }

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

   fUnAllocated->AddAll(fFileNodes);
   fUnAllocNext = fUnAllocated->First();

   fActive->AddAll(fFileNodes);
   fActiveNext = fActive->First();

   nodes.Reset();
   while ( (node = (TFileNode*) nodes.Next()) != 0 ) {
         node->fFileIter = new TIter(node->fFiles);
         node->fActiveNext = node->fActive->First();
   }

   TIter ssi(fSlaveStats);
   while ( TObject *key = ssi() ) {
      TSlaveStat *slstat = (TSlaveStat*) fSlaveStats->GetValue(key);
      slstat->SetFileNode((TFileNode*) fFileNodes->FindObject(slstat->GetName()));
      slstat->fCurFile = 0;
   }


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

      Int_t numev = slstat->fCurElem->GetNum();
      slstat->fProcessed += numev;
      fProcessed += numev;

      fPackets->Add(slstat->fCurElem);
      (*r) >> latency >> proctime >> proccpu;
      PDB(kPacketizer,2) Info("GetNextPacket","slave-%d (%s): %lld %7.3lf %7.3lf %7.3lf",
                              sl->GetOrdinal(), sl->GetName(),
                              numev, latency, proctime, proccpu);

      if (fStat != 0) {
         fStat->PacketEvent(sl->GetName(), slstat->fCurElem->GetFileName(),
                                 sl->GetOrdinal(), numev, latency, proctime,
                                 proccpu);
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

   if ( file == 0 || file->fNextEntry == -1 ) {
      TFileNode *node = 0;
      file = 0;

      // Try its own node first
      if ( (node = slstat->GetFileNode()) != 0 ) {
         file = (TFileStat*) node->fFileIter->Next();
         if ( file == 0 ) {
            slstat->SetFileNode(0);
            if ( fUnAllocNext == node ) {
               fUnAllocNext = fUnAllocated->After(node);
            }
            fUnAllocated->Remove( node );
            if ( fUnAllocNext == 0 ) fUnAllocNext = fUnAllocated->First();
         } else {
            slstat->fCurFile = file;
         }
      }

      while ( fUnAllocNext != 0 && file == 0 ) {
         node = (TFileNode*) fUnAllocNext;
         file = (TFileStat*) node->fFileIter->Next();
         if ( file == 0 ) {
            fUnAllocNext = fUnAllocated->After(node);
            fUnAllocated->Remove( node );
            if ( fUnAllocNext == 0 ) fUnAllocNext = fUnAllocated->First();
         } else {
            slstat->fCurFile = file;
         }
      }

      while ( fActiveNext != 0 && file == 0 ) {
         node = (TFileNode*) fActiveNext;
         file = (TFileStat*) node->fActiveNext;
         if ( file == 0 ) {
            fActiveNext = fActive->After(node);
            fActive->Remove( node );
            if ( fActiveNext == 0 ) fActiveNext = fActive->First();
         } else {
            slstat->fCurFile = file;
         }
      }

      if ( file == 0 ) return 0;
   }

   // get a packet

   TDSetElement *base = file->fElement;
   Long64_t last = base->GetFirst() + base->GetNum();
   Long64_t first;
   Long64_t num = Long64_t(fPacketSize*(Float_t)slstat->fSlave->GetPerfIdx()/fMaxPerfIdx);

   if ( file->fNextEntry + num >= last ) {
      num = last - file->fNextEntry;
      first = file->fNextEntry;
      file->fNextEntry = -1;

      TFileNode *node = file->fNode;
      if ( node->fActiveNext == file )
         node->fActiveNext = node->fActive->After(file);
      node->fActive->Remove(file);
      if ( node->fActiveNext == 0 )
         node->fActive->First();

   } else {
      first = file->fNextEntry;
      file->fNextEntry += num;
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
