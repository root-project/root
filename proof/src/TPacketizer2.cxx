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

ClassImp(TPacketizer2)


class TFileNode;

class TFileStat : public TObject {

friend class TPacketizer2;

private:
   TFileNode     *fNode;         // my FileNode
   TDSetElement  *fElement;      // location of the file and its range
   Int_t          fNextEntry;    // cursor in the range, -1 when done

public:
   TFileStat(TFileNode *node, TDSetElement *elem)
      : fNode(node), fElement(elem), fNextEntry(0) { }
};


class TFileNode : public TObject {

friend class TPacketizer2;

private:
   TString        fNodeName;     // FQDN of the node
   TList         *fFiles;        // TDSetElements (files) stored on this node
   TIter         *fFileIter;     // iterator on fFiles
   TList         *fActive;       // files with work remaining
   TObject       *fActiveNext;      // cursor in fActive

public:

   TFileNode(const char *name)
      : fNodeName(name), fFiles(new TList), fFileIter(0), fActive(new TList), fActiveNext(0)
   {
      fFiles->SetOwner();
      fActive->SetOwner(kFALSE);
   }
  ~TFileNode() { delete fFiles; delete fFileIter; delete fActive; }

   const char *GetName() const { return fNodeName.Data(); }
   void Add(TDSetElement *elem)
   {
      TFileStat *f = new TFileStat(this,elem);
      fFiles->Add(f);
      fActive->Add(f);
   }

};


class TSlaveStat : public TObject {

friend class TPacketizer2;

private:
   TSlave        *fSlave;        // corresponding TSlave record
   TFileNode     *fFileNode;     // corresponding node or 0
   TFileStat     *fCurFile;      // file currently being processed
   TDSetElement  *fCurElem;      // TDSetElement currently being processed
   Int_t          fProcessed;    // number of entries processed

public:
   TSlaveStat(TSlave *slave)
      : fSlave(slave), fFileNode(0), fCurFile(0), fCurElem(0), fProcessed(0) { }

   TFileNode  *GetFileNode() const { return fFileNode; }
   const char *GetName() const { return fSlave->GetName(); }
   Int_t       GetEntriesProcessed() const { return fProcessed; }

   void        SetFileNode(TFileNode *node) { fFileNode = node; }
};


//______________________________________________________________________________
TPacketizer2::TPacketizer2(TDSet *dset, TList *slaves, Long64_t first, Long64_t num)
{
   PDB(kPacketizer,1) Info("TPacketizer2", "Enter");

   fValid = kTRUE;

   fFileNodes = new TList;
   fFileNodes->SetOwner();

   // Split into per host entries
   dset->Reset();
   for( TDSetElement *e = (TDSetElement*)dset->Next(); e != 0 ; e = (TDSetElement*)dset->Next() ) {
      TUrl url = e->GetFileName();

      // TODO: Names must be in rootd URL format, check where?
      if ( !url.IsValid() || !strncmp(url.GetProtocol(),"root", 4) == 0 ) {
         Error("TPacketizer2","Filename not in rootd URL format (%s)",
                e->GetFileName() );
         fValid = kFALSE;
         return;
      }

      TFileNode *node = (TFileNode*) fFileNodes->FindObject( url.GetHost() );

      if ( node == 0 ) {
         node = new TFileNode( url.GetHost() );
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

   TIter si(slaves);
   for( TSlave *slave = (TSlave*)si.Next(); slave ; slave = (TSlave*)si.Next() ) {
      TSlaveStat *slstat = new TSlaveStat(slave);
      fSlaveStats->Add( slave, slstat );
      slstat->SetFileNode((TFileNode*) fFileNodes->FindObject(slstat->GetName()));
   }

   // Check existence of file/dir/tree an get number of entries

   TList *workers = new TList;
   workers->AddAll( slaves );
   TMap     slaves_by_sock;
   TMonitor mon;

   // Setup the communication infrastructure
   si.Reset();
   for (TSlave *s = (TSlave*) si.Next(); s ; s = (TSlave*) si.Next()) {
Info("TPacketizer2","Socket added to monitor: %p", s->GetSocket());
      mon.Add(s->GetSocket());
      slaves_by_sock.Add(s->GetSocket(),s);
   }

   mon.DeActivateAll();

   gProof->DeActivateAsyncInput();

   while(kTRUE) {
      Bool_t   done = kFALSE;

      // send work
      while( TSlave *s = (TSlave*) workers->First() ) {

         workers->Remove(s);

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
Info("TPacketizer2","sent via %p (%s) reportsize on %s %s %s %s", s->GetSocket(),
      s->GetName(), dset->IsTree() ? "tree" : "not tree", elem->GetFileName(),
       elem->GetDirectory(), elem->GetObjName() );
         } else {
            // Done
            done = kTRUE;
            workers->Clear();
         }

      }

      if ( mon.GetActive() == 0 ) break; // nothing to wait for anymore

      TSocket *sock = mon.Select();
      mon.DeActivate(sock);

Info("TPacketizer2","Select returned: %p", sock);

      TMessage *reply;

      if ( sock->Recv(reply) <= 0 ) {
         // Help! lost a slave?

      }

      if ( reply->What() != kPROOF_REPORTSIZE ) {
         // Help! unexpected message type
      }

      TSlave *slave = (TSlave *) slaves_by_sock.GetValue( sock );
      TSlaveStat *slavestat = (TSlaveStat*) fSlaveStats->GetValue( slave );
      TDSetElement *e = slavestat->fCurFile->fElement;
      Long64_t entries;
      Int_t    r;

      (*reply) >> r >> entries;

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
         fValid = kFALSE; // ???

         // TODO: Error("...") or Warning("...") ?
      }

      if ( !done ) {
         workers->Add(slave); // Ready for the next job
      }
   }

   gProof->ActivateAsyncInput();

   // report output from slaves??

   // Apply overal range to list and rebuild structure

   Int_t files = 0;
   fTotalEntries = 0;
   fFileNodes->Clear(); // Delete all objects

   dset->Reset();
   Long64_t cur = 0;
   for( TDSetElement *e = (TDSetElement*)dset->Next(); e != 0 ; e = (TDSetElement*)dset->Next() ) {
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

      TFileNode *node = (TFileNode*) fFileNodes->FindObject( url.GetHost() );

      if ( node == 0 ) {
         node = new TFileNode( url.GetHost() );
         fFileNodes->Add( node );
      }

      ++files;
      fTotalEntries += e->GetNum();
      node->Add( e );
   }

   PDB(kGlobal,1) Info("TPacketizer2","Processing %ld entries in %d files on %d hosts",
         fTotalEntries, files, fFileNodes->GetSize() );

   fUnAllocated->Clear();
   fUnAllocated->AddAll(fFileNodes);
   fUnAllocNext = fUnAllocated->First();

   fActive->Clear();
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

   PDB(kPacketizer,1) Info("TPacketizer2", "return");
}


//______________________________________________________________________________
TPacketizer2::~TPacketizer2()
{
   if (fSlaveStats) {
      TList l;
      TIter n(fSlaveStats);
      while( TObject *key = n() ) {
         l.Add(fSlaveStats->GetValue(key));
      }
      l.SetOwner(kTRUE);
      // The destructor of l will delete the values of fSlaveStats
   }

   delete fSlaveStats;
   delete fUnAllocated;
   delete fActive;
   delete fFileNodes;
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
TDSetElement *TPacketizer2::GetNextPacket(TSlave *sl)
{
   if ( !fValid ) {
      return 0;
   }

   // find slave

   TSlaveStat *slstat = (TSlaveStat*) fSlaveStats->GetValue( sl );

   Assert( slstat != 0 );

   // update stats & free old element

   if ( slstat->fCurElem != 0 ) {
         slstat->fProcessed += slstat->fCurElem->GetNum();
         delete slstat->fCurElem;
         slstat->fCurElem = 0;
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
   Int_t last = base->GetFirst() + base->GetNum();
   Int_t first;
   Int_t num = 1000;  // target packet size TODO: variable packet size

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

