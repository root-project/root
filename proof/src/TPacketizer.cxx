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
// TPacketizer                                                          //
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


#include "TPacketizer.h"

#include "TObject.h"
#include "TSlave.h"
#include "TDSet.h"
#include "TUrl.h"
#include "TError.h"
#include "TProofDebug.h"

ClassImp(TPacketizer)


class TFileNode;

class TFileStat : public TObject {

friend class TPacketizer;

private:
   TFileNode     *fNode;         // my FileNode
   TDSetElement  *fElement;      // location of the file and its range
   Int_t          fNextEntry;    // cursor in the range, -1 when done

public:
   TFileStat(TFileNode *node, TDSetElement *elem)
      : fNode(node), fElement(elem), fNextEntry(elem->GetFirst()) { }
};


class TFileNode : public TObject {

friend class TPacketizer;

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

friend class TPacketizer;

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
TPacketizer::TPacketizer(TDSet *dset, TList *slaves, Long64_t first, Long64_t num)
{
   fValid = kTRUE;

   fFileNodes = new TList;
   fFileNodes->SetOwner();
   fTotalEntries = 0;
   Int_t files = 0;

   dset->Reset();
   Long64_t cur = 0;
   for( TDSetElement *e = (TDSetElement*)dset->Next(); e != 0 ; e = (TDSetElement*)dset->Next() ) {
      TUrl url = e->GetFileName();

      Long64_t n = GetEntries(dset->IsTree(), e);

      if ( n == -1 ) {
         fValid = kFALSE;
         return;
      }

      if ( e->GetFirst() > n ) {
         Error("TPacketizer","First (%d) higher then number of entries (%d) in %d",
               e->GetFirst(), n, e->GetFileName() );
         fValid = kFALSE;
         return;
      }

      if ( e->GetNum() == -1 ) {
         e->SetNum( n - e->GetFirst() );
      } else if ( e->GetFirst() + e->GetNum() > n ) {
         Error("TPacketizer",
            "Num (%d) + First (%d) larger then number of keys (%d) in %s",
            e->GetNum(), e->GetFirst(), n, e->GetFileName() );
         e->SetNum( n - e->GetFirst() );
      }

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

      // TODO: Names must be in rootd URL format, check where?
      if ( !url.IsValid() || !strncmp(url.GetProtocol(),"root", 4) == 0 ) {
         Error("TPacketizer","Filename not in rootd URL format (%s)",
                e->GetFileName() );
         fValid = kFALSE;
         return;
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

   PDB(kGlobal,1) Info("TPacketizer","Processing %ld entries in %d files on %d hosts",
         fTotalEntries, files, fFileNodes->GetSize() );

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

   fSlaves = new TList;
   fSlaves->SetOwner();
   TIter sss(slaves);
   for( TSlave *s = (TSlave*) sss.Next(); s ; s = (TSlave*) sss.Next() ) {
      TSlaveStat *slave = new TSlaveStat(s);
      fSlaves->Add( slave );
      slave->SetFileNode( (TFileNode*) fFileNodes->FindObject(slave->GetName()) );

   }
}


//______________________________________________________________________________
TPacketizer::~TPacketizer()
{
   delete fSlaves;
   delete fUnAllocated;
   delete fActive;
   delete fFileNodes;
}


//______________________________________________________________________________
Long64_t TPacketizer::GetEntriesProcessed(TSlave *sl) const
{
   if ( fSlaves == 0 ) return 0;

   TSlaveStat *slave = (TSlaveStat*) fSlaves->FindObject( sl->GetName() );

   if ( slave == 0 ) return 0;

   return slave->GetEntriesProcessed();
}


//______________________________________________________________________________
TDSetElement *TPacketizer::GetNextPacket(TSlave *sl)
{
   if ( !fValid ) {
      return 0;
   }

   // find slave

   TSlaveStat *slave = (TSlaveStat*) fSlaves->FindObject( sl->GetName() );

   Assert( slave != 0 );

   // update stats & free old element

   if ( slave->fCurElem != 0 ) {
         slave->fProcessed += slave->fCurElem->GetNum();
         delete slave->fCurElem;
         slave->fCurElem = 0;
   }

   // get a file if needed


   TFileStat *file = slave->fCurFile;

   if ( file == 0 || file->fNextEntry == -1 ) {
      TFileNode *node = 0;
      file = 0;

      // Try its own node first
      if ( (node = slave->GetFileNode()) != 0 ) {
         file = (TFileStat*) node->fFileIter->Next();
         if ( file == 0 ) {
            slave->SetFileNode(0);
            if ( fUnAllocNext == node ) {
               fUnAllocNext = fUnAllocated->After(node);
            }
            fUnAllocated->Remove( node );
            if ( fUnAllocNext == 0 ) fUnAllocNext = fUnAllocated->First();
         } else {
            slave->fCurFile = file;
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
            slave->fCurFile = file;
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
            slave->fCurFile = file;
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

   slave->fCurElem = new TDSetElement(0,base->GetFileName(),base->GetObjName(),
                                       base->GetDirectory(),first,num);

   return slave->fCurElem;
}
