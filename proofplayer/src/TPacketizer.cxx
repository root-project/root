// @(#)root/proof:$Name:  $:$Id: $
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
      : fNode(node), fElement(elem), fNextEntry(0) { }
};


class TFileNode : public TObject {

friend class TPacketizer;

private:
   TString        fNodeName;     // FQDN of the node
   TList         *fFiles;        // TDSetElements stored on this node
   TIter         *fNextFile;     // iterator on fFiles
   Int_t          fActiveFiles;  // number of files with work remaining

public:

   TFileNode(const char *name)
      : fNodeName(name), fFiles(new TList), fNextFile(0), fActiveFiles(0) { }
  ~TFileNode() { delete fFiles; delete fNextFile; }

   const char *GetName() const { return fNodeName.Data(); }
   void Add(TDSetElement *elem)
      { fFiles->Add( new TFileStat(this,elem) ); ++fActiveFiles; }

};


class TSlaveStat : public TObject {

friend class TPacketizer;

private:
   TSlave        *fSlave;        // corresponding TSlave record
   TFileNode     *fFileNode;     // correcponding node or 0
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
TPacketizer::TPacketizer(TDSet *dset, TList *slaves)
{
   fFileNodes = new TList;
   fFileNodes->SetOwner();

   dset->Reset();
   for( TDSetElement *e = (TDSetElement*) dset->Next(); e ; e = (TDSetElement*) dset->Next() ) {
      TUrl url = e->GetFileName();

      // TODO: Names must be in rootd URL format, check where?
      Assert( url.IsValid() && strncmp(url.GetProtocol(),"root", 4) == 0 );

      TFileNode *node = (TFileNode*) fFileNodes->FindObject( url.GetHost() );

      if ( node == 0 ) {
         node = new TFileNode( url.GetHost() );
         fFileNodes->Add( node );
      }

      // Get number of entries and adjust element
      // TODO: e->GetEntries ???

      node->Add( e );
   }

   // Is there an easier way to do shallow copy ?
   TIter nodes(fFileNodes);
   fUnAllocated = new TList;
   for( TObject *o = nodes(); o != 0 ; o = nodes() ) { fUnAllocated->Add(o); }
   fUnAllocated->SetOwner(kFALSE);
   fUnAllocNext = fUnAllocated->First();

   nodes.Reset();
   fActive = new TList;
   for( TObject *o = nodes(); o != 0 ; o = nodes() ) { fActive->Add(o); }
   fActive->SetOwner(kFALSE);
   fActiveNext = fActive->First();

   nodes.Reset();
   TFileNode *node;
   while ( (node = (TFileNode*) nodes.Next()) != 0 ) {
         node->fNextFile = new TIter(node->fFiles);
         node->fActiveFiles = node->fFiles->GetSize();
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
Int_t TPacketizer::GetEntriesProcessed(TSlave *sl) const
{
   if ( fSlaves == 0 ) return 0;

   TSlaveStat *slave = (TSlaveStat*) fSlaves->FindObject( sl->GetName() );

   if ( slave == 0 ) return 0;

   return slave->GetEntriesProcessed();
}


//______________________________________________________________________________
TDSetElement *TPacketizer::GetNextPacket(TSlave *sl)
{
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
         file = (TFileStat*) node->fNextFile->Next();
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
         file = (TFileStat*) node->fNextFile->Next();
         if ( file == 0 ) {
            fUnAllocNext = fUnAllocated->After(node);
            fUnAllocated->Remove( node );
            if ( fUnAllocNext == 0 ) fUnAllocNext = fUnAllocated->First();
         } else {
            slave->fCurFile = file;
         }
      }

      // TODO: look for files with remaining work
      if ( file == 0 ) return 0;
   }

   // get a packet

   TDSetElement *base = file->fElement;
   Int_t last = base->GetFirst() + base->GetNum();
   Int_t first;
   Int_t num = 500;  // target packet size TODO: variable packet size

   if ( file->fNextEntry + num >= last ) {
      num = last - file->fNextEntry;
      first = file->fNextEntry;
      file->fNextEntry = -1;
      --file->fNode->fActiveFiles;
   } else {
      first = file->fNextEntry;
      file->fNextEntry += num;
   }

   slave->fCurElem = new TDSetElement(0,base->GetFileName(),base->GetObjName(),
                                       base->GetDirectory(),first,num);

   return slave->fCurElem;
}


