// @(#)root/proof:$Name:  $:$Id$
// Author: Maarten Ballintijn    18/03/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPacketizer
#define ROOT_TPacketizer

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPacketizer                                                         //
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

#ifndef ROOT_TVirtualPacketizer
#include "TVirtualPacketizer.h"
#endif


class TMessage;
class TTimer;
class TTree;
class TMap;
class TProofStats;


class TPacketizer : public TVirtualPacketizer {

private:

   class TFileNode;
   class TFileStat;
   class TSlaveStat;

   Long64_t       fProcessed;    // number of entries processed
   TList         *fPackets;      // all processed packets
   TProofStats   *fStat;         // PROOF runtime statistics

   Long64_t       fTotalEntries; // total number of entries to be distributed

   TList         *fFileNodes;    // nodes with files
   TList         *fUnAllocated;  // nodes with unallocated files
   TObject       *fUnAllocNext;  // cursor in fUnAllocated
   TList         *fActive;       // nodes with unfinished files
   TObject       *fActiveNext;   // cursor in fActive
   TMap          *fSlaveStats;   // slave status, keyed by correspondig TSlave
   TTimer        *fProgress;     // progress updates timer

   Long64_t       fPacketSize;   // global base packet size
   Int_t          fMaxPerfIdx;   // maximum of our slaves' performance index

   TPacketizer();
   TPacketizer(const TPacketizer &);    // no implementation, will generate
   void operator=(const TPacketizer &);  // error on accidental usage

   virtual Bool_t HandleTimer(TTimer *timer);

   TFileNode     *NextUnAllocNode();
   void           RemoveUnAllocNode(TFileNode *);

   TFileNode     *NextActiveNode();
   void           RemoveActiveNode(TFileNode *);

   TFileStat     *GetNextUnAlloc(TFileNode *node = 0);
   TFileStat     *GetNextActive();
   void           RemoveActive(TFileStat *file);

   void           Reset();
   void           ValidateFiles(TDSet *dset, TList *slaves);


public:
   TPacketizer(TDSet *dset, TList *slaves, Long64_t first, Long64_t num);
   virtual ~TPacketizer();

   Long64_t      GetEntriesProcessed() const { return fProcessed; }
   Long64_t      GetEntriesProcessed(TSlave *sl) const;
   TDSetElement *GetNextPacket(TSlave *sl, TMessage *r);


   ClassDef(TPacketizer,0)  //Generate work packets for parallel processing
};

#endif
