// @(#)root/proofplayer:$Name:  $:$Id: TPacketizerDev.h,v 1.1 2006/07/01 12:05:49 rdm Exp $
// Author: Maarten Ballintijn    18/03/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPacketizerDev
#define ROOT_TPacketizerDev

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPacketizerDev                                                       //
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


class TPacketizerDev : public TVirtualPacketizer {

public:              // public because of Sun CC bug
   class TFileNode;
   class TFileStat;
   class TSlaveStat;

private:
   Long64_t       fProcessed;    // number of entries processed
   TList         *fPackets;      // all processed packets

   Long64_t       fTotalEntries; // total number of entries to be distributed

   TList         *fFileNodes;    // nodes with files
   TList         *fUnAllocated;  // nodes with unallocated files
   TList         *fActive;       // nodes with unfinished files
   TMap          *fSlaveStats;   // slave status, keyed by correspondig TSlave
   TTimer        *fProgress;     // progress updates timer

   Long64_t       fPacketSize;   // global base packet size
   Int_t          fMaxPerfIdx;   // maximum of our slaves' performance index

   Int_t          fMaxSlaveCnt;  // maximum number of slaves per filenode

   TPacketizerDev();
   TPacketizerDev(const TPacketizerDev&);  // no implementation, will generate
   void operator=(const TPacketizerDev&);  // error on accidental usage

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
   void           SplitEventList(TDSet *dset);
   TDSetElement  *CreateNewPacket(TDSetElement* base, Long64_t first, Long64_t num);

public:
   TPacketizerDev(TDSet *dset, TList *slaves, Long64_t first, Long64_t num,
                TList *input);
   virtual ~TPacketizerDev();

   Long64_t      GetEntriesProcessed() const { return fProcessed; }
   Long64_t      GetEntriesProcessed(TSlave *sl) const;
   TDSetElement *GetNextPacket(TSlave *sl, TMessage *r);

   ClassDef(TPacketizerDev,0)  //Generate work packets for parallel processing
};

#endif
