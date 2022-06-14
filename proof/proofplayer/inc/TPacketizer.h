// @(#)root/proofplayer:$Id$
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

#include "TVirtualPacketizer.h"


class TMessage;
class TTimer;
class TTree;
class TMap;
class TProofProgressStatus;
class TProofStats;

class TPacketizer : public TVirtualPacketizer {

public:              // public because of Sun CC bug
   class TFileNode;
   class TFileStat;
   class TSlaveStat;

private:
   TList    *fPackets;      // all processed packets

   TList    *fFileNodes;    // nodes with files
   TList    *fUnAllocated;  // nodes with unallocated files
   TList    *fActive;       // nodes with unfinished files

   Long64_t  fPacketSize;   // global base packet size
                                 // It can be set with PROOF_PacketSize
                                 // parameter, in the input list.
   Int_t     fMaxPerfIdx;   // maximum of our slaves' performance index

   Long_t    fMaxSlaveCnt;  // maximum number of workers per filenode (Long_t to avoid
                            // warnings from backward compatibility support)
   Int_t     fPacketAsAFraction; // used to calculate the packet size
                                 // fPacketSize = fTotalEntries / (fPacketAsAFraction * nslaves)
                                 // fPacketAsAFraction can be interpreted as follows:
                                 // assuming all slaves have equal processing rate, packet size
                                 // is (#events processed by 1 slave) / fPacketSizeAsAFraction.
                                 // It can be set with PROOF_PacketAsAFraction in input list.

   // Add workers controls
   Bool_t fHeuristicPSiz;   // Whether the packet size is calculated heuristically
   Bool_t fDefMaxWrkNode;   // Whether the default is used for the max workers per node

   TPacketizer();
   TPacketizer(const TPacketizer&);     // no implementation, will generate
   void operator=(const TPacketizer&);  // error on accidental usage

   TFileNode     *NextUnAllocNode();
   void           RemoveUnAllocNode(TFileNode *);

   TFileNode     *NextActiveNode();
   void           RemoveActiveNode(TFileNode *);

   TFileStat     *GetNextUnAlloc(TFileNode *node = 0);
   TFileStat     *GetNextActive();
   void           RemoveActive(TFileStat *file);

   void           Reset();
   void           ValidateFiles(TDSet *dset, TList *slaves, Long64_t maxent = -1, Bool_t byfile = kFALSE);

public:
   TPacketizer(TDSet *dset, TList *slaves, Long64_t first, Long64_t num,
                TList *input, TProofProgressStatus *st);
   virtual ~TPacketizer();

   Int_t         AddWorkers(TList *workers);
   TDSetElement *GetNextPacket(TSlave *sl, TMessage *r);
   Long64_t      GetEntriesProcessed(TSlave *sl) const;

   Float_t       GetCurrentRate(Bool_t &all);
   Int_t         GetActiveWorkers();

   ClassDef(TPacketizer,0)  //Generate work packets for parallel processing
};

#endif
