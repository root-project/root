// @(#)root/proofplayer:$Name:  $:$Id: TPacketizer.h,v 1.20 2007/03/19 10:46:10 rdm Exp $
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

#ifndef ROOT_TVirtualPacketizer
#include "TVirtualPacketizer.h"
#endif


class TMessage;
class TTimer;
class TTree;
class TMap;
class TNtupleD;
class TProofStats;


class TPacketizer : public TVirtualPacketizer {

public:              // public because of Sun CC bug
   class TFileNode;
   class TFileStat;
   class TSlaveStat;

private:
   Long64_t  fProcessed;    // number of entries processed
   Long64_t  fBytesRead;    // number of bytes processed
   TList    *fPackets;      // all processed packets

   Long64_t  fTotalEntries; // total number of entries to be distributed

   // Members for progress info
   Long_t    fStartTime;    // time offset
   Float_t   fInitTime;     // time before processing
   Float_t   fProcTime;     // time since start of processing
   Float_t   fTimeUpdt;     // time between updates
   TNtupleD *fCircProg;     // Keeps circular info for "instantenous"
                            // rate calculations
   Long_t    fCircN;        // Circularity

   TList    *fFileNodes;    // nodes with files
   TList    *fUnAllocated;  // nodes with unallocated files
   TList    *fActive;       // nodes with unfinished files
   TMap     *fSlaveStats;   // slave status, keyed by correspondig TSlave
   TTimer   *fProgress;     // progress updates timer

   Long64_t  fPacketSize;   // global base packet size
                                 // It can be set with PROOF_PacketSize
                                 // parameter, in the input list.
   Int_t     fMaxPerfIdx;   // maximum of our slaves' performance index

   Int_t     fMaxSlaveCnt;  // maximum number of slaves per filenode
   Int_t     fPacketAsAFraction; // used to calculate the packet size
                                 // fPacketSize = fTotalEntries / (fPacketAsAFraction * nslaves)
                                 // fPacketAsAFraction can be interpreted as follows:
                                 // assuming all slaves have equal processing rate, packet size
                                 // is (#events processed by 1 slave) / fPacketSizeAsAFraction.
                                 // It can be set with PROOF_PacketAsAFraction in input list.

   TPacketizer();
   TPacketizer(const TPacketizer&);     // no implementation, will generate
   void operator=(const TPacketizer&);  // error on accidental usage

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
   TPacketizer(TDSet *dset, TList *slaves, Long64_t first, Long64_t num,
                TList *input);
   virtual ~TPacketizer();

   Long64_t      GetEntriesProcessed() const { return fProcessed; }
   Long64_t      GetEntriesProcessed(TSlave *sl) const;
   TDSetElement *GetNextPacket(TSlave *sl, TMessage *r);

   Long64_t      GetBytesRead() const { return fBytesRead; }
   Float_t       GetInitTime() const { return fInitTime; }
   Float_t       GetProcTime() const { return fProcTime; }

   ClassDef(TPacketizer,0)  //Generate work packets for parallel processing
};

#endif
