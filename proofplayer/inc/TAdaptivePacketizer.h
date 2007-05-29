// @(#)root/proofplayer:$Name:  $:$Id: TAdaptivePacketizer.h,v 1.3 2007/03/19 10:46:10 rdm Exp $
// Author: Jan Iwaszkiewicz   11/12/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAdaptivePacketizer
#define ROOT_TAdaptivePacketizer

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAdaptivePacketizer                                                  //
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
//   packetizer prevents the situation where the query can’t finish     //
//   because of one slow node.                                          //
//                                                                      //
// The data structures: TFileStat, TFileNode and TSlaveStat are         //
// enriched + changed and TFileNode::Compare method is changed.         //
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
class TRandom;


class TAdaptivePacketizer : public TVirtualPacketizer {

public:              // public because of Sun CC bug
   class TFileNode;
   class TFileStat;
   class TSlaveStat;

private:
   TList         *fPackets;      // all processed packets

   TList         *fFileNodes;    // nodes with files
   TList         *fUnAllocated;  // nodes with unallocated files
   TList         *fActive;       // nodes with unfinished files
   TMap          *fSlaveStats;   // slave status, keyed by correspondig TSlave

   Int_t          fMaxPerfIdx;   // maximum of our slaves' performance index

   Float_t        fFractionOfRemoteFiles; // fraction of TDSetElements
                                          // that are on non slaves
   Long64_t       fNEventsOnRemLoc;       // number of events in currently
                                          // unalloc files on non-worker loc.

   Float_t        fBaseLocalPreference;   // indicates how much more likely
   // the nodes will be to open their local files (1 means indifferent)

   TAdaptivePacketizer();
   TAdaptivePacketizer(const TAdaptivePacketizer&);    // no implementation, will generate
   void operator=(const TAdaptivePacketizer&);         // error on accidental usage

   TFileNode     *NextNode();
   void           RemoveUnAllocNode(TFileNode *);

   TFileNode     *NextActiveNode();
   void           RemoveActiveNode(TFileNode *);

   TFileStat     *GetNextUnAlloc(TFileNode *node = 0);
   TFileStat     *GetNextActive();
   void           RemoveActive(TFileStat *file);

   void           Reset();
   void           ValidateFiles(TDSet *dset, TList *slaves);

public:
   static Int_t   fgMaxSlaveCnt;  // maximum number of slaves per filenode

   TAdaptivePacketizer(TDSet *dset, TList *slaves, Long64_t first, Long64_t num,
                       TList *input);
   virtual ~TAdaptivePacketizer();

   Long64_t      GetEntriesProcessed(TSlave *sl) const;
   Int_t         CalculatePacketSize(TObject *slstat);
   TDSetElement *GetNextPacket(TSlave *sl, TMessage *r);

   ClassDef(TAdaptivePacketizer,0)  //Generate work packets for parallel processing
};

#endif
