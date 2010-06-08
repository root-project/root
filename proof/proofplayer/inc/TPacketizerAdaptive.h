// @(#)root/proofplayer:$Id$
// Author: Jan Iwaszkiewicz   11/12/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPacketizerAdaptive
#define ROOT_TPacketizerAdaptive

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPacketizerAdaptive                                                  //
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
//   packetizer prevents the situation where the query can't finish     //
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


class TPacketizerAdaptive : public TVirtualPacketizer {

public:              // public because of Sun CC bug
   class TFileNode;
   class TFileStat;
   class TSlaveStat;

private:
   TList         *fFileNodes;    // nodes with files
   TList         *fUnAllocated;  // nodes with unallocated files
   TList         *fActive;       // nodes with unfinished files
   Int_t          fMaxPerfIdx;   // maximum of our slaves' performance index

   Float_t        fFractionOfRemoteFiles; // fraction of TDSetElements
                                          // that are on non slaves
   Long64_t       fNEventsOnRemLoc;       // number of events in currently
                                          // unalloc files on non-worker loc.
   Float_t        fBaseLocalPreference;   // indicates how much more likely
   // the nodes will be to open their local files (1 means indifferent)
   Bool_t         fForceLocal;    // if 1 - eliminate the remote processing

   TPacketizerAdaptive();
   TPacketizerAdaptive(const TPacketizerAdaptive&);    // no implementation, will generate
   void           InitStats();                         // initialise the stats
   void operator=(const TPacketizerAdaptive&);         // error on accidental usage

   TFileNode     *NextNode();
   void           RemoveUnAllocNode(TFileNode *);

   TFileNode     *NextActiveNode();
   void           RemoveActiveNode(TFileNode *);

   TFileStat     *GetNextUnAlloc(TFileNode *node = 0);
   TFileStat     *GetNextActive();
   void           RemoveActive(TFileStat *file);

   void           Reset();
   void           ValidateFiles(TDSet *dset, TList *slaves, Long64_t maxent = -1, Bool_t byfile = kFALSE);
   Int_t          ReassignPacket(TDSetElement *e, TList **listOfMissingFiles);
   void           SplitPerHost(TList *elements, TList **listOfMissingFiles);

public:
   static Long_t   fgMaxSlaveCnt;  // maximum number of workers per filenode (Long_t to avoid
                                   // warnings from backward compatibility support)
   static Int_t    fgPacketAsAFraction; // used to calculate the packet size
                                  // fPacketSize = fTotalEntries / (fPacketAsAFraction * nslaves)
                                  // fPacketAsAFraction can be interpreted as follows:
                                  // assuming all slaves have equal processing rate, packet size
                                  // is (#events processed by 1 slave) / fPacketSizeAsAFraction.
                                  // It can be set with PROOF_PacketAsAFraction in input list.
   static Double_t fgMinPacketTime; // minimum packet time
   static Double_t fgMaxPacketTime; // maximum packet time
   static Int_t    fgStrategy;    // 0 means the classic and 1 (default) - the adaptive strategy

   TPacketizerAdaptive(TDSet *dset, TList *slaves, Long64_t first, Long64_t num,
                       TList *input, TProofProgressStatus *st);
   virtual ~TPacketizerAdaptive();

   Int_t         AddProcessed(TSlave *sl, TProofProgressStatus *st,
                               Double_t latency, TList **listOfMissingFiles = 0);
   Int_t         GetEstEntriesProcessed(Float_t, Long64_t &ent, Long64_t &bytes, Long64_t &calls);
   Float_t       GetCurrentRate(Bool_t &all);
   Int_t         CalculatePacketSize(TObject *slstat, Long64_t cachesz, Int_t learnent);
   TDSetElement *GetNextPacket(TSlave *sl, TMessage *r);
   void          MarkBad(TSlave *s, TProofProgressStatus *status, TList **missingFiles);

   Int_t         GetActiveWorkers();

   ClassDef(TPacketizerAdaptive,0)  //Generate work packets for parallel processing
};

#endif
