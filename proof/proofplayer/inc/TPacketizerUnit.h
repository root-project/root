// @(#)root/proofplayer:$Id$
// Author: Long Tran-Thanh    22/07/07
// Revised: G. Ganis, May 2011

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPacketizerUnit
#define ROOT_TPacketizerUnit

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPacketizerUnit                                                      //
//                                                                      //
// This packetizer generates packets of generic units, representing the //
// number of times an operation cycle has to be repeated by the worker  //
// node, e.g. the number of Monte carlo events to be generated.         //
// Packets sizes are generated taking into account the performance of   //
// worker nodes, based on the time needed to process previous packets,  //
// with the goal of having all workers ending at the same time.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualPacketizer.h"
#include "TMap.h"


class TMessage;
class TTimer;
class TTree;
class TProofStats;
class TStopwatch;


class TPacketizerUnit : public TVirtualPacketizer {

public:              // public because of Sun CC bug
   class TSlaveStat;

private:
   TList      *fPackets;         // All processed packets
   TMap       *fWrkStats;        // Worker status, keyed by correspondig TSlave
   TList      *fWrkExcluded;     // List of nodes excluded from distribution
                                 // (submasters with no active workers)
   TStopwatch *fStopwatch;       // For measuring the start time of each packet
   Long64_t    fProcessing;      // Event being processed
   Long64_t    fAssigned;        // Entries processed or being processed.
   Double_t    fCalibFrac;       // Size of the calibrating packet as fraction of Ntot/Nwrk
   Long64_t    fNumPerWorker;    // Number of cycles per worker, if this option
                                 // is chosen
   Bool_t      fFixedNum;        // Whether we must assign a fixed number of cycles per worker

   Long64_t    fPacketSeq;       // Sequential number of the last packet assigned

   TPacketizerUnit();
   TPacketizerUnit(const TPacketizerUnit&);     // no implementation, will generate
   void operator=(const TPacketizerUnit&);      // error on accidental usage

public:
   TPacketizerUnit(TList *slaves, Long64_t num, TList *input, TProofProgressStatus *st = 0);
   virtual ~TPacketizerUnit();

   Int_t         AssignWork(TDSet* /*dset*/, Long64_t /*first*/, Long64_t num);
   TDSetElement *GetNextPacket(TSlave *sl, TMessage *r);

   Double_t      GetCurrentTime();

   Float_t       GetCurrentRate(Bool_t &all);
   Int_t         GetActiveWorkers() { return fWrkStats->GetSize(); }

   Int_t         AddWorkers(TList *workers);

   ClassDef(TPacketizerUnit,0)  //Generate work packets for parallel processing
};

#endif
