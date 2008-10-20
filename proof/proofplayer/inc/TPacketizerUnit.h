// @(#)root/proofplayer:$Id$
// Author: Long Tran-Thanh    22/07/07

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
// worker nodes, based on the time needed to process previous packets.  //
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
class TStopwatch;


class TPacketizerUnit : public TVirtualPacketizer {

public:              // public because of Sun CC bug
   class TSlaveStat;

private:
   TList    *fPackets;           // all processed packets
   TMap     *fSlaveStats;        // Slave status, keyed by correspondig TSlave
   Long64_t  fPacketSize;        // Global base packet size
                                 // It can be set with PROOF_PacketSize
                                 // parameter, in the input list.
   Int_t     fPacketAsAFraction; // Used to calculate the packet size
                                 // fPacketSize = fTotalEntries / (fPacketAsAFraction * nslaves)
                                 // fPacketAsAFraction can be interpreted as follows:
                                 // assuming all slaves have equal processing rate, packet size
                                 // is (#events processed by 1 slave) / fPacketSizeAsAFraction.
                                 // It can be set with PROOF_PacketAsAFraction in input list.
   TStopwatch *fStopwatch;       // For measuring the start time of each packet
   Long64_t    fProcessing;      // Event being processed
   Long64_t    fAssigned;        // no. entries processed or being processed.
   Double_t    fTimeLimit;       // Packet time limit

   TPacketizerUnit();
   TPacketizerUnit(const TPacketizerUnit&);     // no implementation, will generate
   void operator=(const TPacketizerUnit&);  // error on accidental usage

public:
   TPacketizerUnit(TList *slaves, Long64_t num, TList *input, TProofProgressStatus *st = 0);
   virtual ~TPacketizerUnit();

   TDSetElement *GetNextPacket(TSlave *sl, TMessage *r);

   Double_t      GetCurrentTime();

   ClassDef(TPacketizerUnit,0)  //Generate work packets for parallel processing
};

#endif
