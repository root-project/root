// @(#)root/treeplayer:$Name$:$Id$
// Author: Fons Rademakers   28/03/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPacketGenerator
#define ROOT_TPacketGenerator


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPacketGenerator                                                     //
//                                                                      //
// This class generates packets to be processed on PROOF slave servers. //
// A packet is an event range (begin entry and number of entries).       //
// Packets are generated taking into account the performance of the     //
// remote machine, the time it took to process a previous packet on     //
// the remote machine, the locality of the database files, etc.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TTime
#include "TTime.h"
#endif
#ifndef ROOT_Htypes
#include "Htypes.h"
#endif

class TObjArray;
class TList;
class TSlave;
class TTree;


class TPacketGenerator : public TObject {

private:
   Stat_t     fNextEntry;         //next entry to be processed
   Stat_t     fLastEntry;         //last entry to be processed
   Stat_t     fEntriesProcessed;  //number of entries processed
   Int_t      fInitialPacketSize; //initial packet size
   Int_t      fMinPacketSize;     //minimum packet size
   Int_t      fSmallestPacket;    //smallest packet generated
   TTime      fMinTimeDiff;       //time difference between last two packets for fastest slave
   Int_t      fMaxOrd;            //highest ordinal slave number
   TTree     *fTree;              //tree or chain being processed
   TList     *fSlaves;            //list of slaves
   Stat_t    *fEntriesPerSlave;   //array containing number of entries processed per slave
   TObjArray *fLastPackets;       //pointer to the last packet for each slave
   TList     *fPackets;           //list of packets that have been processed

   TPacketGenerator();
   TPacketGenerator(const TPacketGenerator &) { }
   void operator=(const TPacketGenerator &) { }
   void Init();

public:
   TPacketGenerator(Stat_t firstEntry, Stat_t lastEntry, TTree *tree, TList *slaves);
   TPacketGenerator(Stat_t firstEntry, Stat_t lastEntry, Int_t packetSize, TList *slaves);
   virtual ~TPacketGenerator();

   Stat_t      GetEntriesProcessed() const { return fEntriesProcessed; }
   Stat_t      GetEntriesProcessed(TSlave *sl) const;
   Int_t       GetInitialPacketSize() const { return fInitialPacketSize; }
   Stat_t      GetNextEntry() const { return fNextEntry; }
   Bool_t      GetNextPacket(TSlave *sl, Int_t &nentry, Stat_t &first);
   Int_t       GetNoPackets() const;
   TList      *GetPackets() const { return fPackets; }
   Stat_t      GetLastEntry() const { return fLastEntry; }

   void        SetMinPacketSize(Int_t minps);

   void        Print(Option_t *option="");
   void        Reset(Stat_t firstEntry, Stat_t lastEntry, TList *slaves);

   ClassDef(TPacketGenerator,0)  //Generate packets of entries for parallel processing
};

#endif
