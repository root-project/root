// @(#)root/proof:$Name:  $:$Id: TPacketizer2.h,v 1.4 2002/07/17 12:29:37 rdm Exp $
// Author: Maarten Ballintijn    18/03/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPacketizer2
#define ROOT_TPacketizer2

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPacketizer2                                                         //
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


class TMap;


class TPacketizer2 : public TVirtualPacketizer {

private:
   Int_t    fProcessed;       // number of entries processed
   Long64_t fTotalEntries;    // Total number of entries to be distributed

   TList   *fFileNodes;       // nodes with files
   TList   *fUnAllocated;     // nodes with unallocated files
   TObject *fUnAllocNext;     // cursor in fUnAllocated
   TList   *fActive;          // nodes with unfinished files
   TObject *fActiveNext;      // cursor in fActive
   TMap    *fSlaveStats;      // slave status, keyed by correspondig TSlave

   TPacketizer2();
   TPacketizer2(const TPacketizer2 &);    // no implementation, will generate
   void operator=(const TPacketizer2 &);  // error on accidental usage

public:
   TPacketizer2(TDSet *dset, TList *slaves, Long64_t first, Long64_t num);
   virtual ~TPacketizer2();

   Long64_t      GetEntriesProcessed() const { return fProcessed; }
   Long64_t      GetEntriesProcessed(TSlave *sl) const;
   TDSetElement *GetNextPacket(TSlave *sl);

   ClassDef(TPacketizer2,0)  //Generate work packets for parallel processing
};

#endif
