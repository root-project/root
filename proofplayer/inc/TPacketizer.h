// @(#)root/proof:$Name:  $:$Id: TPacketizer.h,v 1.3 2002/06/14 15:14:31 rdm Exp $
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


class TPacketizer : public TVirtualPacketizer {

private:
   Int_t    fProcessed;       // number of entries processed
   Long64_t fTotalEntries;    // Total number of entries to be distributed

   TList   *fFileNodes;       // nodes with files
   TList   *fUnAllocated;     // nodes with unallocated files
   TObject *fUnAllocNext;     // cursor in fUnAllocated
   TList   *fActive;          // nodes with unfinished files
   TObject *fActiveNext;      // cursor in fActive
   TList   *fSlaves;          // slaves processing

   TPacketizer();
   TPacketizer(const TPacketizer &);     // no implementation, will generate
   void operator=(const TPacketizer &);  // error on accidental usage

public:
   TPacketizer(TDSet *dset, TList *slaves, Long64_t first, Long64_t num);
   virtual ~TPacketizer();

   Long64_t      GetEntriesProcessed() const { return fProcessed; }
   Long64_t      GetEntriesProcessed(TSlave *sl) const;
   TDSetElement  *GetNextPacket(TSlave *sl);

   ClassDef(TPacketizer,0)  //Generate work packets for parallel processing
};

#endif
