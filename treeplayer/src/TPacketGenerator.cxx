// @(#)root/treeplayer:$Name:  $:$Id: TPacketGenerator.cxx,v 1.1.1.1 2000/05/16 17:00:44 rdm Exp $
// Author: Fons Rademakers   28/03/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPacketGenerator                                                     //
//                                                                      //
// This class generates packets to be processed on PROOF slave servers. //
// A packet is an entry range (begin entry and number of entries).      //
// Packets are generated taking into account the performance of the     //
// remote machine, the time it took to process a previous packet on     //
// the remote machine, the locality of the database files, etc.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TPacketGenerator.h"
#include "TObjArray.h"
#include "TList.h"
#include "TSlave.h"
#include "TTree.h"
#include "TTime.h"
#include "TSystem.h"


// TPacket class used by the TPacketGenerator.

class TPacket : public TObject {

private:
   TSlave   *fSlave;    // slave on which packet has been processed
   Stat_t    fFirst;    // entry number of first entry in packet
   Int_t     fNentries; // number of entries in packet
   TTime     fTime;     // time when packet was generated

public:
   TPacket(TSlave *sl, Stat_t first, Int_t nentries);
   ~TPacket() { }

   Stat_t  GetFirstEntry() const { return fFirst; }
   Int_t   GetPacketSize() const { return fNentries; }
   TSlave *GetSlave() const { return fSlave; }
   TTime   GetTime() const { return fTime; }
   void    Print(Option_t *option ="");
};

TPacket::TPacket(TSlave *sl, Stat_t first, Int_t nentries)
{
   fSlave    = sl;
   fFirst    = first;
   fNentries = nentries;
   fTime     = gSystem->Now();
}

void TPacket::Print(Option_t *)
{
   Printf("    Processed on host:     %s",   fSlave->GetName());
   Printf("    Begin entry:           %.0f", fFirst);
   Printf("    Number of entries:     %d",   fNentries);
   Printf("    Starting time:         %lu",  long(fTime));
}


ClassImp(TPacketGenerator)

//______________________________________________________________________________
TPacketGenerator::TPacketGenerator()
{
   // Private default ctor.

   fTree            = 0;
   fSlaves          = 0;
   fLastPackets     = 0;
   fPackets         = 0;
   fEntriesPerSlave = 0;
}

//______________________________________________________________________________
TPacketGenerator::TPacketGenerator(Stat_t firstEntry, Stat_t lastEntry, TTree *tree, TList *slaves)
{
   // Initialize TPacketGenerator using a TTree and a list of slave processors.

   fNextEntry         = firstEntry;
   fLastEntry         = lastEntry;
   fTree              = tree;
   fInitialPacketSize = fTree->GetPacketSize();
   fSlaves            = slaves;

   Init();
}

//______________________________________________________________________________
TPacketGenerator::TPacketGenerator(Stat_t firstEntry, Stat_t lastEntry, Int_t packetSize, TList *slaves)
{
   // Initialize TPacketGenerator using the total number of entries, the
   // initial packet size and the list of slaves.

   fNextEntry         = firstEntry;
   fLastEntry         = lastEntry;
   fInitialPacketSize = packetSize;
   fSlaves            = slaves;
   fTree              = 0;

   Init();
}

//______________________________________________________________________________
TPacketGenerator::~TPacketGenerator()
{
   // TPacketGenerator dtor. Deletes the list of packets.

   if (fPackets) fPackets->Delete();
   delete fPackets;
   delete fLastPackets;
   delete [] fEntriesPerSlave;
}

//______________________________________________________________________________
void TPacketGenerator::Init()
{
   // Initialize the TPacketGenerator. Called by the ctors.

   // loop over all slaves and find highest ordinal number
   TIter   next(fSlaves);
   TSlave *sl;
   Int_t   mxo = 0;
   while ((sl = (TSlave*) next()))
      mxo = TMath::Max(mxo, sl->GetOrdinal());
   fMaxOrd = mxo;
   mxo++;

   fEntriesProcessed = 0;
   fMinPacketSize    = fInitialPacketSize/5;  // 20% of initial packet size
   fSmallestPacket   = fInitialPacketSize;
   fEntriesPerSlave  = new Stat_t[mxo];
   fLastPackets      = new TObjArray(mxo);
   fPackets          = new TList;
}

//______________________________________________________________________________
Stat_t TPacketGenerator::GetEntriesProcessed(TSlave *sl) const
{
   // Return number of entries processed by slave sl.

   return fEntriesPerSlave[sl->GetOrdinal()];
}

//______________________________________________________________________________
Bool_t TPacketGenerator::GetNextPacket(TSlave *sl, Int_t &nentry, Stat_t &first)
{
   // Get next packet for slave sl. Returns kTRUE and sets first and nentry.
   // In case no more packets return kFALSE (and set first and nentry == 0).
   // The packet size for a slave is a linear function of the time diff between
   // the processing of two packets relative to the fastest slave (i.e. the one
   // with the smallest time diff). The fastest slave always processes
   // packets of size fTree->GetPacketSize() (or of the size specified in
   // the TPacketGenerator ctor).

   Int_t packetSize = fInitialPacketSize;

   // Update the statistics and calculate new packet size for the slave
   if (fLastPackets->UncheckedAt(sl->GetOrdinal())) {
      TPacket *p = (TPacket*)fLastPackets->UncheckedAt(sl->GetOrdinal());
      fEntriesPerSlave[sl->GetOrdinal()] += p->GetPacketSize();
      fEntriesProcessed += p->GetPacketSize();
      packetSize = p->GetPacketSize();
      TTime tdiff = gSystem->Now() - p->GetTime();
      if (long(fMinTimeDiff) == 0)
         fMinTimeDiff = tdiff;
      else if (tdiff < fMinTimeDiff) {
         if (packetSize == fInitialPacketSize)
            fMinTimeDiff = tdiff;
         else {
            Double_t frac = long(fMinTimeDiff - tdiff);
            frac /= long(fMinTimeDiff);
            packetSize += Int_t(frac*packetSize);
            if (packetSize > fInitialPacketSize)
               packetSize = fInitialPacketSize;
         }
      } else {
         Double_t frac = long(tdiff - fMinTimeDiff);
         frac /= long(fMinTimeDiff);
         packetSize -= Int_t(frac*packetSize);
         if (packetSize < fMinPacketSize) packetSize = fMinPacketSize;
         if (packetSize < fSmallestPacket) fSmallestPacket = packetSize;
      }
   }

   // If all entries have been processed return kFALSE
   if (fNextEntry > fLastEntry) {
      fLastPackets->AddAt(0, sl->GetOrdinal());
      first  = 0;
      nentry = -1;
      return kFALSE;
   }

   if (fNextEntry + packetSize > fLastEntry+1)
      packetSize = Int_t(fLastEntry - fNextEntry + 1);

   TPacket *pn = new TPacket(sl, fNextEntry, packetSize);
   fPackets->Add(pn);
   fLastPackets->AddAt(pn, sl->GetOrdinal());
   fNextEntry += packetSize;

   first  = pn->GetFirstEntry();
   nentry = pn->GetPacketSize();
   return kTRUE;
}

//______________________________________________________________________________
Int_t TPacketGenerator::GetNoPackets() const
{
   // Number of packets generated.

   if (fPackets)
      return fPackets->GetSize();
   return 0;
}

//______________________________________________________________________________
void TPacketGenerator::Print(Option_t *option) const
{
   // Prints info about the packet generator. If option is "all", print also
   // info about all packets.

   Float_t tott = 0.0;
   Float_t aves = 0.0;
   Float_t avet = 0.0;
   if (fPackets) {
      tott = (long(((TPacket*)fPackets->Last())->GetTime()) -
              long(((TPacket*)fPackets->First())->GetTime())) / 1000.0;
      aves = fEntriesProcessed / fPackets->GetSize();
      avet = (long(((TPacket*)fPackets->Last())->GetTime()) -
              long(((TPacket*)fPackets->First())->GetTime())) /
              Float_t(fPackets->GetSize());
   }

   // Print generator summary
   Printf("Total entries processed:            %.0f", fEntriesProcessed);
   Printf("Total number of packets:            %d",   fPackets ? fPackets->GetSize() : 0);
   Printf("Default packet size:                %d",   fInitialPacketSize);
   Printf("Smallest packet size:               %d",   fSmallestPacket);
   Printf("Average packet size:                %.2f", aves);
   Printf("Total time (s):                     %.2f", tott);
   Printf("Average time between packets (ms):  %.2f", avet);
   Printf("Shortest time for packet (ms):      %ld",  long(fMinTimeDiff));
   Printf("Number of active slaves:            %d",   fSlaves->GetSize());

   TIter   next(fSlaves);
   TSlave *sl;
   int     i = 0;
   while ((sl = (TSlave*) next())) {
      Printf("   Number of entries processed by slave %d:  %.0f", i, fEntriesPerSlave[sl->GetOrdinal()]);
      i++;
   }

   if (fPackets && !strcasecmp(option,"all")) {
      fPackets->ForEach(TPacket,Print)(option);
   }
}

//______________________________________________________________________________
void TPacketGenerator::Reset(Stat_t firstEntry, Stat_t lastEntry, TList *slaves)
{
   // Reset the packet generator for a new cycle.

   // Find highest ordinal slave number
   TIter   next(slaves);
   TSlave *sl;
   Int_t   mxo = 0;
   while ((sl = (TSlave*) next()))
      mxo = TMath::Max(mxo, sl->GetOrdinal());
   mxo++;

   if (fPackets) fPackets->Delete();

   if (mxo != fMaxOrd+1) {
      delete fLastPackets;
      delete [] fEntriesPerSlave;
      fLastPackets      = new TObjArray(mxo);
      fEntriesPerSlave  = new Stat_t[mxo];
      fMaxOrd = mxo - 1;
   } else {
      for (int i = 0; i < mxo; i++) {
         fLastPackets->AddAt(0,i);
         fEntriesPerSlave[i] = 0;
      }
   }

   if (fTree)
      fInitialPacketSize = fTree->GetPacketSize();

   fNextEntry        = firstEntry;
   fLastEntry        = lastEntry;
   fSlaves           = slaves;
   fEntriesProcessed = 0;
   fMinPacketSize    = fInitialPacketSize/5;  // 20% of initial packet size
   fMinTimeDiff      = 0;
   fSmallestPacket   = fInitialPacketSize;
}

//______________________________________________________________________________
void TPacketGenerator::SetMinPacketSize(Int_t minps)
{
   // Set minimum packet size. The default is 20% of the initial packet size
   // (in case of TTree's the initial packet size is set in via
   // TTree::SetPacketSize() and its default is 100).

   if (minps > 0 && minps < fInitialPacketSize)
      fMinPacketSize = minps;
}
