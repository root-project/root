// @(#)root/proofplayer:$Name:  $:$Id: TVirtualPacketizer.cxx,v 1.10 2007/05/29 16:06:55 ganis Exp $
// Author: Maarten Ballintijn    9/7/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualPacketizer                                                   //
//                                                                      //
// XXX update Comment XXX                                               //
// Packetizer generates packets to be processed on PROOF worker servers.//
// A packet is an event range (begin entry and number of entries) or    //
// object range (first object and number of objects) in a TTree         //
// (entries) or a directory (objects) in a file.                        //
// Packets are generated taking into account the performance of the     //
// remote machine, the time it took to process a previous packet on     //
// the remote machine, the locality of the database files, etc.         //
//                                                                      //
// TVirtualPacketizer includes common parts of PROOF packetizers.       //
// Look in subclasses for details.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TVirtualPacketizer.h"
#include "TFile.h"
#include "TTree.h"
#include "TKey.h"
#include "TDSet.h"
#include "TError.h"
#include "TEventList.h"
#include "TMap.h"
#include "TMessage.h"
#include "TObjString.h"

#include "TProofDebug.h"
#include "TProofPlayer.h"
#include "TProofServ.h"
#include "TSlave.h"
#include "TSocket.h"
#include "TTimer.h"
#include "TUrl.h"
#include "TMath.h"
#include "TMonitor.h"
#include "TNtupleD.h"
#include "TPerfStats.h"

ClassImp(TVirtualPacketizer)

//______________________________________________________________________________
TVirtualPacketizer::TVirtualPacketizer()
{
   // Constructor.

   fValid = kTRUE;
   fStop = kFALSE;
   ResetBit(TVirtualPacketizer::kIsInitializing);
}

//______________________________________________________________________________
Long64_t TVirtualPacketizer::GetEntries(Bool_t tree, TDSetElement *e)
{
   // Get entries.

   Long64_t entries;
   TFile *file = TFile::Open(e->GetFileName());

   if ( file->IsZombie() ) {
      Error("GetEntries","Cannot open file: %s (%s)",
            e->GetFileName(), strerror(file->GetErrno()) );
      return -1;
   }

   TDirectory *dirsave = gDirectory;
   if ( ! file->cd(e->GetDirectory()) ) {
      Error("GetEntries","Cannot cd to: %s", e->GetDirectory() );
      delete file;
      return -1;
   }
   TDirectory *dir = gDirectory;
   dirsave->cd();

   if ( tree ) {
      TKey *key = dir->GetKey(e->GetObjName());
      if ( key == 0 ) {
         Error("GetEntries","Cannot find tree \"%s\" in %s",
               e->GetObjName(), e->GetFileName() );
         delete file;
         return -1;
      }
      TTree *tree = (TTree *) key->ReadObj();
      if ( tree == 0 ) {
         // Error always reported?
         delete file;
         return -1;
      }
      entries = (Long64_t) tree->GetEntries();
      delete tree;

   } else {
      TList *keys = dir->GetListOfKeys();
      entries = keys->GetSize();
   }

   delete file;

   return entries;
}

//______________________________________________________________________________
Long64_t TVirtualPacketizer::GetEntriesProcessed(TSlave *) const
{
   // Get Entries processed by the given slave.

   AbstractMethod("GetEntriesProcessed");
   return 0;
}

//______________________________________________________________________________
TDSetElement *TVirtualPacketizer::GetNextPacket(TSlave *, TMessage *)
{
   // Get next packet.

   AbstractMethod("GetNextPacket");
   return 0;
}

//______________________________________________________________________________
void TVirtualPacketizer::StopProcess(Bool_t /*abort*/)
{
   // Stop process.

   fStop = kTRUE;
}

//______________________________________________________________________________
void TVirtualPacketizer::SplitEventList(TDSet *dset)
{
   // Splits the eventlist into parts for each file.
   // Each part is assigned to the apropriate TDSetElement.

   TEventList *mainList = dset->GetEventList();
   R__ASSERT(mainList);

   TIter next(dset->GetListOfElements());
   TDSetElement *el, *prev;

   prev = dynamic_cast<TDSetElement*> (next());
   if (!prev)
      return;
   Long64_t low = prev->GetTDSetOffset();
   Long64_t high = low;
   Long64_t currPos = 0;
   do {
      el = dynamic_cast<TDSetElement*> (next());
      if (el == 0)
         high = kMaxLong64;         // infinity
      else
         high = el->GetTDSetOffset();

#ifdef DEBUG
      while (currPos < mainList->GetN() && mainList->GetEntry(currPos) < low) {
         Error("SplitEventList", "event outside of the range of any of the TDSetElements");
         currPos++;        // unnecessary check
      }
#endif

      TEventList* newEventList = new TEventList();
      while (currPos < mainList->GetN() && mainList->GetEntry((Int_t)currPos) < high) {
         newEventList->Enter(mainList->GetEntry((Int_t)currPos) - low);
         currPos++;
      }
      prev->SetEventList(newEventList);
      prev->SetNum(newEventList->GetN());
      low = high;
      prev = el;
   } while (el);
}

//______________________________________________________________________________
TDSetElement* TVirtualPacketizer::CreateNewPacket(TDSetElement* base,
                                                  Long64_t first, Long64_t num)
{
   // Creates a new TDSetElement from from base packet starting from
   // the first entry with num entries.
   // The function returns a new created objects which have to be deleted.

   TDSetElement* elem = new TDSetElement(base->GetFileName(), base->GetObjName(),
                                         base->GetDirectory(), first, num);

   // create TDSetElements for all the friends of elem.
   TList *friends = base->GetListOfFriends();
   if (friends) {
      TIter nxf(friends);
      TPair *p = 0;
      while ((p = (TPair *) nxf())) {
         TDSetElement *fe = (TDSetElement *) p->Key();
         elem->AddFriend(new TDSetElement(fe->GetFileName(), fe->GetObjName(),
                                          fe->GetDirectory(), first, num),
                                         ((TObjString *)(p->Value()))->GetName());
      }
   }

   return elem;
}

//______________________________________________________________________________
Bool_t TVirtualPacketizer::HandleTimer(TTimer *)
{
   // Send progress message to client.

   if (fProgress == 0) return kFALSE; // timer stopped already

   // Message to be sent over
   TMessage m(kPROOF_PROGRESS);

   if (gProofServ->GetProtocol() > 11) {

      // Prepare progress info
      TTime tnow = gSystem->Now();
      Float_t now = (Float_t) (Long_t(tnow) - fStartTime) / (Double_t)1000.;
      Double_t evts = (Double_t) fProcessed;
      Double_t mbs = (fBytesRead > 0) ? fBytesRead / TMath::Power(2.,20.) : 0.; //�--> MB

      // Times and counters
      Float_t evtrti = -1., mbrti = -1.;
      if (TestBit(TVirtualPacketizer::kIsInitializing)) {
         // Initialization
         fInitTime = now;
      } else {
         // Fill the reference as first
         if (fCircProg->GetEntries() <= 0) {
            fCircProg->Fill((Double_t)0., 0., 0.);
            // Best estimation of the init time
            fInitTime = (now + fInitTime) / 2.;
         }
         // Time between updates
         fTimeUpdt = now - fProcTime;
         // Update proc time
         fProcTime = now - fInitTime;
         // Good entry
         fCircProg->Fill((Double_t)fProcTime, evts, mbs);
         // Instantaneous rates (at least 5 reports)
         if (fCircProg->GetEntries() > 4) {
            Double_t *ar = fCircProg->GetArgs();
            fCircProg->GetEntry(0);
            Double_t dt = (Double_t)fProcTime - ar[0];
            evtrti = (dt > 0) ? (Float_t) (evts - ar[1]) / dt : -1. ;
            mbrti = (dt > 0) ? (Float_t) (mbs - ar[2]) / dt : -1. ;
            if (gPerfStats != 0)
               gPerfStats->RateEvent((Double_t)fProcTime, dt,
                                     (Long64_t) (evts - ar[1]),
                                     (Long64_t) ((mbs - ar[2])*TMath::Power(2.,20.)));
         }
      }

      // Fill the message now
      m << fTotalEntries << fProcessed << fBytesRead << fInitTime << fProcTime
        << evtrti << mbrti;


   } else {
      // Old format
      m << fTotalEntries << fProcessed;
   }

   // send message to client;
   gProofServ->GetSocket()->Send(m);

   return kFALSE; // ignored?
}

//______________________________________________________________________________
void TVirtualPacketizer::SetInitTime()
{
   // Set the initialization time

   if (TestBit(TVirtualPacketizer::kIsInitializing)) {
      fInitTime = (Float_t) (Long_t(gSystem->Now()) - fStartTime) / (Double_t)1000.;
      ResetBit(TVirtualPacketizer::kIsInitializing);
   }
   PDB(kPacketizer,2)
      Info("SetInitTime","fInitTime: %f s", fInitTime);
}
