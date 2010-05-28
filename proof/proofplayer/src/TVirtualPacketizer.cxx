// @(#)root/proofplayer:$Id$
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
// The packetizer is a load balancing object created for each query.    //
// It generates packets to be processed on PROOF worker servers.        //
// A packet is an event range (begin entry and number of entries) or    //
// object range (first object and number of objects) in a TTree         //
// (entries) or a directory (objects) in a file.                        //
// Packets are generated taking into account the performance of the     //
// remote machine, the time it took to process a previous packet on     //
// the remote machine, the locality of the database files, etc.         //
//                                                                      //
// TVirtualPacketizer includes common parts of PROOF packetizers.       //
// Look in subclasses for details.                                      //
// The default packetizer is TPacketizerAdaptive.                       //
// To use an alternative one, for instance - the TPacketizer, call:     //
// proof->SetParameter("PROOF_Packetizer", "TPacketizer");              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TVirtualPacketizer.h"
#include "TEnv.h"
#include "TFile.h"
#include "TTree.h"
#include "TKey.h"
#include "TDSet.h"
#include "TError.h"
#include "TEventList.h"
#include "TEntryList.h"
#include "TMap.h"
#include "TMessage.h"
#include "TObjString.h"

#include "TProof.h"
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
TVirtualPacketizer::TVirtualPacketizer(TList *input, TProofProgressStatus *st)
{
   // Constructor.

   fProgressStatus = st;
   if (!fProgressStatus) {
      Error("TVirtualPacketizer", "No progress status");
      return;
   }
   fTotalEntries = 0;
   fValid = kTRUE;
   fStop = kFALSE;
   fFailedPackets = 0;
   fDataSet = "";
   fSlaveStats = 0;

   // Performance monitoring
   TTime tnow = gSystem->Now();
   fStartTime = Long_t(tnow);
   SetBit(TVirtualPacketizer::kIsInitializing);
   ResetBit(TVirtualPacketizer::kIsDone);
   fInitTime = 0;
   fProcTime = 0;
   fTimeUpdt = -1.;

   // Init circularity ntple for performance calculations
   fCircProg = new TNtupleD("CircNtuple","Circular progress info","tm:ev:mb:rc:al");
   fCircN = 5;
   TProof::GetParameter(input, "PROOF_ProgressCircularity", fCircN);
   fCircProg->SetCircular(fCircN);

   // Check if we need to start the progress timer (multi-packetizers do not want
   // timers from the packetizers they control ...)
   TString startProgress("yes");
   TProof::GetParameter(input, "PROOF_StartProgressTimer", startProgress);

   // Init progress timer, if requested
   fProgress = 0;
   if (startProgress == "yes") {
      Long_t period = 500;
      TProof::GetParameter(input, "PROOF_ProgressPeriod", period);
      fProgress = new TTimer;
      fProgress->SetObject(this);
      fProgress->Start(period, kFALSE);
   }

   // Whether to send estimated values for the progress info
   TString estopt;
   TProof::GetParameter(input, "PROOF_RateEstimation", estopt);
   if (estopt.IsNull()) {
      // Parse option from the env
      estopt = gEnv->GetValue("Proof.RateEstimation", "");
   }
   fUseEstOpt = kEstOff;
   if (estopt == "current")
      fUseEstOpt = kEstCurrent;
   else if (estopt == "average")
      fUseEstOpt = kEstAverage;
}

//______________________________________________________________________________
TVirtualPacketizer::~TVirtualPacketizer()
{
   // Destructor.

   SafeDelete(fCircProg);
   SafeDelete(fProgress);
   SafeDelete(fFailedPackets);
   fProgressStatus = 0; // belongs to the player
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
      TTree *t = (TTree *) key->ReadObj();
      if ( t == 0 ) {
         // Error always reported?
         delete file;
         return -1;
      }
      entries = (Long64_t) t->GetEntries();
      delete t;

   } else {
      TList *keys = dir->GetListOfKeys();
      entries = keys->GetSize();
   }

   delete file;

   return entries;
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
TDSetElement* TVirtualPacketizer::CreateNewPacket(TDSetElement* base,
                                                  Long64_t first, Long64_t num)
{
   // Creates a new TDSetElement from from base packet starting from
   // the first entry with num entries.
   // The function returns a new created objects which have to be deleted.

   TDSetElement* elem = new TDSetElement(base->GetFileName(), base->GetObjName(),
                                         base->GetDirectory(), first, num,
                                         0, fDataSet.Data());

   // create TDSetElements for all the friends of elem.
   TList *friends = base->GetListOfFriends();
   if (friends) {
      TIter nxf(friends);
      TDSetElement *fe = 0;
      while ((fe = (TDSetElement *) nxf())) {
         TDSetElement *xfe = new TDSetElement(fe->GetFileName(), fe->GetObjName(),
                                              fe->GetDirectory(), first, num);
         // The alias, if any, is in the element name options ('friend_alias=<alias>|')
         elem->AddFriend(xfe, 0);
      }
   }

   return elem;
}

//______________________________________________________________________________
Bool_t TVirtualPacketizer::HandleTimer(TTimer *)
{
   // Send progress message to client.

   PDB(kPacketizer,2)
      Info("HandleTimer", "fProgress: %p, isDone: %d",
                          fProgress, TestBit(TVirtualPacketizer::kIsDone));

   if (fProgress == 0 || TestBit(TVirtualPacketizer::kIsDone))
      return kFALSE; // timer stopped already or reports completed

   // Prepare progress info
   TTime tnow = gSystem->Now();
   Float_t now = (Float_t) (Long_t(tnow) - fStartTime) / (Double_t)1000.;
   Long64_t estent = GetEntriesProcessed();
   Long64_t estmb = GetBytesRead();
   Long64_t estrc = GetReadCalls();

   // Times and counters
   Float_t evtrti = -1., mbrti = -1.;
   if (TestBit(TVirtualPacketizer::kIsInitializing)) {
      // Initialization
      fInitTime = now;
   } else {
      // Fill the reference as first
      if (fCircProg->GetEntries() <= 0) {
         fCircProg->Fill((Double_t)0., 0., 0., 0., 0.);
      }
      // Time between updates
      fTimeUpdt = now - fProcTime;
      // Update proc time
      fProcTime = now - fInitTime;
      // Get the last entry
      Double_t *ar = fCircProg->GetArgs();
      fCircProg->GetEntry(fCircProg->GetEntries()-1);
      // The current rate
      Bool_t all = kTRUE;
      evtrti = GetCurrentRate(all);
      Double_t xall = (all) ? 1. : 0.;
      GetEstEntriesProcessed(0, estent, estmb, estrc);
      // Fill entry
      Double_t evts = (Double_t) estent;
      Double_t mbs = (estmb > 0) ? estmb / TMath::Power(2.,20.) : 0.; //--> MB
      Double_t rcs = (Double_t) estrc;
      fCircProg->Fill((Double_t)fProcTime, evts, mbs, rcs, xall);
      fCircProg->GetEntry(fCircProg->GetEntries()-2);
      if (all) {
         Double_t dt = (Double_t)fProcTime - ar[0];
         Long64_t de = (evts > ar[1]) ? (Long64_t) (evts - ar[1]) : 0;
         Long64_t db = (mbs > ar[2]) ? (Long64_t) ((mbs - ar[2])*TMath::Power(2.,20.)) : 0;
         if (gPerfStats)
            gPerfStats->RateEvent((Double_t)fProcTime, dt, de, db);
         // Get the last to spot the cache readings
         Double_t rc = (Double_t)estrc - ar[3];
         mbrti = (rc > 0 && mbs > ar[2]) ? (Float_t) (mbs - ar[2]) / rc : 0. ;
      }
      // Final report only once (to correctly determine the proc time)
      if (fTotalEntries > 0 && GetEntriesProcessed() >= fTotalEntries)
         SetBit(TVirtualPacketizer::kIsDone);
      PDB(kPacketizer,2)
         Info("HandleTimer", "ent:%lld, bytes:%lld, proct:%f, evtrti:%f, mbrti:%f (%f,%f)",
                             estent, estmb, fProcTime, evtrti, mbrti, mbs, ar[2]);
   }

   if (gProofServ) {
      // Message to be sent over
      TMessage m(kPROOF_PROGRESS);
      if (gProofServ->GetProtocol() > 25) {
         // Fill the message now
         TProofProgressInfo pi(fTotalEntries, estent, estmb, fInitTime,
                               fProcTime, evtrti, mbrti, GetActiveWorkers(),
                               gProofServ->GetActSessions(), gProofServ->GetEffSessions());
         m << &pi;
      } else if (gProofServ->GetProtocol() > 11) {
         // Fill the message now
         m << fTotalEntries << estent << estmb << fInitTime << fProcTime
           << evtrti << mbrti;
      } else {
         // Old format
         m << fTotalEntries << GetEntriesProcessed();
      }
      // send message to client;
      gProofServ->GetSocket()->Send(m);

   } else {
      if (gProof && gProof->GetPlayer()) {
         // Log locally
         gProof->GetPlayer()->Progress(fTotalEntries, estent, estmb,
                                       fInitTime, fProcTime, evtrti, mbrti);
      }
   }

   // Final report only once (to correctly determine the proc time)
   if (fTotalEntries > 0 && GetEntriesProcessed() >= fTotalEntries)
      SetBit(TVirtualPacketizer::kIsDone);

   return kFALSE; // ignored?
}

//______________________________________________________________________________
void TVirtualPacketizer::SetInitTime()
{
   // Set the initialization time

   if (TestBit(TVirtualPacketizer::kIsInitializing)) {
      fInitTime = (Float_t) (Long_t(gSystem->Now()) - fStartTime) / (Double_t)1000.;
      ResetBit(TVirtualPacketizer::kIsInitializing);
      PDB(kPacketizer,2)
         Info("SetInitTime","fInitTime set to %f s", fInitTime);
   }
}
