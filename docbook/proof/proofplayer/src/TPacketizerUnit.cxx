// @(#)root/proofplayer:$Id$
// Author: Long Tran-Thanh    22/07/07

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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


#include "TPacketizerUnit.h"

#include "Riostream.h"
#include "TDSet.h"
#include "TError.h"
#include "TEventList.h"
#include "TMap.h"
#include "TMessage.h"
#include "TMonitor.h"
#include "TNtupleD.h"
#include "TObject.h"
#include "TParameter.h"
#include "TPerfStats.h"
#include "TProofDebug.h"
#include "TProof.h"
#include "TProofPlayer.h"
#include "TProofServ.h"
#include "TSlave.h"
#include "TSocket.h"
#include "TStopwatch.h"
#include "TTimer.h"
#include "TUrl.h"
#include "TClass.h"
#include "TMath.h"
#include "TObjString.h"


using namespace TMath;
//
// The following utility class manage the state of the
// work to be performed and the slaves involved in the process.
//
// The list of TSlaveStat(s) keep track of the work (being) done
// by each slave
//

//------------------------------------------------------------------------------

class TPacketizerUnit::TSlaveStat : public TVirtualPacketizer::TVirtualSlaveStat {

friend class TPacketizerUnit;

private:
   Long64_t  fLastProcessed; // number of processed entries of the last packet
   Double_t  fSpeed;         // estimated current average speed of the processing slave
   Double_t  fTimeInstant;   // stores the time instant when the current packet started
   TNtupleD *fCircNtp;       // Keeps circular info for speed calculations
   Long_t    fCircLvl;       // Circularity level

public:
   TSlaveStat(TSlave *sl, TList *input);
   ~TSlaveStat();

   void        GetCurrentTime();

   void        UpdatePerformance(Double_t time);
   TProofProgressStatus *AddProcessed(TProofProgressStatus *st);
};

//______________________________________________________________________________
TPacketizerUnit::TSlaveStat::TSlaveStat(TSlave *slave, TList *input)
                            : fLastProcessed(0),
                              fSpeed(0), fTimeInstant(0), fCircLvl(5)
{
   // Main constructor

   // Initialize the circularity ntple for speed calculations
   fCircNtp = new TNtupleD("Speed Circ Ntp", "Circular process info","tm:ev");
   TProof::GetParameter(input, "PROOF_TPacketizerUnitCircularity", fCircLvl);
   fCircLvl = (fCircLvl > 0) ? fCircLvl : 5;
   fCircNtp->SetCircular(fCircLvl);
   fSlave = slave;
   fStatus = new TProofProgressStatus();
}

//______________________________________________________________________________
TPacketizerUnit::TSlaveStat::~TSlaveStat()
{
   // Destructor

   SafeDelete(fCircNtp);
}

//______________________________________________________________________________
void TPacketizerUnit::TSlaveStat::UpdatePerformance(Double_t time)
{
   // Update the circular ntple

   Double_t ttot = time;
   Double_t *ar = fCircNtp->GetArgs();
   Int_t ne = fCircNtp->GetEntries();
   if (ne <= 0) {
      // First call: just fill one ref entry and return
      fCircNtp->Fill(0., 0);
      fSpeed = 0.;
      return;
   }
   // Fill the entry
   fCircNtp->GetEntry(ne-1);
   ttot = ar[0] + time;
   fCircNtp->Fill(ttot, GetEntriesProcessed());

   // Calculate the speed
   fCircNtp->GetEntry(0);
   Double_t dtime = (ttot > ar[0]) ? ttot - ar[0] : ne+1 ;
   Long64_t nevts = GetEntriesProcessed() - (Long64_t)ar[1];
   fSpeed = nevts / dtime;
   PDB(kPacketizer,2)
      Info("UpdatePerformance", "time:%f, dtime:%f, nevts:%lld, speed: %f",
                                time, dtime, nevts, fSpeed);

}

//______________________________________________________________________________
TProofProgressStatus *TPacketizerUnit::TSlaveStat::AddProcessed(TProofProgressStatus *st)
{
   // Update the status info to the 'st'.
   // return the difference (*st - *fStatus)

   if (st) {
      // The entriesis not correct in 'st'
      Long64_t lastEntries = st->GetEntries() - fStatus->GetEntries();
      // The last proc time should not be added
      fStatus->SetLastProcTime(0.);
      // Get the diff
      TProofProgressStatus *diff = new TProofProgressStatus(*st - *fStatus);
      *fStatus += *diff;
      // Set the correct value
      fStatus->SetLastEntries(lastEntries);
      return diff;
   } else {
      Error("AddProcessed", "status arg undefined");
      return 0;
   }
}

//------------------------------------------------------------------------------

ClassImp(TPacketizerUnit)

//______________________________________________________________________________
TPacketizerUnit::TPacketizerUnit(TList *slaves, Long64_t num, TList *input,
                                 TProofProgressStatus *st)
                : TVirtualPacketizer(input, st)
{
   // Constructor

   PDB(kPacketizer,1) Info("TPacketizerUnit", "enter (num %lld)", num);

   // Init pointer members
   fSlaveStats = 0;
   fPackets = 0;

   Int_t fixednum = -1;
   if (TProof::GetParameter(input, "PROOF_PacketizerFixedNum", fixednum) != 0 || fixednum <= 0)
      fixednum = 0;
   if (fixednum == 1)
      Info("TPacketizerUnit", "forcing the same cycles on each worker");

   if (TProof::GetParameter(input, "PROOF_PacketizerCalibNum", fCalibNum) != 0 || fCalibNum <= 0)
      fCalibNum = 5;
   PDB(kPacketizer,1)
      Info("TPacketizerUnit", "size of the calibration packets: %lld", fCalibNum);

   fMaxPacketTime = 1.;
   Double_t timeLimit = -1;
   if (TProof::GetParameter(input, "PROOF_PacketizerTimeLimit", timeLimit) == 0) {
      fMaxPacketTime = timeLimit;
      Warning("TPacketizerUnit", "PROOF_PacketizerTimeLimit is deprecated: use PROOF_MaxPacketTime instead");
   }
   PDB(kPacketizer,1)
      Info("TPacketizerUnit", "time limit is %lf", fMaxPacketTime);
   fProcessing = 0;
   fAssigned = 0;

   fStopwatch = new TStopwatch();

   fPackets = new TList;
   fPackets->SetOwner();

   fSlaveStats = new TMap;
   fSlaveStats->SetOwner(kFALSE);

   TSlave *slave;
   TIter si(slaves);
   while ((slave = (TSlave*) si.Next()))
      fSlaveStats->Add(slave, new TSlaveStat(slave, input));

   fTotalEntries = num;
   fNumPerWorker = -1;
   if (fixednum == 1 && fSlaveStats->GetSize() > 0) {
      // Approximate number: the exact number is determined in GetNextPacket
      fNumPerWorker = fTotalEntries / fSlaveStats->GetSize();
      if (fNumPerWorker == 0) fNumPerWorker = 1;
      if (fCalibNum >= fNumPerWorker) fCalibNum = 1;
   }

   // Save the config parameters in the dedicated list so that they will be saved
   // in the outputlist and therefore in the relevant TQueryResult
   fConfigParams->Add(new TParameter<Long64_t>("PROOF_PacketizerFixedNum", fNumPerWorker));
   fConfigParams->Add(new TParameter<Long64_t>("PROOF_PacketizerCalibNum", fCalibNum));

   fStopwatch->Start();
   PDB(kPacketizer,1) Info("TPacketizerUnit", "return");
}

//______________________________________________________________________________
TPacketizerUnit::~TPacketizerUnit()
{
   // Destructor.

   if (fSlaveStats)
      fSlaveStats->DeleteValues();
   SafeDelete(fSlaveStats);
   SafeDelete(fPackets);
   SafeDelete(fStopwatch);
}

//______________________________________________________________________________
Double_t TPacketizerUnit::GetCurrentTime()
{
   // Get current time

   Double_t retValue = fStopwatch->RealTime();
   fStopwatch->Continue();
   return retValue;
}

//______________________________________________________________________________
Float_t TPacketizerUnit::GetCurrentRate(Bool_t &all)
{
   // Get Estimation of the current rate; just summing the current rates of
   // the active workers

   all = kTRUE;
   // Loop over the workers
   Float_t currate = 0.;
   if (fSlaveStats && fSlaveStats->GetSize() > 0) {
      TIter nxw(fSlaveStats);
      TObject *key;
      while ((key = nxw()) != 0) {
         TSlaveStat *slstat = (TSlaveStat *) fSlaveStats->GetValue(key);
         if (slstat && slstat->GetProgressStatus() && slstat->GetEntriesProcessed() > 0) {
            // Sum-up the current rates
            currate += slstat->GetProgressStatus()->GetCurrentRate();
         } else {
            all = kFALSE;
         }
      }
   }
   // Done
   return currate;
}

//______________________________________________________________________________
TDSetElement *TPacketizerUnit::GetNextPacket(TSlave *sl, TMessage *r)
{
   // Get next packet

   if (!fValid)
      return 0;

   // Find slave
   TSlaveStat *slstat = (TSlaveStat*) fSlaveStats->GetValue(sl);
   R__ASSERT(slstat != 0);

   PDB(kPacketizer,2)
      Info("GetNextPacket","worker-%s: fAssigned %lld\t", sl->GetOrdinal(), fAssigned);

   // Update stats & free old element
   Double_t latency = 0., proctime = 0., proccpu = 0.;
   Long64_t bytesRead = -1;
   Long64_t totalEntries = -1; // used only to read an old message type
   Long64_t totev = 0;
   Long64_t numev = -1;

   TProofProgressStatus *status = 0;
   if (sl->GetProtocol() > 18) {
      (*r) >> latency;
      (*r) >> status;

      // Calculate the progress made in the last packet
      TProofProgressStatus *progress = 0;
      if (status) {
         // upadte the worker status
         numev = status->GetEntries() - slstat->GetEntriesProcessed();
         progress = slstat->AddProcessed(status);
         if (progress) {
            // (*fProgressStatus) += *progress;
            proctime = progress->GetProcTime();
            proccpu  = progress->GetCPUTime();
            totev  = status->GetEntries(); // for backward compatibility
            bytesRead  = progress->GetBytesRead();
            delete progress;
         }
         delete status;
      } else
          Error("GetNextPacket", "no status came in the kPROOF_GETPACKET message");
   } else {

      (*r) >> latency >> proctime >> proccpu;

      // only read new info if available
      if (r->BufferSize() > r->Length()) (*r) >> bytesRead;
      if (r->BufferSize() > r->Length()) (*r) >> totalEntries;
      if (r->BufferSize() > r->Length()) (*r) >> totev;

      numev = totev - slstat->GetEntriesProcessed();
      slstat->GetProgressStatus()->IncEntries(numev);
   }

   fProgressStatus->IncEntries(numev);

   fProcessing = 0;

   PDB(kPacketizer,2)
      Info("GetNextPacket","worker-%s (%s): %lld %7.3lf %7.3lf %7.3lf %lld",
                           sl->GetOrdinal(), sl->GetName(),
                           numev, latency, proctime, proccpu, bytesRead);

   if (gPerfStats != 0) {
      gPerfStats->PacketEvent(sl->GetOrdinal(), sl->GetName(), "", numev,
                              latency, proctime, proccpu, bytesRead);
   }

   if (fNumPerWorker > 0 && slstat->GetEntriesProcessed() >= fNumPerWorker) {
      PDB(kPacketizer,2)
         Info("GetNextPacket","worker-%s (%s) is done (%lld cycles)",
                           sl->GetOrdinal(), sl->GetName(), slstat->GetEntriesProcessed());
      return 0;
   }

   if (fAssigned == fTotalEntries) {
      // Send last timer message
      HandleTimer(0);
      return 0;
   }

   if (fStop) {
      // Send last timer message
      HandleTimer(0);
      return 0;
   }


   Long64_t num;

   // Get the current time
   Double_t cTime = GetCurrentTime();

   if (slstat->fCircNtp->GetEntries() <= 0) {
      // The calibration phase
      PDB(kPacketizer,2)
         Info("GetNextPacket", "calibration: total entries %lld, workers %d",
                               fTotalEntries, fSlaveStats->GetSize());
      Long64_t avg = fTotalEntries / fSlaveStats->GetSize();
      num = (avg > fCalibNum) ? fCalibNum : avg;

      // Create a reference entry
      slstat->UpdatePerformance(0.);

   } else {
      if (fNumPerWorker < 0) {
         // Schedule tasks for workers based on the currently estimated processing speeds

         // Update performances
         // slstat->fStatus was updated before;
         slstat->UpdatePerformance(proctime);

         TMapIter *iter = (TMapIter *)fSlaveStats->MakeIterator();
         TSlave * tmpSlave;
         TSlaveStat * tmpStat;

         Double_t sumSpeed = 0;
         Double_t sumBusy = 0;

         // The basic idea is to estimate the optimal finishing time for the process, assuming
         // that the currently measured processing speeds of the slaves remain the same in the future.
         // The optTime can be calculated as follows.
         // Let s_i be the speed of worker i. We assume that worker i will be busy in the
         // next b_i time instant. Therefore we have the following equation:
         //                 SUM((optTime-b_i)*s_i) = (remaining_entries),
         // Hence optTime can be calculated easily:
         //                 optTime = ((remaining_entries) + SUM(b_i*s_i))/SUM(s_i)

         while ((tmpSlave = (TSlave *)iter->Next())) {
            tmpStat = (TSlaveStat *)fSlaveStats->GetValue(tmpSlave);
            // If the slave doesn't response for a long time, its service rate will be considered as 0
            if ((cTime - tmpStat->fTimeInstant) > 4*fMaxPacketTime)
               tmpStat->fSpeed = 0;
            PDB(kPacketizer,2)
               Info("GetNextPacket",
                  "worker-%s: speed %lf", tmpSlave->GetOrdinal(), tmpStat->fSpeed);
            if (tmpStat->fSpeed > 0) {
               // Calculating the SUM(s_i)
               sumSpeed += tmpStat->fSpeed;
               // There is nothing to do if slave_i is not calibrated or slave i is the current slave
               if ((tmpStat->fTimeInstant) && (cTime - tmpStat->fTimeInstant > 0)) {
                  // Calculating the SUM(b_i*s_i)
                  //      s_i = tmpStat->fSpeed
                  //      b_i = tmpStat->fTimeInstant + tmpStat->fLastProcessed/s_i - cTime)
                  //  b_i*s_i = (tmpStat->fTimeInstant - cTime)*s_i + tmpStat->fLastProcessed
                  Double_t busyspeed =
                     ((tmpStat->fTimeInstant - cTime)*tmpStat->fSpeed + tmpStat->fLastProcessed);
                  if (busyspeed > 0)
                     sumBusy += busyspeed;
               }
            }
         }
         SafeDelete(iter);
         PDB(kPacketizer,2)
            Info("GetNextPacket", "worker-%s: sum speed: %lf, sum busy: %f",
                                 sl->GetOrdinal(), sumSpeed, sumBusy);
         // firstly the slave will try to get all of the remaining entries
         if (slstat->fSpeed > 0 &&
            (fTotalEntries - fAssigned)/(slstat->fSpeed) < fMaxPacketTime) {
            num = (fTotalEntries - fAssigned);
         } else {
            if (slstat->fSpeed > 0) {
               // calculating the optTime
               Double_t optTime = (fTotalEntries - fAssigned + sumBusy)/sumSpeed;
               // if optTime is greater than the official time limit, then the slave gets a number
               // of entries that still fit into the time limit, otherwise it uses the optTime as
               // a time limit
               num = (optTime > fMaxPacketTime) ? Nint(fMaxPacketTime*(slstat->fSpeed))
                                          : Nint(optTime*(slstat->fSpeed));
            PDB(kPacketizer,2)
               Info("GetNextPacket", "opTime %lf num %lld speed %lf", optTime, num, slstat->fSpeed);
            } else {
               Long64_t avg = fTotalEntries/(fSlaveStats->GetSize());
               num = (avg > 5) ? 5 : avg;
            }
         }
      } else {
         // Fixed number of cycles per worker
         num = fNumPerWorker - slstat->fLastProcessed;
         if (num > 1 && slstat->fSpeed > 0 && num / slstat->fSpeed > fMaxPacketTime) {
            num = (Long64_t) (slstat->fSpeed * fMaxPacketTime);
         }
      }
   }
   // Minimum packet size
   num = (num > 1) ? num : 1;
   fProcessing = (num < (fTotalEntries - fAssigned)) ? num
                                                     : (fTotalEntries - fAssigned);

   // Set the informations of the current slave
   slstat->fLastProcessed = fProcessing;
   // Set the start time of the current packet
   slstat->fTimeInstant = cTime;

   PDB(kPacketizer,2)
      Info("GetNextPacket", "worker-%s: num %lld, processing %lld, remaining %lld",sl->GetOrdinal(),
                            num, fProcessing, (fTotalEntries - fAssigned - fProcessing));
   TDSetElement *elem = new TDSetElement("", "", "", fAssigned, fProcessing);
   elem->SetBit(TDSetElement::kEmpty);

   // Update the total counter
   fAssigned += slstat->fLastProcessed;

   return elem;
}
