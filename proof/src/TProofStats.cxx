// @(#)root/proof:$Name:  $:$Id: TProofStats.cxx,v 1.1 2004/05/18 11:32:49 rdm Exp $
// Author: Kristjan Gulbrandsen   11/05/04

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofStats                                                          //
//                                                                      //
// Provides the interface for the PROOF internal performance measurment //
// and event tracing                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TProofStats.h"

#include "TTree.h"
#include "TCollection.h"
#include "TSlave.h"
#include "TH1.h"
#include "TH2.h"
#include "TProofDebug.h"


ClassImp(TProofEvent)
ClassImp(TProofStats)


//------------------------------------------------------------------------------
TProofStats::TProofStats(Int_t nslaves, TList *output, Bool_t doHist, Bool_t doTrace)
   : fTrace(0), fProofEvent(0), fPacketsHist(0), fEventsHist(0), fLatencyHist(0),
      fProcTimeHist(0), fCpuTimeHist(0)
{
   // Normal Constructor

   PDB(kGlobal,1) Info("TProofStats", "Statistics for %d slave(s)", nslaves);

   if (doTrace) {
      // Construct tree
      fTrace = new TTree("proofstats","PROOF Statistics");
      fTrace->SetDirectory(0);
      fTrace->Bronch("ProofEvents","TProofEvent",&fProofEvent,64000,0);
      output->Add(fTrace);
   }

   if (doHist) {
      // Make Histograms
      Double_t time_per_bin = 1e-2; // 10ms
      Double_t min_time = 0;
      Int_t ntime_bins = 200;

      fPacketsHist = new TH1D("PacketsHist", "Packets processed per Slave",
                              nslaves, 0, nslaves);
      fPacketsHist->SetDirectory(0);
      output->Add(fPacketsHist);

      fEventsHist = new TH1D("EventsHist","Events processed per Slave",
                             nslaves, 0, nslaves);
      fEventsHist->SetDirectory(0);
      output->Add(fEventsHist);

      fNodeHist = new TH1D("NodeHist","Slaves per Fileserving Node",
                             0, 0, 0);
      fNodeHist->SetDirectory(0);
      output->Add(fNodeHist);

      fLatencyHist = new TH2D("LatencyHist","GetPacket Latency per Slave",
                              nslaves, 0, nslaves,
                              ntime_bins, min_time, time_per_bin*ntime_bins);
      fLatencyHist->SetDirectory(0);
      output->Add(fLatencyHist);

      fProcTimeHist = new TH2D("ProcTimeHist","Packet Processing Time per Slave",
                               nslaves, 0, nslaves,
                               ntime_bins, min_time, time_per_bin*ntime_bins);
      fProcTimeHist->SetDirectory(0);
      output->Add(fProcTimeHist);

      fCpuTimeHist = new TH2D("CpuTimeHist","Packet CPU Time per Slave",
                              nslaves, 0, nslaves,
                              ntime_bins, min_time, time_per_bin*ntime_bins);
      fCpuTimeHist->SetDirectory(0);
      output->Add(fCpuTimeHist);
   }
}


//______________________________________________________________________________
void TProofStats::SimpleEvent(TProofEvent::EEventType type)
{
   if (type == TProofEvent::kStop && fPacketsHist != 0) {
      fNodeHist->LabelsDeflate("X");
      fNodeHist->LabelsOption("auv","X");
   }

   if (fTrace == 0) return;

   TProofEvent pe; // sets timestamp

   pe.fType = type;

   fProofEvent = &pe;
   fTrace->Fill();
   fProofEvent = 0;
}


//______________________________________________________________________________
void TProofStats::PacketEvent(Int_t slave, const char* slavename, const char* filename,
                              Long64_t eventsprocessed, Double_t latency, Double_t proctime,
                              Double_t cputime)
{

   if (fTrace != 0) {
      TProofEvent pe; // sets timestamp

      pe.fType = TProofEvent::kPacket;
      pe.fSlaveName = slavename;
      pe.fFileName = filename;
      pe.fSlave = slave;
      pe.fEventsProcessed = eventsprocessed;
      pe.fLatency = latency;
      pe.fProcTime = proctime;
      pe.fCpuTime = cputime;

      fProofEvent = &pe;
      fTrace->Fill();
      fProofEvent = 0;
   }

   if (fPacketsHist != 0) {
      fPacketsHist->Fill(slave);
      fEventsHist->Fill(slave, eventsprocessed);
      fLatencyHist->Fill(slave, latency);
      fProcTimeHist->Fill(slave, proctime);
      fCpuTimeHist->Fill(slave, cputime);
   }
}


//______________________________________________________________________________
void TProofStats::FileEvent(Int_t slave, const char *slavename, const char *nodename,
                            const char *filename, Bool_t isStart)
{
   if (fTrace != 0) {
      TProofEvent pe; // sets timestamp

      pe.fType = TProofEvent::kFile;
      pe.fSlaveName = slavename;
      pe.fNodeName = nodename;
      pe.fFileName = filename;
      pe.fSlave = slave;
      pe.fIsStart = isStart;

      fProofEvent = &pe;
      fTrace->Fill();
      fProofEvent = 0;
   }

   if (fPacketsHist != 0) {
      fNodeHist->Fill(nodename, isStart ? 1 : -1);
   }
}
