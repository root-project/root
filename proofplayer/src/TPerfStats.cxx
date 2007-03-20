// @(#)root/proofplayer:$Name:  $:$Id: TPerfStats.cxx,v 1.10 2006/11/28 12:10:52 rdm Exp $
// Author: Kristjan Gulbrandsen   11/05/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPerfStats                                                           //
//                                                                      //
// Provides the interface for the PROOF internal performance measurment //
// and event tracing.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TPerfStats.h"

#include "Riostream.h"
#include "TCollection.h"
#include "TEnv.h"
#include "TError.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProofDebug.h"
#include "TProof.h"
#include "TProofServ.h"
#include "TSlave.h"
#include "TTree.h"


ClassImp(TPerfEvent)
ClassImp(TPerfStats)


//------------------------------------------------------------------------------

//______________________________________________________________________________
TPerfEvent::TPerfEvent(TTimeStamp *offset)
   : fEvtNode("-3"), fType(TVirtualPerfStats::kUnDefined), fSlave(),
     fEventsProcessed(0), fBytesRead(0), fLen(0), fLatency(0.0), fProcTime(0.0), fCpuTime(0.0),
     fIsStart(kFALSE), fIsOk(kFALSE)
{
   // Constructor

   if (gProofServ != 0) {
      fEvtNode = gProofServ->GetOrdinal();
   } else {
      fEvtNode = "-2"; // not on a PROOF server
   }

   if (offset != 0) {
      fTimeStamp = TTimeStamp(fTimeStamp.GetSec() - offset->GetSec(),
                     fTimeStamp.GetNanoSec() - offset->GetNanoSec());
   }
}


//______________________________________________________________________________
Int_t TPerfEvent::Compare(const TObject *obj) const
{
   // Compare method. Must return -1 if this is smaller than obj,
   // 0 if objects are equal and 1 if this is larger than obj.

   const TPerfEvent *pe = dynamic_cast<const TPerfEvent*>(obj);

   R__ASSERT(pe != 0);

   if (fTimeStamp < pe->fTimeStamp) {
      return -1;
   } else if (fTimeStamp == pe->fTimeStamp) {
      return 0;
   } else {
      return 1;
   }
}


//______________________________________________________________________________
void TPerfEvent::Print(Option_t *) const
{
   // Dump content of this instance

   cout << "TPerfEvent: ";

   if ( fEvtNode == -2 ) {
      cout << "StandAlone ";
   } else if ( fEvtNode == -1 ) {
      cout << "Master ";
   } else {
      cout << "Slave " << fEvtNode << " ";
   }
   cout << TVirtualPerfStats::EventType(fType) << " "
        << double(fTimeStamp)
        << endl;
}


//------------------------------------------------------------------------------

//______________________________________________________________________________
TPerfStats::TPerfStats(TList *input, TList *output)
   : fTrace(0), fPerfEvent(0), fPacketsHist(0), fEventsHist(0), fLatencyHist(0),
      fProcTimeHist(0), fCpuTimeHist(0), fDoHist(0),
      fDoTrace(0), fDoTraceRate(0), fDoSlaveTrace(0)
{
   // Normal Constructor.

   Int_t nslaves = 0;
   TProof *proof = gProofServ->GetProof();
   TList *l = proof ? proof->GetSlaveInfo() : 0 ;
   TIter nextslaveinfo(l);
   while (TSlaveInfo *si = dynamic_cast<TSlaveInfo*>(nextslaveinfo()))
      if (si->fStatus == TSlaveInfo::kActive) nslaves++;

   PDB(kGlobal,1) Info("TPerfStats", "Statistics for %d slave(s)", nslaves);

   fDoHist = (input->FindObject("PROOF_StatsHist") != 0);
   fDoTrace = (input->FindObject("PROOF_StatsTrace") != 0);
   fDoTraceRate = (input->FindObject("PROOF_RateTrace") != 0);
   fDoSlaveTrace = (input->FindObject("PROOF_SlaveStatsTrace") != 0);

   if ((gProofServ->IsMaster() && (fDoTrace || fDoTraceRate)) ||
       (!gProofServ->IsMaster() && fDoSlaveTrace)) {
      // Construct tree
      fTrace = new TTree("PROOF_PerfStats", "PROOF Statistics");
      fTrace->SetDirectory(0);
      fTrace->Bronch("PerfEvents", "TPerfEvent", &fPerfEvent, 64000, 0);
      output->Add(fTrace);
   }

   if (fDoHist && gProofServ->IsMaster()) {
      // Make Histograms
      Double_t time_per_bin = 1e-3; // 10ms
      Double_t min_time = 0;
      Int_t ntime_bins = 1000;

      fPacketsHist = new TH1D("PROOF_PacketsHist", "Packets processed per Slave",
                              nslaves, 0, nslaves);
      fPacketsHist->SetDirectory(0);
      fPacketsHist->SetMinimum(0);
      output->Add(fPacketsHist);

      fEventsHist = new TH1D("PROOF_EventsHist", "Events processed per Slave",
                             nslaves, 0, nslaves);
      fEventsHist->SetDirectory(0);
      fEventsHist->SetMinimum(0);
      output->Add(fEventsHist);

      fNodeHist = new TH1D("PROOF_NodeHist", "Slaves per Fileserving Node",
                           nslaves, 0, nslaves);
      fNodeHist->SetDirectory(0);
      fNodeHist->SetMinimum(0);
      fNodeHist->SetBit(TH1::kCanRebin);
      output->Add(fNodeHist);

      fLatencyHist = new TH2D("PROOF_LatencyHist", "GetPacket Latency per Slave",
                              nslaves, 0, nslaves,
                              ntime_bins, min_time, time_per_bin);
      fLatencyHist->SetDirectory(0);
      fLatencyHist->SetMarkerStyle(4);
      fLatencyHist->SetBit(TH1::kCanRebin);
      output->Add(fLatencyHist);

      fProcTimeHist = new TH2D("PROOF_ProcTimeHist", "Packet Processing Time per Slave",
                               nslaves, 0, nslaves,
                               ntime_bins, min_time, time_per_bin);
      fProcTimeHist->SetDirectory(0);
      fProcTimeHist->SetMarkerStyle(4);
      fProcTimeHist->SetBit(TH1::kCanRebin);
      output->Add(fProcTimeHist);

      fCpuTimeHist = new TH2D("PROOF_CpuTimeHist", "Packet CPU Time per Slave",
                              nslaves, 0, nslaves,
                              ntime_bins, min_time, time_per_bin);
      fCpuTimeHist->SetDirectory(0);
      fCpuTimeHist->SetMarkerStyle(4);
      fCpuTimeHist->SetBit(TH1::kCanRebin);
      output->Add(fCpuTimeHist);

      nextslaveinfo.Reset();
      Int_t slavebin=1;
      while (TSlaveInfo *si = dynamic_cast<TSlaveInfo*>(nextslaveinfo())) {
         if (si->fStatus == TSlaveInfo::kActive) {
            fPacketsHist->GetXaxis()->SetBinLabel(slavebin, si->GetOrdinal());
            fEventsHist->GetXaxis()->SetBinLabel(slavebin, si->GetOrdinal());
            fLatencyHist->GetXaxis()->SetBinLabel(slavebin, si->GetOrdinal());
            fProcTimeHist->GetXaxis()->SetBinLabel(slavebin, si->GetOrdinal());
            fCpuTimeHist->GetXaxis()->SetBinLabel(slavebin, si->GetOrdinal());
            slavebin++;
         }
      }
   }
}


//______________________________________________________________________________
void TPerfStats::SimpleEvent(EEventType type)
{
   // Simple event

   if (type == kStop && fPacketsHist != 0) {
      fNodeHist->LabelsDeflate("X");
      fNodeHist->LabelsOption("auv","X");
   }

   if (fTrace == 0) return;

   TPerfEvent pe(&fTzero);
   pe.fType = type;

   fPerfEvent = &pe;
   fTrace->SetBranchAddress("PerfEvents",&fPerfEvent);
   fTrace->Fill();
   fPerfEvent = 0;
}


//______________________________________________________________________________
void TPerfStats::PacketEvent(const char *slave, const char* slavename, const char* filename,
                              Long64_t eventsprocessed, Double_t latency, Double_t proctime,
                              Double_t cputime, Long64_t bytesRead)
{
   // Packet event

   if (fDoTrace && fTrace != 0) {
      TPerfEvent pe(&fTzero);

      pe.fType = kPacket;
      pe.fSlaveName = slavename;
      pe.fFileName = filename;
      pe.fSlave = slave;
      pe.fEventsProcessed = eventsprocessed;
      pe.fBytesRead = bytesRead;
      pe.fLatency = latency;
      pe.fProcTime = proctime;
      pe.fCpuTime = cputime;

      fPerfEvent = &pe;
      fTrace->SetBranchAddress("PerfEvents",&fPerfEvent);
      fTrace->Fill();
      fPerfEvent = 0;
   }

   if (fDoHist && fPacketsHist != 0) {
      fPacketsHist->Fill(slave, 1);
      fEventsHist->Fill(slave, eventsprocessed);
      fLatencyHist->Fill(slave, latency, 1);
      fProcTimeHist->Fill(slave, proctime, 1);
      fCpuTimeHist->Fill(slave, cputime, 1);
   }
}


//______________________________________________________________________________
void TPerfStats::FileEvent(const char *slave, const char *slavename, const char *nodename,
                            const char *filename, Bool_t isStart)
{
   // File event

   if (fDoTrace && fTrace != 0) {
      TPerfEvent pe(&fTzero);

      pe.fType = kFile;
      pe.fSlaveName = slavename;
      pe.fNodeName = nodename;
      pe.fFileName = filename;
      pe.fSlave = slave;
      pe.fIsStart = isStart;

      fPerfEvent = &pe;
      fTrace->SetBranchAddress("PerfEvents",&fPerfEvent);
      fTrace->Fill();
      fPerfEvent = 0;
   }

   if (fDoHist && fPacketsHist != 0) {
      fNodeHist->Fill(nodename, isStart ? 1 : -1);
   }
}


//______________________________________________________________________________
void TPerfStats::FileOpenEvent(TFile *file, const char *filename, Double_t proctime)
{
   // Open file event

   if (fDoTrace && fTrace != 0) {
      TPerfEvent pe(&fTzero);

      pe.fType = kFileOpen;
      pe.fFileName = filename;
      pe.fFileClass = file != 0 ? file->ClassName() : "none";
      pe.fProcTime = proctime;
      pe.fIsOk = (file != 0);

      fPerfEvent = &pe;
      fTrace->SetBranchAddress("PerfEvents",&fPerfEvent);
      fTrace->Fill();
      fPerfEvent = 0;
   }
}


//______________________________________________________________________________
void TPerfStats::FileReadEvent(TFile *file, Int_t len, Double_t proctime)
{
   // Read file event

   if (fDoTrace && fTrace != 0) {
      TPerfEvent pe(&fTzero);

      pe.fType = kFileRead;
      pe.fFileName = file->GetName();
      pe.fFileClass = file->ClassName();
      pe.fLen = len;
      pe.fProcTime = proctime;

      fPerfEvent = &pe;
      fTrace->SetBranchAddress("PerfEvents",&fPerfEvent);
      fTrace->Fill();
      fPerfEvent = 0;
   }
}

//______________________________________________________________________________
void TPerfStats::RateEvent(Double_t proctime, Double_t deltatime,
                           Long64_t eventsprocessed, Long64_t bytesRead)
{
   // Rate event

   if ((fDoTrace || fDoTraceRate) && fTrace != 0) {
      TPerfEvent pe(&fTzero);

      pe.fType = kRate;
      pe.fEventsProcessed = eventsprocessed;
      pe.fBytesRead = bytesRead;
      pe.fProcTime = proctime;
      pe.fLatency = deltatime;

      fPerfEvent = &pe;
      fTrace->SetBranchAddress("PerfEvents",&fPerfEvent);
      fTrace->Fill();
      fPerfEvent = 0;
   }
}

//______________________________________________________________________________
void TPerfStats::SetBytesRead(Long64_t num)
{
   // Set number of bytes read

   fBytesRead = num;
}


//______________________________________________________________________________
Long64_t TPerfStats::GetBytesRead() const
{
   // Get number of bytes read

   return fBytesRead;
}


//______________________________________________________________________________
void TPerfStats::Setup(TList *input)
{
   // Setup the PROOF input list with requested statistics and tracing options.

   const Int_t ntags=3;
   const Char_t *tags[ntags] = {"StatsHist",
                                "StatsTrace",
                                "SlaveStatsTrace"};

   for (Int_t i=0; i<ntags; i++) {
      TString envvar = "Proof.";
      envvar += tags[i];
      TString inputname = "PROOF_";
      inputname += tags[i];
      TObject* obj = input->FindObject(inputname.Data());
      if (gEnv->GetValue(envvar.Data(), 0)) {
         if (!obj)
            input->Add(new TNamed(inputname.Data(),""));
      } else {
         if (obj) {
            input->Remove(obj);
            delete obj;
         }
      }
   }
}


//______________________________________________________________________________
void TPerfStats::Start(TList *input, TList *output)
{
   // Initialize PROOF statistics run.

   if (gPerfStats != 0) {
      delete gPerfStats;
   }

   gPerfStats = new TPerfStats(input, output);

   gPerfStats->SimpleEvent(TVirtualPerfStats::kStart);
}


//______________________________________________________________________________
void TPerfStats::Stop()
{
   // Terminate the PROOF statistics run.

   if (gPerfStats == 0) return;

   gPerfStats->SimpleEvent(TVirtualPerfStats::kStop);

   delete gPerfStats;
   gPerfStats = 0;
}
