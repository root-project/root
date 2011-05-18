// @(#)root/peac:$Id$
// Author: Maarten Ballintijn    21/10/2004
// Author: Kris Gulbrandsen      21/10/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofPEAC                                                           //
//                                                                      //
// This class implements the setup of a PROOF session using PEAC        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClarens.h"
#include "TCondor.h"
#include "TDSet.h"
#include "TEnv.h"
#include "TError.h"
#include "TList.h"
#include "TLM.h"
#include "TMonitor.h"
#include "TProofPEAC.h"
#include "TProofServ.h"
#include "TSlave.h"
#include "TSystem.h"
#include "TTimer.h"
#include "TUrl.h"

ClassImp(TProofPEAC)


//______________________________________________________________________________
TProofPEAC::TProofPEAC(const char *masterurl, const char *sessionid,
                       const char *confdir, Int_t loglevel,
                       const char *, TProofMgr *mgr)
   : fCondor(0), fTimer(0)
{
   // Start PEAC proof session

   // This may be needed during init
   fManager = mgr;

   if (!strncasecmp(sessionid, "peac:", 5))
      sessionid+=5;

   Init(masterurl, sessionid, confdir, loglevel);

}

//______________________________________________________________________________
TProofPEAC::~TProofPEAC()
{
   // Destroy PEAC proof session

   delete fCondor;
   delete fTimer;
   if (fLM) {
      delete fHeartbeatTimer;
      fHeartbeatTimer = 0;
      fLM->EndSession(fSession);
      delete fLM;
      fLM = 0;
   }

}

//------------------------------------------------------------------------------
Bool_t TProofPEAC::StartSlaves(Bool_t,Bool_t)
{

   if (IsMaster()) {

      TClarens::Init();
      const Char_t *lmUrl = gEnv->GetValue("PEAC.LmUrl",
                                           "http://localhost:8080/clarens/");
      fLM = gClarens->CreateLM(lmUrl);
      if (!fLM) {
         Error("StartSlaves", "Could not connect to local manager for url '%s'",
                              lmUrl);
         return kFALSE;
      }

      TUrl url(lmUrl);
      TString lm = url.GetHost();
      Int_t lmPort = url.GetPort();
      fSession = fConfFile;

      PDB(kGlobal,1) Info("StartSlaves", "PEAC mode: host: %s  port: %d  session: %s",
                          lm.Data(), lmPort, fSession.Data());

      TList* config = 0;
      if(!fLM->StartSession(fSession, config, fHBPeriod)) {
         Error("StartSlaves", "Could not start session '%s' for local manager '%s'",
                              fSession.Data(), lmUrl);
         return kFALSE;
      }

      TList csl;

      TIter NextSlave(config);
      Int_t ord = 0;
      TString jobad;
      while (TLM::TSlaveParams *sp = dynamic_cast<TLM::TSlaveParams*>(NextSlave())) {

         PDB(kGlobal,1) Info("StartSlaves", "node: %s", sp->fNode.Data());

         // create slave server

         if (sp->fType == "inetd") {
            TString fullord = TString(gProofServ->GetOrdinal()) + "." + ((Long_t) ord);
            ord++;
            TSlave *slave = CreateSlave(sp->fNode, fullord,
                                        sp->fPerfidx, sp->fImg, Form("~/%s", kPROOF_WorkDir));
            fSlaves->Add(slave);
            if (slave->IsValid()) {
               fAllMonitor->Add(slave->GetSocket());
               PDB(kGlobal,3)
                 Info("StartSlaves", "slave on host %s created and added to list",
                      sp->fNode.Data());
            } else {
               fBadSlaves->Add(slave);
               PDB(kGlobal,3)
                 Info("StartSlaves", "slave on host %s created and added to list of bad slaves",
                      sp->fNode.Data());
            }
         } else if (sp->fType == "cod") {
            if (fCondor == 0) {
               fCondor = new TCondor;
               jobad = GetJobAd();

               fImage = fCondor->GetImage(gSystem->HostName());
               if (fImage.Length() == 0) {
                  Error("StartSlaves", "no image found for node %s",
                        gSystem->HostName());
                  delete fCondor;
                  fCondor = 0;
               }
            }

            if (fCondor != 0) {
               TCondorSlave *c = fCondor->Claim(sp->fNode, jobad);

               if (c != 0) {
                  csl.Add(c);
               } else {
                  Info("StartSlaves", "node: %s not claimed", sp->fNode.Data());
               }
            }
         } else {
            Error("StartSlaves", "unknown slave type (%s)", sp->fType.Data());
         }
      }
      delete config;

      TIter next(&csl);
      TCondorSlave *cs;
      while ((cs = (TCondorSlave*)next()) != 0) {
         // Get slave FQDN ...
         TString SlaveFqdn;
         TInetAddress SlaveAddr = gSystem->GetHostByName(cs->fHostname);
         if (SlaveAddr.IsValid())
            SlaveFqdn = SlaveAddr.GetHostName();

         // who do we believe for perf & img, Condor for the moment
         TString fullord = TString(gProofServ->GetOrdinal()) + "." + ((Long_t) ord);
         ord++;
         TSlave *slave = CreateSlave(cs->fHostname, fullord,
                                     cs->fPerfIdx, cs->fImage, Form("~/%s", kPROOF_WorkDir));

         fSlaves->Add(slave);
         if (slave->IsValid()) {
            fAllMonitor->Add(slave->GetSocket());
            PDB(kGlobal,3)
              Info("StartSlaves", "slave on host %s created and added to list (port %d)",
               cs->fHostname.Data(),cs->fPort);
         } else {
            fBadSlaves->Add(slave);
            PDB(kGlobal,3)
              Info("StartSlaves", "slave on host %s created and added to list of bad slaves (port %d)",
                cs->fHostname.Data(),cs->fPort);
         }
      }

      //create and start heartbeat timer
      fHeartbeatTimer = new TTimer;
      fHeartbeatTimer->Connect("Timeout()", "TProofPEAC", this, "SendHeartbeat()");
      fHeartbeatTimer->Start(fHBPeriod*1000, kFALSE);
   } else {
      return TProof::StartSlaves(kTRUE);
   }

   return kTRUE;
}

//______________________________________________________________________________
void TProofPEAC::Close(Option_t *option)
{

   TProof::Close(option);

   if (fLM) {
      delete fHeartbeatTimer;
      fHeartbeatTimer = 0;
      fLM->EndSession(fSession);
      delete fLM;
      fLM = 0;
   }

}

//______________________________________________________________________________
void TProofPEAC::SetActive(Bool_t active)
{
   // Suspend or resume PROOF via Condor.

   if (fCondor) {
      if (fTimer == 0) {
         fTimer = new TTimer();
      }
      if (active) {
         PDB(kCondor,1) Info("SetActive","-- Condor Resume --");
         fTimer->Stop();
         if (fCondor->GetState() == TCondor::kSuspended)
            fCondor->Resume();
      } else {
         Int_t delay = 10000; // milli seconds
         PDB(kCondor,1) Info("SetActive","-- Delayed Condor Suspend (%d msec) --", delay);
         fTimer->Connect("Timeout()", "TCondor", fCondor, "Suspend()");
         fTimer->Start(10000, kTRUE); // single shot
      }
   }
}

//______________________________________________________________________________
TString TProofPEAC::GetJobAd()
{
   TString ad;

   ad = "JobUniverse = 5\n"; // vanilla
   ad += Form("Cmd = \"%s/bin/proofd\"\n", GetConfDir());
   ad += "Iwd = \"/tmp\"\n";
   ad += "In = \"/dev/null\"\n";
   ad += "Out = \"/tmp/proofd.out.$(Port)\"\n";
   ad += "Err = \"/tmp/proofd.err.$(Port)\"\n";
   ad += Form("Args = \"-f -p $(Port) -d %d %s\"\n", GetLogLevel(), GetConfDir());

   return ad;
}

//______________________________________________________________________________
Bool_t TProofPEAC::IsDataReady(Long64_t &totalbytes, Long64_t &bytesready)
{
   Bool_t dataready = kFALSE;
   if (IsMaster()) {
      dataready = fLM ? fLM->DataReady(fSession, bytesready, totalbytes) : kFALSE;
      ////// for testing
      //static Long64_t total = 10;
      //static Long64_t ready = -1;
      //ready++;
      //totalbytes=total;
      //bytesready=ready;
      if (totalbytes>bytesready) dataready=kFALSE;
      //////
   } else {
      dataready = TProof::IsDataReady(totalbytes, bytesready);
   }
   return dataready;
}

//______________________________________________________________________________
void TProofPEAC::SendHeartbeat()
{
   if (fLM) fLM->Heartbeat(fSession);
}
