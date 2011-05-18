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
// TPEAC                                                                //
//                                                                      //
// This class implements the setup of a PROOF session using PEAC        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TPEAC.h"

#include "TClarens.h"
#include "TDSet.h"
#include "TEnv.h"
#include "TGM.h"
#include "TObjString.h"
#include "TPluginManager.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TUrl.h"
#include "TProof.h"


namespace {

//------------------------------------------------------------------------------

struct TPEACStartup {
   TPEACStartup() {TPEAC::Init();}
} PEACStartup;

//------------------------------------------------------------------------------

};


TPEAC *gPEAC = 0;

ClassImp(TPEAC)


//______________________________________________________________________________
TPEAC::TPEAC()
   : fGM(0), fProof(0)
{
   TClarens::Init();
}


//______________________________________________________________________________
TPEAC::~TPEAC()
{
   if (!fSessionID.IsNull()) EndSession();
   delete fGM;
}


//______________________________________________________________________________
void TPEAC::Init()
{
   if (gPEAC == 0) {
      gPEAC = new TPEAC;
   }
}


//______________________________________________________________________________
TDSet *TPEAC::StartSession(const Char_t *dataset)
{
   if (fGM == 0) {

      const Char_t *gmUrl = gEnv->GetValue("PEAC.GmUrl",
                                           "http://localhost:8080/clarens/");

      fGM = gClarens->CreateGM(gmUrl);

      if (!fGM) {
         Error("TPEAC", "Could not get Global Manager for URL: %s", gmUrl);
         return 0;
      }
   }

   if (!fSessionID.IsNull()) {
      Error("StartSession", "Session associated with dataset '%s' still open",
                            fDataSet.Data());
      Error("StartSession", "That session must end before"
                            " starting a new session");
      return 0;
   }

   if (gDebug > 0) fGM->Print();

   TList* files = 0;
   TString sessionid;
   TUrl purl("");
   if (!fGM->CreateSession(dataset, sessionid, files, purl)) {
      delete fGM;
      fGM = 0;
      return 0;
   }

   // session successfully created

   if (gDebug > 0) {
      Info("StartSession", "sessionid = %s @ %s", sessionid.Data(), purl.GetUrl());
      files->Print();
   }

   // construct TDSet
   TDSet *dset = 0;
   TIter NextFile(files);
   while (TGM::TFileParams *fp = dynamic_cast<TGM::TFileParams*>(NextFile())) {

      if (dset == 0) dset = new TDSet(fp->fObjClass, fp->fObjName, fp->fDir);

      dset->Add(fp->fFileName, fp->fObjName, fp->fDir, fp->fFirst, fp->fNum);
   }
   Int_t nfiles = files->GetSize();
   delete files;

   // save session id
   fSessionID = sessionid;
   fDataSet   = dataset;

   // start proof
   fProof = TProof::Open(purl.GetUrl(), Form("peac:%s", sessionid.Data()));

   if (!fProof || !fProof->IsValid()) {
      Error("StartSession", "PROOF session could not be started");
      EndSession();
      delete dset;
      return 0;
   }

   //call EndSession when proof is destroyed
   fProof->Connect("~TProof()", "TPEAC", this, "EndSessionCallback()");

   //wait until data is ready
   Long64_t totalbytes, bytesready;
   Bool_t dataready = fProof->IsDataReady(totalbytes, bytesready);

   //make a progress bar - either the user deletes it or it deletes itself
   if (!gROOT->IsBatch()) {
      if (TPluginManager *pm = gROOT->GetPluginManager()) {
         if (TPluginHandler *h = pm->FindHandler("TProofStartupDialog")) {
            if(h->LoadPlugin() != -1) {
               h->ExecPlugin(4, fProof, dataset, nfiles, totalbytes);
               //trigger progress atleast once
               dataready = fProof->IsDataReady(totalbytes, bytesready);
            }
         }
      }
   }

   if (!dataready) {
      gSystem->Sleep(500);
      while (!fProof->IsDataReady(totalbytes, bytesready)) {
         gSystem->Sleep(500);
      }
   }

   return dset;
}


//______________________________________________________________________________
void TPEAC::EndSessionCallback()
{
   if (fSessionID.IsNull()) {
      Error("EndSession", "No session active. Don't call EndSessionCallback() directly");
      return;
   }

   if (!fGM) {
      Error("EndSession", "Global manager does not exist");
      return;
   }

   if (fProof) {
      fProof->Disconnect("~TProof()", this, "EndSessionCallback()");
      fProof = 0;
   }
   fGM->DestroySession(fSessionID);
   fSessionID = "";
   fDataSet   = "";

}


//______________________________________________________________________________
void TPEAC::EndSession()
{
   if (fSessionID.IsNull()) {
      Info("EndSession", "No session active");
      return;
   }

   if (!fGM) {
      Error("EndSession", "Global manager does not exist");
      return;
   }

   if (fProof) {
      fProof->Disconnect("~TProof()", this, "EndSessionCallback()");
      delete fProof;
      fProof = 0;
   }
   fGM->DestroySession(fSessionID);
   fSessionID = "";
   fDataSet   = "";
}
