// @(#)root/alien:$Name:  $:$Id: TAlienProof.cxx,v 1.0 2003/09/05 11:29:24 peters Exp $
// Author: Andreas Peters 05/09/2003

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienProof                                                          //
//                                                                      //
// Class defining the interface to AliEn GRID PROOF Services.           //
// Objects of this class are created by TAlien methods.                 //
//                                                                      //
// Related classes are TAlien.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include "TUrl.h"
#include "TBrowser.h"
#include "TAlien.h"
#include "TROOT.h"
#include "TAuthenticate.h"

#define ALIEN_POSIX
#include <AliEnAPI++.h>


ClassImp(TAlienProof)

//______________________________________________________________________________
TAlienProof::TAlienProof() : TGridProof()
{
   // Create AliEn PROOF interface.

   for (int i = 0; i < kALIENPROOFMAXSITES; i++) {
      fSites[i]  = 0;
      fNtimes[i] = 0;
   }
   fNsites = 0;

   if (gGrid) {
      gGrid->SetGridProof(this);
   }
}

//______________________________________________________________________________
TAlienProof::~TAlienProof()
{
   // Cleanup Grid PROOF session.

   for (int i = 0; i < kALIENPROOFMAXSITES; i++) {
      delete fSites[i];
      delete fNtimes[i];
   }
}

//______________________________________________________________________________
Bool_t TAlienProof::Request(TDSet *dse)
{
   // Request the startup of PROOF depending on the information in the TDSet.
   // Returns false in case of error.

   if (!gGrid->IsConnected() || !dse)
      return kFALSE;

   // pack the dataset
   dse->GridPack();

   // extract the required sites
   fNsites = DSetMsnExtraction(dse);

   fProofSession = gGrid->RequestProofSession(gGrid->GetUser(), fNsites,
                                  (void **) &fSites, (void **) &fNtimes, 0, 0);

   if (fProofSession &&
       (((TAliEnAPI::ProofSession_T *) fProofSession)->sessionId != 0)) {
      fProofMasterUrl =
          ((TAliEnAPI::ProofSession_T *) fProofSession)->masterurl.c_str();
      fProofLogin =
          ((TAliEnAPI::ProofSession_T *) fProofSession)->login.c_str();
      fProofPasswd =
          ((TAliEnAPI::ProofSession_T *) fProofSession)->password.c_str();
      fProofConfigFile =
          ((TAliEnAPI::ProofSession_T *) fProofSession)->configfile.c_str();
      fProofSiteBlob =
          ((TAliEnAPI::ProofSession_T *) fProofSession)->siteblob.c_str();
      Info("Request", "PROOF request accepted: your session ID is %4d",
             ((TAliEnAPI::ProofSession_T *) fProofSession)->sessionId);
      printf("--------------------------------------------------------\n");
      ps(((TAliEnAPI::ProofSession_T *) fProofSession)->sessionId);
      printf("--------------------------------------------------------\n");
      const char *siteblob = GetSiteBlob();
      DSetProofServiceAnswer(dse);
      DSetRepack(dse);
      return kTRUE;
   } else {
      Info("Request", "PROOF request rejected!");
      return kFALSE;
   }
}

//______________________________________________________________________________
void TAlienProof::Connect()
{
   // Connect to the AliEn prepared PROOF cluster.

   if (fProofSession) {
      printf("MasterUrl:      %s\n", fProofMasterUrl.Data());
      printf("ConfigFile:     %s\n", fProofConfigFile.Data());
      printf("Login:          %s\n", fProofLogin.Data());
      printf("Password:       -------\n");

      TAuthenticate::SetGlobalUser(fProofLogin);
      TAuthenticate::SetGlobalPasswd(fProofPasswd);

      fProof = gROOT->Proof("proof://aliendb1:8155", fProofConfigFile);
      //fProof = gROOT->Proof(fProofMasterUrl, fProofConfigFile);
   } else {
      Error("TAlienProof", "you don't have a valid session ID");
      return;
   }
}

//______________________________________________________________________________
Int_t TAlienProof::DSetMsnExtraction(TDSet *dse)
{
   // Extract sites from the list of files in the TDSet.

   if (!dse)
      return 0;

   Int_t index = 0;

   // loop over the mass storage name list

   TIter next(dse->GetListOfElementsMsn());
   while (TDSetElementMsn *dsemsn = (TDSetElementMsn *) next()) {
      if (index < kALIENPROOFMAXSITES) {
         Info("DSetMsnExtraction", "request %4d daemons for site %s",
              dsemsn->GetNfiles(), dsemsn->GetMsn());
         if (fSites[index] != 0) {
            delete fSites[index];
         }
         if (fNtimes[index] != 0) {
            delete fNtimes[index];
         }
         fSites[index] = new string(dsemsn->GetMsn());
         fNtimes[index] = new string(Form("%d", dsemsn->GetNfiles()));
         index++;
      } else {
         Error("DSetMsnExtraction",
               "number of supported sites overflow (> %d)! List truncated!",
               kALIENPROOFMAXSITES);
      }
   }
   printf("--------------------------------------------------------\n");
   fNsites = index;
   return index;
}

//______________________________________________________________________________
void TAlienProof::DSetProofServiceAnswer(TDSet *dse)
{
   // Build missing entries in fElementsMsn from the siteblob,
   // which has the format <mssA>##<siteA>##<nA>####<mssB><siteB><nB> ....

   if (!dse) return;

   if (fProofSiteBlob.IsNull()) {
      Error("TAlienProof", "it looks like we didn't get any daemons assigned");
      return;
   }

   int nc = fProofSiteBlob.Length();
   const char *siteblob  = fProofSiteBlob;
   const char *infostart = siteblob;
   //  printf("Blob is: %s\n", siteblob);

   for (int i = 0; i < nc; i++) {
      char sitename[1024];
      char mssname[1024];
      char nsite[1024];
      int intnsite = 0;
      if (siteblob[i] != '#')
         continue;
      if (i <= (nc - 4)) {
         if (siteblob[i] == '#' && siteblob[i + 1] == '#' &&
             siteblob[i + 2] == '#' && siteblob[i + 3] == '#') {
            // copy from infostart to siteblob[i-1]
            memcpy(nsite, infostart, (siteblob + i - infostart));
            nsite[siteblob + i - infostart] = 0;
            intnsite = atoi(nsite);
            i += 4;
            infostart = siteblob + i;
            //printf("Nsite %s\n",nsite);
            // find this site in fSiteList
            TIter next(dse->GetListOfElementsMsn());
            Bool_t found = kFALSE;
            while (TDSetElementMsn *dsemsn = (TDSetElementMsn *) next()) {
               if (!(strcmp(dsemsn->GetMsn(), mssname))) {
                  // this is the same site ...
                  // add the daemons up
                  int insitedaemon = dsemsn->GetNSiteDaemons();
                  Info("DSetProofServiceAnswer",
                       "this %s site has already %d daemons", dsemsn->GetMsn(), dsemsn->GetNSiteDaemons());
                  dsemsn->SetNSiteDaemons(intnsite+((insitedaemon>-1)?insitedaemon:0));
                  dsemsn->Print();
                  found = kTRUE;
                  break;
               }
            }
            if (!found) {
               Error("DSetProofServiceAnswer",
                     "strange ... Mss %s not found in fSiteList!",
                     mssname);
            } else {
               ;  //printf("I found it\n");
            }
         } else {
            if (siteblob[i] == '#' && siteblob[i + 1] == '#' && siteblob[i + 2] == '#' ) {
               memcpy(sitename, infostart, (siteblob + i - infostart));
               sitename[siteblob + i - infostart] = 0;
               i += 3;
               infostart = siteblob + i;
               printf("CE Name is %s\n", sitename);

            } else {
               if (siteblob[i] == '#' && siteblob[i + 1] == '#') {
                  memcpy(mssname, infostart, (siteblob + i - infostart));
                  mssname[siteblob + i - infostart] = 0;
                  i += 2;
                  infostart = siteblob + i;
                  printf("Mss Name is %s\n",mssname);
               }
            }
         }
      }
   }
}

//______________________________________________________________________________
void TAlienProof::DSetRepack(TDSet *dse)
{
   // Assign Computing Elements to the TDSetElementPfn elements for PROOF
   // processing.

   if (!dse) return;

   // loop over all sites and calculate the #files per daemon[site]
   TIter next(dse->GetListOfElementsMsn());
   while (TDSetElementMsn *dsemsn = (TDSetElementMsn *) next()) {
      int nFile = dsemsn->GetNfiles();
      int nProofd = dsemsn->GetNSiteDaemons();
      int nFilePerProof =
          (nProofd) ? (int) (float(nFile) / nProofd - 0.5) : 0;
      int nCE = 0;
      int nCEi = 0;
      int nCEa = 0;
      // loop over the LFNs
      dse->Reset();
      while (TDSetElement * lfnE = dse->Next()) {
         if (nCE >= nProofd) {
            break;
         }
         if (nCEa >= nFile) {
            break;
         }
         lfnE->Reset();
         // take the 1st PFN
         // (for the moment, we just consider the primary location of a file)
         if (TDSetElementPfn * pfnE = lfnE->Next()) {
            // check if this is for the SiteElement dsemsn
            if (!(strcmp(dsemsn->GetName(), pfnE->GetMsn()))) {
               // assign an indexed CE name
               TString newCenScalar =
                   TString(dsemsn->GetName()) + TString(";");
               newCenScalar += nCE;
               TString newCen = newCenScalar;
               printf("Setting new ProofCE: %s\n", newCen.Data());
               // store the proofd CE index name
               pfnE->SetCen(newCen);
               nCEi++;
               nCEa++;
               // if we have already enough files/proof, take the next proof
               if ((nCEi > nFilePerProof) && (((nCE != (nProofd - 1))))) {
                  nCE++;
                  nCEi = 0;
               }
            }
         }
      }
   }
   printf("--------------------------------------------------------\n");
}
