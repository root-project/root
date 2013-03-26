// @(#)root/proof:$Id$
// Author: G. Ganis, Mar 2010

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// pq2wrappers                                                          //
//                                                                      //
// This file implements the wrapper functions used in PQ2               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include <stdlib.h>

#include "pq2wrappers.h"
#include "redirguard.h"

#include "TDataSetManager.h"
#include "TEnv.h"
#include "TFileInfo.h"
#include "TPluginManager.h"
#include "TProof.h"
#include "TROOT.h"

TDataSetManager *gDataSetManager = 0;

// Global variables defined by other PQ2 components
extern TUrl    gUrl;
extern Bool_t  gIsProof;
extern TString flog;
extern TString ferr;
extern TString fres;
extern Int_t gverbose;

// How to start PROOF (new VerifyDataSet wants it parallel)
static bool doParallel = kFALSE;

Int_t getProof(const char *where, Int_t verbose = 1);
Int_t getDSMgr(const char *where);

//_______________________________________________________________________________________
void DataSetCache(bool clear, const char *ds)
{
   // ShowCache wrapper
   if (gIsProof) {
      doParallel = kFALSE;
      if (!gProof && getProof("DataSetCache", 0) != 0) return;
      return (clear ? gProof->ClearDataSetCache(ds) : gProof->ShowDataSetCache(ds));
   } else {
      if (!gDataSetManager && getDSMgr("DataSetCache") != 0) return;
      Int_t rc = (clear) ? gDataSetManager->ClearCache(ds) : gDataSetManager->ShowCache(ds);
      if (rc != 0)
         Printf("DataSetCache: problems running '%s'", (clear ? "clear" : "show"));
      return;
   }
   // Done
   return;
}

//_______________________________________________________________________________________
void ShowDataSets(const char *ds, const char *opt)
{
   // ShowDataSets wrapper
   if (gIsProof) {
      doParallel = kFALSE;
      if (!gProof && getProof("ShowDataSets", 0) != 0) return;
      return gProof->ShowDataSets(ds, opt);
   } else {
      if (!gDataSetManager && getDSMgr("ShowDataSets") != 0) return;
      return gDataSetManager->ShowDataSets(ds, opt);
   }
   // Done
   return;
}

//_______________________________________________________________________________________
TFileCollection *GetDataSet(const char *ds, const char *server)
{
   // GetDataSet wrapper
   TFileCollection *fc = 0;
   if (gIsProof) {
      doParallel = kFALSE;
      if (!gProof && getProof("GetDataSet") != 0) return fc;
      return gProof->GetDataSet(ds, server);
   } else {
      if (!gDataSetManager && getDSMgr("ShowDataSets") != 0) return fc;
      return gDataSetManager->GetDataSet(ds, server);
   }
   // Done
   return fc;
}

//_______________________________________________________________________________________
TMap *GetDataSets(const char *owner, const char *server, const char *opt)
{
   // GetDataSets wrapper
   TMap *dss = 0;
   if (gIsProof) {
      doParallel = kFALSE;
      if (!gProof && getProof("GetDataSets") != 0) return dss;
      return gProof->GetDataSets(owner, server);
   } else {
      if (!gDataSetManager && getDSMgr("GetDataSets") != 0) return dss;
      // Get the datasets and fill a map
      UInt_t oo = (opt && !strcmp(opt, "list")) ? (UInt_t)TDataSetManager::kList
                                                : (UInt_t)TDataSetManager::kExport;
      dss = gDataSetManager->GetDataSets(owner, oo);
      // If defines, option gives the name of a server for which to extract the information
      if (dss) {
         if (server && strlen(server) > 0) {
            // The return map will be in the form   </group/user/datasetname> --> <dataset>
            TMap *rmap = new TMap;
            TObject *k = 0;
            TFileCollection *fc = 0, *xfc = 0;
            TIter nxd(dss);
            while ((k = nxd()) && (fc = (TFileCollection *) dss->GetValue(k))) {
               // Get subset on specified server, if any
               if ((xfc = fc->GetFilesOnServer(server))) {
                  rmap->Add(new TObjString(k->GetName()), xfc);
               }
            }
            dss->DeleteAll();
            delete dss;
            dss = 0;
            if (rmap->GetSize() > 0) {
               dss = rmap;
            } else {
               Printf("GetDataSets: no dataset found on server '%s' for owner '%s'", server, owner);
               delete rmap;
            }
         }
      } else  {
         Printf("GetDataSets: no dataset found for '%s'", owner);
      }
   }
   // Done
   return dss;
}

//_______________________________________________________________________________________
Int_t RemoveDataSet(const char *dsname)
{
   // RemoveDataSet wrapper
   if (gIsProof) {
      doParallel = kFALSE;
      if (!gProof && getProof("RemoveDataSet") != 0) return -1;
      return gProof->RemoveDataSet(dsname);
   } else {
      if (!gDataSetManager && getDSMgr("RemoveDataSet") != 0) return -1;
      return gDataSetManager->RemoveDataSet(dsname);
   }
   // Done
   return -1;
}

//_______________________________________________________________________________________
Int_t VerifyDataSet(const char *dsname, const char *opt, const char *redir)
{
   // VerifyDataSet wrapper

   Int_t rc = -1;
   // Honour the 'redir' if required
   TString srvmaps;
   if (redir && strlen(redir) > 0) srvmaps.Form("|%s", redir);
   if (gIsProof) {
      // Honour the 'redir' if required
      if (!(srvmaps.IsNull())) {
         TProof::AddEnvVar("DATASETSRVMAPS", srvmaps);
      }
      TString sopt(opt);
      doParallel = (sopt.Contains("S")) ? kFALSE : kTRUE;
      if (gProof && doParallel && gProof->GetParallel() == 0) {
         gProof->Close();
         delete gProof;
         gProof = 0;
      }
      if (!gProof && getProof("VerifyDataSet") != 0) return -1;
      if ((rc = gProof->VerifyDataSet(dsname, opt)) == 0) {
         // Success; partial at least. Check if all files are staged
         TFileCollection *fcs = gProof->GetDataSet(dsname, "S:");
         if (fcs && fcs->GetStagedPercentage() < 99.99999) rc = 1;
      }
   } else {
      // Honour the 'redir' if required
      if (!(srvmaps.IsNull())) {
         gEnv->SetValue("DataSet.SrvMaps", srvmaps);
      }
      if (!gDataSetManager && getDSMgr("VerifyDataSet") != 0) return -1;
      if ((rc = gDataSetManager->ScanDataSet(dsname, opt)) == 0) {
         // Success; partial at least. Check if all files are staged
         TFileCollection *fcs = gDataSetManager->GetDataSet(dsname, "S:");
         if (fcs && fcs->GetStagedPercentage() < 99.99999) rc = 1;
      }
   }
   // Done
   return rc;
}

//_______________________________________________________________________________________
Bool_t ExistsDataSet(const char *dsname)
{
   // ExistsDataSet wrapper
   if (gIsProof) {
      doParallel = kFALSE;
      if (!gProof && getProof("ExistsDataSet") != 0) return kFALSE;
      return gProof->ExistsDataSet(dsname);
   } else {
      if (!gDataSetManager && getDSMgr("ExistsDataSet") != 0) return kFALSE;
      return gDataSetManager->ExistsDataSet(dsname);
   }
   return kFALSE;
}

//_______________________________________________________________________________________
Int_t RegisterDataSet(const char *dsname, TFileCollection *fc, const char* opt)
{
   // RegisterDataSet wrapper
   if (gIsProof) {
      doParallel = kFALSE;
      if (!gProof && getProof("GetDataSet") != 0) return -1;
      return gProof->RegisterDataSet(dsname, fc, opt);
   } else {
      if (!gDataSetManager && getDSMgr("RegisterDataSet") != 0) return -1;
      return gDataSetManager->RegisterDataSet(dsname, fc, opt);
   }
   return -1;
}

//_______________________________________________________________________________________
Int_t getProof(const char *where, Int_t verbose)
{
   // Open a PROOF session at gUrl

   {  redirguard rog(flog.Data(), "a", verbose);
      const char *popt = (doParallel) ? "" : "masteronly";
      TProof::Open(gUrl.GetUrl(), popt);
   }
   if (!gProof || !gProof->IsValid()) {
      Printf("getProof:%s: problems starting a PROOF session at '%s'", where, gUrl.GetUrl());
      return -1;
   }
   if (gverbose >= 2) gProof->SetLogLevel(2);
   // Done
   return 0;
}

//_______________________________________________________________________________________
Int_t getDSMgr(const char *where)
{
   // Open a dataset manager for gUrl

   Int_t rc = -1;
   if (gROOT->GetPluginManager()) {
      // Find the appropriate handler
      TPluginHandler *h = gROOT->GetPluginManager()->FindHandler("TDataSetManager", "file");
      if (h && h->LoadPlugin() != -1) {
         TString group(getenv("PQ2GROUP")), user(getenv("PQ2USER"));
         TString dsm, opt("opt:-Ar:-Av:");
         const char *o = getenv("PQ2DSMGROPTS");
         if (o) {
            opt = "";
            if (strlen(o) > 0) opt.Form("opt:%s", o);
         }
         dsm.Form("file dir:%s %s", gUrl.GetUrl(), opt.Data());
         gDataSetManager = reinterpret_cast<TDataSetManager*>(h->ExecPlugin(3,
                                                              group.Data(), user.Data(),
                                                              dsm.Data()));
         if (gDataSetManager) {
            rc = 0;
         } else {
            Printf("getDSMgr:%s: problems creating a dataset manager at '%s'", where, gUrl.GetUrl());
         }
      }
   }
   // Done
   return rc;
}
