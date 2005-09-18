// @(#)root/proof:$Name:  $:$Id: TProofSuperMaster.cxx,v 1.4 2005/09/17 13:52:55 rdm Exp $
// Author: Fons Rademakers   13/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofSuperMaster                                                    //
//                                                                      //
// This class controls a Parallel ROOT Facility, PROOF, cluster.        //
// It fires the slave servers, it keeps track of how many slaves are    //
// running, it keeps track of the slaves running status, it broadcasts  //
// messages to all slaves, it collects results, etc.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofSuperMaster.h"
#include "TString.h"
#include "TError.h"
#include "TList.h"
#include "TSortedList.h"
#include "TSlave.h"
#include "TMap.h"
#include "TProofServ.h"
#include "TSocket.h"
#include "TMonitor.h"
#include "TSemaphore.h"
#include "TDSet.h"
#include "TPluginManager.h"
#include "TProofPlayer.h"
#include "TMessage.h"


ClassImp(TProofSuperMaster)

//______________________________________________________________________________
TProofSuperMaster::TProofSuperMaster(const char *masterurl, const char *conffile,
                                     const char *confdir, Int_t loglevel)
  : TProof(masterurl, conffile, confdir, loglevel)
{
   // Start super master PROOF session.

   if (!conffile || strlen(conffile) == 0)
      conffile = kPROOF_ConfFile;
   else if (!strncasecmp(conffile, "sm:", 3))
      conffile+=3;
   if (!confdir  || strlen(confdir) == 0)
      confdir = kPROOF_ConfDir;

   Init(masterurl, conffile, confdir, loglevel);
}

//______________________________________________________________________________
Bool_t TProofSuperMaster::StartSlaves(Bool_t parallel)
{
   // Start up PROOF submasters.

   // If this is a supermaster server, find the config file and start
   // submaster servers as specified in the config file.
   // There is a difference in startup between a slave and a submaster
   // in which the submaster will issue a kPROOF_LOGFILE and
   // then a kPROOF_LOGDONE message (which must be collected)
   // while slaves do not.

   TString fconf;
   fconf.Form("%s/.%s", gSystem->Getenv("HOME"), fConfFile.Data());
   PDB(kGlobal,2) Info("StartSlaves", "checking PROOF config file %s", fconf.Data());
   if (gSystem->AccessPathName(fconf, kReadPermission)) {
      fconf.Form("%s/proof/etc/%s", fConfDir.Data(), fConfFile.Data());
      PDB(kGlobal,2) Info("StartSlaves", "checking PROOF config file %s", fconf.Data());
      if (gSystem->AccessPathName(fconf, kReadPermission)) {
         Error("StartSlaves", "no PROOF config file found");
         return kFALSE;
      }
   }

   PDB(kGlobal,1) Info("StartSlaves", "using PROOF config file: %s", fconf.Data());

   TList validSlaves;
   TList validPairs;
   UInt_t nSlaves = 0;

   FILE *pconf;
   if ((pconf = fopen(fconf, "r"))) {

      TString fconfname = fConfFile;
      fConfFile = fconf;

      // read the config file
      char line[1024];
      TString host = gSystem->GetHostByName(gSystem->HostName()).GetHostName();
      int  ord = 0;

      // check for valid master line
      while (fgets(line, sizeof(line), pconf)) {
         char word[12][128];
         if (line[0] == '#') continue;   // skip comment lines
         int nword = sscanf(line, "%s %s %s %s %s %s %s %s %s %s %s %s",
             word[0], word[1],
             word[2], word[3], word[4], word[5], word[6],
             word[7], word[8], word[9], word[10], word[11]);

         // see if master may run on this node, accept both old "node"
         // and new "master" lines
         if (nword >= 2 &&
             (!strcmp(word[0], "node") || !strcmp(word[0], "master")) &&
             !fImage.Length()) {
            TInetAddress a = gSystem->GetHostByName(word[1]);
            if (!host.CompareTo(a.GetHostName()) ||
                !strcmp(word[1], "localhost")) {
               const char *image = word[1];
               for (int i = 2; i < nword; i++) {

                  if (!strncmp(word[i], "image=", 6))
                     image = word[i]+6;

               }
               fImage = image;
            }
         } else if (nword >= 2 && (!strcmp(word[0], "submaster")))
            nSlaves++;
      }

      if (fImage.Length() == 0) {
         fclose(pconf);
         Error("StartSlaves", "no appropriate master line found in %s", fconf.Data());
         return kFALSE;
      }

      // Init arrays for threads, if neeeded
      std::vector<TProofThread *> thrHandlers;
      if (parallel) {
         thrHandlers.reserve(nSlaves);
         if (thrHandlers.max_size() < nSlaves) {
            PDB(kGlobal,1)
               Info("StartSlaves","cannot reserve enough space thread"
                    " handlers - switch to serial startup");
            parallel = kFALSE;
         }
      }

      // check for valid submaster lines and start them
      rewind(pconf);
      while (fgets(line, sizeof(line), pconf)) {
         char word[12][128];
         if (line[0] == '#') continue;   // skip comment lines
         int nword = sscanf(line, "%s %s %s %s %s %s %s %s %s %s %s %s",
             word[0], word[1],
             word[2], word[3], word[4], word[5], word[6],
             word[7], word[8], word[9], word[10], word[11]);

         // find all submaster servers
         if (nword >= 2 &&
             !strcmp(word[0], "submaster")) {
            int sport    = fPort;

            const char *conffile = 0;
            const char *image = word[1];
            const char *msd = 0;
            for (int i = 2; i < nword; i++) {

               if (!strncmp(word[i], "image=", 6))
                  image = word[i]+6;
               if (!strncmp(word[i], "port=", 5))
                  sport = atoi(word[i]+5);
               if (!strncmp(word[i], "config=", 7))
                  conffile = word[i]+7;
               if (!strncmp(word[i], "msd=", 4))
                  msd = word[i]+4;

            }

            // Get slave FQDN ...
            TString slaveFqdn;
            TInetAddress slaveAddr = gSystem->GetHostByName(word[1]);
            if (slaveAddr.IsValid()) {
               slaveFqdn = slaveAddr.GetHostName();
               if (slaveFqdn == "UnNamedHost")
               slaveFqdn = slaveAddr.GetHostAddress();
            }

            TString fullord =
               TString(gProofServ->GetOrdinal()) + "." + ((Long_t) ord);
            if (parallel) {

               // Prepare arguments
               TProofThreadArg *ta =
                   new TProofThreadArg(word[1], sport,
                                       fullord, image, conffile, msd,
                                       fSlaves, this);
               if (ta) {
                  // Change default type
                  ta->fType = TSlave::kMaster;
                  // The type of the thread func makes it a detached thread
                  TThread *th = new TThread(SlaveStartupThread,ta);
                  if (!th) {
                     Info("StartSlaves","Can't create startup thread:"
                                        " out of system resources");
                     SafeDelete(ta);
                  } else {
                     // Save in vector
                     thrHandlers.push_back(new TProofThread(th,ta));
                     // Run the thread
                     th->Run();
                  }
               } else {
                  Info("StartSlaves","Can't create thread arguments object:"
                                     " out of system resources");
               }

            } else {

               // create submaster server
               TSlave *slave = CreateSubmaster(word[1], sport, fullord,
                                               image, msd);

               // Add to global list (we will add to the monitor list after
               // finalizing the server startup)
               fSlaves->Add(slave);
               if (slave->IsValid()) {
                  validPairs.Add(new TPair(slave,new TObjString(conffile)));
               } else {
                  fBadSlaves->Add(slave);
               }

               PDB(kGlobal,3)
                  Info("StartSlaves","submaster on host %s created and"
                                     " added to list", word[1]);
            }
            ord++;
         }

      }

      if (parallel) {

         // Wait completion of startup operations
         std::vector<TProofThread *>::iterator i;
         for (i = thrHandlers.begin(); i != thrHandlers.end(); ++i) {
            TProofThread *pt = *i;
            // Wait on this condition
            if (pt && pt->fThread->GetState() == TThread::kRunningState) {
               PDB(kGlobal,3)
                  Info("StartSlaves",
                       "parallel startup: waiting for submaster %s (%s:%d)",
                        pt->fArgs->fOrd.Data(), pt->fArgs->fHost.Data(),
                        pt->fArgs->fPort);
               pt->fThread->Join();
            }
         }

         TIter next(fSlaves);
         TSlave *sl = 0;
         while ((sl = (TSlave *)next())) {
            if (sl->IsValid()) {
               if (fProtocol == 1) {
                  Error("StartSlaves", "master and submaster protocols"
                        " not compatible (%d and %d)",
                        kPROOF_Protocol, fProtocol);
                  fBadSlaves->Add(sl);
               } else {
                  fAllMonitor->Add(sl->GetSocket());
                  validSlaves.Add(sl);
               }
            } else {
               fBadSlaves->Add(sl);
            }
         }

         // We can cleanup now
         while (!thrHandlers.empty()) {
            i = thrHandlers.end()-1;
            if (*i) {
               SafeDelete(*i);
               thrHandlers.erase(i);
            }
         }

      } else {

         // Here we finalize the server startup: in this way the bulk
         // of remote operations are almost parallelized
         TIter nxsc(&validPairs);
         TPair *sc = 0;
         while ((sc = (TPair *) nxsc())) {

            // Finalize setup of the server
            TSlave *sl = (TSlave *) sc->Key();
            TObjString *cf = (TObjString *) sc->Value();

            sl->SetupServ(TSlave::kMaster, cf->GetName());

            // Monitor good slaves
            if (sl->IsValid()) {
               // check protocol compatability
               // protocol 1 is not supported anymore
               if (fProtocol == 1) {
                  Error("StartSlaves", "master and submaster protocols"
                        " not compatible (%d and %d)",
                        kPROOF_Protocol, fProtocol);
                  fBadSlaves->Add(sl);
               } else {
                  fAllMonitor->Add(sl->GetSocket());
                  validSlaves.Add(sl);
               }
            } else {
               fBadSlaves->Add(sl);
            }

            // Drop the temporary pairs
            validPairs.Remove(sc);
            delete sc;
         }
      }
   }
   fclose(pconf);
   Collect(kAll); //Get kPROOF_LOGFILE and kPROOF_LOGDONE messages
   TIter nextSlave(&validSlaves);
   while (TSlave* sl = dynamic_cast<TSlave*>(nextSlave())){
      if (sl->GetStatus() == -99) {
         Error("StartSlaves", "not allowed to connect to PROOF master server");
         fBadSlaves->Add(sl);
         continue;
      }

      if (!sl->IsValid()) {
         Error("StartSlaves", "failed to setup connection with PROOF master server");
         fBadSlaves->Add(sl);
         continue;
      }

   }
   return kTRUE;
}

//______________________________________________________________________________
Int_t TProofSuperMaster::Process(TDSet *set, const char *selector, Option_t *option,
                                 Long64_t nentries, Long64_t first, TEventList *evl)
{
   // Process a data set (TDSet) using the specified selector (.C) file.
   // Returns -1 in case of error, 0 otherwise.

   if (!IsValid()) return -1;

   Assert(GetPlayer());

   if (GetProgressDialog())
      GetProgressDialog()->ExecPlugin(5, this, selector, set->GetListOfElements()->GetSize(),
                                  first, nentries);

   return GetPlayer()->Process(set, selector, option, nentries, first, evl);
}

//______________________________________________________________________________
void TProofSuperMaster::ValidateDSet(TDSet *dset)
{
   // Validate a TDSet.

   if (dset->ElementsValid()) return;

   TList msds;
   msds.SetOwner();

   TList smholder;
   smholder.SetOwner();
   TList elemholder;
   elemholder.SetOwner();

   // build nodelist with slaves and elements
   TIter nextSlave(GetListOfActiveSlaves());
   while (TSlave *sl = dynamic_cast<TSlave*>(nextSlave())) {
      TList *smlist = 0;
      TPair *p = dynamic_cast<TPair*>(msds.FindObject(sl->GetMsd()));
      if (!p) {
         smlist = new TList;
         smlist->SetName(sl->GetMsd());
         smholder.Add(smlist);
         TList *elemlist = new TSortedList(kSortDescending);
         elemlist->SetName(TString(sl->GetMsd())+"_elem");
         elemholder.Add(elemlist);
         msds.Add(new TPair(smlist, elemlist));
      } else {
         smlist = dynamic_cast<TList*>(p->Key());
      }
      smlist->Add(sl);
   }

   TIter nextElem(dset->GetListOfElements());
   while (TDSetElement *elem = dynamic_cast<TDSetElement*>(nextElem())) {
      if (elem->GetValid()) continue;
      TPair *p = dynamic_cast<TPair*>(msds.FindObject(elem->GetMsd()));
      if (p) {
         dynamic_cast<TList*>(p->Value())->Add(elem);
      } else {
         Error("ValidateDSet", "no mass storage domain '%s' associated"
                               " with available submasters",
                               elem->GetMsd());
         return;
      }
   }

   // send to slaves
   TList usedsms;
   TIter nextSM(&msds);
   SetDSet(dset); // set dset to be validated in Collect()
   while (TPair *msd = dynamic_cast<TPair*>(nextSM())) {
      TList *sms = dynamic_cast<TList*>(msd->Key());
      TList *setelements = dynamic_cast<TList*>(msd->Value());

      // distribute elements over the slaves
      Int_t nsms = sms->GetSize();
      Int_t nelements = setelements->GetSize();
      for (Int_t i=0; i<nsms; i++) {

         TDSet set(dset->GetType(), dset->GetObjName(),
                   dset->GetDirectory());
         for (Int_t j = (i*nelements)/nsms;
                    j < ((i+1)*nelements)/nsms;
                    j++) {
            TDSetElement *elem =
               dynamic_cast<TDSetElement*>(setelements->At(j));
            set.Add(elem->GetFileName(), elem->GetObjName(),
                    elem->GetDirectory(), elem->GetFirst(),
                    elem->GetNum(), elem->GetMsd());
         }

         if (set.GetListOfElements()->GetSize()>0) {
            TMessage mesg(kPROOF_VALIDATE_DSET);
            mesg << &set;

            TSlave *sl = dynamic_cast<TSlave*>(sms->At(i));
            PDB(kGlobal,1) Info("ValidateDSet",
                                "Sending TDSet with %d elements to slave %s"
                                " to be validated",
                                set.GetListOfElements()->GetSize(),
                                sl->GetOrdinal());
            sl->GetSocket()->Send(mesg);
            usedsms.Add(sl);
         }
      }
   }

   PDB(kGlobal,1) Info("ValidateDSet","Calling Collect");
   Collect(&usedsms);
   SetDSet(0);
}

//______________________________________________________________________________
TProofPlayer *TProofSuperMaster::MakePlayer()
{
   // Construct a TProofPlayer object.

   SetPlayer(new TProofPlayerSuperMaster(this));
   return GetPlayer();
}

