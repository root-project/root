// @(#)root/proof:$Name:  $:$Id: TProofSuperMaster.cxx,v 1.14 2006/05/15 09:45:03 brun Exp $
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
#include "TUrl.h"
#include "TProofResourcesStatic.h"
#include "TProofNodeInfo.h"

ClassImp(TProofSuperMaster)

//______________________________________________________________________________
TProofSuperMaster::TProofSuperMaster(const char *masterurl, const char *conffile,
                                     const char *confdir, Int_t loglevel)
{
   // Start super master PROOF session.

   fUrl = TUrl(masterurl);

   if (!conffile || strlen(conffile) == 0)
      conffile = kPROOF_ConfFile;
   else if (!strncasecmp(conffile, "sm:", 3))
      conffile+=3;
   if (!confdir  || strlen(confdir) == 0)
      confdir = kPROOF_ConfDir;

   Init(masterurl, conffile, confdir, loglevel);
}

//______________________________________________________________________________
Bool_t TProofSuperMaster::StartSlaves(Bool_t parallel, Bool_t)
{
   // Start up PROOF submasters.

   // If this is a supermaster server, find the config file and start
   // submaster servers as specified in the config file.
   // There is a difference in startup between a slave and a submaster
   // in which the submaster will issue a kPROOF_LOGFILE and
   // then a kPROOF_LOGDONE message (which must be collected)
   // while slaves do not.

   // Parse the config file
   TProofResourcesStatic *resources = new TProofResourcesStatic(fConfDir, fConfFile);
   fConfFile = resources->GetFileName(); // Update the global file name (with path)
   PDB(kGlobal,1) Info("StartSlaves", "using PROOF config file: %s", fConfFile.Data());

   // Get the master
   TProofNodeInfo *master = resources->GetMaster();
   if (master)
      fImage = master->GetImage();
   if (!master || (fImage.Length() == 0)) {
      Error("StartSlaves",
            "no appropriate master line found in %s", fConfFile.Data());
      return kFALSE;
   }

   TList *submasterList = resources->GetSubmasters();
   UInt_t nSubmasters = submasterList->GetSize();
   UInt_t nSubmastersDone = 0;
   Int_t ord = 0;
   TList validSubmasters;
   TList validPairs;
   validPairs.SetOwner();

   // Init arrays for threads, if neeeded
   std::vector<TProofThread *> thrHandlers;
   if (parallel) {
      thrHandlers.reserve(nSubmasters);
      if (thrHandlers.max_size() < nSubmasters) {
         PDB(kGlobal,1)
            Info("StartSlaves","cannot reserve enough space thread"
                 " handlers - switch to serial startup");
         parallel = kFALSE;
      }
   }

   // Loop over all submasters and start them
   TListIter next(submasterList);
   TObject *to;
   TProofNodeInfo *submaster;
   while ((to = next())) {
      // Get the next submaster from the list
      submaster = (TProofNodeInfo *)to;
      const Char_t *conffile = submaster->GetConfig();
      const Char_t *image = submaster->GetImage();
      const Char_t *msd = submaster->GetMsd();
      Int_t sport = submaster->GetPort();
      if (sport == -1)
         sport = fUrl.GetPort();

      TString fullord = TString(gProofServ->GetOrdinal()) + "." + ((Long_t) ord);
      if (parallel) {
         // Prepare arguments
         TProofThreadArg *ta =
             new TProofThreadArg(submaster->GetNodeName().Data(), sport,
                                 fullord, image, conffile, msd, fSlaves, this);
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
               // Notify opening of connection
               nSubmastersDone++;
               TMessage m(kPROOF_SERVERSTARTED);
               m << TString("Opening connections to submasters") << nSubmasters
                 << nSubmastersDone << kTRUE;
               gProofServ->GetSocket()->Send(m);
            }
         } else {
            Info("StartSlaves","Can't create thread arguments object:"
                 " out of system resources");
         }
      } // end if (parallel)
      else {
         // create submaster server
         TSlave *slave =
            CreateSubmaster(Form("%s:%d", submaster->GetNodeName().Data(), sport),
                            fullord, image, msd);

         // Add to global list (we will add to the monitor list after
         // finalizing the server startup)
         Bool_t submasterOk = kTRUE;
         fSlaves->Add(slave);
         if (slave->IsValid()) {
            validPairs.Add(new TPair(slave, new TObjString(conffile)));
         } else {
            submasterOk = kFALSE;
            fBadSlaves->Add(slave);
         }

         PDB(kGlobal,3)
            Info("StartSlaves","submaster on host %s created and"
                 " added to list", submaster->GetNodeName().Data());

         // Notify opening of connection
         nSubmastersDone++;
         TMessage m(kPROOF_SERVERSTARTED);
         m << TString("Opening connections to submasters") << nSubmasters
           << nSubmastersDone << submasterOk;
         gProofServ->GetSocket()->Send(m);
      } // end else (create submaster server)

      ord++;

   } // end loop over all submasters

   // Cleanup
   delete resources;
   resources = 0;

   nSubmastersDone = 0;
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
                    pt->fArgs->fOrd.Data(), pt->fArgs->fUrl->GetHost(),
                    pt->fArgs->fUrl->GetPort());
            pt->fThread->Join();
         }

         // Notify end of startup operations
         nSubmastersDone++;
         TMessage m(kPROOF_SERVERSTARTED);
         m << TString("Setting up submasters") << nSubmasters
           << nSubmastersDone << kTRUE;
         gProofServ->GetSocket()->Send(m);
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
               validSubmasters.Add(sl);
            }
         } else {
            fBadSlaves->Add(sl);
         }
      } // end while

      // We can cleanup now
      while (!thrHandlers.empty()) {
         i = thrHandlers.end()-1;
         if (*i) {
            SafeDelete(*i);
            thrHandlers.erase(i);
         }
      }
   } // end if (parallel)
   else {
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
         Bool_t submasterOk = kTRUE;
         if (sl->IsValid()) {
            // check protocol compatability
            // protocol 1 is not supported anymore
            if (fProtocol == 1) {
               Error("StartSlaves", "master and submaster protocols"
                     " not compatible (%d and %d)",
                     kPROOF_Protocol, fProtocol);
               submasterOk = kFALSE;
               fBadSlaves->Add(sl);
            } else {
               fAllMonitor->Add(sl->GetSocket());
               validSubmasters.Add(sl);
            }
         } else {
            submasterOk = kFALSE;
            fBadSlaves->Add(sl);
         }

         // Notify end of startup operations
         nSubmastersDone++;
         TMessage m(kPROOF_SERVERSTARTED);
         m << TString("Setting up submasters") << nSubmasters
           << nSubmastersDone << submasterOk;
         gProofServ->GetSocket()->Send(m);
      }
   }

   Collect(kAll); //Get kPROOF_LOGFILE and kPROOF_LOGDONE messages
   TIter nextSubmaster(&validSubmasters);
   while (TSlave* sl = dynamic_cast<TSlave*>(nextSubmaster())) {
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
Long64_t TProofSuperMaster::Process(TDSet *set, const char *selector, Option_t *option,
                                    Long64_t nentries, Long64_t first, TEventList *evl)
{
   // Process a data set (TDSet) using the specified selector (.C) file.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.

   if (!IsValid()) return -1;

   R__ASSERT(GetPlayer());

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
   TIter nextSubmaster(GetListOfActiveSlaves());
   while (TSlave *sl = dynamic_cast<TSlave*>(nextSubmaster())) {
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

