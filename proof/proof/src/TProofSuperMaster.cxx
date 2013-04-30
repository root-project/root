// @(#)root/proof:$Id$
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
#include "TObjString.h"
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
#include "TVirtualProofPlayer.h"
#include "TMessage.h"
#include "TUrl.h"
#include "TProofResourcesStatic.h"
#include "TProofNodeInfo.h"
#include "TROOT.h"

ClassImp(TProofSuperMaster)

//______________________________________________________________________________
TProofSuperMaster::TProofSuperMaster(const char *masterurl, const char *conffile,
                                     const char *confdir, Int_t loglevel,
                                     const char *alias, TProofMgr *mgr)
{
   // Start super master PROOF session.

   // Default initializations
   InitMembers();

   // This may be needed during init
   fManager = mgr;

   fUrl = TUrl(masterurl);

   if (!conffile || !conffile[0])
      conffile = kPROOF_ConfFile;
   else if (!strncasecmp(conffile, "sm:", 3))
      conffile+=3;
   if (!confdir  || !confdir[0])
      confdir = kPROOF_ConfDir;

   // Instance type
   fMasterServ = kTRUE;
   ResetBit(TProof::kIsClient);
   SetBit(TProof::kIsMaster);
   SetBit(TProof::kIsTopMaster);

   Init(masterurl, conffile, confdir, loglevel, alias);

   // For Final cleanup
   gROOT->GetListOfProofs()->Add(this);
}

//______________________________________________________________________________
Bool_t TProofSuperMaster::StartSlaves(Bool_t)
{
   // Start up PROOF submasters.

   // If this is a supermaster server, find the config file and start
   // submaster servers as specified in the config file.
   // There is a difference in startup between a slave and a submaster
   // in which the submaster will issue a kPROOF_LOGFILE and
   // then a kPROOF_LOGDONE message (which must be collected)
   // while slaves do not.

   Int_t pc = 0;
   TList *submasterList = new TList;
   // Get list of workers
   if (gProofServ->GetWorkers(submasterList, pc) == TProofServ::kQueryStop) {
      Error("StartSlaves", "getting list of submaster nodes");
      return kFALSE;
   }
   fImage = gProofServ->GetImage();
   if (fImage.IsNull())
      fImage = Form("%s:%s", TUrl(gSystem->HostName()).GetHostFQDN(),
                             gProofServ->GetWorkDir());

   UInt_t nSubmasters = submasterList->GetSize();
   UInt_t nSubmastersDone = 0;
   Int_t ord = 0;
   TList validSubmasters;
   TList validPairs;
   validPairs.SetOwner();

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

      // create submaster server
      TUrl u(Form("%s:%d", submaster->GetNodeName().Data(), sport));
      // Add group info in the password firdl, if any
      if (strlen(gProofServ->GetGroup()) > 0) {
         // Set also the user, otherwise the password is not exported
         if (strlen(u.GetUser()) <= 0)
            u.SetUser(gProofServ->GetUser());
         u.SetPasswd(gProofServ->GetGroup());
      }
      TSlave *slave =
         CreateSubmaster(u.GetUrl(), fullord, image, msd);

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

      ord++;

   } // end loop over all submasters

   // Cleanup
   SafeDelete(submasterList);

   nSubmastersDone = 0;

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
                                    Long64_t nentries, Long64_t first)
{
   // Process a data set (TDSet) using the specified selector (.C) file.
   // Entry- or event-lists should be set in the data set object using
   // TDSet::SetEntryList.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.

   if (!IsValid()) return -1;

   R__ASSERT(GetPlayer());

   if (GetProgressDialog())
      GetProgressDialog()->ExecPlugin(5, this, selector, set->GetListOfElements()->GetSize(),
                                  first, nentries);

   return GetPlayer()->Process(set, selector, option, nentries, first);
}

//______________________________________________________________________________
void TProofSuperMaster::ValidateDSet(TDSet *dset)
{
   // Validate a TDSet.

   if (dset->ElementsValid()) return;

   // We need to recheck after this
   dset->ResetBit(TDSet::kValidityChecked);
   dset->ResetBit(TDSet::kSomeInvalid);

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
      if (smlist) smlist->Add(sl);
   }

   TIter nextElem(dset->GetListOfElements());
   while (TDSetElement *elem = dynamic_cast<TDSetElement*>(nextElem())) {
      if (elem->GetValid()) continue;
      TPair *p = dynamic_cast<TPair*>(msds.FindObject(elem->GetMsd()));
      if (p && p->Value()) {
         TList *xl = dynamic_cast<TList*>(p->Value());
         if (xl) xl->Add(elem);
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
      Int_t nsms = sms ? sms->GetSize() : -1;
      Int_t nelements = setelements ? setelements->GetSize() : -1;
      for (Int_t i=0; i<nsms; i++) {

         TDSet set(dset->GetType(), dset->GetObjName(),
                   dset->GetDirectory());
         for (Int_t j = (i*nelements)/nsms;
                    j < ((i+1)*nelements)/nsms;
                    j++) {
            TDSetElement *elem = setelements ?
               dynamic_cast<TDSetElement*>(setelements->At(j)) : (TDSetElement *)0;
            if (elem) {
               set.Add(elem->GetFileName(), elem->GetObjName(),
                     elem->GetDirectory(), elem->GetFirst(),
                     elem->GetNum(), elem->GetMsd());
            }
         }

         if (set.GetListOfElements()->GetSize()>0) {
            TMessage mesg(kPROOF_VALIDATE_DSET);
            mesg << &set;

            TSlave *sl = dynamic_cast<TSlave*>(sms->At(i));
            if (sl) {
               PDB(kGlobal,1)
                  Info("ValidateDSet",
                     "Sending TDSet with %d elements to worker %s"
                     " to be validated", set.GetListOfElements()->GetSize(),
                                          sl->GetOrdinal());
               sl->GetSocket()->Send(mesg);
               usedsms.Add(sl);
            } else {
               Warning("ValidateDSet", "not a TSlave object");
            }
         }
      }
   }

   PDB(kGlobal,1)
      Info("ValidateDSet","Calling Collect");
   Collect(&usedsms);
   SetDSet(0);
}

//______________________________________________________________________________
TVirtualProofPlayer *TProofSuperMaster::MakePlayer(const char *player, TSocket *s)
{
   // Construct a TProofPlayer object. The player string specifies which
   // player should be created: remote, slave, sm (supermaster) or base.
   // Default is sm. Socket is needed in case a slave player is created.

   if (!player)
      player = "sm";

   SetPlayer(TVirtualProofPlayer::Create(player, this, s));
   return GetPlayer();
}

