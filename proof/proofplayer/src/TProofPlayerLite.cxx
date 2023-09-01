// @(#)root/proofplayer:$Id$
// Author: G. Ganis Mar 2008

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TProofPlayerLite
\ingroup proofkernel

Version of TProofPlayerRemote merges the functionality needed by clients
and masters. It is used in optmized local sessions.

*/

#include "TProofPlayerLite.h"

#include "MessageTypes.h"
#include "TDSet.h"
#include "TDSetProxy.h"
#include "TEntryList.h"
#include "TEventList.h"
#include "THashList.h"
#include "TMap.h"
#include "TMessage.h"
#include "TObjString.h"
#include "TPerfStats.h"
#include "TProofLite.h"
#include "TProofDebug.h"
#include "TProofServ.h"
#include "TROOT.h"
#include "TSelector.h"
#include "TVirtualPacketizer.h"

////////////////////////////////////////////////////////////////////////////////
/// Create the selector object and save the relevant files and binary information
/// in the cache so that the worker can pick it up.
/// Returns 0 and fill fSelector in case of success. Returns -1 and sets
/// fSelector to 0 in case of failure.

Int_t TProofPlayerLite::MakeSelector(const char *selfile)
{
   fSelectorClass = 0;
   SafeDelete(fSelector);
   if (!selfile || strlen(selfile) <= 0) {
      Error("MakeSelector", "input file path or name undefined");
      return -1;
   }

   // If we are just given a name, init the selector and return
   if (!strchr(gSystem->BaseName(selfile), '.')) {
      if (gDebug > 1)
         Info("MakeSelector", "selector name '%s' does not contain a '.':"
              " no file to check, it will be loaded from a library", selfile);
      if (!(fSelector = TSelector::GetSelector(selfile))) {
         Error("MakeSelector", "could not create a %s selector", selfile);
         return -1;
      }
      // Done
      return 0;
   }

   if (((TProofLite*)fProof)->CopyMacroToCache(selfile, 1, &fSelector, TProof::kCp | TProof::kCpBin) < 0)
      return -1;

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process specified TDSet on PROOF.
/// This method is called on client and on the PROOF master.
/// The return value is -1 in case of an error and TSelector::GetStatus() in
/// in case of success.

Long64_t TProofPlayerLite::Process(TDSet *dset, TSelector *selector,
                                   Option_t *option, Long64_t nentries,
                                   Long64_t first)
{
   if (!selector) {
      Error("Process", "selector object undefined");
      return -1;
   }

   // Define fSelector in Client
   if (selector != fSelector) {
      SafeDelete(fSelector);
      fSelector = selector;
   }

   fCreateSelObj = kFALSE;
   Long64_t rc = Process(dset, selector->ClassName(), option, nentries, first);
   fCreateSelObj = kTRUE;

   // Done
   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Process specified TDSet on PROOF.
/// This method is called on client and on the PROOF master.
/// The return value is -1 in case of error and TSelector::GetStatus() in
/// in case of success.

Long64_t TProofPlayerLite::Process(TDSet *dset, const char *selector_file,
                                   Option_t *option, Long64_t nentries,
                                   Long64_t first)
{
   PDB(kGlobal,1) Info("Process","Enter");
   fDSet = dset;
   fExitStatus = kFinished;

   if (!fProgressStatus) {
      Error("Process", "No progress status");
      return -1;
   }
   fProgressStatus->Reset();

   //   delete fOutput;
   if (!fOutput)
      fOutput = new THashList;
   else
      fOutput->Clear();

   TPerfStats::Setup(fInput);
   TPerfStats::Start(fInput, fOutput);

   TStopwatch elapsed;

   TMessage mesg(kPROOF_PROCESS);
   TString fn(gSystem->BaseName(selector_file));

   // Parse option
   Bool_t sync = (fProof->GetQueryMode(option) == TProof::kSync);

   // Make sure that the temporary output list is empty
   if (fOutputLists) {
      fOutputLists->Delete();
      delete fOutputLists;
      fOutputLists = 0;
   }

   if (!sync) {
      gSystem->RedirectOutput(fProof->fLogFileName);
      Printf(" ");
      Info("Process","starting new query");
   }

   if (fCreateSelObj) {
      if (MakeSelector(selector_file) != 0) {
         if (!sync)
            gSystem->RedirectOutput(0);
         return -1;
      }
   }

   fSelectorClass = fSelector->IsA();
   // Add fSelector to inputlist if processing with object
   TList *inputtmp = 0;  // List of temporary input objects
   if (!fCreateSelObj) {
      // In any input list was set into the selector move it to the PROOF
      // input list, because we do not want to stream the selector one
      if (fSelector->GetInputList() && fSelector->GetInputList()->GetSize() > 0) {
         TIter nxi(fSelector->GetInputList());
         TObject *o = 0;
         while ((o = nxi())) {
            if (!fInput->FindObject(o)) {
               fInput->Add(o);
               if (!inputtmp) {
                  inputtmp = new TList;
                  inputtmp->SetOwner(kFALSE);
               }
               inputtmp->Add(o);
            }
         }
      }
      fInput->Add(fSelector);
   }
   // Set the input list for initialization
   fSelector->SetInputList(fInput);
   fSelector->SetOption(option);
   if (fSelector->GetOutputList()) fSelector->GetOutputList()->Clear();

   PDB(kLoop,1) Info("Process","Call Begin(0)");
   fSelector->Begin(0);

   // Send large input data objects, if any
   gProof->SendInputDataFile();

   // Attach to the transient histogram with the assigned packets, if required
   if (fInput->FindObject("PROOF_StatsHist") != 0) {
      if (!(fProcPackets = (TH1 *) fOutput->FindObject("PROOF_ProcPcktHist"))) {
         Warning("Process", "could not attach to histogram 'PROOF_ProcPcktHist'");
      } else {
         PDB(kLoop,1)
            Info("Process", "attached to histogram 'PROOF_ProcPcktHist' to record"
                            " packets being processed");
      }
   }

   PDB(kPacketizer,1) Info("Process","Create Proxy TDSet");
   TDSet *set = new TDSetProxy(dset->GetType(), dset->GetObjName(),
                               dset->GetDirectory());
   if (dset->TestBit(TDSet::kEmpty))
      set->SetBit(TDSet::kEmpty);
   fProof->SetParameter("PROOF_MaxSlavesPerNode", (Long_t) 0);
   if (InitPacketizer(dset, nentries, first, "TPacketizerUnit", "TPacketizer") != 0) {
      Error("Process", "cannot init the packetizer");
      fExitStatus = kAborted;
      return -1;
   }
   // reset start, this is now managed by the packetizer
   first = 0;

   // Negative memlogfreq disable checks.
   // If 0 is passed we try to have 100 messages about memory
   // Otherwise we use the frequency passed.
   Int_t mrc = -1;
   Long64_t memlogfreq = -1, mlf;
   if ((mrc = TProof::GetParameter(fProof->GetInputList(), "PROOF_MemLogFreq", mlf)) == 0) memlogfreq = mlf;
   if (mrc != 0 && gSystem->Getenv("PROOF_MEMLOGFREQ")) {
      TString clf(gSystem->Getenv("PROOF_MEMLOGFREQ"));
      if (clf.IsDigit()) { memlogfreq = clf.Atoi(); mrc = 0; }
   }
   if (memlogfreq == 0) {
      memlogfreq = fPacketizer->GetTotalEntries()/(fProof->GetParallel()*100);
      if (memlogfreq <= 0) memlogfreq = 1;
   }
   if (mrc == 0) fProof->SetParameter("PROOF_MemLogFreq", memlogfreq);

   // Add the unique query tag as TNamed object to the input list
   // so that it is available in TSelectors for monitoring
   fProof->SetParameter("PROOF_QueryTag", fProof->GetName());
   //  ... and the sequential number
   fProof->SetParameter("PROOF_QuerySeqNum", fProof->fSeqNum);

   if (!sync)
      gSystem->RedirectOutput(0);

   TCleanup clean(this);
   SetupFeedback();

   TString opt = option;

   // Workers will get the entry ranges from the packetizer
   Long64_t num = (fProof->IsParallel()) ? -1 : nentries;
   Long64_t fst = (fProof->IsParallel()) ? -1 : first;

   // Entry- or Event- list ?
   TEntryList *enl = (!fProof->IsMaster()) ? dynamic_cast<TEntryList *>(set->GetEntryList())
                                           : (TEntryList *)0;
   TEventList *evl = (!fProof->IsMaster() && !enl) ? dynamic_cast<TEventList *>(set->GetEntryList())
                                           : (TEventList *)0;
   // Reset the merging progress information
   fProof->ResetMergePrg();

   // Broadcast main message
   PDB(kGlobal,1) Info("Process","Calling Broadcast");
   if (fProcessMessage) delete fProcessMessage;
   fProcessMessage = new TMessage(kPROOF_PROCESS);
   mesg << set << fn << fInput << opt << num << fst << evl << sync << enl;
   (*fProcessMessage) << set << fn << fInput << opt << num << fst << evl << sync << enl;
   Int_t nb = fProof->Broadcast(mesg);
   PDB(kGlobal,1) Info("Process", "Broadcast called: %d workers notified", nb);
   fProof->fNotIdle += nb;

   // Redirect logs from master to special log frame
   fProof->fRedirLog = kTRUE;

   if (!sync) {

      // Asynchronous query: just make sure that asynchronous input
      // is enabled and return the prompt
      PDB(kGlobal,1) Info("Process","Asynchronous processing:"
                                    " activating CollectInputFrom");
      fProof->Activate();

      // Return the query sequential number
      return fProof->fSeqNum;

   } else {

      // Wait for processing
      PDB(kGlobal,1) Info("Process","Synchronous processing: calling Collect");
      fProof->Collect();

      // Restore prompt logging (Collect leaves things as they were
      // at the time it was called)
      fProof->fRedirLog = kFALSE;

      if (!TSelector::IsStandardDraw(fn))
         HandleTimer(0); // force an update of final result
      if (fPacketizer) {
         fPacketizer->StopProcess(kFALSE, kTRUE);
         // The progress timer will now stop itself at the next call
         fPacketizer->SetBit(TVirtualPacketizer::kIsDone);
         // Store process info
         elapsed.Stop();
         if (fQuery)
            fQuery->SetProcessInfo(0, 0., fPacketizer->GetBytesRead(),
                                          fPacketizer->GetInitTime(),
                                          elapsed.RealTime());
      }
      StopFeedback();

      Long64_t rc = -1;
      if (GetExitStatus() != TProofPlayer::kAborted)
         rc = Finalize(kFALSE, sync);

      // Remove temporary input objects, if any
      if (inputtmp) {
         TIter nxi(inputtmp);
         TObject *o = 0;
         while ((o = nxi())) fInput->Remove(o);
         SafeDelete(inputtmp);
      }

       // Done
      return rc;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Finalize a query.
/// Returns -1 in case error, 0 otherwise.

Long64_t TProofPlayerLite::Finalize(Bool_t force, Bool_t sync)
{
   if (fOutputLists == 0) {
      if (force && fQuery)
         return fProof->Finalize(Form("%s:%s", fQuery->GetTitle(),
                                               fQuery->GetName()), force);
   }

   Long64_t rv = 0;

   TPerfStats::Stop();

   if (!fQuery) {
      Info("Finalize", "query is undefined!");
      return -1;
   }

   // Some objects (e.g. histos in autobin) may not have been merged yet
   // do it now
   MergeOutput();

   if (fExitStatus != kAborted) {

      if (!sync) {
         // Reinit selector (with multi-sessioning we must do this until
         // TSelector::GetSelector() is optimized to i) avoid reloading of an
         // unchanged selector and ii) invalidate existing instances of
         // reloaded selector)
         if (ReinitSelector(fQuery) == -1) {
            Info("Finalize", "problems reinitializing selector \"%s\"",
                  fQuery->GetSelecImp()->GetName());
            return -1;
         }
      }

      // Some input parameters may be needed in Terminate
      fSelector->SetInputList(fInput);

      TList *output = fSelector->GetOutputList();
      if (output) {
         TIter next(fOutput);
         while(TObject* obj = next()) {
            if (fProof->IsParallel() || DrawCanvas(obj) == 1)
               // Either parallel or not a canvas or not able to display it:
               // just add to the list
               output->Add(obj);
         }
      } else {
         Warning("Finalize", "undefined output list in the selector! Protocol error?");
      }

      SetSelectorDataMembersFromOutputList();

      PDB(kLoop,1) Info("Finalize","Call Terminate()");
      fOutput->Clear("nodelete");
      // This is the end of merging
      SetMerging(kFALSE);
      // We measure the merge time
      fProof->fQuerySTW.Reset();
      // Call Terminate now
      fSelector->Terminate();

      rv = fSelector->GetStatus();

      // copy the output list back and clean the selector's list
      TIter it(output);
      while(TObject* o = it()) {
         fOutput->Add(o);
      }

      // Save the output list in the current query, if any
      if (fQuery) {
         fQuery->SetOutputList(fOutput);
         // Set in finalized state (cannot be done twice)
         fQuery->SetFinalized();
      } else {
         Warning("Finalize","current TQueryResult object is undefined!");
      }

      if (!fCreateSelObj) {
         fInput->Remove(fSelector);
         fOutput->Remove(fSelector);
         if (output) output->Remove(fSelector);
         fSelector = 0;
      }

      // We have transferred copy of the output objects in TQueryResult,
      // so now we can cleanup the selector, making sure that we do not
      // touch the output objects
      if (output) output->SetOwner(kFALSE);
      SafeDelete(fSelector);

      // Delete fOutput (not needed anymore, cannot be finalized twice),
      // making sure that the objects saved in TQueryResult are not deleted
      fOutput->SetOwner(kFALSE);
      SafeDelete(fOutput);
   } else {

      // Cleanup
      fOutput->SetOwner();
      SafeDelete(fSelector);
      if (!fCreateSelObj) fSelector = 0;
   }

   PDB(kGlobal,1) Info("Finalize","exit");
   return rv;
}

////////////////////////////////////////////////////////////////////////////////
/// Send feedback objects to client.

Bool_t TProofPlayerLite::HandleTimer(TTimer *)
{
   PDB(kFeedback,2)
      Info("HandleTimer","Entry: %p", fFeedbackTimer);

   if (fFeedbackTimer == 0) return kFALSE; // timer already switched off


   // process local feedback objects

   TList *fb = new TList;
   fb->SetOwner();

   TIter next(fFeedback);
   while( TObjString *name = (TObjString*) next() ) {
      TObject *o = fOutput->FindObject(name->GetName());
      if (o != 0) fb->Add(o->Clone());
   }

   if (fb->GetSize() > 0)
      StoreFeedback(this, fb); // adopts fb
   else
      delete fb;

   if (fFeedbackLists == 0) {
      fFeedbackTimer->Start(fFeedbackPeriod, kTRUE);   // maybe next time
      return kFALSE;
   }

   fb = MergeFeedback();

   Feedback(fb);
   fb->SetOwner();
   delete fb;

   fFeedbackTimer->Start(fFeedbackPeriod, kTRUE);

   return kFALSE; // ignored?
}

////////////////////////////////////////////////////////////////////////////////
/// Setup reporting of feedback objects.

void TProofPlayerLite::SetupFeedback()
{
   fFeedback = (TList*) fInput->FindObject("FeedbackList");

   if (fFeedback) {
      PDB(kFeedback,1)
         Info("SetupFeedback","\"FeedbackList\" found: %d objects", fFeedback->GetSize());
   } else {
      PDB(kFeedback,1)
         Info("SetupFeedback","\"FeedbackList\" NOT found");
   }

   if (fFeedback == 0 || fFeedback->GetSize() == 0) return;

   // OK, feedback was requested, setup the timer
   SafeDelete(fFeedbackTimer);
   fFeedbackPeriod = 2000;
   TProof::GetParameter(fInput, "PROOF_FeedbackPeriod", fFeedbackPeriod);
   fFeedbackTimer = new TTimer;
   fFeedbackTimer->SetObject(this);
   fFeedbackTimer->Start(fFeedbackPeriod, kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Store feedback results from the specified slave.

void TProofPlayerLite::StoreFeedback(TObject *slave, TList *out)
{
   PDB(kFeedback,1)
      Info("StoreFeedback","Enter (%p,%p,%d)", fFeedbackLists, out, (out ? out->GetSize() : -1));

   if ( out == 0 ) {
      PDB(kFeedback,1)
         Info("StoreFeedback","Leave (empty)");
      return;
   }

   if (fFeedbackLists == 0) {
      PDB(kFeedback,2) Info("StoreFeedback","Create fFeedbackLists");
      fFeedbackLists = new TList;
      fFeedbackLists->SetOwner();
   }

   TIter next(out);
   out->SetOwner(kFALSE);  // take ownership of the contents

   TObject *obj;
   while( (obj = next()) ) {
      PDB(kFeedback,2)
         Info("StoreFeedback","Find '%s'", obj->GetName() );

      TMap *map = (TMap*) fFeedbackLists->FindObject(obj->GetName());
      if ( map == 0 ) {
         PDB(kFeedback,2)
            Info("StoreFeedback", "map for '%s' not found (creating)", obj->GetName());
         // map must not be owner (ownership is with regards to the keys (only))
         map = new TMap;
         map->SetName(obj->GetName());
         fFeedbackLists->Add(map);
      } else {
         PDB(kFeedback,2)
            Info("StoreFeedback","removing previous value");
         if (map->GetValue(slave))
            delete map->GetValue(slave);
         map->Remove(slave);
      }
      map->Add(slave, obj);
   }

   delete out;
   PDB(kFeedback,1)
      Info("StoreFeedback","Leave");
}
