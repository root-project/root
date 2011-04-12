// @(#)root/proofplayer:$Id$
// Author: Maarten Ballintijn   07/01/02

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofPlayer                                                         //
//                                                                      //
// This internal class and its subclasses steer the processing in PROOF.//
// Instances of the TProofPlayer class are created on the worker nodes  //
// per session and do the processing.                                   //
// Instances of its subclass - TProofPlayerRemote are created per each  //
// query on the master(s) and on the client. On the master(s),          //
// TProofPlayerRemote coordinate processing, check the dataset, create  //
// the packetizer and take care of merging the results of the workers.  //
// The instance on the client collects information on the input         //
// (dataset and selector), it invokes the Begin() method and finalizes  //
// the query by calling Terminate().                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofDraw.h"
#include "TProofPlayer.h"
#include "THashList.h"
#include "TEnv.h"
#include "TEventIter.h"
#include "TVirtualPacketizer.h"
#include "TSelector.h"
#include "TSocket.h"
#include "TProofServ.h"
#include "TProof.h"
#include "TProofOutputFile.h"
#include "TProofSuperMaster.h"
#include "TSlave.h"
#include "TClass.h"
#include "TROOT.h"
#include "TError.h"
#include "TException.h"
#include "MessageTypes.h"
#include "TMessage.h"
#include "TDSetProxy.h"
#include "TString.h"
#include "TSystem.h"
#include "TFile.h"
#include "TFileCollection.h"
#include "TFileInfo.h"
#include "TFileMerger.h"
#include "TProofDebug.h"
#include "TTimer.h"
#include "TMap.h"
#include "TPerfStats.h"
#include "TStatus.h"
#include "TEventList.h"
#include "TProofLimitsFinder.h"
#include "TSortedList.h"
#include "TTree.h"
#include "TEntryList.h"
#include "TDSet.h"
#include "TDrawFeedback.h"
#include "TNamed.h"
#include "TObjString.h"
#include "TQueryResult.h"
#include "TMD5.h"
#include "TMethodCall.h"
#include "TObjArray.h"
#include "TMutex.h"
#include "TH1.h"
#include "TVirtualMonitoring.h"
#include "TParameter.h"
#include "TOutputListSelectorDataMap.h"

// Timeout exception
#define kPEX_STOPPED  1001
#define kPEX_ABORTED  1002

// To flag an abort condition: use a local static variable to avoid
// warnings about problems with longjumps
static Bool_t gAbort = kFALSE;

class TAutoBinVal : public TNamed {
private:
   Double_t fXmin, fXmax, fYmin, fYmax, fZmin, fZmax;

public:
   TAutoBinVal(const char *name, Double_t xmin, Double_t xmax, Double_t ymin,
               Double_t ymax, Double_t zmin, Double_t zmax) : TNamed(name,"")
   {
      fXmin = xmin; fXmax = xmax;
      fYmin = ymin; fYmax = ymax;
      fZmin = zmin; fZmax = zmax;
   }
   void GetAll(Double_t& xmin, Double_t& xmax, Double_t& ymin,
               Double_t& ymax, Double_t& zmin, Double_t& zmax)
   {
      xmin = fXmin; xmax = fXmax;
      ymin = fYmin; ymax = fYmax;
      zmin = fZmin; zmax = fZmax;
   }

};

//
// Special timer to dispatch pending events while processing
//______________________________________________________________________________
class TDispatchTimer : public TTimer {
private:
   TProofPlayer    *fPlayer;

public:
   TDispatchTimer(TProofPlayer *p) : TTimer(1000, kFALSE), fPlayer(p) { }

   Bool_t Notify();
};
//______________________________________________________________________________
Bool_t TDispatchTimer::Notify()
{
   // Handle expiration of the timer associated with dispatching pending
   // events while processing. We must act as fast as possible here, so
   // we just set a flag submitting a request for dispatching pending events

   if (gDebug > 0)
      Info ("Notify","called!");

   fPlayer->SetBit(TProofPlayer::kDispatchOneEvent);

   // Needed for the next shot
   Reset();
   return kTRUE;
}

//
// Special timer to handle stop/abort request via exception raising
//______________________________________________________________________________
class TStopTimer : public TTimer {
private:
   Bool_t           fAbort;
   TProofPlayer    *fPlayer;

public:
   TStopTimer(TProofPlayer *p, Bool_t abort, Int_t to);

   Bool_t Notify();
};

//______________________________________________________________________________
TStopTimer::TStopTimer(TProofPlayer *p, Bool_t abort, Int_t to)
           : TTimer(((to <= 0 || to > 864000) ? 10 : to * 1000), kFALSE)
{
   // Constructor for the timer to stop/abort processing.
   // The 'timeout' is in seconds.
   // Make sure that 'to' make sense, i.e. not larger than 10 days;
   // the minimum value is 10 ms (0 does not seem to start the timer ...).

   if (gDebug > 0)
      Info ("TStopTimer","enter: %d, timeout: %d", abort, to);

   fPlayer = p;
   fAbort = abort;

   if (gDebug > 1)
      Info ("TStopTimer","timeout set to %s ms", fTime.AsString());
}

//______________________________________________________________________________
Bool_t TStopTimer::Notify()
{
   // Handle the signal coming from the expiration of the timer
   // associated with an abort or stop request.
   // We raise an exception which will be processed in the
   // event loop.

   if (gDebug > 0)
      Info ("Notify","called!");

   if (fAbort)
      Throw(kPEX_ABORTED);
   else
      Throw(kPEX_STOPPED);

   return kTRUE;
}

//------------------------------------------------------------------------------

ClassImp(TProofPlayer)

THashList *TProofPlayer::fgDrawInputPars = 0;

//______________________________________________________________________________
TProofPlayer::TProofPlayer(TProof *)
   : fAutoBins(0), fOutput(0), fSelector(0), fSelectorClass(0),
     fFeedbackTimer(0), fFeedbackPeriod(2000),
     fEvIter(0), fSelStatus(0),
     fTotalEvents(0), fQueryResults(0), fQuery(0), fDrawQueries(0),
     fMaxDrawQueries(1), fStopTimer(0), fStopTimerMtx(0), fDispatchTimer(0)
{
   // Default ctor.

   fInput         = new TList;
   fExitStatus    = kFinished;
   fProgressStatus = new TProofProgressStatus();
   SetProcessing(kFALSE);

   static Bool_t initLimitsFinder = kFALSE;
   if (!initLimitsFinder && gProofServ && !gProofServ->IsMaster()) {
      THLimitsFinder::SetLimitsFinder(new TProofLimitsFinder);
      initLimitsFinder = kTRUE;
   }
}

//______________________________________________________________________________
TProofPlayer::~TProofPlayer()
{
   // Destructor.

   fInput->Clear("nodelete");
   SafeDelete(fInput);
   // The output list is owned by fSelector and destroyed in there
   SafeDelete(fSelector);
   SafeDelete(fFeedbackTimer);
   SafeDelete(fEvIter);
   SafeDelete(fQueryResults);
   SafeDelete(fDispatchTimer);
   SafeDelete(fStopTimer);
}

//______________________________________________________________________________
void TProofPlayer::SetProcessing(Bool_t on)
{
   // Set processing bit according to 'on'

   if (on)
      SetBit(TProofPlayer::kIsProcessing);
   else
      ResetBit(TProofPlayer::kIsProcessing);
}

//______________________________________________________________________________
void TProofPlayer::StopProcess(Bool_t abort, Int_t timeout)
{
   // Stop the process after this event. If timeout is positive, start
   // a timer firing after timeout seconds to hard-stop time-expensive
   // events.

   if (gDebug > 0)
      Info ("StopProcess","abort: %d, timeout: %d", abort, timeout);

   if (fEvIter != 0)
      fEvIter->StopProcess(abort);
   Long_t to = 1;
   if (abort == kTRUE) {
      fExitStatus = kAborted;
   } else {
      fExitStatus = kStopped;
      to = timeout;
   }
   // Start countdown, if needed
   if (to > 0)
      SetStopTimer(kTRUE, abort, to);
}

//______________________________________________________________________________
void TProofPlayer::SetDispatchTimer(Bool_t on)
{
   // Enable/disable the timer to dispatch pening events while processing.

   SafeDelete(fDispatchTimer);
   ResetBit(TProofPlayer::kDispatchOneEvent);
   if (on) {
      fDispatchTimer = new TDispatchTimer(this);
      fDispatchTimer->Start();
   }
}

//______________________________________________________________________________
void TProofPlayer::SetStopTimer(Bool_t on, Bool_t abort, Int_t timeout)
{
   // Enable/disable the timer to stop/abort processing.
   // The 'timeout' is in seconds.

   fStopTimerMtx = (fStopTimerMtx) ? fStopTimerMtx : new TMutex(kTRUE);
   R__LOCKGUARD(fStopTimerMtx);

   // Clean-up the timer
   SafeDelete(fStopTimer);
   if (on) {
      // create timer
      fStopTimer = new TStopTimer(this, abort, timeout);
      // Start the countdown
      fStopTimer->Start();
      if (gDebug > 0)
         Info ("SetStopTimer", "%s timer STARTED (timeout: %d)",
                               (abort ? "ABORT" : "STOP"), timeout);
   } else {
      if (gDebug > 0)
         Info ("SetStopTimer", "timer STOPPED");
   }
}

//______________________________________________________________________________
void TProofPlayer::AddQueryResult(TQueryResult *q)
{
   // Add query result to the list, making sure that there are no
   // duplicates.

   if (!q) {
      Warning("AddQueryResult","query undefined - do nothing");
      return;
   }

   // Treat differently normal and draw queries
   if (!(q->IsDraw())) {
      if (!fQueryResults) {
         fQueryResults = new TList;
         fQueryResults->Add(q);
      } else {
         TIter nxr(fQueryResults);
         TQueryResult *qr = 0;
         TQueryResult *qp = 0;
         while ((qr = (TQueryResult *) nxr())) {
            // If same query, remove old version and break
            if (*qr == *q) {
               fQueryResults->Remove(qr);
               delete qr;
               break;
            }
            // Record position according to start time
            if (qr->GetStartTime().Convert() <= q->GetStartTime().Convert())
               qp = qr;
         }

         if (!qp) {
            fQueryResults->AddFirst(q);
         } else {
            fQueryResults->AddAfter(qp, q);
         }
      }
   } else if (IsClient()) {
      // If max reached, eliminate first the oldest one
      if (fDrawQueries == fMaxDrawQueries && fMaxDrawQueries > 0) {
         TIter nxr(fQueryResults);
         TQueryResult *qr = 0;
         while ((qr = (TQueryResult *) nxr())) {
            // If same query, remove old version and break
            if (qr->IsDraw()) {
               fDrawQueries--;
               fQueryResults->Remove(qr);
               delete qr;
               break;
            }
         }
      }
      // Add new draw query
      if (fDrawQueries >= 0 && fDrawQueries < fMaxDrawQueries) {
         fDrawQueries++;
         if (!fQueryResults)
            fQueryResults = new TList;
         fQueryResults->Add(q);
      }
   }
}

//______________________________________________________________________________
void TProofPlayer::RemoveQueryResult(const char *ref)
{
   // Remove all query result instances referenced 'ref' from
   // the list of results.

   if (fQueryResults) {
      TIter nxq(fQueryResults);
      TQueryResult *qr = 0;
      while ((qr = (TQueryResult *) nxq())) {
         if (qr->Matches(ref)) {
            fQueryResults->Remove(qr);
            delete qr;
         }
      }
   }
}

//______________________________________________________________________________
TQueryResult *TProofPlayer::GetQueryResult(const char *ref)
{
   // Get query result instances referenced 'ref' from
   // the list of results.

   if (fQueryResults) {
      if (ref && strlen(ref) > 0) {
         TIter nxq(fQueryResults);
         TQueryResult *qr = 0;
         while ((qr = (TQueryResult *) nxq())) {
            if (qr->Matches(ref))
               return qr;
         }
      } else {
         // Get last
         return (TQueryResult *) fQueryResults->Last();
      }
   }

   // Nothing found
   return (TQueryResult *)0;
}

//______________________________________________________________________________
void TProofPlayer::SetCurrentQuery(TQueryResult *q)
{
   // Set current query and save previous value.

   fPreviousQuery = fQuery;
   fQuery = q;
}

//______________________________________________________________________________
void TProofPlayer::AddInput(TObject *inp)
{
   // Add object to input list.

   fInput->Add(inp);
}

//______________________________________________________________________________
void TProofPlayer::ClearInput()
{
   // Clear input list.

   fInput->Clear();
}

//______________________________________________________________________________
TObject *TProofPlayer::GetOutput(const char *name) const
{
   // Get output object by name.

   if (fOutput)
      return fOutput->FindObject(name);
   return 0;
}

//______________________________________________________________________________
TList *TProofPlayer::GetOutputList() const
{
   // Get output list.

   TList *ol = fOutput;
   if (!ol && fQuery)
      ol = fQuery->GetOutputList();
   return ol;
}

//______________________________________________________________________________
Int_t TProofPlayer::ReinitSelector(TQueryResult *qr)
{
   // Reinitialize fSelector using the selector files in the query result.
   // Needed when Finalize is called after a Process execution for the same
   // selector name.

   Int_t rc = 0;

   // Make sure we have a query
   if (!qr) {
      Info("ReinitSelector", "query undefined - do nothing");
      return -1;
   }

   // Selector name
   TString selec = qr->GetSelecImp()->GetName();
   if (selec.Length() <= 0) {
      Info("ReinitSelector", "selector name undefined - do nothing");
      return -1;
   }

   // Find out if this is a standard selection used for Draw actions
   Bool_t stdselec = TSelector::IsStandardDraw(selec);

   // Find out if this is a precompiled selector: in such a case we do not
   // have the code in TMacros, so we must rely on local libraries
   Bool_t compselec = (selec.Contains(".") || stdselec) ? kFALSE : kTRUE;

   // If not, find out if it needs to be expanded
   TString ipathold;
   if (!stdselec && !compselec) {
      // Check checksums for the versions of the selector files
      Bool_t expandselec = kTRUE;
      TString dir, ipath;
      char *selc = gSystem->Which(TROOT::GetMacroPath(), selec, kReadPermission);
      if (selc) {
         // Check checksums
         TMD5 *md5icur = 0, *md5iold = 0, *md5hcur = 0, *md5hold = 0;
         // Implementation files
         md5icur = TMD5::FileChecksum(selc);
         md5iold = qr->GetSelecImp()->Checksum();
         // Header files
         TString selh(selc);
         Int_t dot = selh.Last('.');
         if (dot != kNPOS) selh.Remove(dot);
         selh += ".h";
         if (!gSystem->AccessPathName(selh, kReadPermission))
            md5hcur = TMD5::FileChecksum(selh);
         md5hold = qr->GetSelecHdr()->Checksum();

         // If nothing has changed nothing to do
         if (md5hcur && md5hold && md5icur && md5iold)
            if (*md5hcur == *md5hold && *md5icur == *md5iold)
               expandselec = kFALSE;

         SafeDelete(md5icur);
         SafeDelete(md5hcur);
         SafeDelete(md5iold);
         SafeDelete(md5hold);
         if (selc) delete [] selc;
      }

      Bool_t ok = kTRUE;
      // Expand selector files, if needed
      if (expandselec) {

         ok = kFALSE;
         // Expand files in a temporary directory
         TUUID u;
         dir = Form("%s/%s",gSystem->TempDirectory(),u.AsString());
         if (!(gSystem->MakeDirectory(dir))) {

            // Export implementation file
            selec = Form("%s/%s",dir.Data(),selec.Data());
            qr->GetSelecImp()->SaveSource(selec);

            // Export header file
            TString seleh = Form("%s/%s",dir.Data(),qr->GetSelecHdr()->GetName());
            qr->GetSelecHdr()->SaveSource(seleh);

            // Adjust include path
            ipathold = gSystem->GetIncludePath();
            ipath = Form("-I%s %s", dir.Data(), gSystem->GetIncludePath());
            gSystem->SetIncludePath(ipath.Data());

            ok = kTRUE;
         }
      }
      TString opt(qr->GetOptions());
      Ssiz_t id = opt.Last('#');
      if (id != kNPOS && id < opt.Length() - 1)
         selec += opt(id + 1, opt.Length());

      if (!ok) {
         Info("ReinitSelector", "problems locating or exporting selector files");
         return -1;
      }
   }

   // Cleanup previous stuff
   SafeDelete(fSelector);
   fSelectorClass = 0;

   // Init the selector now
   Int_t iglevelsave = gErrorIgnoreLevel;
   if (compselec)
      // Silent error printout on first attempt
      gErrorIgnoreLevel = kBreak;

   if ((fSelector = TSelector::GetSelector(selec))) {
      if (compselec)
         gErrorIgnoreLevel = iglevelsave; // restore ignore level
      fSelectorClass = fSelector->IsA();
      fSelector->SetOption(qr->GetOptions());

   } else {
      if (compselec) {
         gErrorIgnoreLevel = iglevelsave; // restore ignore level
         // Retry by loading first the libraries listed in TQueryResult, if any
         if (strlen(qr->GetLibList()) > 0) {
            TString sl(qr->GetLibList());
            TObjArray *oa = sl.Tokenize(" ");
            if (oa) {
               Bool_t retry = kFALSE;
               TIter nxl(oa);
               TObjString *os = 0;
               while ((os = (TObjString *) nxl())) {
                  TString lib = gSystem->BaseName(os->GetName());
                  if (lib != "lib") {
                     lib.ReplaceAll("-l", "lib");
                     if (gSystem->Load(lib) == 0)
                        retry = kTRUE;
                  }
               }
               // Retry now, if the case
               if (retry)
                  fSelector = TSelector::GetSelector(selec);
            }
         }
      }
      if (!fSelector) {
         if (compselec)
            Info("ReinitSelector", "compiled selector re-init failed:"
                                   " automatic reload unsuccessful:"
                                   " please load manually the correct library");
         rc = -1;
      }
   }
   if (fSelector) {
      // Draw needs to reinit temp histos
      fSelector->SetInputList(qr->GetInputList());
      if (stdselec) {
         ((TProofDraw *)fSelector)->DefVar();
      } else {
         // variables may have been initialized in Begin()
         fSelector->Begin(0);
      }
   }

   // Restore original include path, if needed
   if (ipathold.Length() > 0)
      gSystem->SetIncludePath(ipathold.Data());

   return rc;
}

//______________________________________________________________________________
Int_t TProofPlayer::AddOutputObject(TObject *)
{
   // Incorporate output object (may not be used in this class).

   MayNotUse("AddOutputObject");
   return -1;
}

//______________________________________________________________________________
void TProofPlayer::AddOutput(TList *)
{
   // Incorporate output list (may not be used in this class).

   MayNotUse("AddOutput");
}

//______________________________________________________________________________
void TProofPlayer::StoreOutput(TList *)
{
   // Store output list (may not be used in this class).

   MayNotUse("StoreOutput");
}

//______________________________________________________________________________
void TProofPlayer::StoreFeedback(TObject *, TList *)
{
   // Store feedback list (may not be used in this class).

   MayNotUse("StoreFeedback");
}

//______________________________________________________________________________
void TProofPlayer::Progress(Long64_t /*total*/, Long64_t /*processed*/)
{
   // Report progress (may not be used in this class).

   MayNotUse("Progress");
}

//______________________________________________________________________________
void TProofPlayer::Progress(Long64_t /*total*/, Long64_t /*processed*/,
                            Long64_t /*bytesread*/,
                            Float_t /*evtRate*/, Float_t /*mbRate*/,
                            Float_t /*evtrti*/, Float_t /*mbrti*/)
{
   // Report progress (may not be used in this class).

   MayNotUse("Progress");
}

//______________________________________________________________________________
void TProofPlayer::Progress(TProofProgressInfo * /*pi*/)
{
   // Report progress (may not be used in this class).

   MayNotUse("Progress");
}

//______________________________________________________________________________
void TProofPlayer::Feedback(TList *)
{
   // Set feedback list (may not be used in this class).

   MayNotUse("Feedback");
}

//______________________________________________________________________________
TDrawFeedback *TProofPlayer::CreateDrawFeedback(TProof *p)
{
   // Draw feedback creation proxy. When accessed via TProof avoids
   // link dependency on libProofPlayer.

   return new TDrawFeedback(p);
}

//______________________________________________________________________________
void TProofPlayer::SetDrawFeedbackOption(TDrawFeedback *f, Option_t *opt)
{
   // Set draw feedback option.

   if (f)
      f->SetOption(opt);
}

//______________________________________________________________________________
void TProofPlayer::DeleteDrawFeedback(TDrawFeedback *f)
{
   // Delete draw feedback object.

   delete f;
}

//______________________________________________________________________________
Long64_t TProofPlayer::Process(TDSet *dset, const char *selector_file,
                               Option_t *option, Long64_t nentries,
                               Long64_t first)
{
   // Process specified TDSet on PROOF worker.
   // The return value is -1 in case of error and TSelector::GetStatus()
   // in case of success.

   PDB(kGlobal,1) Info("Process","Enter");

   fExitStatus = kFinished;
   fOutput = 0;

   TCleanup clean(this);

   SafeDelete(fSelector);
   fSelectorClass = 0;
   Int_t version = -1;
   TRY {
      // Get selector files from cache
      if (gProofServ) {
         gProofServ->GetCacheLock()->Lock();
         gProofServ->CopyFromCache(selector_file, 1);
      }

      if (!(fSelector = TSelector::GetSelector(selector_file))) {
         Error("Process", "cannot load: %s", selector_file );
         gProofServ->GetCacheLock()->Unlock();
         return -1;
      }

      // Save binaries to cache, if any
      if (gProofServ) {
         gProofServ->CopyToCache(selector_file, 1);
         gProofServ->GetCacheLock()->Unlock();
      }

      fSelectorClass = fSelector->IsA();
      version = fSelector->Version();

      fOutput = fSelector->GetOutputList();

      if (gProofServ)
         TPerfStats::Start(fInput, fOutput);

      fSelStatus = new TStatus;
      fOutput->Add(fSelStatus);

      fSelector->SetOption(option);
      fSelector->SetInputList(fInput);

      // If in sequential (0-PROOF) mode validate the data set to get
      // the number of entries
      fTotalEvents = nentries;
      if (fTotalEvents < 0 && gProofServ &&
         gProofServ->IsMaster() && !gProofServ->IsParallel()) {
         dset->Validate();
         dset->Reset();
         TDSetElement *e = 0;
         while ((e = dset->Next())) {
            fTotalEvents += e->GetNum();
         }
      }

      dset->Reset();

      // Set parameters controlling the iterator behaviour
      Int_t useTreeCache = 1;
      if (TProof::GetParameter(fInput, "PROOF_UseTreeCache", useTreeCache) == 0) {
         if (useTreeCache > -1 && useTreeCache < 2)
            gEnv->SetValue("ProofPlayer.UseTreeCache", useTreeCache);
      }
      Long64_t cacheSize = -1;
      if (TProof::GetParameter(fInput, "PROOF_CacheSize", cacheSize) == 0) {
         TString sz = TString::Format("%lld", cacheSize);
         gEnv->SetValue("ProofPlayer.CacheSize", sz.Data());
      }
      // Parallel unzipping
      Int_t useParallelUnzip = 0;
      if (TProof::GetParameter(fInput, "PROOF_UseParallelUnzip", useParallelUnzip) == 0) {
         if (useParallelUnzip > -1 && useParallelUnzip < 2)
            gEnv->SetValue("ProofPlayer.UseParallelUnzip", useParallelUnzip);
      }
      // OS file caching (Mac Os X only)
      Int_t dontCacheFiles = 0;
      if (TProof::GetParameter(fInput, "PROOF_DontCacheFiles", dontCacheFiles) == 0) {
         if (dontCacheFiles == 1)
            gEnv->SetValue("ProofPlayer.DontCacheFiles", 1);
      }
      fEvIter = TEventIter::Create(dset, fSelector, first, nentries);

      if (version == 0) {
         PDB(kLoop,1) Info("Process","Call Begin(0)");
         fSelector->Begin(0);
      } else {
         if (IsClient()) {
            // on client (for local run)
            PDB(kLoop,1) Info("Process","Call Begin(0)");
            fSelector->Begin(0);
         }
         if (fSelStatus->IsOk()) {
            PDB(kLoop,1) Info("Process","Call SlaveBegin(0)");
            fSelector->SlaveBegin(0);  // Init is called explicitly
                                       // from GetNextEvent()
         }
      }

   } CATCH(excode) {
      SetProcessing(kFALSE);
      Error("Process","exception %d caught", excode);
      gProofServ->GetCacheLock()->Unlock();
      return -1;
   } ENDTRY;

   // Create feedback lists, if required
   SetupFeedback();

   if (gMonitoringWriter)
      gMonitoringWriter->SendProcessingStatus("STARTED",kTRUE);

   PDB(kLoop,1)
      Info("Process","Looping over Process()");

   // get the byte read counter at the beginning of processing
   Long64_t readbytesatstart = TFile::GetFileBytesRead();
   Long64_t readcallsatstart = TFile::GetFileReadCalls();
   // force the first monitoring info
   if (gMonitoringWriter)
      gMonitoringWriter->SendProcessingProgress(0,0,kTRUE);

   // Start asynchronous timer to dispatch pending events
   SetDispatchTimer(kTRUE);

   // Loop over range
   gAbort = kFALSE;
   Long64_t entry;
   fProgressStatus->Reset();
   if (gProofServ) gProofServ->ResetBit(TProofServ::kHighMemory);

   TRY {

      // Get the frequency for checking memory consumption and logging information
      TParameter<Long64_t> *par = (TParameter<Long64_t>*)fInput->FindObject("PROOF_MemLogFreq");
      Long64_t singleshot = 1, memlogfreq = (par) ? par->GetVal() : 100;
      Bool_t warnHWMres = kTRUE, warnHWMvir = kTRUE;
      TString lastMsg, wmsg;

      // Initial memory footprint
      if (!CheckMemUsage(singleshot, warnHWMres, warnHWMvir, wmsg)) {
         Error("Process", "%s", wmsg.Data());
         wmsg.Insert(0, TString::Format("ERROR:%s, after SlaveBegin(), ", gProofServ->GetOrdinal()));
         fSelStatus->Add(wmsg.Data());
         if (gProofServ) {
            gProofServ->SendAsynMessage(wmsg.Data());
            gProofServ->SetBit(TProofServ::kHighMemory);
         }
         fExitStatus = kStopped;
         SetProcessing(kFALSE);
      } else if (!wmsg.IsNull()) {
         Warning("Process", "%s", wmsg.Data());
      }

      TPair *currentElem = 0;
      // The event loop on the worker
      while ((entry = fEvIter->GetNextEvent()) >= 0 && fSelStatus->IsOk() &&
              fSelector->GetAbort() == TSelector::kContinue) {

         // This is needed by the inflate infrastructure to calculate
         // sleeping times
         SetProcessing(kTRUE);

         // Give the possibility to the selector to access additional info in the
         // incoming packet
         lastMsg = "(unfortunately no detailed info is available about current packet)";
         if (dset->Current()) {
            if (!currentElem) {
               currentElem = new TPair(new TObjString("PROOF_CurrentElement"), dset->Current());
               fInput->Add(currentElem);
            } else {
               if (currentElem->Value() != dset->Current()) {
                  currentElem->SetValue(dset->Current());
               } else if (dset->Current()->TestBit(TDSetElement::kNewRun)) {
                  dset->Current()->ResetBit(TDSetElement::kNewRun);
               }
            }
            if (dset->TestBit(TDSet::kEmpty)) {
               lastMsg.Form("while processing cycle:%lld - check logs for possible stacktrace", entry);
            } else {
               TDSetElement *elem = dynamic_cast<TDSetElement *>(currentElem->Value());
               TString fn = (elem) ? elem->GetFileName() : "<undef>";
               lastMsg.Form("while processing dset:'%s', file:'%s', event:%lld"
                            " - check logs for possible stacktrace", dset->GetName(), fn.Data(), entry);
            }
         }
         // This will be sent to clients in case of exceptions ...
         TProofServ::SetLastMsg(lastMsg);

         if (version == 0) {
            PDB(kLoop,3)
               Info("Process","Call ProcessCut(%lld)", entry);
            if (fSelector->ProcessCut(entry)) {
               PDB(kLoop,3)
                  Info("Process","Call ProcessFill(%lld)", entry);
               fSelector->ProcessFill(entry);
            }
         } else {
            PDB(kLoop,3)
               Info("Process","Call Process(%lld)", entry);
            fSelector->Process(entry);
            if (fSelector->GetAbort() == TSelector::kAbortProcess) {
               SetProcessing(kFALSE);
               break;
            } else if (fSelector->GetAbort() == TSelector::kAbortFile) {
               Info("Process", "packet processing aborted following the selector settings:\n%s",
                               lastMsg.Data());
               fEvIter->InvalidatePacket();
               fProgressStatus->SetBit(TProofProgressStatus::kFileCorrupted);
            }
         }

         if (fSelStatus->IsOk()) {
            fProgressStatus->IncEntries();
            fProgressStatus->SetBytesRead(TFile::GetFileBytesRead()-readbytesatstart);
            fProgressStatus->SetReadCalls(TFile::GetFileReadCalls()-readcallsatstart);
            if (gMonitoringWriter)
               gMonitoringWriter->SendProcessingProgress(fProgressStatus->GetEntries(),
                       TFile::GetFileBytesRead()-readbytesatstart, kFALSE);
         }
         // Check the memory footprint, if required
         if (!CheckMemUsage(memlogfreq, warnHWMres, warnHWMvir, wmsg)) {
            Error("Process", "%s", wmsg.Data());
            if (gProofServ) {
               wmsg.Insert(0, TString::Format("ERROR:%s, entry:%lld, ", gProofServ->GetOrdinal(), entry));
               gProofServ->SendAsynMessage(wmsg.Data());
            }
            fExitStatus = kStopped;
            SetProcessing(kFALSE);
            if (gProofServ) gProofServ->SetBit(TProofServ::kHighMemory);
            break;
         } else {
            if (!wmsg.IsNull()) {
               Warning("Process", "%s", wmsg.Data());
               if (gProofServ) {
                  wmsg.Insert(0, TString::Format("WARNING:%s, entry:%lld, ", gProofServ->GetOrdinal(), entry));
                  gProofServ->SendAsynMessage(wmsg.Data());
               }
            }
         }

         if (TestBit(TProofPlayer::kDispatchOneEvent)) {
            gSystem->DispatchOneEvent(kTRUE);
            ResetBit(TProofPlayer::kDispatchOneEvent);
         }
         SetProcessing(kFALSE);
         if (!fSelStatus->IsOk() || gROOT->IsInterrupted()) break;

         // Make sure that the selector abort status is reset
         if (fSelector->GetAbort() == TSelector::kAbortFile)
            fSelector->Abort("status reset", TSelector::kContinue);
      }

   } CATCH(excode) {
      if (excode == kPEX_STOPPED) {
         Info("Process","received stop-process signal");
         fExitStatus = kStopped;
      } else if (excode == kPEX_ABORTED) {
         gAbort = kTRUE;
         Info("Process","received abort-process signal");
         fExitStatus = kAborted;
      } else {
         Error("Process","exception %d caught", excode);
         // Perhaps we need a dedicated status code here ...
         gAbort = kTRUE;
         fExitStatus = kAborted;
      }
      SetProcessing(kFALSE);
   } ENDTRY;

   // Clean-up the envelop for the current element
   TPair *currentElem = 0;
   if ((currentElem = (TPair *) fInput->FindObject("PROOF_CurrentElement"))) {
      fInput->Remove(currentElem);
      delete currentElem->Key();
      delete currentElem;
   }

   // Final memory footprint
   Long64_t singleshot = 1;
   Bool_t warnHWMres = kTRUE, warnHWMvir = kTRUE;
   TString wmsg;
   Bool_t shrc = CheckMemUsage(singleshot, warnHWMres, warnHWMvir, wmsg);
   if (!wmsg.IsNull()) Warning("Process", "%s (%s)", wmsg.Data(), shrc ? "warn" : "hwm");

   PDB(kGlobal,2)
      Info("Process","%lld events processed", fProgressStatus->GetEntries());

   if (gMonitoringWriter) {
      gMonitoringWriter->SendProcessingProgress(fProgressStatus->GetEntries(),
                                                TFile::GetFileBytesRead()-readbytesatstart, kFALSE);
      gMonitoringWriter->SendProcessingStatus("DONE");
   }

   // Stop active timers
   SetDispatchTimer(kFALSE);
   if (fStopTimer != 0)
      SetStopTimer(kFALSE, gAbort);
   if (fFeedbackTimer != 0)
      HandleTimer(0);

   StopFeedback();

   SafeDelete(fEvIter);

   // Finalize

   if (fExitStatus != kAborted) {

      TIter nxo(GetOutputList());
      TObject *o = 0;
      while ((o = nxo())) {
         // Special treatment for files
         if (o->IsA() == TProofOutputFile::Class()) {
            ((TProofOutputFile *)o)->SetWorkerOrdinal(gProofServ->GetOrdinal());
            if (!strcmp(((TProofOutputFile *)o)->GetDir(),""))
               ((TProofOutputFile *)o)->SetDir(gProofServ->GetSessionDir());
         }
      }

      MapOutputListToDataMembers();

      if (fSelStatus->IsOk()) {
         if (version == 0) {
            PDB(kLoop,1) Info("Process","Call Terminate()");
            fSelector->Terminate();
         } else {
            PDB(kLoop,1) Info("Process","Call SlaveTerminate()");
            fSelector->SlaveTerminate();
            if (IsClient() && fSelStatus->IsOk()) {
               PDB(kLoop,1) Info("Process","Call Terminate()");
               fSelector->Terminate();
            }
         }
      }
      if (gProofServ && !gProofServ->IsParallel()) {  // put all the canvases onto the output list
         TIter nxc(gROOT->GetListOfCanvases());
         while (TObject *c = nxc())
            fOutput->Add(c);
      }
   }

   if (gProofServ)
      TPerfStats::Stop();

   return 0;
}

//______________________________________________________________________________
Bool_t TProofPlayer::CheckMemUsage(Long64_t &mfreq, Bool_t &w80r,
                                   Bool_t &w80v, TString &wmsg)
{
   // Check the memory usage, if requested.
   // Return kTRUE if OK, kFALSE if above 95% of at least one between virtual or
   // resident limits are depassed.

   if (mfreq > 0 && GetEventsProcessed()%mfreq == 0) {
      // Record the memory information
      ProcInfo_t pi;
      if (!gSystem->GetProcInfo(&pi)){
         wmsg = "";
         Info("CheckMemUsage|Svc", "Memory %ld virtual %ld resident event %lld",
                                   pi.fMemVirtual, pi.fMemResident, GetEventsProcessed());
         // Save info in TStatus
         fSelStatus->SetMemValues(pi.fMemVirtual, pi.fMemResident);
         // Apply limit on virtual memory, if any: warn if above 80%, stop if above 95% of max
         if (TProofServ::GetVirtMemMax() > 0) {
            if (pi.fMemVirtual > TProofServ::GetMemStop() * TProofServ::GetVirtMemMax()) {
               wmsg.Form("using more than %d%% of allowed virtual memory (%ld kB)"
                         " - STOP processing", (Int_t) (TProofServ::GetMemStop() * 100), pi.fMemVirtual);
               return kFALSE;
            } else if (pi.fMemVirtual > TProofServ::GetMemHWM() * TProofServ::GetVirtMemMax() && w80v) {
               // Refine monitoring
               mfreq = 1;
               wmsg.Form("using more than %d%% of allowed virtual memory (%ld kB)",
                         (Int_t) (TProofServ::GetMemHWM() * 100), pi.fMemVirtual);
               w80v = kFALSE;
            }
         }
         // Apply limit on resident memory, if any: warn if above 80%, stop if above 95% of max
         if (TProofServ::GetResMemMax() > 0) {
            if (pi.fMemResident > TProofServ::GetMemStop() * TProofServ::GetResMemMax()) {
               wmsg.Form("using more than %d%% of allowed resident memory (%ld kB)"
                         " - STOP processing", (Int_t) (TProofServ::GetMemStop() * 100), pi.fMemResident);
               return kFALSE;
            } else if (pi.fMemResident > TProofServ::GetMemHWM() * TProofServ::GetResMemMax() && w80r) {
               // Refine monitoring
               mfreq = 1;
               if (wmsg.Length() > 0) {
                  wmsg.Form("using more than %d%% of allowed both virtual and resident memory ({%ld,%ld} kB)",
                            (Int_t) (TProofServ::GetMemHWM() * 100), pi.fMemVirtual, pi.fMemResident);
               } else {
                  wmsg.Form("using more than %d%% of allowed resident memory (%ld kB)",
                            (Int_t) (TProofServ::GetMemHWM() * 100), pi.fMemResident);
               }
               w80r = kFALSE;
            }
         }
      }
   }
   // Done
   return kTRUE;
}

//______________________________________________________________________________
Long64_t TProofPlayer::Finalize(Bool_t, Bool_t)
{
   // Finalize query (may not be used in this class).

   MayNotUse("Finalize");
   return -1;
}

//______________________________________________________________________________
Long64_t TProofPlayer::Finalize(TQueryResult *)
{
   // Finalize query (may not be used in this class).

   MayNotUse("Finalize");
   return -1;
}
//______________________________________________________________________________
void TProofPlayer::MergeOutput()
{
   // Merge output (may not be used in this class).

   MayNotUse("MergeOutput");
   return;
}

//______________________________________________________________________________
void TProofPlayer::MapOutputListToDataMembers() const
{
   TOutputListSelectorDataMap* olsdm = new TOutputListSelectorDataMap(fSelector);
   fOutput->Add(olsdm);
}

//______________________________________________________________________________
void TProofPlayer::UpdateAutoBin(const char *name,
                                 Double_t& xmin, Double_t& xmax,
                                 Double_t& ymin, Double_t& ymax,
                                 Double_t& zmin, Double_t& zmax)
{
   // Update automatic binning parameters for given object "name".

   if ( fAutoBins == 0 ) {
      fAutoBins = new THashList;
   }

   TAutoBinVal *val = (TAutoBinVal*) fAutoBins->FindObject(name);

   if ( val == 0 ) {
      //look for info in higher master
      if (gProofServ && !gProofServ->IsTopMaster()) {
         TString key = name;
         TProofLimitsFinder::AutoBinFunc(key,xmin,xmax,ymin,ymax,zmin,zmax);
      }

      val = new TAutoBinVal(name,xmin,xmax,ymin,ymax,zmin,zmax);
      fAutoBins->Add(val);
   } else {
      val->GetAll(xmin,xmax,ymin,ymax,zmin,zmax);
   }
}

//______________________________________________________________________________
TDSetElement *TProofPlayer::GetNextPacket(TSlave *, TMessage *)
{
   // Get next packet (may not be used in this class).

   MayNotUse("GetNextPacket");
   return 0;
}

//______________________________________________________________________________
void TProofPlayer::SetupFeedback()
{
   // Set up feedback (may not be used in this class).

   MayNotUse("SetupFeedback");
}

//______________________________________________________________________________
void TProofPlayer::StopFeedback()
{
   // Stop feedback (may not be used in this class).

   MayNotUse("StopFeedback");
}

//______________________________________________________________________________
Long64_t TProofPlayer::DrawSelect(TDSet * /*set*/, const char * /*varexp*/,
                                  const char * /*selection*/, Option_t * /*option*/,
                                  Long64_t /*nentries*/, Long64_t /*firstentry*/)
{
   // Draw (may not be used in this class).

   MayNotUse("DrawSelect");
   return -1;
}

//______________________________________________________________________________
void TProofPlayer::HandleGetTreeHeader(TMessage *)
{
   // Handle tree header request.

   MayNotUse("HandleGetTreeHeader|");
}

//______________________________________________________________________________
void TProofPlayer::HandleRecvHisto(TMessage *mess)
{
   // Receive histo from slave.

   TObject *obj = mess->ReadObject(mess->GetClass());
   if (obj->InheritsFrom(TH1::Class())) {
      TH1 *h = (TH1*)obj;
      h->SetDirectory(0);
      TH1 *horg = (TH1*)gDirectory->GetList()->FindObject(h->GetName());
      if (horg)
         horg->Add(h);
      else
         h->SetDirectory(gDirectory);
   }
}

//______________________________________________________________________________
Int_t TProofPlayer::DrawCanvas(TObject *obj)
{
   // Draw the object if it is a canvas.
   // Return 0 in case of success, 1 if it is not a canvas or libProofDraw
   // is not available.

   static Int_t (*gDrawCanvasHook)(TObject *) = 0;

   // Load the library the first time
   if (!gDrawCanvasHook) {
      // Load library needed for graphics ...
      TString drawlib = "libProofDraw";
      char *p = 0;
      if ((p = gSystem->DynamicPathName(drawlib, kTRUE))) {
         delete[] p;
         if (gSystem->Load(drawlib) != -1) {
            // Locate DrawCanvas
            Func_t f = 0;
            if ((f = gSystem->DynFindSymbol(drawlib,"DrawCanvas")))
               gDrawCanvasHook = (Int_t (*)(TObject *))(f);
            else
               Warning("DrawCanvas", "can't find DrawCanvas");
         } else
            Warning("DrawCanvas", "can't load %s", drawlib.Data());
      } else
         Warning("DrawCanvas", "can't locate %s", drawlib.Data());
   }
   if (gDrawCanvasHook && obj)
      return (*gDrawCanvasHook)(obj);
   // No drawing hook or object undefined
   return 1;
}

//______________________________________________________________________________
Int_t TProofPlayer::GetDrawArgs(const char *var, const char *sel, Option_t *opt,
                                TString &selector, TString &objname)
{
   // Parse the arguments from var, sel and opt and fill the selector and
   // object name accordingly.
   // Return 0 in case of success, 1 if libProofDraw is not available.

   static Int_t (*gGetDrawArgsHook)(const char *, const char *, Option_t *,
                                    TString &, TString &) = 0;

   // Load the library the first time
   if (!gGetDrawArgsHook) {
      // Load library needed for graphics ...
      TString drawlib = "libProofDraw";
      char *p = 0;
      if ((p = gSystem->DynamicPathName(drawlib, kTRUE))) {
         delete[] p;
         if (gSystem->Load(drawlib) != -1) {
            // Locate GetDrawArgs
            Func_t f = 0;
            if ((f = gSystem->DynFindSymbol(drawlib,"GetDrawArgs")))
               gGetDrawArgsHook = (Int_t (*)(const char *, const char *, Option_t *,
                                             TString &, TString &))(f);
            else
               Warning("GetDrawArgs", "can't find GetDrawArgs");
         } else
            Warning("GetDrawArgs", "can't load %s", drawlib.Data());
      } else
         Warning("GetDrawArgs", "can't locate %s", drawlib.Data());
   }
   if (gGetDrawArgsHook)
      return (*gGetDrawArgsHook)(var, sel, opt, selector, objname);
   // No parser hook or object undefined
   return 1;
}

//______________________________________________________________________________
void TProofPlayer::FeedBackCanvas(const char *name, Bool_t create)
{
   // Create/destroy a named canvas for feedback

   static void (*gFeedBackCanvasHook)(const char *, Bool_t) = 0;

   // Load the library the first time
   if (!gFeedBackCanvasHook) {
      // Load library needed for graphics ...
      TString drawlib = "libProofDraw";
      char *p = 0;
      if ((p = gSystem->DynamicPathName(drawlib, kTRUE))) {
         delete[] p;
         if (gSystem->Load(drawlib) != -1) {
            // Locate FeedBackCanvas
            Func_t f = 0;
            if ((f = gSystem->DynFindSymbol(drawlib,"FeedBackCanvas")))
               gFeedBackCanvasHook = (void (*)(const char *, Bool_t))(f);
            else
               Warning("FeedBackCanvas", "can't find FeedBackCanvas");
         } else
            Warning("FeedBackCanvas", "can't load %s", drawlib.Data());
      } else
         Warning("FeedBackCanvas", "can't locate %s", drawlib.Data());
   }
   if (gFeedBackCanvasHook) (*gFeedBackCanvasHook)(name, create);
   // No parser hook or object undefined
   return;
}

//______________________________________________________________________________
Long64_t TProofPlayer::GetCacheSize()
{
   // Return the size in bytes of the cache

   if (fEvIter) return fEvIter->GetCacheSize();
   return -1;
}

//______________________________________________________________________________
Int_t TProofPlayer::GetLearnEntries()
{
   // Return the number of entries in the learning phase

   if (fEvIter) return fEvIter->GetLearnEntries();
   return -1;
}

//------------------------------------------------------------------------------

ClassImp(TProofPlayerLocal)


//------------------------------------------------------------------------------

ClassImp(TProofPlayerRemote)


//______________________________________________________________________________
TProofPlayerRemote::~TProofPlayerRemote()
{
   // Destructor.

   SafeDelete(fOutput);      // owns the output list
   SafeDelete(fOutputLists);

   // Objects stored in maps are already deleted when merging the feedback
   SafeDelete(fFeedbackLists);
   SafeDelete(fPacketizer);
}

//______________________________________________________________________________
Int_t TProofPlayerRemote::InitPacketizer(TDSet *dset, Long64_t nentries,
                                         Long64_t first, const char *defpackunit,
                                         const char *defpackdata)
{
   // Init the packetizer
   // Return 0 on success (fPacketizer is correctly initialized), -1 on failure.

   SafeDelete(fPacketizer);
   PDB(kGlobal,1) Info("Process","Enter");
   fDSet = dset;
   fExitStatus = kFinished;

   Bool_t noData = dset->TestBit(TDSet::kEmpty) ? kTRUE : kFALSE;

   TString packetizer;
   TList *listOfMissingFiles = 0;

   TMethodCall callEnv;
   TClass *cl;
   noData = dset->TestBit(TDSet::kEmpty) ? kTRUE : kFALSE;

   if (noData) {

      if (TProof::GetParameter(fInput, "PROOF_Packetizer", packetizer) != 0)
         packetizer = defpackunit;
      else
         Info("InitPacketizer", "using alternate packetizer: %s", packetizer.Data());

      // Get linked to the related class
      cl = TClass::GetClass(packetizer);
      if (cl == 0) {
         Error("InitPacketizer", "class '%s' not found", packetizer.Data());
         fExitStatus = kAborted;
         return -1;
      }

      // Init the constructor
      callEnv.InitWithPrototype(cl, cl->GetName(),"TList*,Long64_t,TList*,TProofProgressStatus*");
      if (!callEnv.IsValid()) {
         Error("InitPacketizer",
               "cannot find correct constructor for '%s'", cl->GetName());
         fExitStatus = kAborted;
         return -1;
      }
      callEnv.ResetParam();
      callEnv.SetParam((Long_t) fProof->GetListOfActiveSlaves());
      callEnv.SetParam((Long64_t) nentries);
      callEnv.SetParam((Long_t) fInput);
      callEnv.SetParam((Long_t) fProgressStatus);

   } else if (dset->TestBit(TDSet::kMultiDSet)) {

      // We have to process many datasets in one go, keeping them separate
      if (fProof->GetRunStatus() != TProof::kRunning) {
         // We have been asked to stop
         Error("InitPacketizer", "received stop/abort request");
         fExitStatus = kAborted;
         return -1;
      }

      // The multi packetizer
      packetizer = "TPacketizerMulti";

      // Get linked to the related class
      cl = TClass::GetClass(packetizer);
      if (cl == 0) {
         Error("InitPacketizer", "class '%s' not found", packetizer.Data());
         fExitStatus = kAborted;
         return -1;
      }

      // Init the constructor
      callEnv.InitWithPrototype(cl, cl->GetName(),"TDSet*,TList*,Long64_t,Long64_t,TList*,TProofProgressStatus*");
      if (!callEnv.IsValid()) {
         Error("InitPacketizer", "cannot find correct constructor for '%s'", cl->GetName());
         fExitStatus = kAborted;
         return -1;
      }
      callEnv.ResetParam();
      callEnv.SetParam((Long_t) dset);
      callEnv.SetParam((Long_t) fProof->GetListOfActiveSlaves());
      callEnv.SetParam((Long64_t) first);
      callEnv.SetParam((Long64_t) nentries);
      callEnv.SetParam((Long_t) fInput);
      callEnv.SetParam((Long_t) fProgressStatus);

      // We are going to test validity during the packetizer initialization
      dset->SetBit(TDSet::kValidityChecked);
      dset->ResetBit(TDSet::kSomeInvalid);

   } else {

      // Lookup - resolve the end-point urls to optmize the distribution.
      // The lookup was previously called in the packetizer's constructor.
      // A list for the missing files may already have been added to the
      // output list; otherwise, if needed it will be created inside
      if ((listOfMissingFiles = (TList *)fInput->FindObject("MissingFiles"))) {
         // Move it to the output list
         fInput->Remove(listOfMissingFiles);
      } else {
         listOfMissingFiles = new TList;
      }
      // Do the lookup; we only skip it if explicitely requested so.
      TString lkopt;
      if (TProof::GetParameter(fInput, "PROOF_LookupOpt", lkopt) != 0 || lkopt != "none")
         dset->Lookup(kTRUE, &listOfMissingFiles);

      if (fProof->GetRunStatus() != TProof::kRunning) {
         // We have been asked to stop
         Error("InitPacketizer", "received stop/abort request");
         fExitStatus = kAborted;
         return -1;
      }

      if (!(dset->GetListOfElements()) ||
          !(dset->GetListOfElements()->GetSize())) {
         if (gProofServ)
            gProofServ->SendAsynMessage("InitPacketizer: No files from the data set were found - Aborting");
         Error("InitPacketizer", "No files from the data set were found - Aborting");
         fExitStatus = kAborted;
         if (listOfMissingFiles) {
            listOfMissingFiles->SetOwner();
            fOutput->Remove(listOfMissingFiles);
            SafeDelete(listOfMissingFiles);
         }
         return -1;
      }

      if (TProof::GetParameter(fInput, "PROOF_Packetizer", packetizer) != 0)
         // Using standard packetizer TAdaptivePacketizer
         packetizer = defpackdata;
      else
         Info("InitPacketizer", "using alternate packetizer: %s", packetizer.Data());

      // Get linked to the related class
      cl = TClass::GetClass(packetizer);
      if (cl == 0) {
         Error("InitPacketizer", "class '%s' not found", packetizer.Data());
         fExitStatus = kAborted;
         return -1;
      }

      // Init the constructor
      callEnv.InitWithPrototype(cl, cl->GetName(),"TDSet*,TList*,Long64_t,Long64_t,TList*,TProofProgressStatus*");
      if (!callEnv.IsValid()) {
         Error("InitPacketizer", "cannot find correct constructor for '%s'", cl->GetName());
         fExitStatus = kAborted;
         return -1;
      }
      callEnv.ResetParam();
      callEnv.SetParam((Long_t) dset);
      callEnv.SetParam((Long_t) fProof->GetListOfActiveSlaves());
      callEnv.SetParam((Long64_t) first);
      callEnv.SetParam((Long64_t) nentries);
      callEnv.SetParam((Long_t) fInput);
      callEnv.SetParam((Long_t) fProgressStatus);

      // We are going to test validity during the packetizer initialization
      dset->SetBit(TDSet::kValidityChecked);
      dset->ResetBit(TDSet::kSomeInvalid);
   }

   // Get an instance of the packetizer
   Long_t ret = 0;
   callEnv.Execute(ret);
   if ((fPacketizer = (TVirtualPacketizer *)ret) == 0) {
      Error("InitPacketizer", "cannot construct '%s'", cl->GetName());
      fExitStatus = kAborted;
      return -1;
   }

   if (!fPacketizer->IsValid()) {
      Error("InitPacketizer",
            "instantiated packetizer object '%s' is invalid", cl->GetName());
      fExitStatus = kAborted;
      SafeDelete(fPacketizer);
      return -1;
   }

   // In multi mode retrieve the list of missing files
   if (!noData && dset->TestBit(TDSet::kMultiDSet)) {
      if ((listOfMissingFiles = (TList *) fInput->FindObject("MissingFiles"))) {
         // Remove it; it will be added to the output list
         fInput->Remove(listOfMissingFiles);
      }
   }

   if (!noData) {
      // Add invalid elements to the list of missing elements
      TDSetElement *elem = 0;
      if (dset->TestBit(TDSet::kSomeInvalid)) {
         TIter nxe(dset->GetListOfElements());
         while ((elem = (TDSetElement *)nxe())) {
            if (!elem->GetValid()) {
               if (!listOfMissingFiles)
                  listOfMissingFiles = new TList;
               listOfMissingFiles->Add(elem->GetFileInfo(dset->GetType()));
               dset->Remove(elem, kFALSE);
            }
         }
         // The invalid elements have been removed
         dset->ResetBit(TDSet::kSomeInvalid);
      }

      // Record the list of missing or invalid elements in the output list
      if (listOfMissingFiles && listOfMissingFiles->GetSize() > 0) {
         TIter missingFiles(listOfMissingFiles);
         TString msg;
         if (gDebug > 0) {
            TFileInfo *fi = 0;
            while ((fi = (TFileInfo *) missingFiles.Next())) {
               if (fi->GetCurrentUrl()) {
                  msg = Form("File not found: %s - skipping!",
                                                fi->GetCurrentUrl()->GetUrl());
               } else {
                  msg = Form("File not found: %s - skipping!", fi->GetName());
               }
               if (gProofServ) gProofServ->SendAsynMessage(msg.Data());
            }
         }
         // Make sure it will be sent back
         if (!GetOutput("MissingFiles")) {
            listOfMissingFiles->SetName("MissingFiles");
            AddOutputObject(listOfMissingFiles);
         }
         TStatus *tmpStatus = (TStatus *)GetOutput("PROOF_Status");
         if (!tmpStatus) {
            tmpStatus = new TStatus();
            AddOutputObject(tmpStatus);
         }
         // Estimate how much data are missing
         Int_t ngood = dset->GetListOfElements()->GetSize();
         Int_t nbad = listOfMissingFiles->GetSize();
         Double_t xb = Double_t(nbad) / Double_t(ngood + nbad);
         msg = Form(" About %.2f %c of the requested files (%d out of %d) were missing or unusable; details in"
                    " the 'missingFiles' list", xb * 100., '%', nbad, nbad + ngood);
         tmpStatus->Add(msg.Data());
         msg = Form(" +++\n"
                    " +++ About %.2f %c of the requested files (%d out of %d) are missing or unusable; details in"
                    " the 'MissingFiles' list\n"
                    " +++", xb * 100., '%', nbad, nbad + ngood);
         if (gProofServ) gProofServ->SendAsynMessage(msg.Data());
      } else {
         // Cleanup
         SafeDelete(listOfMissingFiles);
      }
   }

   // Done
   return 0;
}

//______________________________________________________________________________
Long64_t TProofPlayerRemote::Process(TDSet *dset, const char *selector_file,
                                     Option_t *option, Long64_t nentries,
                                     Long64_t first)
{
   // Process specified TDSet on PROOF.
   // This method is called on client and on the PROOF master.
   // The return value is -1 in case of an error and TSelector::GetStatus() in
   // in case of success.

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
      fOutput = new TList;
   else
      fOutput->Clear();

   SafeDelete(fFeedbackLists);

   if (fProof->IsMaster()){
      TPerfStats::Start(fInput, fOutput);
   } else {
      TPerfStats::Setup(fInput);
   }

   if(!SendSelector(selector_file)) return -1;

   TMessage mesg(kPROOF_PROCESS);
   TString fn(gSystem->BaseName(selector_file));

   // Parse option
   Bool_t sync = (fProof->GetQueryMode(option) == TProof::kSync);

   TDSet *set = dset;
   if (fProof->IsMaster()) {

      PDB(kPacketizer,1) Info("Process","Create Proxy TDSet");
      set = new TDSetProxy( dset->GetType(), dset->GetObjName(),
                            dset->GetDirectory() );
      if (dset->TestBit(TDSet::kEmpty))
         set->SetBit(TDSet::kEmpty);

      const char *datapack = (fProof->IsLite()) ? "TPacketizer" : "TPacketizerAdaptive";
      if (InitPacketizer(dset, nentries, first, "TPacketizerUnit", datapack) != 0) {
         Error("Process", "cannot init the packetizer");
         fExitStatus = kAborted;
         return -1;
      }

      // Reset start, this is now managed by the packetizer
      first = 0;
      // Try to have 100 messages about memory, unless a different number is given by the user
      if (!fProof->GetParameter("PROOF_MemLogFreq")){
         Long64_t memlogfreq = fPacketizer->GetTotalEntries()/(fProof->GetParallel()*100);
         memlogfreq = (memlogfreq > 0) ? memlogfreq : 1;
         fProof->SetParameter("PROOF_MemLogFreq", memlogfreq);
      }

      // Send input data, if any
      TString emsg;
      if (TProof::SendInputData(fQuery, fProof, emsg) != 0)
         Warning("Process", "could not forward input data: %s", emsg.Data());

   } else {

      // Check whether we have to enforce the use of submergers
      if (gEnv->Lookup("Proof.UseMergers") && !fInput->FindObject("PROOF_UseMergers")) {
         Int_t smg = gEnv->GetValue("Proof.UseMergers",-1);
         if (smg >= 0) fInput->Add(new TParameter<Int_t>("PROOF_UseMergers", smg));
      }

      // For a new query clients should make sure that the temporary
      // output list is empty
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

      SafeDelete(fSelector);
      fSelectorClass = 0;
      if (!(fSelector = TSelector::GetSelector(selector_file))) {
         if (!sync)
            gSystem->RedirectOutput(0);
         return -1;
      }
      fSelectorClass = fSelector->IsA();
      fSelector->SetInputList(fInput);
      fSelector->SetOption(option);

      PDB(kLoop,1) Info("Process","Call Begin(0)");
      fSelector->Begin(0);

      // Send large input data objects, if any
      fProof->SendInputDataFile();

      if (!sync)
         gSystem->RedirectOutput(0);
   }

   TCleanup clean(this);
   SetupFeedback();

   TString opt = option;

   // Old servers need a dedicated streamer
   if (fProof->fProtocol < 13)
      dset->SetWriteV3(kTRUE);

   // Workers will get the entry ranges from the packetizer
   Long64_t num = (gProofServ && gProofServ->IsMaster() && gProofServ->IsParallel()) ? -1 : nentries;
   Long64_t fst = (gProofServ && gProofServ->IsMaster() && gProofServ->IsParallel()) ? -1 : first;

   // Entry- or Event- list ?
   TEntryList *enl = (!fProof->IsMaster()) ? dynamic_cast<TEntryList *>(set->GetEntryList())
                                           : (TEntryList *)0;
   TEventList *evl = (!fProof->IsMaster() && !enl) ? dynamic_cast<TEventList *>(set->GetEntryList())
                                           : (TEventList *)0;
   if (fProof->fProtocol > 14) {
      mesg << set << fn << fInput << opt << num << fst << evl << sync << enl;
   } else {
      mesg << set << fn << fInput << opt << num << fst << evl << sync;
      if (enl)
         // Not supported remotely
         Warning("Process","entry lists not supported by the server");
   }

   // Reset the merging progress information
   fProof->ResetMergePrg();

   PDB(kGlobal,1) Info("Process","Calling Broadcast");
   fProof->Broadcast(mesg);

   // Reset streamer choice
   if (fProof->fProtocol < 13)
      dset->SetWriteV3(kFALSE);

   // Redirect logs from master to special log frame
   if (IsClient())
      fProof->fRedirLog = kTRUE;

   if (!IsClient()){
      // Signal the start of finalize for the memory log grepping
      Info("Process|Svc", "Start merging Memory information");
   }

   if (!sync) {
      if (IsClient()) {
         // Asynchronous query: just make sure that asynchronous input
         // is enabled and return the prompt
         PDB(kGlobal,1) Info("Process","Asynchronous processing:"
                                       " activating CollectInputFrom");
         fProof->Activate();

         // Receive the acknowledgement and query sequential number
         fProof->Collect();

         return fProof->fSeqNum;

      } else {
         PDB(kGlobal,1) Info("Process","Calling Collect");
         fProof->Collect();

         HandleTimer(0); // force an update of final result
         StopFeedback();

         return Finalize(kFALSE,sync);
      }
   } else {

      PDB(kGlobal,1) Info("Process","Synchronous processing: calling Collect");
      fProof->Collect();
      if (!(fProof->IsSync())) {
         // The server required to switch to asynchronous mode
         Info("Process", "switching to the asynchronous mode ...");
         return fProof->fSeqNum;
      }

      // Restore prompt logging, for clients (Collect leaves things as they were
      // at the time it was called)
      if (IsClient())
         fProof->fRedirLog = kFALSE;

      if (!IsClient()) {
         // Force an update of final result
         HandleTimer(0);
         // This forces a last call to TPacketizer::HandleTimer via the second argument
         // (the first is ignored). This is needed when some events were skipped so that
         // the total number of entries is not the one requested. The packetizer has no
         // way in such a case to understand that processing is finished: it must be told.
         if (fPacketizer) fPacketizer->StopProcess(kFALSE, kTRUE);
         // Store process info
         if (fPacketizer && fQuery)
            fQuery->SetProcessInfo(0, 0., fPacketizer->GetBytesRead(),
                                          fPacketizer->GetInitTime(),
                                          fPacketizer->GetProcTime());
      }
      StopFeedback();

      if (!IsClient() || GetExitStatus() != TProofPlayer::kAborted)
         return Finalize(kFALSE,sync);
      else
         return -1;
   }
}

//______________________________________________________________________________
Bool_t TProofPlayerRemote::MergeOutputFiles()
{
   // Merge output in files

   TList *rmList = 0;
   if (fMergeFiles) {
      TIter nxo(fOutput);
      TObject *o = 0;
      TProofOutputFile *pf = 0;
      while ((o = nxo())) {
         if ((pf = dynamic_cast<TProofOutputFile*>(o))) {

            if (pf->IsMerge()) {

               // Point to the merger
               TFileMerger *filemerger = pf->GetFileMerger();
               if (!filemerger) {
                  Error("MergeOutputFiles", "file merger is null in TProofOutputFile! Protocol error?");
                  pf->Print();
                  continue;
               }
               // Set the output file
               if (!filemerger->OutputFile(pf->GetOutputFileName())) {
                  Error("MergeOutputFiles", "cannot open the output file");
                  continue;
               }
               // If only one instance the list in the merger is not yet created: do it now
               if (!pf->IsMerged()) {
                  TString fileLoc = TString::Format("%s/%s", pf->GetDir(), pf->GetFileName());
                  filemerger->AddFile(fileLoc);
               }
               // Merge
               if (!filemerger->Merge()) {
                  Error("MergeOutputFiles", "cannot merge the output files");
                  continue;
               }
               // Remove the files
               TList *fileList = filemerger->GetMergeList();
               if (fileList) {
                  TIter next(fileList);
                  TObjString *url = 0;
                  while((url = (TObjString*)next())) {
                     gSystem->Unlink(url->GetString());
                  }
               }
               // Reset the merger
               filemerger->Reset();

            } else {

               // Point to the dataset
               TFileCollection *fc = pf->GetFileCollection();
               if (!fc) {
                  Error("MergeOutputFiles", "file collection is null in TProofOutputFile! Protocol error?");
                  pf->Print();
                  continue;
               }
               // Add the collection to the output list for registration and/or to be returned
               // to the client
               fOutput->Add(fc);
               // Do not cleanup at destruction
               pf->ResetFileCollection();
               // Tell the main thread to register this dataset, if needed
               if (pf->IsRegister()) {
                  TString opt;
                  if ((pf->GetTypeOpt() & TProofOutputFile::kOverwrite)) opt += "O";
                  if ((pf->GetTypeOpt() & TProofOutputFile::kVerify)) opt += "V";
                  if (!fOutput->FindObject("PROOFSERV_RegisterDataSet"))
                     fOutput->Add(new TNamed("PROOFSERV_RegisterDataSet", ""));
                  TString tag = TString::Format("DATASET_%s", pf->GetTitle());
                  fOutput->Add(new TNamed(tag, opt));
               }
               // Remove this object from the output list and schedule it for distruction
               fOutput->Remove(pf);
               if (!rmList) rmList = new TList;
               rmList->Add(pf);
            }
         }
      }
   }

   // Remove objects scheduled for removal
   if (rmList && rmList->GetSize() > 0) {
      TIter nxo(rmList);
      TObject *o = 0;
      while((o = nxo())) {
         fOutput->Remove(o);
      }
      rmList->SetOwner(kTRUE);
      delete rmList;
   }

   // Done
   return kTRUE;
}


//______________________________________________________________________________
void TProofPlayerRemote::SetSelectorDataMembersFromOutputList()
{
   // Set the selector's data members:
   // find the mapping of data members to otuput list entries in the output list
   // and apply it.
   TOutputListSelectorDataMap* olsdm
      = TOutputListSelectorDataMap::FindInList(fOutput);
   if (!olsdm) {
      PDB(kOutput,1) Warning("SetSelectorDataMembersFromOutputList","Failed to find map object in output list!");
      return;
   }

   olsdm->SetDataMembers(fSelector);
}

//______________________________________________________________________________
Long64_t TProofPlayerRemote::Finalize(Bool_t force, Bool_t sync)
{

   // Finalize a query.
   // Returns -1 in case of an error, 0 otherwise.

   if (IsClient()) {
      if (fOutputLists == 0) {
         if (force)
            if (fQuery)
               return fProof->Finalize(Form("%s:%s", fQuery->GetTitle(),
                                                     fQuery->GetName()), force);
      } else {
         // Make sure the all objects are in the output list
         PDB(kGlobal,1) Info("Finalize","Calling Merge Output to finalize the output list");
         MergeOutput();
      }
   }

   Long64_t rv = 0;
   if (fProof->IsMaster()) {
      TPerfStats::Stop();

      PDB(kOutput,1) Info("Finalize","Calling Merge Output");
      // Some objects (e.g. histos in autobin) may not have been merged yet
      // do it now
      MergeOutput();

      // Merge the output files created on workers, if any
      MergeOutputFiles();

      fOutput->SetOwner();

      // Add the active-wrks-vs-proctime info from the packetizer
      if (fPacketizer) {
         TObject *pperf = (TObject *) fPacketizer->GetProgressPerf(kTRUE);
         if (pperf) fOutput->Add(pperf);
         TList *parms = fPacketizer->GetConfigParams(kTRUE);
         if (parms) {
            TIter nxo(parms);
            TObject *o = 0;
            while ((o = nxo())) fOutput->Add(o);
         }
         
         // If other invalid elements were found during processing, add them to the
         // list of missing elements
         TDSetElement *elem = 0;
         if (fPacketizer->GetFailedPackets()) {
            TString type = (fPacketizer->TestBit(TVirtualPacketizer::kIsTree)) ? "TTree" : "";
            TList *listOfMissingFiles = (TList *) fOutput->FindObject("MissingFiles");
            if (!listOfMissingFiles) {
               listOfMissingFiles = new TList;
               listOfMissingFiles->SetName("MissingFiles");
            }
            TIter nxe(fPacketizer->GetFailedPackets());
            while ((elem = (TDSetElement *)nxe()))
               listOfMissingFiles->Add(elem->GetFileInfo(type));
            if (!fOutput->FindObject(listOfMissingFiles)) fOutput->Add(listOfMissingFiles);
         }
      }

      SafeDelete(fSelector);
   } else {
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

         if (fPacketizer)
            if (TList *failedPackets = fPacketizer->GetFailedPackets()) {
               fPacketizer->SetFailedPackets(0);
               failedPackets->SetName("FailedPackets");
               AddOutputObject(failedPackets);

               TStatus *status = (TStatus *)GetOutput("PROOF_Status");
               if (!status) {
                  status = new TStatus();
                  AddOutputObject(status);
               }
               status->Add("Some packets were not processed! Check the the"
                           " 'FailedPackets' list in the output list");
            }

         // Some input parameters may be needed in Terminate
         fSelector->SetInputList(fInput);

         TIter next(fOutput);
         TList *output = fSelector->GetOutputList();
         while(TObject* obj = next()) {
            if (fProof->IsParallel() || DrawCanvas(obj) == 1)
               // Either parallel or not a canvas or not able to display it:
               // just add to the list
               output->Add(obj);
         }

         SetSelectorDataMembersFromOutputList();

         PDB(kLoop,1) Info("Finalize","Call Terminate()");
         fOutput->Clear("nodelete");
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

         // We have transferred copy of the output objects in TQueryResult,
         // so now we can cleanup the selector, making sure that we do not
         // touch the output objects
         output->SetOwner(kFALSE);
         SafeDelete(fSelector);

         // Delete fOutput (not needed anymore, cannot be finalized twice),
         // making sure that the objects saved in TQueryResult are not deleted
         fOutput->SetOwner(kFALSE);
         SafeDelete(fOutput);
      }
   }
   PDB(kGlobal,1) Info("Process","exit");

   if (!IsClient()) {
      Info("Finalize", "finalization on %s finished", gProofServ->GetPrefix());
   }
   fProof->FinalizationDone();

   return rv;
}

//______________________________________________________________________________
Long64_t TProofPlayerRemote::Finalize(TQueryResult *qr)
{
   // Finalize the results of a query already processed.

   PDB(kGlobal,1) Info("Finalize(TQueryResult *)","Enter");

   if (!IsClient()) {
      Info("Finalize(TQueryResult *)",
           "method to be executed only on the clients");
      return -1;
   }

   if (!qr) {
      Info("Finalize(TQueryResult *)", "query undefined");
      return -1;
   }

   if (qr->IsFinalized()) {
      Info("Finalize(TQueryResult *)", "query already finalized");
      return -1;
   }

   // Reset the list
   if (!fOutput)
      fOutput = new TList;
   else
      fOutput->Clear();

   // Make sure that the temporary output list is empty
   if (fOutputLists) {
      fOutputLists->Delete();
      delete fOutputLists;
      fOutputLists = 0;
   }

   // Re-init the selector
   gSystem->RedirectOutput(fProof->fLogFileName);

   // Import the output list
   TList *tmp = (TList *) qr->GetOutputList();
   if (!tmp) {
      gSystem->RedirectOutput(0);
      Info("Finalize(TQueryResult *)", "ouputlist is empty");
      return -1;
   }
   TList *out = fOutput;
   if (fProof->fProtocol < 11)
      out = new TList;
   TIter nxo(tmp);
   TObject *o = 0;
   while ((o = nxo()))
      out->Add(o->Clone());

   // Adopts the list
   if (fProof->fProtocol < 11) {
      out->SetOwner();
      StoreOutput(out);
   }
   gSystem->RedirectOutput(0);

   SetSelectorDataMembersFromOutputList();

   // Finalize it
   SetCurrentQuery(qr);
   Long64_t rc = Finalize();
   RestorePreviousQuery();

   return rc;
}

//______________________________________________________________________________
Bool_t TProofPlayerRemote::SendSelector(const char* selector_file)
{
   // Send the selector file(s) to master or worker nodes.

   // Check input
   if (!selector_file) {
      Info("SendSelector", "Invalid input: selector (file) name undefined");
      return kFALSE;
   }

   if (!strchr(gSystem->BaseName(selector_file), '.')) {
      if (gDebug > 1)
         Info("SendSelector", "selector name '%s' does not contain a '.':"
              " nothing to send, it will be loaded from a library", selector_file);
      return kTRUE;
   }

   // Extract the fine name first
   TString selec = selector_file;
   TString aclicMode;
   TString arguments;
   TString io;
   selec = gSystem->SplitAclicMode(selec, aclicMode, arguments, io);

   // Expand possible envs or '~'
   gSystem->ExpandPathName(selec);

   // Update the macro path
   TString mp(TROOT::GetMacroPath());
   TString np(gSystem->DirName(selec));
   if (!np.IsNull()) {
      np += ":";
      if (!mp.BeginsWith(np) && !mp.Contains(":"+np)) {
         Int_t ip = (mp.BeginsWith(".:")) ? 2 : 0;
         mp.Insert(ip, np);
         TROOT::SetMacroPath(mp);
         if (gDebug > 0)
            Info("SendSelector", "macro path set to '%s'", TROOT::GetMacroPath());
      }
   }

   // Header file
   TString header = selec;
   header.Remove(header.Last('.'));
   header += ".h";
   if (gSystem->AccessPathName(header, kReadPermission)) {
      TString h = header;
      header.Remove(header.Last('.'));
      header += ".hh";
      if (gSystem->AccessPathName(header, kReadPermission)) {
         Info("SendSelector",
              "header file not found: tried: %s %s", h.Data(), header.Data());
         return kFALSE;
      }
   }

   // Send files now
   if (fProof->SendFile(selec, (TProof::kBinary | TProof::kForward | TProof::kCp | TProof::kCpBin)) == -1) {
      Info("SendSelector", "problems sending implementation file %s", selec.Data());
      return kFALSE;
   }
   if (fProof->SendFile(header, (TProof::kBinary | TProof::kForward | TProof::kCp)) == -1) {
      Info("SendSelector", "problems sending header file %s", header.Data());
      return kFALSE;
   }

   return kTRUE;
}

//______________________________________________________________________________
void TProofPlayerRemote::MergeOutput()
{
   // Merge objects in output the lists.

   PDB(kOutput,1) Info("MergeOutput","Enter");

   if (fOutputLists == 0) {
      PDB(kOutput,1) Info("MergeOutput","Leave (no output)");
      return;
   }

   TIter next(fOutputLists);

   TList *list;
   while ( (list = (TList *) next()) ) {

      TObject *obj = fOutput->FindObject(list->GetName());

      if (obj == 0) {
         obj = list->First();
         list->Remove(obj);
         fOutput->Add(obj);
      }

      if ( list->IsEmpty() ) continue;

      TMethodCall callEnv;
      if (obj->IsA())
         callEnv.InitWithPrototype(obj->IsA(), "Merge", "TCollection*");
      if (callEnv.IsValid()) {
         callEnv.SetParam((Long_t) list);
         callEnv.Execute(obj);
      } else {
         // No Merge interface, return individual objects
         while ( (obj = list->First()) ) {
            fOutput->Add(obj);
            list->Remove(obj);
         }
      }
   }

   SafeDelete(fOutputLists);

   PDB(kOutput,1) Info("MergeOutput","Leave (%d object(s))", fOutput->GetSize());
}

//______________________________________________________________________________
void TProofPlayerRemote::Progress(Long64_t total, Long64_t processed)
{
   // Progress signal.

   if (IsClient()) {
      fProof->Progress(total, processed);
   } else {
      // Send to the previous tier
      TMessage m(kPROOF_PROGRESS);
      m << total << processed;
      gProofServ->GetSocket()->Send(m);
   }
}

//______________________________________________________________________________
void TProofPlayerRemote::Progress(Long64_t total, Long64_t processed,
                                  Long64_t bytesread,
                                  Float_t initTime, Float_t procTime,
                                  Float_t evtrti, Float_t mbrti)
{
   // Progress signal.

   PDB(kGlobal,1)
      Info("Progress","%lld %lld %lld %f %f %f %f", total, processed, bytesread,
                                             initTime, procTime, evtrti, mbrti);

   if (IsClient()) {
      fProof->Progress(total, processed, bytesread, initTime, procTime, evtrti, mbrti);
   } else {
      // Send to the previous tier
      TMessage m(kPROOF_PROGRESS);
      m << total << processed << bytesread << initTime << procTime << evtrti << mbrti;
      gProofServ->GetSocket()->Send(m);
   }
}

//______________________________________________________________________________
void TProofPlayerRemote::Progress(TProofProgressInfo *pi)
{
   // Progress signal.

   if (pi) {
      PDB(kGlobal,1)
         Info("Progress","%lld %lld %lld %f %f %f %f %d %f", pi->fTotal, pi->fProcessed, pi->fBytesRead,
                           pi->fInitTime, pi->fProcTime, pi->fEvtRateI, pi->fMBRateI,
                           pi->fActWorkers, pi->fEffSessions);

      if (IsClient()) {
         fProof->Progress(pi->fTotal, pi->fProcessed, pi->fBytesRead,
                           pi->fInitTime, pi->fProcTime,
                           pi->fEvtRateI, pi->fMBRateI,
                           pi->fActWorkers, pi->fTotSessions, pi->fEffSessions);
      } else {
         // Send to the previous tier
         TMessage m(kPROOF_PROGRESS);
         m << pi;
         gProofServ->GetSocket()->Send(m);
      }
   } else {
      Warning("Progress","TProofProgressInfo object undefined!");
   }
}


//______________________________________________________________________________
void TProofPlayerRemote::Feedback(TList *objs)
{
   // Feedback signal.

   fProof->Feedback(objs);
}

//______________________________________________________________________________
void TProofPlayerRemote::StopProcess(Bool_t abort, Int_t)
{
   // Stop process after this event.

   if (fPacketizer != 0)
      fPacketizer->StopProcess(abort, kTRUE);
   if (abort == kTRUE)
      fExitStatus = kAborted;
   else
      fExitStatus = kStopped;
}

//______________________________________________________________________________
Int_t TProofPlayerRemote::AddOutputObject(TObject *obj)
{
   // Incorporate the received object 'obj' into the output list fOutput.
   // The latter is created if not existing.
   // This method short cuts 'StoreOutput + MergeOutput' optimizing the memory
   // consumption.
   // Returns -1 in case of error, 1 if the object has been merged into another
   // one (so that its ownership has not been taken and can be deleted), and 0
   // otherwise.

   PDB(kOutput,1)
      Info("AddOutputObject","Enter: %p (%s)", obj, obj ? obj->ClassName() : "undef");

   // We must something to process
   if (!obj) {
      PDB(kOutput,1) Info("AddOutputObject","Invalid input (obj == 0x0)");
      return -1;
   }

   // Create the output list, if not yet done
   if (!fOutput)
      fOutput = new TList;

   // Flag about merging
   Bool_t merged = kTRUE;

   // Process event lists first
   TList *elists = dynamic_cast<TList *> (obj);
   if (elists && !strcmp(elists->GetName(), "PROOF_EventListsList")) {

      // Create a global event list, result of merging the event lists
      // coresponding to the various data set elements
      TEventList *evlist = new TEventList("PROOF_EventList");

      // Iterate the list of event list segments
      TIter nxevl(elists);
      TEventList *evl = 0;
      while ((evl = dynamic_cast<TEventList *> (nxevl()))) {

         // Find the file offset (fDSet is the current TDSet instance)
         // locating the element by name
         TIter nxelem(fDSet->GetListOfElements());
         TDSetElement *elem = 0;
         while ((elem = dynamic_cast<TDSetElement *> (nxelem()))) {
            if (!strcmp(elem->GetFileName(), evl->GetName()))
               break;
         }
         if (!elem) {
            Error("AddOutputObject", "Found an event list for %s, but no object with"
                                     " the same name in the TDSet", evl->GetName());
            continue;
         }
         Long64_t offset = elem->GetTDSetOffset();

         // Shift the list by the number of first event in that file
         Long64_t *arr = evl->GetList();
         Int_t num = evl->GetN();
         if (arr && offset > 0)
            for (Int_t i = 0; i < num; i++)
               arr[i] += offset;

         // Add to the global event list
         evlist->Add(evl);
      }

      // Incorporate the resulting global list in fOutput
      SetLastMergingMsg(evlist);
      Incorporate(evlist, fOutput, merged);
      NotifyMemory(evlist);

      // Delete the global list if merged
      if (merged)
         SafeDelete(evlist);

      // The original object has been transformed in something else; we do
      // not have ownership on it
      return 1;
   }

   // Check if we need to merge files
   TProofOutputFile *pf = dynamic_cast<TProofOutputFile*>(obj);
   if (pf) {
      fMergeFiles = kTRUE;
      if (!IsClient()) {
         if (pf->IsMerge()) {
            // Fill the output file name, if not done by the client
            if (strlen(pf->GetOutputFileName()) <= 0 ||
                !pf->TestBit(TProofOutputFile::kOutputFileNameSet)) {
               TString of;
               if (gSystem->Getenv("LOCALDATASERVER")) {
                  of = gSystem->Getenv("LOCALDATASERVER");
               } else {
                  // Assume an xroot server running on the machine
                  of.Form("root://%s", gSystem->HostName());
                  if (gSystem->Getenv("XRDPORT")) {
                     TString sp(gSystem->Getenv("XRDPORT"));
                     if (sp.IsDigit())
                        of += Form(":%s", sp.Data());
                  }
               }
               TString sessionPath(gProofServ->GetSessionDir());
               // Take into account a prefix, if included and if xrootd
               TString sproto = TUrl(sessionPath).GetProtocol();
               TString pfx  = gEnv->GetValue("Path.Localroot","");
               if (!pfx.IsNull() && sessionPath.BeginsWith(pfx) &&
                  (sproto == "root" || sproto == "xrd"))
                  sessionPath.Remove(0, pfx.Length());
               of += TString::Format("/%s/%s", sessionPath.Data(), pf->GetFileName());
               pf->SetOutputFileName(of);
            }
            // Notify
            pf->Print();
         }
      } else {
         // On clients notify the output path
         Printf("Output file: %s", pf->GetOutputFileName());
      }
   }

   // For other objects we just run the incorporation procedure
   SetLastMergingMsg(obj);
   Incorporate(obj, fOutput, merged);
   NotifyMemory(obj);

   // We are done
   return (merged ? 1 : 0);
}

//______________________________________________________________________________
void TProofPlayerRemote::RedirectOutput(Bool_t on)
{
   // Control output redirection to TProof::fLogFileW

   if (on && fProof && fProof->fLogFileW) {
      TProofServ::SetErrorHandlerFile(fProof->fLogFileW);
      fErrorHandler = SetErrorHandler(TProofServ::ErrorHandler);
   } else if (!on) {
      if (fErrorHandler) {
         TProofServ::SetErrorHandlerFile(0);
         SetErrorHandler(fErrorHandler);
      }
   }
}

//______________________________________________________________________________
void TProofPlayerRemote::AddOutput(TList *out)
{
   // Incorporate the content of the received output list 'out' into the final
   // output list fOutput. The latter is created if not existing.
   // This method short cuts 'StoreOutput + MergeOutput' limiting the memory
   // consumption.

   PDB(kOutput,1) Info("AddOutput","Enter");

   // We must something to process
   if (!out) {
      PDB(kOutput,1) Info("AddOutput","Invalid input (out == 0x0)");
      return;
   }

   // Create the output list, if not yet done
   if (!fOutput)
      fOutput = new TList;

   // Process event lists first
   Bool_t merged = kTRUE;
   TList *elists = dynamic_cast<TList *> (out->FindObject("PROOF_EventListsList"));
   if (elists) {

      // Create a global event list, result of merging the event lists
      // corresponding to the various data set elements
      TEventList *evlist = new TEventList("PROOF_EventList");

      // Iterate the list of event list segments
      TIter nxevl(elists);
      TEventList *evl = 0;
      while ((evl = dynamic_cast<TEventList *> (nxevl()))) {

         // Find the file offset (fDSet is the current TDSet instance)
         // locating the element by name
         TIter nxelem(fDSet->GetListOfElements());
         TDSetElement *elem = 0;
         while ((elem = dynamic_cast<TDSetElement *> (nxelem()))) {
            if (!strcmp(elem->GetFileName(), evl->GetName()))
               break;
         }
         if (!elem) {
            Error("AddOutput", "Found an event list for %s, but no object with"
                               " the same name in the TDSet", evl->GetName());
            continue;
         }
         Long64_t offset = elem->GetTDSetOffset();

         // Shift the list by the number of first event in that file
         Long64_t *arr = evl->GetList();
         Int_t num = evl->GetN();
         if (arr && offset > 0)
            for (Int_t i = 0; i < num; i++)
               arr[i] += offset;

         // Add to the global event list
         evlist->Add(evl);
      }

      // Remove and delete the events lists object to avoid spoiling iteration
      // during next steps
      out->Remove(elists);
      delete elists;

      // Incorporate the resulting global list in fOutput
      SetLastMergingMsg(evlist);
      Incorporate(evlist, fOutput, merged);
      NotifyMemory(evlist);
   }

   // Iterate on the remaining objects in the received list
   TIter nxo(out);
   TObject *obj = 0;
   while ((obj = nxo())) {
      SetLastMergingMsg(obj);
      Incorporate(obj, fOutput, merged);
      // If not merged, drop from the temporary list, as the ownership
      // passes to fOutput
      if (!merged)
         out->Remove(obj);
      NotifyMemory(obj);
   }

   // Done
   return;
}

//______________________________________________________________________________
void TProofPlayerRemote::NotifyMemory(TObject *obj)
{
   // Printout the memory record after merging object 'obj'
   // This record is used by the memory monitor

   if (fProof && (!IsClient() || fProof->IsLite())){
      ProcInfo_t pi;
      if (!gSystem->GetProcInfo(&pi)){
         // For PROOF-Lite we redirect this output to a the open log file so that the
         // memory monitor can pick these messages up
         RedirectOutput(fProof->IsLite());
         Info("NotifyMemory|Svc", "Memory %ld virtual %ld resident after merging object %s",
                                  pi.fMemVirtual, pi.fMemResident, obj->GetName());
         RedirectOutput(0);
      }
      // Record also values for monitoring
      TPerfStats::SetMemValues();
   }
}

//______________________________________________________________________________
void TProofPlayerRemote::SetLastMergingMsg(TObject *obj)
{
   // Set the message to be notified in case of exception

   TString lastMsg = TString::Format("while merging object '%s'", obj->GetName());
   TProofServ::SetLastMsg(lastMsg);
}

//______________________________________________________________________________
Int_t TProofPlayerRemote::Incorporate(TObject *newobj, TList *outlist, Bool_t &merged)
{
   // Incorporate object 'newobj' in the list 'outlist'.
   // The object is merged with an object of the same name already existing in
   // the list, or just added.
   // The boolean merged is set to kFALSE when the object is just added to 'outlist';
   // this happens if the Merge() method does not exist or if a object named as 'obj'
   // is not already in the list. If the obj is not 'merged' than it should not be
   // deleted, unless outlist is not owner of its objects.
   // Return 0 on success, -1 on error.

   merged = kTRUE;

   PDB(kOutput,1)
      Info("Incorporate", "enter: obj: %p (%s), list: %p",
                          newobj, newobj ? newobj->ClassName() : "undef", outlist);

   // The object and list must exist
   if (!newobj || !outlist) {
      Error("Incorporate","Invalid inputs: obj: %p, list: %p", newobj, outlist);
      return -1;
   }

   // Special treatment for histograms in autobin mode
   Bool_t specialH =
      (!fProof || !fProof->TestBit(TProof::kIsClient) || fProof->IsLite()) ? kTRUE : kFALSE;
   if (specialH && newobj->InheritsFrom(TH1::Class())) {
      if (!HandleHistogram(newobj)) {
         PDB(kOutput,1) Info("Incorporate", "histogram object '%s' added to the"
                             " appropriate list for delayed merging", newobj->GetName());
         merged = kFALSE;
         return 0;
      }
   }

   // Check if an object with the same name exists already
   TObject *obj = outlist->FindObject(newobj->GetName());

   // If no, add the new object and return
   if (!obj) {
      outlist->Add(newobj);
      merged = kFALSE;
      // Done
      return 0;
   }

   // Locate the Merge(TCollection *) method
   TMethodCall callEnv;
   if (obj->IsA())
      callEnv.InitWithPrototype(obj->IsA(), "Merge", "TCollection*");
   if (callEnv.IsValid()) {
      // Found: put the object in a one-element list
      static TList *xlist = new TList;
      xlist->Add(newobj);
      // Call the method
      callEnv.SetParam((Long_t) xlist);
      callEnv.Execute(obj);
      // Ready for next call
      xlist->Clear();
   } else {
      // Not found: return individual objects
      outlist->Add(newobj);
      merged = kFALSE;
   }

   // Done
   return 0;
}

//______________________________________________________________________________
TObject *TProofPlayerRemote::HandleHistogram(TObject *obj)
{
   // Low statistic histograms need a special treatment when using autobin

   TH1 *h = dynamic_cast<TH1 *>(obj);
   if (!h) {
      // Not an histo
      return obj;
   }

   // Does is still needs binning ?
   Bool_t tobebinned = (h->GetBuffer()) ? kTRUE : kFALSE;

   // Number of entries
   Int_t nent = h->GetBufferLength();
   PDB(kOutput,2) Info("HandleHistogram", "h:%s ent:%d, buffer size: %d",
                       h->GetName(), nent, h->GetBufferSize());

   // Attach to the list in the outputlists, if any
   TList *list = 0;
   if (!fOutputLists) {
      PDB(kOutput,2) Info("HandleHistogram", "create fOutputLists");
      fOutputLists = new TList;
      fOutputLists->SetOwner();
   }
   list = (TList *) fOutputLists->FindObject(h->GetName());

   TH1 *href = 0;
   if (tobebinned) {

      // The histogram needs to be projected in a reasonable range: we
      // do this at the end with all the histos, so we need to create
      // a list here
      if (!list) {
         // Create the list
         list = new TList;
         list->SetName(h->GetName());
         list->SetOwner();
         fOutputLists->Add(list);
         // Move in it any previously merged object from the output list
         if (fOutput && (href = (TH1 *) fOutput->FindObject(h->GetName()))) {
            fOutput->Remove(href);
            list->Add(href);
         }
      }
      TIter nxh(list);
      while ((href = (TH1 *) nxh())) {
         if (href->GetBuffer() && href->GetBufferLength() < nent) break;
      }
      if (href) {
         list->AddBefore(href, h);
      } else {
         list->Add(h);
      }
      // Done
      return (TObject *)0;

   } else {

      if (list) {
         TIter nxh(list);
         while ((href = (TH1 *) nxh())) {
            if (href->GetBuffer() || href->GetEntries() < nent) break;
         }
         if (href) {
            list->AddBefore(href, h);
         } else {
            list->Add(h);
         }
         // Done
         return (TObject *)0;

      } else {
         // Histogram has already been projected
         Int_t hsz = h->GetNbinsX() * h->GetNbinsY() * h->GetNbinsZ();
         if (gProofServ && hsz > gProofServ->GetMsgSizeHWM()) {
            // Large histo: merge one-by-one
            return obj;
         } else {
            // Create the list to merge in one-go at the end (more efficient
            // than merging one by one)
            list = new TList;
            list->SetName(h->GetName());
            list->SetOwner();
            fOutputLists->Add(list);
            list->Add(h);
            // Done
            return (TObject *)0;
         }
      }
   }
   PDB(kOutput,1) Info("HandleHistogram", "leaving");
}

//______________________________________________________________________________
void TProofPlayerRemote::StoreOutput(TList *out)
{
   // Store received output list.

   PDB(kOutput,1) Info("StoreOutput","Enter");

   if ( out == 0 ) {
      PDB(kOutput,1) Info("StoreOutput","Leave (empty)");
      return;
   }

   TIter next(out);
   out->SetOwner(kFALSE);  // take ownership of the contents

   if (fOutputLists == 0) {
      PDB(kOutput,2) Info("StoreOutput","Create fOutputLists");
      fOutputLists = new TList;
      fOutputLists->SetOwner();
   }
   // process eventlists first
   TList* lists = dynamic_cast<TList*> (out->FindObject("PROOF_EventListsList"));
   if (lists) {
      out->Remove(lists);
      TEventList *mainList = new TEventList("PROOF_EventList");
      out->Add(mainList);
      TIter it(lists);
      TEventList *aList;
      while ( (aList = dynamic_cast<TEventList*> (it())) ) {
         // find file offset
         TIter nxe(fDSet->GetListOfElements());
         TDSetElement *elem;
         while ( (elem = dynamic_cast<TDSetElement*> (nxe())) ) {
            if (strcmp(elem->GetFileName(), aList->GetName()) == 0)
               break;
         }
         if (!elem) {
            Error("StoreOutput", "found the EventList for %s, but no object with that name "
                                 "in the TDSet", aList->GetName());
            continue;
         }
         Long64_t offset = elem->GetTDSetOffset();

         // shift the list by the number of first event in that file
         Long64_t *arr = aList->GetList();
         Int_t num = aList->GetN();
         if (arr && offset)
            for (int i = 0; i < num; i++)
               arr[i] += offset;

         mainList->Add(aList);           // add to the main list
      }
      delete lists;
   }

   TObject *obj;
   while( (obj = next()) ) {
      PDB(kOutput,2) Info("StoreOutput","find list for '%s'", obj->GetName() );

      TList *list = (TList *) fOutputLists->FindObject( obj->GetName() );
      if ( list == 0 ) {
         PDB(kOutput,2) Info("StoreOutput", "list for '%s' not found (creating)", obj->GetName());
         list = new TList;
         list->SetName( obj->GetName() );
         list->SetOwner();
         fOutputLists->Add( list );
      }
      list->Add( obj );
   }

   delete out;
   PDB(kOutput,1) Info("StoreOutput", "leave");
}

//______________________________________________________________________________
TList *TProofPlayerRemote::MergeFeedback()
{
   // Merge feedback lists.

   PDB(kFeedback,1)
      Info("MergeFeedback","Enter");

   if ( fFeedbackLists == 0 ) {
      PDB(kFeedback,1)
         Info("MergeFeedback","Leave (no output)");
      return 0;
   }

   TList *fb = new TList;   // collection of feedback objects
   fb->SetOwner();

   TIter next(fFeedbackLists);

   TMap *map;
   while ( (map = (TMap*) next()) ) {

      PDB(kFeedback,2)
         Info("MergeFeedback", "map %s size: %d", map->GetName(), map->GetSize());

      // turn map into list ...

      TList *list = new TList;
      TIter keys(map);

#ifndef R__TH1MERGEFIXED
      Int_t nbmx = -1;
      TObject *oref = 0;
#endif
      while ( TObject *key = keys() ) {
         TObject *o = map->GetValue(key);
         TH1 *h = dynamic_cast<TH1 *>(o);
#ifndef R__TH1MERGEFIXED
         // Temporary fix for to cope with the problem in TH1::Merge.
         // We need to use a reference histo the one with the largest number
         // of bins so that the histos from all submasters can be correctly
         // fit in
         if (h && !strncmp(o->GetName(),"PROOF_",6)) {
            if (h->GetNbinsX() > nbmx) {
               nbmx=  h->GetNbinsX();
               oref = o;
            }
         }
#endif
         if (h) {
            TIter nxh(list);
            TH1 *href= 0;
            while ((href = (TH1 *)nxh())) {
               if (h->GetBuffer()) {
                  if (href->GetBuffer() && href->GetBufferLength() < h->GetBufferLength()) break;
               } else {
                  if (href->GetBuffer() || href->GetEntries() < h->GetEntries()) break;
               }
            }
            if (href) {
               list->AddBefore(href, h);
            } else {
               list->Add(h);
            }
         } else {
            list->Add(o);
         }
      }

      // clone first object, remove from list
#ifdef R__TH1MERGEFIXED
      TObject *obj = list->First();
#else
      TObject *obj = (oref) ? oref : list->First();
#endif
      list->Remove(obj);
      obj = obj->Clone();
      fb->Add(obj);

      if ( list->IsEmpty() ) {
         delete list;
         continue;
      }

      // merge list with clone
      TMethodCall callEnv;
      if (obj->IsA())
         callEnv.InitWithPrototype(obj->IsA(), "Merge", "TCollection*");
      if (callEnv.IsValid()) {
         callEnv.SetParam((Long_t) list);
         callEnv.Execute(obj);
      } else {
         // No Merge interface, return copy of individual objects
         while ( (obj = list->First()) ) {
            fb->Add(obj->Clone());
            list->Remove(obj);
         }
      }

      delete list;
   }

   PDB(kFeedback,1)
      Info("MergeFeedback","Leave (%d object(s))", fb->GetSize());

   return fb;
}

//______________________________________________________________________________
void TProofPlayerRemote::StoreFeedback(TObject *slave, TList *out)
{
   // Store feedback results from the specified slave.

   PDB(kFeedback,1)
      Info("StoreFeedback","Enter");

   if ( out == 0 ) {
      PDB(kFeedback,1)
         Info("StoreFeedback","Leave (empty)");
      return;
   }

   if ( IsClient() ) {
      // in client
      Feedback(out);
      delete out;
      return;
   }

   if (fFeedbackLists == 0) {
      PDB(kFeedback,2) Info("StoreFeedback","Create fFeedbackLists");
      fFeedbackLists = new TList;
      fFeedbackLists->SetOwner();
   }

   TIter next(out);
   out->SetOwner(kFALSE);  // take ownership of the contents

   const char *ord = ((TSlave*) slave)->GetOrdinal();

   TObject *obj;
   while( (obj = next()) ) {
      PDB(kFeedback,2)
         Info("StoreFeedback","%s: Find '%s'", ord, obj->GetName() );
      TMap *map = (TMap*) fFeedbackLists->FindObject(obj->GetName());
      if ( map == 0 ) {
         PDB(kFeedback,2)
            Info("StoreFeedback", "%s: map for '%s' not found (creating)", ord, obj->GetName());
         // Map must not be owner (ownership is with regards to the keys (only))
         map = new TMap;
         map->SetName(obj->GetName());
         fFeedbackLists->Add(map);
      } else {
         PDB(kFeedback,2)
            Info("StoreFeedback","%s: removing previous value", ord);
         if (map->GetValue(slave))
            delete map->GetValue(slave);
         map->Remove(slave);
      }
      map->Add(slave, obj);
      PDB(kFeedback,2)
         Info("StoreFeedback","%s: %s, size: %d", ord, obj->GetName(), map->GetSize());
   }

   delete out;
   PDB(kFeedback,1)
      Info("StoreFeedback","Leave");
}

//______________________________________________________________________________
void TProofPlayerRemote::SetupFeedback()
{
   // Setup reporting of feedback objects.

   if (IsClient()) return; // Client does not need timer

   fFeedback = (TList*) fInput->FindObject("FeedbackList");

   PDB(kFeedback,1) Info("SetupFeedback","\"FeedbackList\" %sfound",
      fFeedback == 0 ? "NOT ":"");

   if (fFeedback == 0 || fFeedback->GetSize() == 0) return;

   // OK, feedback was requested, setup the timer
   SafeDelete(fFeedbackTimer);
   fFeedbackPeriod = 2000;
   TProof::GetParameter(fInput, "PROOF_FeedbackPeriod", fFeedbackPeriod);
   fFeedbackTimer = new TTimer;
   fFeedbackTimer->SetObject(this);
   fFeedbackTimer->Start(fFeedbackPeriod, kTRUE);
}

//______________________________________________________________________________
void TProofPlayerRemote::StopFeedback()
{
   // Stop reporting of feedback objects.

   if (fFeedbackTimer == 0) return;

   PDB(kFeedback,1) Info("StopFeedback","Stop Timer");

   SafeDelete(fFeedbackTimer);
}

//______________________________________________________________________________
Bool_t TProofPlayerRemote::HandleTimer(TTimer *)
{
   // Send feedback objects to client.

   PDB(kFeedback,2) Info("HandleTimer","Entry");

   if (fFeedbackTimer == 0) return kFALSE; // timer already switched off

   // process local feedback objects

   TList *fb = new TList;
   fb->SetOwner();

   TIter next(fFeedback);
   while( TObjString *name = (TObjString*) next() ) {
      TObject *o = fOutput->FindObject(name->GetName());
      if (o != 0) {
         fb->Add(o->Clone());
         // remove the corresponding entry from the feedback list
         TMap *m = 0;
         if (fFeedbackLists &&
            (m = (TMap *) fFeedbackLists->FindObject(name->GetName()))) {
            fFeedbackLists->Remove(m);
            m->DeleteValues();
            delete m;
         }
      }
   }

   if (fb->GetSize() > 0) {
      StoreFeedback(this, fb); // adopts fb
   } else {
      delete fb;
   }

   if (fFeedbackLists == 0) {
      fFeedbackTimer->Start(fFeedbackPeriod, kTRUE);   // maybe next time
      return kFALSE;
   }

   fb = MergeFeedback();

   PDB(kFeedback,2) Info("HandleTimer","Sending %d objects", fb->GetSize());

   TMessage m(kPROOF_FEEDBACK);
   m << fb;

   // send message to client;
   gProofServ->GetSocket()->Send(m);

   delete fb;

   fFeedbackTimer->Start(fFeedbackPeriod, kTRUE);

   return kFALSE; // ignored?
}

//______________________________________________________________________________
TDSetElement *TProofPlayerRemote::GetNextPacket(TSlave *slave, TMessage *r)
{
   // Get next packet for specified slave.

   // The first call to this determines the end of initialization
   SetInitTime();

   TDSetElement *e = fPacketizer->GetNextPacket( slave, r );

   if (e == 0) {
      PDB(kPacketizer,2) Info("GetNextPacket","%s: done!", slave->GetOrdinal());
   } else if (e == (TDSetElement*) -1) {
      PDB(kPacketizer,2) Info("GetNextPacket","%s: waiting ...", slave->GetOrdinal());
   } else {
      PDB(kPacketizer,2)
         Info("GetNextPacket","%s (%s): '%s' '%s' '%s' %lld %lld",
              slave->GetOrdinal(), slave->GetName(), e->GetFileName(),
              e->GetDirectory(), e->GetObjName(), e->GetFirst(), e->GetNum());
   }

   return e;
}

//______________________________________________________________________________
Bool_t TProofPlayerRemote::IsClient() const
{
   // Is the player running on the client?

   return fProof ? fProof->TestBit(TProof::kIsClient) : kFALSE;
}

//______________________________________________________________________________
Long64_t TProofPlayerRemote::DrawSelect(TDSet *set, const char *varexp,
                                        const char *selection, Option_t *option,
                                        Long64_t nentries, Long64_t firstentry)
{
   // Draw (support for TChain::Draw()).
   // Returns -1 in case of error or number of selected events in case of success.

   if (!fgDrawInputPars) {
      fgDrawInputPars = new THashList;
      fgDrawInputPars->Add(new TObjString("FeedbackList"));
      fgDrawInputPars->Add(new TObjString("PROOF_LineColor"));
      fgDrawInputPars->Add(new TObjString("PROOF_LineStyle"));
      fgDrawInputPars->Add(new TObjString("PROOF_LineWidth"));
      fgDrawInputPars->Add(new TObjString("PROOF_MarkerColor"));
      fgDrawInputPars->Add(new TObjString("PROOF_MarkerStyle"));
      fgDrawInputPars->Add(new TObjString("PROOF_MarkerSize"));
      fgDrawInputPars->Add(new TObjString("PROOF_FillColor"));
      fgDrawInputPars->Add(new TObjString("PROOF_FillStyle"));
   }

   TString selector, objname;
   if (GetDrawArgs(varexp, selection, option, selector, objname) != 0) {
      Error("DrawSelect", "parsing arguments");
      return -1;
   }

   TNamed *varexpobj = new TNamed("varexp", varexp);
   TNamed *selectionobj = new TNamed("selection", selection);

   // Save the current input list
   TObject *o = 0;
   TList *savedInput = new TList;
   TIter nxi(fInput);
   while ((o = nxi())) {
      savedInput->Add(o);
      TString n(o->GetName());
      if (fgDrawInputPars && !fgDrawInputPars->FindObject(o->GetName())) fInput->Remove(o);
   }

   fInput->Add(varexpobj);
   fInput->Add(selectionobj);

   // Make sure we have an object name
   if (objname == "") objname = "htemp";

   fProof->AddFeedback(objname);
   Long64_t r = Process(set, selector, option, nentries, firstentry);
   fProof->RemoveFeedback(objname);

   fInput->Remove(varexpobj);
   fInput->Remove(selectionobj);
   if (TNamed *opt = dynamic_cast<TNamed*> (fInput->FindObject("PROOF_OPTIONS"))) {
      fInput->Remove(opt);
      delete opt;
   }

   delete varexpobj;
   delete selectionobj;

   // Restore the input list
   fInput->Clear();
   TIter nxsi(savedInput);
   while ((o = nxsi()))
      fInput->Add(o);
   savedInput->SetOwner(kFALSE);
   delete savedInput;

   return r;
}

//______________________________________________________________________________
void TProofPlayerRemote::SetInitTime()
{
   // Set init time

   if (fPacketizer)
      fPacketizer->SetInitTime();
}

//------------------------------------------------------------------------------


ClassImp(TProofPlayerSlave)

//______________________________________________________________________________
void TProofPlayerSlave::SetupFeedback()
{
   // Setup feedback.

   TList *fb = (TList*) fInput->FindObject("FeedbackList");
   if (fb) {
      PDB(kFeedback,1)
         Info("SetupFeedback","\"FeedbackList\" found: %d objects", fb->GetSize());
   } else {
      PDB(kFeedback,1)
         Info("SetupFeedback","\"FeedbackList\" NOT found");
   }

   if (fb == 0 || fb->GetSize() == 0) return;

   // OK, feedback was requested, setup the timer

   SafeDelete(fFeedbackTimer);
   fFeedbackPeriod = 2000;
   TProof::GetParameter(fInput, "PROOF_FeedbackPeriod", fFeedbackPeriod);
   fFeedbackTimer = new TTimer;
   fFeedbackTimer->SetObject(this);
   fFeedbackTimer->Start(fFeedbackPeriod, kTRUE);

   fFeedback = fb;
}

//______________________________________________________________________________
void TProofPlayerSlave::StopFeedback()
{
   // Stop feedback.

   if (fFeedbackTimer == 0) return;

   PDB(kFeedback,1) Info("StopFeedback","Stop Timer");

   SafeDelete(fFeedbackTimer);
}

//______________________________________________________________________________
Bool_t TProofPlayerSlave::HandleTimer(TTimer *)
{
   // Handle timer event.

   PDB(kFeedback,2) Info("HandleTimer","Entry");

   // If in sequential (0-slave-PROOF) mode we do not have a packetizer
   // so we also send the info to update the progress bar.
   if (gProofServ) {
      Bool_t sendm = kFALSE;
      TMessage m(kPROOF_PROGRESS);
      if (gProofServ->IsMaster() && !gProofServ->IsParallel()) {
         sendm = kTRUE;
         if (gProofServ->GetProtocol() > 25) {
            m << GetProgressStatus();
         } else if (gProofServ->GetProtocol() > 11) {
            TProofProgressStatus *ps = GetProgressStatus();
            m << fTotalEvents << ps->GetEntries() << ps->GetBytesRead()
              << (Float_t) -1. << (Float_t) ps->GetProcTime()
              << (Float_t) ps->GetRate() << (Float_t) -1.;
         } else {
            m << fTotalEvents << GetEventsProcessed();
         }
      }
      if (sendm) gProofServ->GetSocket()->Send(m);
   }

   if (fFeedback == 0) return kFALSE;

   TList *fb = new TList;
   fb->SetOwner(kFALSE);

   if (fOutput == 0) {
      fOutput = fSelector->GetOutputList();
   }

   if (fOutput) {
      TIter next(fFeedback);
      while( TObjString *name = (TObjString*) next() ) {
         // TODO: find object in memory ... maybe allow only in fOutput ?
         TObject *o = fOutput->FindObject(name->GetName());
         if (o != 0) fb->Add(o);
      }
   }

   PDB(kFeedback,2) Info("HandleTimer","Sending %d objects", fb->GetSize());

   TMessage m(kPROOF_FEEDBACK);
   m << fb;

   // send message to client;
   gProofServ->GetSocket()->Send(m);

   delete fb;

   fFeedbackTimer->Start(fFeedbackPeriod, kTRUE);

   return kFALSE; // ignored?
}

//______________________________________________________________________________
void TProofPlayerSlave::HandleGetTreeHeader(TMessage *mess)
{
   // Handle tree header request.

   TMessage answ(kPROOF_GETTREEHEADER);

   TDSet *dset;
   (*mess) >> dset;
   dset->Reset();
   TDSetElement *e = dset->Next();
   Long64_t entries = 0;
   TFile *f = 0;
   TTree *t = 0;
   if (!e) {
      PDB(kGlobal, 1) Info("HandleGetTreeHeader", "empty TDSet");
   } else {
      f = TFile::Open(e->GetFileName());
      t = 0;
      if (f) {
         t = (TTree*) f->Get(e->GetObjName());
         if (t) {
            t->SetMaxVirtualSize(0);
            t->DropBaskets();
            entries = t->GetEntries();

            // compute #entries in all the files
            while ((e = dset->Next()) != 0) {
               TFile *f1 = TFile::Open(e->GetFileName());
               if (f1) {
                  TTree *t1 = (TTree*) f1->Get(e->GetObjName());
                  if (t1) {
                     entries += t1->GetEntries();
                     delete t1;
                  }
                  delete f1;
               }
            }
            t->SetMaxEntryLoop(entries);   // this field will hold the total number of entries ;)
         }
      }
   }
   if (t)
      answ << TString("Success") << t;
   else
      answ << TString("Failed") << t;

   fSocket->Send(answ);

   SafeDelete(t);
   SafeDelete(f);
}


//------------------------------------------------------------------------------

ClassImp(TProofPlayerSuperMaster)

//______________________________________________________________________________
Long64_t TProofPlayerSuperMaster::Process(TDSet *dset, const char *selector_file,
                                          Option_t *option, Long64_t nentries,
                                          Long64_t first)
{
   // Process specified TDSet on PROOF. Runs on super master.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.

   fProgressStatus->Reset();
   PDB(kGlobal,1) Info("Process","Enter");

   TProofSuperMaster *proof = dynamic_cast<TProofSuperMaster*>(GetProof());
   if (!proof) return -1;

   delete fOutput;
   fOutput = new TList;

   TPerfStats::Start(fInput, fOutput);

   if (!SendSelector(selector_file)) {
      Error("Process", "sending selector %s", selector_file);
      return -1;
   }

   TCleanup clean(this);
   SetupFeedback();

   if (proof->IsMaster()) {

      // make sure the DSet is valid
      if (!dset->ElementsValid()) {
         proof->ValidateDSet(dset);
         if (!dset->ElementsValid()) {
            Error("Process", "could not validate TDSet");
            return -1;
         }
      }

      TList msds;
      msds.SetOwner(); // This will delete TPairs

      TList keyholder; // List to clean up key part of the pairs
      keyholder.SetOwner();
      TList valueholder; // List to clean up value part of the pairs
      valueholder.SetOwner();

      // Construct msd list using the slaves
      TIter nextslave(proof->GetListOfActiveSlaves());
      while (TSlave *sl = dynamic_cast<TSlave*>(nextslave())) {
         TList *submasters = 0;
         TPair *msd = dynamic_cast<TPair*>(msds.FindObject(sl->GetMsd()));
         if (!msd) {
            submasters = new TList;
            submasters->SetName(sl->GetMsd());
            keyholder.Add(submasters);
            TList *setelements = new TSortedList(kSortDescending);
            setelements->SetName(TString(sl->GetMsd())+"_Elements");
            valueholder.Add(setelements);
            msds.Add(new TPair(submasters, setelements));
         } else {
            submasters = dynamic_cast<TList*>(msd->Key());
         }
         if (submasters) submasters->Add(sl);
      }

      // Add TDSetElements to msd list
      Long64_t cur = 0; //start of next element
      TIter nextelement(dset->GetListOfElements());
      while (TDSetElement *elem = dynamic_cast<TDSetElement*>(nextelement())) {

         if (elem->GetNum()<1) continue; // get rid of empty elements

         if (nentries !=-1 && cur>=first+nentries) {
            // we are done
            break;
         }

         if (cur+elem->GetNum()-1<first) {
            //element is before first requested entry
            cur+=elem->GetNum();
            continue;
         }

         if (cur<first) {
            //modify element to get proper start
            elem->SetNum(elem->GetNum()-(first-cur));
            elem->SetFirst(elem->GetFirst()+first-cur);
            cur=first;
         }

         if (nentries==-1 || cur+elem->GetNum()<=first+nentries) {
            cur+=elem->GetNum();
         } else {
            //modify element to get proper end
            elem->SetNum(first+nentries-cur);
            cur=first+nentries;
         }

         TPair *msd = dynamic_cast<TPair*>(msds.FindObject(elem->GetMsd()));
         if (!msd) {
            Error("Process", "data requires mass storage domain '%s'"
                  " which is not accessible in this proof session",
                  elem->GetMsd());
            return -1;
         } else {
            TList *elements = dynamic_cast<TList*>(msd->Value());
            if (elements) elements->Add(elem);
         }
      }

      TList usedmasters;
      TIter nextmsd(msds.MakeIterator());
      while (TPair *msd = dynamic_cast<TPair*>(nextmsd())) {
         TList *submasters = dynamic_cast<TList*>(msd->Key());
         TList *setelements = dynamic_cast<TList*>(msd->Value());

         // distribute elements over the masters
         Int_t nmasters = submasters ? submasters->GetSize() : -1;
         Int_t nelements = setelements ? setelements->GetSize() : -1;
         for (Int_t i=0; i<nmasters; i++) {

            Long64_t nent = 0;
            TDSet set(dset->GetType(), dset->GetObjName(),
                      dset->GetDirectory());
            for (Int_t j = (i*nelements)/nmasters;
                       j < ((i+1)*nelements)/nmasters;
                       j++) {
               TDSetElement *elem = setelements ?
                  dynamic_cast<TDSetElement*>(setelements->At(j)) : (TDSetElement *)0;
               if (elem) {
                  set.Add(elem->GetFileName(), elem->GetObjName(),
                        elem->GetDirectory(), elem->GetFirst(),
                        elem->GetNum(), elem->GetMsd());
                  nent += elem->GetNum();
               } else {
                  Warning("Process", "not a TDSetElement object");
               }
            }

            if (set.GetListOfElements()->GetSize()>0) {
               TMessage mesg(kPROOF_PROCESS);
               TString fn(gSystem->BaseName(selector_file));
               TString opt = option;
               mesg << &set << fn << fInput << opt << Long64_t(-1) << Long64_t(0);

               TSlave *sl = dynamic_cast<TSlave*>(submasters->At(i));
               if (sl) {
                  PDB(kGlobal,1) Info("Process",
                                    "Sending TDSet with %d elements to submaster %s",
                                    set.GetListOfElements()->GetSize(),
                                    sl->GetOrdinal());
                  sl->GetSocket()->Send(mesg);
                  usedmasters.Add(sl);

                  // setup progress info
                  fSlaves.AddLast(sl);
                  fSlaveProgress.Set(fSlaveProgress.GetSize()+1);
                  fSlaveProgress[fSlaveProgress.GetSize()-1] = 0;
                  fSlaveTotals.Set(fSlaveTotals.GetSize()+1);
                  fSlaveTotals[fSlaveTotals.GetSize()-1] = nent;
                  fSlaveBytesRead.Set(fSlaveBytesRead.GetSize()+1);
                  fSlaveBytesRead[fSlaveBytesRead.GetSize()-1] = 0;
                  fSlaveInitTime.Set(fSlaveInitTime.GetSize()+1);
                  fSlaveInitTime[fSlaveInitTime.GetSize()-1] = -1.;
                  fSlaveProcTime.Set(fSlaveProcTime.GetSize()+1);
                  fSlaveProcTime[fSlaveProcTime.GetSize()-1] = -1.;
                  fSlaveEvtRti.Set(fSlaveEvtRti.GetSize()+1);
                  fSlaveEvtRti[fSlaveEvtRti.GetSize()-1] = -1.;
                  fSlaveMBRti.Set(fSlaveMBRti.GetSize()+1);
                  fSlaveMBRti[fSlaveMBRti.GetSize()-1] = -1.;
                  fSlaveActW.Set(fSlaveActW.GetSize()+1);
                  fSlaveActW[fSlaveActW.GetSize()-1] = 0;
                  fSlaveTotS.Set(fSlaveTotS.GetSize()+1);
                  fSlaveTotS[fSlaveTotS.GetSize()-1] = 0;
                  fSlaveEffS.Set(fSlaveEffS.GetSize()+1);
                  fSlaveEffS[fSlaveEffS.GetSize()-1] = 0.;
               } else {
                  Warning("Process", "not a TSlave object");
               }
            }
         }
      }

      if ( !IsClient() ) HandleTimer(0);
      PDB(kGlobal,1) Info("Process","Calling Collect");
      proof->Collect(&usedmasters);
      HandleTimer(0);

   }

   StopFeedback();

   PDB(kGlobal,1) Info("Process","Calling Merge Output");
   MergeOutput();

   TPerfStats::Stop();

   return 0;
}

//______________________________________________________________________________
void TProofPlayerSuperMaster::Progress(TSlave *sl, Long64_t total, Long64_t processed)
{
   // Report progress.

   Int_t idx = fSlaves.IndexOf(sl);
   fSlaveProgress[idx] = processed;
   if (fSlaveTotals[idx] != total)
      Warning("Progress", "total events has changed for slave %s", sl->GetName());
   fSlaveTotals[idx] = total;

   Long64_t tot = 0;
   Int_t i;
   for (i = 0; i < fSlaveTotals.GetSize(); i++) tot += fSlaveTotals[i];
   Long64_t proc = 0;
   for (i = 0; i < fSlaveProgress.GetSize(); i++) proc += fSlaveProgress[i];

   Progress(tot, proc);
}

//______________________________________________________________________________
void TProofPlayerSuperMaster::Progress(TSlave *sl, Long64_t total,
                                       Long64_t processed, Long64_t bytesread,
                                       Float_t initTime, Float_t procTime,
                                       Float_t evtrti, Float_t mbrti)
{
   // Report progress.

   PDB(kGlobal,2)
      Info("Progress","%s: %lld %lld %f %f %f %f", sl->GetName(),
                      processed, bytesread, initTime, procTime, evtrti, mbrti);

   Int_t idx = fSlaves.IndexOf(sl);
   if (fSlaveTotals[idx] != total)
      Warning("Progress", "total events has changed for slave %s", sl->GetName());
   fSlaveTotals[idx] = total;
   fSlaveProgress[idx] = processed;
   fSlaveBytesRead[idx] = bytesread;
   fSlaveInitTime[idx] = (initTime > -1.) ? initTime : fSlaveInitTime[idx];
   fSlaveProcTime[idx] = (procTime > -1.) ? procTime : fSlaveProcTime[idx];
   fSlaveEvtRti[idx] = (evtrti > -1.) ? evtrti : fSlaveEvtRti[idx];
   fSlaveMBRti[idx] = (mbrti > -1.) ? mbrti : fSlaveMBRti[idx];

   Int_t i;
   Long64_t tot = 0;
   Long64_t proc = 0;
   Long64_t bytes = 0;
   Float_t init = -1.;
   Float_t ptime = -1.;
   Float_t erti = 0.;
   Float_t srti = 0.;
   Int_t nerti = 0;
   Int_t nsrti = 0;
   for (i = 0; i < fSlaveTotals.GetSize(); i++) {
      tot += fSlaveTotals[i];
      if (i < fSlaveProgress.GetSize())
         proc += fSlaveProgress[i];
      if (i < fSlaveBytesRead.GetSize())
         bytes += fSlaveBytesRead[i];
      if (i < fSlaveInitTime.GetSize())
         if (fSlaveInitTime[i] > -1. && (init < 0. || fSlaveInitTime[i] < init))
            init = fSlaveInitTime[i];
      if (i < fSlaveProcTime.GetSize())
         if (fSlaveProcTime[i] > -1. && (ptime < 0. || fSlaveProcTime[i] > ptime))
            ptime = fSlaveProcTime[i];
      if (i < fSlaveEvtRti.GetSize())
         if (fSlaveEvtRti[i] > -1.) {
            erti += fSlaveEvtRti[i];
            nerti++;
         }
      if (i < fSlaveMBRti.GetSize())
         if (fSlaveMBRti[i] > -1.) {
            srti += fSlaveMBRti[i];
            nsrti++;
         }
   }
   srti = (nsrti > 0) ? srti / nerti : 0.;

   Progress(tot, proc, bytes, init, ptime, erti, srti);
}

//______________________________________________________________________________
void TProofPlayerSuperMaster::Progress(TSlave *wrk, TProofProgressInfo *pi)
{
   // Progress signal.

   if (pi) {
      PDB(kGlobal,2)
         Info("Progress","%s: %lld %lld %lld %f %f %f %f %d %f", wrk->GetOrdinal(),
                         pi->fTotal, pi->fProcessed, pi->fBytesRead,
                         pi->fInitTime, pi->fProcTime, pi->fEvtRateI, pi->fMBRateI,
                         pi->fActWorkers, pi->fEffSessions);

      Int_t idx = fSlaves.IndexOf(wrk);
      if (fSlaveTotals[idx] != pi->fTotal)
         Warning("Progress", "total events has changed for worker %s", wrk->GetName());
      fSlaveTotals[idx] = pi->fTotal;
      fSlaveProgress[idx] = pi->fProcessed;
      fSlaveBytesRead[idx] = pi->fBytesRead;
      fSlaveInitTime[idx] = (pi->fInitTime > -1.) ? pi->fInitTime : fSlaveInitTime[idx];
      fSlaveProcTime[idx] = (pi->fProcTime > -1.) ? pi->fProcTime : fSlaveProcTime[idx];
      fSlaveEvtRti[idx] = (pi->fEvtRateI > -1.) ? pi->fEvtRateI : fSlaveEvtRti[idx];
      fSlaveMBRti[idx] = (pi->fMBRateI > -1.) ? pi->fMBRateI : fSlaveMBRti[idx];
      fSlaveActW[idx] = (pi->fActWorkers > -1) ? pi->fActWorkers : fSlaveActW[idx];
      fSlaveTotS[idx] = (pi->fTotSessions > -1) ? pi->fTotSessions : fSlaveTotS[idx];
      fSlaveEffS[idx] = (pi->fEffSessions > -1.) ? pi->fEffSessions : fSlaveEffS[idx];

      Int_t i;
      Int_t nerti = 0;
      Int_t nsrti = 0;
      TProofProgressInfo pisum(0, 0, 0, -1., -1., 0., 0., 0, 0, 0.);
      for (i = 0; i < fSlaveTotals.GetSize(); i++) {
         pisum.fTotal += fSlaveTotals[i];
         if (i < fSlaveProgress.GetSize())
            pisum.fProcessed += fSlaveProgress[i];
         if (i < fSlaveBytesRead.GetSize())
            pisum.fBytesRead += fSlaveBytesRead[i];
         if (i < fSlaveInitTime.GetSize())
            if (fSlaveInitTime[i] > -1. && (pisum.fInitTime < 0. || fSlaveInitTime[i] < pisum.fInitTime))
               pisum.fInitTime = fSlaveInitTime[i];
         if (i < fSlaveProcTime.GetSize())
            if (fSlaveProcTime[i] > -1. && (pisum.fProcTime < 0. || fSlaveProcTime[i] > pisum.fProcTime))
               pisum.fProcTime = fSlaveProcTime[i];
         if (i < fSlaveEvtRti.GetSize())
            if (fSlaveEvtRti[i] > -1.) {
               pisum.fEvtRateI += fSlaveEvtRti[i];
               nerti++;
            }
         if (i < fSlaveMBRti.GetSize())
            if (fSlaveMBRti[i] > -1.) {
               pisum.fMBRateI += fSlaveMBRti[i];
               nsrti++;
            }
         if (i < fSlaveActW.GetSize())
            pisum.fActWorkers += fSlaveActW[i];
         if (i < fSlaveTotS.GetSize())
            if (fSlaveTotS[i] > -1 && (pisum.fTotSessions < 0. || fSlaveTotS[i] > pisum.fTotSessions))
               pisum.fTotSessions = fSlaveTotS[i];
         if (i < fSlaveEffS.GetSize())
            if (fSlaveEffS[i] > -1. && (pisum.fEffSessions < 0. || fSlaveEffS[i] > pisum.fEffSessions))
               pisum.fEffSessions = fSlaveEffS[i];
      }
      pisum.fMBRateI = (nsrti > 0) ? pisum.fMBRateI / nerti : 0.;

      Progress(&pisum);
   }
}

//______________________________________________________________________________
Bool_t TProofPlayerSuperMaster::HandleTimer(TTimer *)
{
   // Send progress and feedback to client.

   if (fFeedbackTimer == 0) return kFALSE; // timer stopped already

   Int_t i;
   Long64_t tot = 0;
   Long64_t proc = 0;
   Long64_t bytes = 0;
   Float_t init = -1.;
   Float_t ptime = -1.;
   Float_t erti = 0.;
   Float_t srti = 0.;
   Int_t nerti = 0;
   Int_t nsrti = 0;
   for (i = 0; i < fSlaveTotals.GetSize(); i++) {
      tot += fSlaveTotals[i];
      if (i < fSlaveProgress.GetSize())
         proc += fSlaveProgress[i];
      if (i < fSlaveBytesRead.GetSize())
         bytes += fSlaveBytesRead[i];
      if (i < fSlaveInitTime.GetSize())
         if (fSlaveInitTime[i] > -1. && (init < 0. || fSlaveInitTime[i] < init))
            init = fSlaveInitTime[i];
      if (i < fSlaveProcTime.GetSize())
         if (fSlaveProcTime[i] > -1. && (ptime < 0. || fSlaveProcTime[i] > ptime))
            ptime = fSlaveProcTime[i];
      if (i < fSlaveEvtRti.GetSize())
         if (fSlaveEvtRti[i] > -1.) {
            erti += fSlaveEvtRti[i];
            nerti++;
         }
      if (i < fSlaveMBRti.GetSize())
         if (fSlaveMBRti[i] > -1.) {
            srti += fSlaveMBRti[i];
            nsrti++;
         }
   }
   erti = (nerti > 0) ? erti / nerti : 0.;
   srti = (nsrti > 0) ? srti / nerti : 0.;

   TMessage m(kPROOF_PROGRESS);
   if (gProofServ->GetProtocol() > 25) {
      // Fill the message now
      TProofProgressInfo pi(tot, proc, bytes, init, ptime,
                            erti, srti, -1,
                            gProofServ->GetTotSessions(), gProofServ->GetEffSessions());
      m << &pi;
   } else {

      m << tot << proc << bytes << init << ptime << erti << srti;
   }

   // send message to client;
   gProofServ->GetSocket()->Send(m);

   if (fReturnFeedback)
      return TProofPlayerRemote::HandleTimer(0);
   else
      return kFALSE;
}

//______________________________________________________________________________
void TProofPlayerSuperMaster::SetupFeedback()
{
   // Setup reporting of feedback objects and progress messages.

   if (IsClient()) return; // Client does not need timer

   TProofPlayerRemote::SetupFeedback();

   if (fFeedbackTimer) {
      fReturnFeedback = kTRUE;
      return;
   } else {
      fReturnFeedback = kFALSE;
   }

   // setup the timer for progress message
   SafeDelete(fFeedbackTimer);
   fFeedbackPeriod = 2000;
   TProof::GetParameter(fInput, "PROOF_FeedbackPeriod", fFeedbackPeriod);
   fFeedbackTimer = new TTimer;
   fFeedbackTimer->SetObject(this);
   fFeedbackTimer->Start(fFeedbackPeriod, kTRUE);
}
