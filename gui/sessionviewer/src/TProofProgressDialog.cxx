// @(#)root/sessionviewer:$Id$
// Author: Fons Rademakers   21/03/03

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofProgressDialog                                                 //
//                                                                      //
// This class provides a query progress bar.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofProgressDialog.h"
#include "TProofProgressLog.h"
#include "TProofProgressMemoryPlot.h"
#include "TEnv.h"
#include "TError.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TGTextBuffer.h"
#include "TGTextEntry.h"
#include "TGProgressBar.h"
#include "TProof.h"
#include "TSlave.h"
#include "TSystem.h"
#include "TTimer.h"
#include "TGraph.h"
#include "TNtuple.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TLine.h"
#include "TAxis.h"
#include "TPaveText.h"
#include "TMath.h"

#ifdef PPD_SRV_NEWER
#undef PPD_SRV_NEWER
#endif
#define PPD_SRV_NEWER(v) (fProof && fProof->GetRemoteProtocol() > v)
#ifdef PPD_SRV_NEWER_REV
#undef PPD_SRV_NEWER_REV
#endif
#define PPD_SRV_NEWER_REV(r) (fSVNRev > r)

Bool_t TProofProgressDialog::fgKeepDefault = kTRUE;
Bool_t TProofProgressDialog::fgLogQueryDefault = kFALSE;
TString TProofProgressDialog::fgTextQueryDefault = "last";

static const Int_t gSVNMemPlot = 25090;

ClassImp(TProofProgressDialog)

//______________________________________________________________________________
TProofProgressDialog::TProofProgressDialog(TProof *proof,
                                           const char *selector,
                                           Int_t files,
                                           Long64_t first,
                                           Long64_t entries)
{
   // Create PROOF processing progress dialog.

   fProof         = proof;
   fFiles         = files;
   fFirst         = first;
   fEntries       = entries;
   fPrevProcessed = 0;
   fPrevTotal     = 0;
   fLogWindow     = 0;
   fMemWindow     = 0;
   fStatus        = kRunning;
   fKeep          = fgKeepDefault;
   fLogQuery      = fgLogQueryDefault;
   fRatePoints    = 0;
   fRateGraph     = 0;
   fProcTime      = 0.;
   fAvgRate       = 0.;
   fAvgMBRate     = 0.;
   fSVNRev        = -1;

   // Make sure we are attached to a good instance
   if (!proof || !(proof->IsValid())) {
      Error("TProofProgressDialog", "proof instance is invalid (%p, %s): protocol error?",
                                    proof, (proof && !(proof->IsValid())) ? "invalid" : "undef");
      return;
   }

   // Have to save this information here, in case gProof is dead when
   // the logs are requested
   fSessionUrl = (proof && proof->GetManager()) ? proof->GetManager()->GetUrl() : "";

   // SVN version run by the master
   TSlave *mst = (TSlave *) proof->GetListOfActiveSlaves()->First();
   if (mst) {
      TString vrs = mst->GetROOTVersion();
      Ssiz_t ib = vrs.Index("|"), from = ib + 2;
      if (ib != kNPOS) {
         TString svnr;
         // Strip of also the 'r' in front of the number
         vrs.Tokenize(svnr, from, "|");
         if (svnr.IsDigit()) {
            if (gDebug)
               Info("TProofProgressDialog", "svn revision run by the master: %s", svnr.Data());
            fSVNRev = svnr.Atoi();
         } else {
            Info("TProofProgressDialog", "could not find svn revision run by the master");
         }
      } else if (gDebug) {
         Info("TProofProgressDialog", "non-standard master version string:'%s'", vrs.Data());
      }
   } else {
      Warning("TProofProgressDialog", "list of active workers is empty!");
   }

   if (PPD_SRV_NEWER(11))
      fRatePoints = new TNtuple("RateNtuple","Rate progress info","tm:evr:mbr");

   fDialog = new TGTransientFrame(0, 0, 10, 10);
   fDialog->Connect("CloseWindow()", "TProofProgressDialog", this, "DoClose()");
   fDialog->DontCallClose();

   // Title label
   TString buf;
   buf = TString::Format("Executing on PROOF cluster \"%s\" with %d parallel workers:",
           fProof ? fProof->GetMaster() : "<dummy>",
           fProof ? fProof->GetParallel() : 0);
   fTitleLab = new TGLabel(fDialog, buf),
   fDialog->AddFrame(fTitleLab,
                     new TGLayoutHints(kLHintsNormal, 10, 10, 20, 0));
   buf = TString::Format("Selector: %s", selector);
   fSelector = new TGLabel(fDialog, buf);
   fDialog->AddFrame(fSelector,
                     new TGLayoutHints(kLHintsNormal, 10, 10, 5, 0));
   buf = TString::Format("%d files, number of events %lld, starting event %lld",
           fFiles, fEntries, fFirst);
   fFilesEvents = new TGLabel(fDialog, buf);
   fDialog->AddFrame(fFilesEvents, new TGLayoutHints(kLHintsNormal, 10, 10, 5, 0));

   // Progress bar
   fBar = new TGHProgressBar(fDialog, TGProgressBar::kFancy, 450);
   fBar->SetBarColor("green");
   fBar->UsePercent();
   fBar->ShowPos(kTRUE);
   fDialog->AddFrame(fBar, new TGLayoutHints(kLHintsTop | kLHintsLeft |
                     kLHintsExpandX, 10, 10, 20, 20));

   // Status labels
   if (PPD_SRV_NEWER(11)) {
      TGHorizontalFrame *hf0 = new TGHorizontalFrame(fDialog, 0, 0);
      TGCompositeFrame *cf0 = new TGCompositeFrame(hf0, 110, 0, kFixedWidth);
      cf0->AddFrame(new TGLabel(cf0, "Initialization time:"));
      hf0->AddFrame(cf0);
      fInit = new TGLabel(hf0, "- secs");
      hf0->AddFrame(fInit, new TGLayoutHints(kLHintsNormal, 10, 10, 0, 0));
      fDialog->AddFrame(hf0, new TGLayoutHints(kLHintsNormal, 10, 10, 5, 0));
   }

   TGHorizontalFrame *hf1 = new TGHorizontalFrame(fDialog, 0, 0);
   TGCompositeFrame *cf1 = new TGCompositeFrame(hf1, 110, 0, kFixedWidth);
   fProcessed = new TGLabel(cf1, "Estimated time left:");
   cf1->AddFrame(fProcessed);
   hf1->AddFrame(cf1);
   fTotal= new TGLabel(hf1, "- sec (- events of - processed)");
   hf1->AddFrame(fTotal, new TGLayoutHints(kLHintsNormal, 10, 10, 0, 0));
   fDialog->AddFrame(hf1, new TGLayoutHints(kLHintsNormal, 10, 10, 5, 0));

   TGHorizontalFrame *hf2 = new TGHorizontalFrame(fDialog, 0, 0);
   TGCompositeFrame *cf2 = new TGCompositeFrame(hf2, 110, 0, kFixedWidth);
   cf2->AddFrame(new TGLabel(cf2, "Processing rate:"));
   hf2->AddFrame(cf2);
   fRate = new TGLabel(hf2, "- events/sec");
   hf2->AddFrame(fRate, new TGLayoutHints(kLHintsNormal, 10, 10, 0, 0));
   fDialog->AddFrame(hf2, new TGLayoutHints(kLHintsNormal, 10, 10, 5, 0));

   // Keep toggle button
   fKeepToggle = new TGCheckButton(fDialog,
                    new TGHotString("Close dialog when processing is complete"));
   if (!fKeep) fKeepToggle->SetState(kButtonDown);
   fKeepToggle->Connect("Toggled(Bool_t)",
                        "TProofProgressDialog", this, "DoKeep(Bool_t)");
   fDialog->AddFrame(fKeepToggle, new TGLayoutHints(kLHintsNormal, 10, 10, 20, 0));

   // Stop, cancel and close buttons
   TGHorizontalFrame *hf3 = new TGHorizontalFrame(fDialog, 60, 20, kFixedWidth);

   UInt_t  nb1 = 0, width1 = 0, height1 = 0;

   fAsyn = new TGTextButton(hf3, "&Run in background");
   if (fProof->GetRemoteProtocol() >= 22 && fProof->IsSync()) {
      fAsyn->SetToolTipText("Continue running in the background (asynchronous mode), releasing the ROOT prompt");
   } else {
      fAsyn->SetToolTipText("Switch to asynchronous mode disabled: functionality not supported by the server");
      fAsyn->SetState(kButtonDisabled);
   }
   fAsyn->Connect("Clicked()", "TProofProgressDialog", this, "DoAsyn()");
   hf3->AddFrame(fAsyn, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 7, 7, 0, 0));
   height1 = TMath::Max(height1, fAsyn->GetDefaultHeight());
   width1  = TMath::Max(width1, fAsyn->GetDefaultWidth()); ++nb1;

   fStop = new TGTextButton(hf3, "&Stop");
   fStop->SetToolTipText("Stop processing, Terminate() will be executed");
   fStop->Connect("Clicked()", "TProofProgressDialog", this, "DoStop()");
   hf3->AddFrame(fStop, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 7, 7, 0, 0));
   height1 = TMath::Max(height1, fStop->GetDefaultHeight());
   width1  = TMath::Max(width1, fStop->GetDefaultWidth()); ++nb1;

   fAbort = new TGTextButton(hf3, "&Cancel");
   fAbort->SetToolTipText("Cancel processing, Terminate() will NOT be executed");
   fAbort->Connect("Clicked()", "TProofProgressDialog", this, "DoAbort()");
   hf3->AddFrame(fAbort, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 7, 7, 0, 0));
   height1 = TMath::Max(height1, fAbort->GetDefaultHeight());
   width1  = TMath::Max(width1, fAbort->GetDefaultWidth()); ++nb1;

   fClose = new TGTextButton(hf3, "&Close");
   fClose->SetToolTipText("Close this dialog");
   fClose->SetState(kButtonDisabled);
   fClose->Connect("Clicked()", "TProofProgressDialog", this, "DoClose()");
   hf3->AddFrame(fClose, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 7, 7, 0, 0));
   height1 = TMath::Max(height1, fClose->GetDefaultHeight());
   width1  = TMath::Max(width1, fClose->GetDefaultWidth()); ++nb1;

   fDialog->AddFrame(hf3, new TGLayoutHints(kLHintsBottom | kLHintsCenterX | kLHintsExpandX, 5, 5, 10, 5));

   UInt_t  nb2 = 0, width2 = 0, height2 = 0;
   TGHorizontalFrame *hf5 = new TGHorizontalFrame(fDialog, 60, 20, kFixedWidth);

   fLog = new TGTextButton(hf5, "&Show Logs");
   fLog->SetToolTipText("Show query log messages");
   fLog->Connect("Clicked()", "TProofProgressDialog", this, "DoLog()");
   hf5->AddFrame(fLog, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 7, 7, 0, 0));
   height2 = TMath::Max(height2, fLog->GetDefaultHeight());
   width2  = TMath::Max(width2, fLog->GetDefaultWidth()); ++nb2;

   if (PPD_SRV_NEWER(11)) {
      fRatePlot = new TGTextButton(hf5, "&Rate plot");
      fRatePlot->SetToolTipText("Show processing rate vs time");
      fRatePlot->SetState(kButtonDisabled);
      fRatePlot->Connect("Clicked()", "TProofProgressDialog", this, "DoPlotRateGraph()");
      hf5->AddFrame(fRatePlot, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 7, 7, 0, 0));
      height2 = TMath::Max(height2, fRatePlot->GetDefaultHeight());
      width2  = TMath::Max(width2, fRatePlot->GetDefaultWidth()); ++nb2;
   }

   fMemPlot = new TGTextButton(hf5, "Memory Plot");
   fMemPlot->Connect("Clicked()", "TProofProgressDialog", this, "DoMemoryPlot()");
   fMemPlot->SetToolTipText("Show memory consumption vs entry / merging phase");
   hf5->AddFrame(fMemPlot, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 10, 10, 0, 0));
   height2 = TMath::Max(height2, fMemPlot->GetDefaultHeight());
   width2  = TMath::Max(width2, fMemPlot->GetDefaultWidth()); ++nb2;

   fDialog->AddFrame(hf5, new TGLayoutHints(kLHintsBottom | kLHintsCenterX | kLHintsExpandX, 5, 5, 10, 5));

   // Only enable if master supports it
   if (!PPD_SRV_NEWER(18)) {
      fMemPlot->SetState(kButtonDisabled);
      TString tip = TString::Format("Not supported by the master: required protocol 19 > %d",
                                    (fProof ? fProof->GetRemoteProtocol() : -1));
      fMemPlot->SetToolTipText(tip.Data());
   } else {
      fMemPlot->SetToolTipText("Show memory consumption");
   }

   // Keep buttons centered and with the same width
   UInt_t width, height, nb;
   width = TMath::Max(width1, width2);
   height = TMath::Max(height1, height2);
   nb = TMath::Max(nb1, nb2);

   // Connect slot to proof progress signal
   if (fProof) {
      fProof->Connect("Progress(Long64_t,Long64_t)", "TProofProgressDialog",
                      this, "Progress(Long64_t,Long64_t)");
      fProof->Connect("Progress(Long64_t,Long64_t,Long64_t,Float_t,Float_t,Float_t,Float_t)",
                      "TProofProgressDialog", this,
                      "Progress(Long64_t,Long64_t,Long64_t,Float_t,Float_t,Float_t,Float_t)");
      fProof->Connect("StopProcess(Bool_t)", "TProofProgressDialog", this,
                      "IndicateStop(Bool_t)");
      fProof->Connect("ResetProgressDialog(const char*,Int_t,Long64_t,Long64_t)",
                      "TProofProgressDialog", this,
                      "ResetProgressDialog(const char*,Int_t,Long64_t,Long64_t)");
      fProof->Connect("CloseProgressDialog()", "TProofProgressDialog", this, "DoClose()");
      fProof->Connect("DisableGoAsyn()", "TProofProgressDialog", this, "DisableAsyn()");
   }

   // Set dialog title
   if (fProof) {
      if (strlen(fProof->GetUser()) > 0) 
         fDialog->SetWindowName(Form("PROOF Query Progress: %s@%s",
                                     fProof->GetUser(), fProof->GetMaster()));
      else
         fDialog->SetWindowName(Form("PROOF Query Progress: %s", fProof->GetMaster()));
   } else
      fDialog->SetWindowName("PROOF Query Progress: <dummy>");

   // Map all widgets and calculate size of dialog
   fDialog->MapSubwindows();

   width  = fDialog->GetDefaultWidth();
   height = fDialog->GetDefaultHeight();

   // To allow for lengthening of lines when the number are displayed
   width += 100;
   fDialog->Resize(width, height);

   const TGWindow *main = gClient->GetRoot();
   // Position relative to the parent window (which is the root window)
   Window_t wdum;
   int      ax, ay;
   Int_t    mw = ((TGFrame *) main)->GetWidth();
   Int_t    mh = ((TGFrame *) main)->GetHeight();

   gVirtualX->TranslateCoordinates(main->GetId(), main->GetId(),
                          (mw - width), (mh - height) >> 1, ax, ay, wdum);

   // Make the message box non-resizable
   fDialog->SetWMSize(width, height);
   fDialog->SetWMSizeHints(width, height, width, height, 0, 0);

   fDialog->SetMWMHints(kMWMDecorAll | kMWMDecorResizeH  | kMWMDecorMaximize |
                                       kMWMDecorMinimize | kMWMDecorMenu,
                        kMWMFuncAll  | kMWMFuncResize    | kMWMFuncMaximize |
                                       kMWMFuncMinimize,
                        kMWMInputModeless);

   fDialog->Move(ax-10, ay - mh/4);
   fDialog->SetWMPosition(ax-10, ay - mh/4);
   // Popup dialog and wait till user replies
   fDialog->MapWindow();

   fStartTime = gSystem->Now();
}

//______________________________________________________________________________
void TProofProgressDialog::ResetProgressDialog(const char *selec,
                                               Int_t files, Long64_t first,
                                               Long64_t entries)
{
   // Reset dialog box preparing for new query
   TString buf;

   // Update title
   buf = TString::Format("Executing on PROOF cluster \"%s\" with %d parallel workers:",
           fProof ? fProof->GetMaster() : "<dummy>",
           fProof ? fProof->GetParallel() : 0);
   fTitleLab->SetText(buf);

   // Reset members
   fFiles         = files;
   fFirst         = first;
   fEntries       = entries;
   fPrevProcessed = 0;
   fPrevTotal     = 0;
   fStatus        = kRunning;

   // Update selector name
   buf = TString::Format("Selector: %s", selec);
   fSelector->SetText(buf);

   // Reset 'processed' text
   fProcessed->SetText("Estimated time left:");

   // Update numbers
   buf = TString::Format("%d files, number of events %lld, starting event %lld",
           fFiles, fEntries, fFirst);
   fFilesEvents->SetText(buf);

   // Reset progress bar
   fBar->SetBarColor("green");
   fBar->Reset();

   // Reset buttons
   fStop->SetState(kButtonUp);
   fAbort->SetState(kButtonUp);
   fClose->SetState(kButtonDisabled);
   if (fProof->IsSync() && fProof->GetRemoteProtocol() >= 22) {
      fAsyn->SetState(kButtonUp);
   } else {
      fAsyn->SetState(kButtonDisabled);
   }

   // Reconnect the slots
   if (fProof) {
      fProof->Connect("Progress(Long64_t,Long64_t)", "TProofProgressDialog",
                      this, "Progress(Long64_t,Long64_t)");
      fProof->Connect("Progress(Long64_t,Long64_t,Long64_t,Float_t,Float_t,Float_t,Float_t)",
                      "TProofProgressDialog", this,
                      "Progress(Long64_t,Long64_t,Long64_t,Float_t,Float_t,Float_t,Float_t)");
      fProof->Connect("StopProcess(Bool_t)", "TProofProgressDialog", this,
                      "IndicateStop(Bool_t)");
      fProof->Connect("DisableGoAsyn()", "TProofProgressDialog", this, "DisableAsyn()");
   }

   // Reset start time
   fStartTime = gSystem->Now();

   // Clear the list of performances points
   if (PPD_SRV_NEWER(11))
      fRatePoints->Reset();
   SafeDelete(fRateGraph);
   fAvgRate = 0.;
   fAvgMBRate = 0.;
}

//______________________________________________________________________________
void TProofProgressDialog::Progress(Long64_t total, Long64_t processed)
{
   // Update progress bar and status labels.
   // Use "processed == total" or "processed < 0" to indicate end of processing.

   Long_t tt;
   UInt_t hh=0, mm=0, ss=0;
   TString buf;
   char stm[256];
   static const char *cproc[] = { "running", "done",
                                  "STOPPED", "ABORTED", "***EVENTS SKIPPED***"};

   // Update title
   buf = TString::Format("Executing on PROOF cluster \"%s\" with %d parallel workers:",
           fProof ? fProof->GetMaster() : "<dummy>",
           fProof ? fProof->GetParallel() : 0);
   fTitleLab->SetText(buf);

   if (total < 0)
      total = fPrevTotal;
   else
      fPrevTotal = total;

   // Nothing to update
   if (fPrevProcessed == processed)
      return;

   // Number of processed events
   Long64_t evproc = (processed >= 0) ? processed : fPrevProcessed;

   if (fEntries != total) {
      fEntries = total;
      buf = TString::Format("%d files, number of events %lld, starting event %lld",
              fFiles, fEntries, fFirst);
      fFilesEvents->SetText(buf);
   }

   // Update position
   Float_t pos = Float_t(Double_t(evproc * 100)/Double_t(total));
   fBar->SetPosition(pos);

   // get current time
   fEndTime = gSystem->Now();
   TTime tdiff = fEndTime - fStartTime;
   Float_t eta = 0;
   if (evproc > 0)
      eta = ((Float_t)((Long_t)tdiff)*total/Float_t(evproc) - Long_t(tdiff))/1000.;

   if (processed >= 0 && processed >= total) {
      tt = (Long_t(tdiff)/1000);
      if (tt > 0) {
         hh = (UInt_t)(tt / 3600);
         mm = (UInt_t)((tt % 3600) / 60);
         ss = (UInt_t)((tt % 3600) % 60);
      }
      if (hh)
         sprintf(stm, "%d h %d min %d sec", hh, mm, ss);
      else if (mm)
         sprintf(stm, "%d min %d sec", mm, ss);
      else
         sprintf(stm, "%d sec", ss);
      fProcessed->SetText("Processed:");
      buf = TString::Format("%lld events in %s", total, stm);
      fTotal->SetText(buf);

      if (fProof) {
         fProof->Disconnect("Progress(Long64_t,Long64_t)", this,
                            "Progress(Long64_t,Long64_t)");
         fProof->Disconnect("StopProcess(Bool_t)", this,
                            "IndicateStop(Bool_t)");
         fProof->Disconnect("DisableGoAsyn()", this, "DisableAsyn()");
      }

      // Set button state
      fAsyn->SetState(kButtonDisabled);
      fStop->SetState(kButtonDisabled);
      fAbort->SetState(kButtonDisabled);
      fClose->SetState(kButtonUp);
      if (!fKeep) DoClose();

      // Set the status to done
      fStatus = kDone;

   } else {
      // A negative value for process indicates that we are finished,
      // no matter whether the processing was complete
      Bool_t incomplete = (processed < 0 &&
                          (fPrevProcessed < total || fPrevProcessed == 0))
                        ? kTRUE : kFALSE;
      if (incomplete) {
         fStatus = kIncomplete;
         // We use a different color to highlight incompletion
         fBar->SetBarColor("magenta");
      }
      tt = (Long_t)eta;
      if (tt > 0) {
         hh = (UInt_t)(tt / 3600);
         mm = (UInt_t)((tt % 3600) / 60);
         ss = (UInt_t)((tt % 3600) % 60);
      }
      if (hh)
         sprintf(stm, "%d h %d min %d sec", hh, mm, ss);
      else if (mm)
         sprintf(stm, "%d min %d sec", mm, ss);
      else
         sprintf(stm, "%d sec", ss);
      if (fStatus > kDone) {
         buf = TString::Format("%s (%lld events of %lld processed) - %s",
                      stm, evproc, total, cproc[fStatus]);
      } else {
         buf = TString::Format("%s (%lld events of %lld processed)",
                      stm, evproc, total);
      }
      fTotal->SetText(buf);
      buf = TString::Format("%.1f events/sec", Float_t(evproc)/Long_t(tdiff)*1000.);
      fRate->SetText(buf);

      if (processed < 0) {
         // And we disable the buttons
         fAsyn->SetState(kButtonDisabled);
         fStop->SetState(kButtonDisabled);
         fAbort->SetState(kButtonDisabled);
         fClose->SetState(kButtonUp);

         // Set the status to done
         fStatus = kDone;
      }
   }
   fPrevProcessed = evproc;

   fDialog->Layout();
}

//______________________________________________________________________________
void TProofProgressDialog::Progress(Long64_t total, Long64_t processed,
                                    Long64_t bytesread,
                                    Float_t initTime, Float_t procTime,
                                    Float_t evtrti, Float_t mbrti)
{
   // Update progress bar and status labels.
   // Use "processed == total" or "processed < 0" to indicate end of processing.

   Long_t tt;
   UInt_t hh=0, mm=0, ss=0;
   TString buf;
   char stm[256];
   static const char *cproc[] = { "running", "done",
                                  "STOPPED", "ABORTED", "***EVENTS SKIPPED***"};

   // Update title
   buf = TString::Format("Executing on PROOF cluster \"%s\" with %d parallel workers:",
           fProof ? fProof->GetMaster() : "<dummy>",
           fProof ? fProof->GetParallel() : 0);
   fTitleLab->SetText(buf);

   if (initTime >= 0.) {
      // Set init time
      buf = TString::Format("%.1f secs", initTime);
      fInit->SetText(buf);
      fDialog->Layout();
   }

   Bool_t over = kFALSE;
   if (total < 0) {
      total = fPrevTotal;
      over = kTRUE;
   } else {
      fPrevTotal = total;
   }

   // Nothing to update
   if (fPrevProcessed == processed)
      return;

   // Number of processed events
   Long64_t evproc = (processed >= 0) ? processed : fPrevProcessed;
   Float_t mbsproc = bytesread / TMath::Power(2.,20.);

   if (fEntries != total) {
      fEntries = total;
      buf = TString::Format("%d files, number of events %lld, starting event %lld",
              fFiles, fEntries, fFirst);
      fFilesEvents->SetText(buf);
   }

   // Update position
   Float_t pos = Float_t(Double_t(evproc * 100)/Double_t(total));
   fBar->SetPosition(pos);

   Float_t eta = 0;
   if (evproc > 0 && procTime > 0.)
      eta = (Float_t) (total - evproc) / (Double_t)evproc * procTime;

   // Update average rates
   if (procTime > 0.) {
      fProcTime = procTime;
      fAvgRate = Float_t(evproc) / procTime;
      fAvgMBRate = mbsproc / procTime;
   }

   if (over || (processed >= 0 && processed >= total)) {

      // A negative value for process indicates that we are finished,
      // no matter whether the processing was complete
      Bool_t incomplete = (processed < 0 &&
                          (fPrevProcessed < total || fPrevProcessed == 0))
                        ? kTRUE : kFALSE;
      TString st = "";
      if (incomplete) {
         fStatus = kIncomplete;
         // We use a different color to highlight incompletion
         fBar->SetBarColor("magenta");
         st = TString::Format(" %s", cproc[fStatus]);
      }

      tt = (Long_t)fProcTime;
      if (tt > 0) {
         hh = (UInt_t)(tt / 3600);
         mm = (UInt_t)((tt % 3600) / 60);
         ss = (UInt_t)((tt % 3600) % 60);
      }
      if (hh)
         sprintf(stm, "%d h %d min %d sec", hh, mm, ss);
      else if (mm)
         sprintf(stm, "%d min %d sec", mm, ss);
      else
         sprintf(stm, "%d sec", ss);
      fProcessed->SetText("Processed:");
      buf = TString::Format("%lld events (%.2f MBs) in %s %s",
              std::max(fPrevProcessed, processed), fAvgMBRate*fProcTime, stm, st.Data());
      fTotal->SetText(buf);
      buf = TString::Format("%.1f evts/sec (%.1f MBs/sec)", fAvgRate, fAvgMBRate);
      fRate->SetText(buf);
      // Fill rate graph
      Bool_t useAvg = gEnv->GetValue("Proof.RatePlotUseAvg", 0);
      if (useAvg) {
         if (fAvgRate > 0.) {
            fRatePoints->Fill(procTime, fAvgRate, fAvgMBRate);
            fRatePlot->SetState(kButtonUp);
         }
      } else {
         if (evtrti > 0.) {
            fRatePoints->Fill(procTime, evtrti, mbrti);
            fRatePlot->SetState(kButtonUp);
         }
      }

      if (fProof) {
         fProof->Disconnect("Progress(Long64_t,Long64_t)", this,
                            "Progress(Long64_t,Long64_t)");
         fProof->Disconnect("Progress(Long64_t,Long64_t,Long64_t,Float_t,Float_t,Float_t,Float_t)",
                            this,
                            "Progress(Long64_t,Long64_t,Long64_t,Float_t,Float_t,Float_t,Float_t)");
         fProof->Disconnect("StopProcess(Bool_t)", this, "IndicateStop(Bool_t)");
         fProof->Disconnect("DisableGoAsyn()", this, "DisableAsyn()");
      }

      // Set button state
      fAsyn->SetState(kButtonDisabled);
      fStop->SetState(kButtonDisabled);
      fAbort->SetState(kButtonDisabled);
      fClose->SetState(kButtonUp);
      if (!fKeep) DoClose();

      // Set the status to done
      fStatus = kDone;

   } else {
      // A negative value for process indicates that we are finished,
      // no matter whether the processing was complete
      Bool_t incomplete = (processed < 0 &&
                          (fPrevProcessed < total || fPrevProcessed == 0))
                        ? kTRUE : kFALSE;
      if (incomplete) {
         fStatus = kIncomplete;
         // We use a different color to highlight incompletion
         fBar->SetBarColor("magenta");
      }
      tt = (Long_t)eta;
      if (tt > 0) {
         hh = (UInt_t)(tt / 3600);
         mm = (UInt_t)((tt % 3600) / 60);
         ss = (UInt_t)((tt % 3600) % 60);
      }
      if (hh)
         sprintf(stm, "%d h %d min %d sec", hh, mm, ss);
      else if (mm)
         sprintf(stm, "%d min %d sec", mm, ss);
      else
         sprintf(stm, "%d sec", ss);
      if (fStatus > kDone) {
         buf = TString::Format("%s (processed %lld events out of %lld - %.2f MBs of data) - %s",
                      stm, evproc, total, mbsproc, cproc[fStatus]);
      } else {
         buf = TString::Format("%s (processed %lld events out of %lld - %.2f MBs of data)",
                      stm, evproc, total, mbsproc);
      }
      fTotal->SetText(buf);

      // Post
      if (evtrti > 0.) {
         buf = TString::Format("%.1f evts/sec (%.1f MBs/sec) - avg: %.1f evts/sec (%.1f MBs/sec)",
                      evtrti, mbrti, fAvgRate, fAvgMBRate);
         fRatePoints->Fill(procTime, evtrti, mbrti);
         fRatePlot->SetState(kButtonUp);
      } else {
         buf = TString::Format("avg: %.1f evts/sec (%.1f MBs/sec)", fAvgRate, fAvgMBRate);
      }
      fRate->SetText(buf);

      if (processed < 0) {
         // And we disable the buttons
         fAsyn->SetState(kButtonDisabled);
         fStop->SetState(kButtonDisabled);
         fAbort->SetState(kButtonDisabled);
         fClose->SetState(kButtonUp);

         // Set the status to done
         fStatus = kDone;
      }
   }
   fPrevProcessed = evproc;

   fDialog->Layout();
}

//______________________________________________________________________________
TProofProgressDialog::~TProofProgressDialog()
{
   // Cleanup dialog.

   if (fProof) {
      fProof->Disconnect("Progress(Long64_t,Long64_t)", this,
                         "Progress(Long64_t,Long64_t)");
      fProof->Disconnect("Progress(Long64_t,Long64_t,Long64_t,Float_t,Float_t,Float_t,Float_t)",
                         this,
                         "Progress(Long64_t,Long64_t,Long64_t,Float_t,Float_t,Float_t,Float_t)");
      fProof->Disconnect("StopProcess(Bool_t)", this, "IndicateStop(Bool_t)");
      fProof->Disconnect("DisableGoAsyn()", this, "DisableAsyn()");
      fProof->Disconnect("ResetProgressDialog(const char*,Int_t,Long64_t,Long64_t)",
                         this,
                         "ResetProgressDialog(const char*,Int_t,Long64_t,Long64_t)");
      fProof->Disconnect("CloseProgressDialog()", this, "CloseProgressDialog()");
      fProof->ResetProgressDialogStatus();
      // We are called after a TProofDetach: we delete the instance
      if (!fProof->IsValid())
         SafeDelete(fProof);
   }
   if (fLogWindow)
      delete fLogWindow;
   if (fMemWindow)
      delete fMemWindow;
   fDialog->Cleanup();
   delete fDialog;
}

//______________________________________________________________________________
void TProofProgressDialog::CloseWindow()
{
   // Called when dialog is closed.

   delete this;
}

//______________________________________________________________________________
void TProofProgressDialog::DisableAsyn()
{
   // Disable the asyn switch when an external request for going asynchronous is issued

   fProof->Disconnect("DisableGoAsyn()", this, "DisableAsyn()");
   fAsyn->SetState(kButtonDisabled);
}

//______________________________________________________________________________
void TProofProgressDialog::IndicateStop(Bool_t aborted)
{
   // Indicate that Cancel or Stop was clicked.

   if (aborted == kTRUE)
      fBar->SetBarColor("red");
   else
      fBar->SetBarColor("yellow");

   if (fProof) {
      fProof->Disconnect("Progress(Long64_t,Long64_t)", this,
                         "Progress(Long64_t,Long64_t)");
      fProof->Disconnect("Progress(Long64_t,Long64_t,Long64_t,Float_t,Float_t,Float_t,Float_t)",
                         this,
                         "Progress(Long64_t,Long64_t,Long64_t,Float_t,Float_t,Float_t,Float_t)");
      fProof->Disconnect("StopProcess(Bool_t)", this, "IndicateStop(Bool_t)");
      fProof->Disconnect("DisableGoAsyn()", this, "DisableAsyn()");
      // These buttons are meaningless at this point
      fAsyn->SetState(kButtonDisabled);
      fStop->SetState(kButtonDisabled);
      fAbort->SetState(kButtonDisabled);
   }

   fClose->SetState(kButtonUp);
   if (!fKeep)
      DoClose();
}

//______________________________________________________________________________
void TProofProgressDialog::LogMessage(const char *msg, Bool_t all)
{
   // Load/append a log msg in the log frame, if open

   if (fLogWindow) {
      if (all) {
         // load buffer
         fLogWindow->LoadBuffer(msg);
      } else {
         // append
         fLogWindow->AddBuffer(msg);
      }
   }
}

//______________________________________________________________________________
void TProofProgressDialog::DoClose()
{
   // Close dialog.

   fClose->SetState(kButtonDisabled);
   TTimer::SingleShot(50, "TProofProgressDialog", this, "CloseWindow()");
}

//______________________________________________________________________________
void TProofProgressDialog::DoLog()
{
   // Ask proof session for logs

   if (fProof) {
      if (!fLogWindow) {
         fLogWindow = new TProofProgressLog(this);
         if (!fLogWindow->TestBit(TObject::kInvalidObject))
            fLogWindow->DoLog();
      } else {
         // Clear window
         if (!fLogWindow->TestBit(TObject::kInvalidObject)) {
            fLogWindow->Clear();
            fLogWindow->DoLog();
         }
      }
   }
}

//______________________________________________________________________________
void TProofProgressDialog::DoKeep(Bool_t)
{
   // Handle keep toggle button.

   fKeep = !fKeep;

   // Last choice will be the default for the future
   fgKeepDefault = fKeep;
}

//______________________________________________________________________________
void TProofProgressDialog::DoSetLogQuery(Bool_t)
{
   // Handle log-current-query-only toggle button.

   fLogQuery = !fLogQuery;
   fEntry->SetEnabled(fLogQuery);
   if (fLogQuery)
      fEntry->SetToolTipText("Enter the query number ('last' for the last query)",50);
   else
      fEntry->SetToolTipText(0);

   // Last choice will be the default for the future
   fgLogQueryDefault = fLogQuery;
}

//______________________________________________________________________________
void TProofProgressDialog::DoStop()
{
   // Handle Stop button.

   // Do not wait for ever, but al least 10 seconds
   Long_t timeout = gEnv->GetValue("Proof.ShutdownTimeout", 60) / 2;
   timeout = (timeout > 10) ? timeout : 10;
   fProof->StopProcess(kFALSE, timeout);
   fStatus = kStopped;

   // Set buttons states
   fAsyn->SetState(kButtonDisabled);
   fStop->SetState(kButtonDisabled);
   fAbort->SetState(kButtonDisabled);
   fClose->SetState(kButtonUp);
}

//______________________________________________________________________________
void TProofProgressDialog::DoAbort()
{
   // Handle Cancel button.

   fProof->StopProcess(kTRUE);
   fStatus = kAborted;

   // Set buttons states
   fAsyn->SetState(kButtonDisabled);
   fStop->SetState(kButtonDisabled);
   fAbort->SetState(kButtonDisabled);
   fClose->SetState(kButtonUp);
}

//______________________________________________________________________________
void TProofProgressDialog::DoAsyn()
{
   // Handle Asyn button.

   fProof->GoAsynchronous();

   // Set buttons states
   fAsyn->SetState(kButtonDisabled);
}

//______________________________________________________________________________
void TProofProgressDialog::DoPlotRateGraph()
{
   // Handle Plot Rate Graph.

   // We must have some point to plot
   if (!fRatePoints || fRatePoints->GetEntries() <= 0) {
      Info("DoPlotRateGraph","list is empty!");
      return;
   }

   // Create a canvas
   TCanvas *c1 = new TCanvas("c1","Rate vs Time",200,10,700,500);
   c1->SetFillColor(0);
   c1->SetGrid();
   c1->SetBorderMode(0);
   c1->SetFrameBorderMode(0);

   // Fill TGraph
   Int_t np = (Int_t)fRatePoints->GetEntries();
   Double_t ymx = -1.;
   SafeDelete(fRateGraph);
   fRateGraph = new TGraph(np);
   Float_t *nar = fRatePoints->GetArgs();
   Int_t ii = 0;
   for ( ; ii < np; ++ii) {
      fRatePoints->GetEntry(ii);
      fRateGraph->SetPoint(ii, (Double_t) nar[0], (Double_t) nar[1]);
      ymx = (nar[1] > ymx) ? nar[1] : ymx;
   }

   fRateGraph->SetMinimum(0.);
   fRateGraph->SetMaximum(ymx*1.1);
   fRateGraph->SetLineColor(2);
   fRateGraph->SetLineWidth(4);
   fRateGraph->SetMarkerColor(4);
   fRateGraph->SetMarkerStyle(21);
   fRateGraph->SetTitle("Processing rate (evts/sec)");
   fRateGraph->GetXaxis()->SetTitle("elapsed time (sec)");
   fRateGraph->Draw("ALP");

   // Line with average
   TLine *line = new TLine(fRateGraph->GetXaxis()->GetXmin(),fAvgRate,
                           fRateGraph->GetXaxis()->GetXmax(),fAvgRate);
   Int_t ci;   // for color index setting
   ci = TColor::GetColor("#008200");
   line->SetLineColor(ci);
   line->SetLineWidth(2);
   line->Draw("P");

   // Label
   Double_t xax0 = fRateGraph->GetXaxis()->GetXmin();
   Double_t xax1 = fRateGraph->GetXaxis()->GetXmax();
   Double_t yax0 = 0.;
   Double_t yax1 = ymx*1.1;
   Double_t x0 = xax0 + 0.05 * (xax1 - xax0);
   Double_t x1 = xax0 + 0.60 * (xax1 - xax0);
   Double_t y0 = yax0 + 0.10 * (yax1 - yax0);
   Double_t y1 = yax0 + 0.20 * (yax1 - yax0);
   TPaveText *pt = new TPaveText(x0, y0, x1, y1, "br");
   pt->SetFillColor(0);
   pt->AddText(Form("Global average: %.2f evts/sec", fAvgRate));
   pt->Draw();

   c1->Modified();
}

//______________________________________________________________________________
void TProofProgressDialog::DoMemoryPlot()
{
   // Do a memory plot

   if (!fMemWindow) {
      fMemWindow = new TProofProgressMemoryPlot(this, 500, 300);
      fMemWindow->DoPlot();
   } else {
      // Clear window
      fMemWindow->Clear();
      fMemWindow->DoPlot();
   }
}
