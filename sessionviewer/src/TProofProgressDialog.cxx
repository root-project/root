// @(#)root/proof:$Name:  $:$Id: TProofProgressDialog.cxx,v 1.22 2006/06/05 22:51:13 rdm Exp $
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
#include "TEnv.h"
#include "TError.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TGTextBuffer.h"
#include "TGTextEntry.h"
#include "TGProgressBar.h"
#include "TProof.h"
#include "TSystem.h"
#include "TTimer.h"


Bool_t TProofProgressDialog::fgKeepDefault = kTRUE;
Bool_t TProofProgressDialog::fgLogQueryDefault = kFALSE;
TString TProofProgressDialog::fgTextQueryDefault = "last";


ClassImp(TProofProgressDialog)

//______________________________________________________________________________
TProofProgressDialog::TProofProgressDialog(TVirtualProof *proof,
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
   fStatus        = kRunning;
   fKeep          = fgKeepDefault;
   fLogQuery      = fgLogQueryDefault;

   const TGWindow *main = gClient->GetRoot();
   fDialog = new TGTransientFrame(main, main, 10, 10);
   fDialog->Connect("CloseWindow()", "TProofProgressDialog", this, "DoClose()");
   fDialog->DontCallClose();

   // title label
   char buf[256];
   sprintf(buf, "Executing on PROOF cluster \"%s\" with %d parallel workers:",
           fProof ? fProof->GetMaster() : "<dummy>",
           fProof ? fProof->GetParallel() : 0);
   fTitleLab = new TGLabel(fDialog, buf),
   fDialog->AddFrame(fTitleLab,
                     new TGLayoutHints(kLHintsNormal, 10, 10, 20, 0));
   sprintf(buf,"Selector: %s", selector);
   fSelector = new TGLabel(fDialog, buf);
   fDialog->AddFrame(fSelector,
                     new TGLayoutHints(kLHintsNormal, 10, 10, 5, 0));
   sprintf(buf, "%d files, number of events %lld, starting event %lld",
           fFiles, fEntries, fFirst);
   fFilesEvents = new TGLabel(fDialog, buf);
   fDialog->AddFrame(fFilesEvents, new TGLayoutHints(kLHintsNormal, 10, 10, 5, 0));

   // progress bar
   fBar = new TGHProgressBar(fDialog, TGProgressBar::kFancy, 450);
   fBar->SetBarColor("green");
   fDialog->AddFrame(fBar, new TGLayoutHints(kLHintsTop | kLHintsLeft |
                     kLHintsExpandX, 10, 10, 20, 20));

   // status labels
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

   // logs-from-given-query-only toggle button
   TGHorizontalFrame *hflog = new TGHorizontalFrame(fDialog, 200, 20, kFixedWidth);
   fLogQueryToggle = new TGCheckButton(hflog,
                        new TGHotString("Show only logs from query"));
   if (!fLogQuery) fLogQueryToggle->SetState(kButtonUp);
   fLogQueryToggle->Connect("Toggled(Bool_t)",
                            "TProofProgressDialog", this, "DoSetLogQuery(Bool_t)");
   hflog->AddFrame(fLogQueryToggle,
                   new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 0, 0, 0, 0));
   // text entry for logs-from-given-query-only toggle button
   fTextQuery = new TGTextBuffer();
   fTextQuery->AddText(0, fgTextQueryDefault);
   fEntry = new TGTextEntry(hflog, fTextQuery);
   if (fLogQuery)
      fEntry->SetToolTipText("Enter the query number ('last' for the last query)",50);
   fEntry->SetEnabled(fLogQuery);
   hflog->AddFrame(fEntry, new TGLayoutHints(kLHintsCenterY |
                                             kLHintsExpandX, 2, 0, 0, 0));

   fDialog->AddFrame(hflog, new TGLayoutHints(kLHintsNormal, 10, 10, 5, 0));

   // stop, cancel and close buttons
   TGHorizontalFrame *hf3 = new TGHorizontalFrame(fDialog, 60, 20, kFixedWidth);

   UInt_t  nb = 0, width = 0, height = 0;

   fStop = new TGTextButton(hf3, "&Stop");
   fStop->SetToolTipText("Stop processing, Terminate() will be executed");
   fStop->Connect("Clicked()", "TProofProgressDialog", this, "DoStop()");
   hf3->AddFrame(fStop, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 7, 7, 0, 0));
   height = TMath::Max(height, fStop->GetDefaultHeight());
   width  = TMath::Max(width, fStop->GetDefaultWidth()); ++nb;

   fAbort = new TGTextButton(hf3, "&Cancel");
   fAbort->SetToolTipText("Cancel processing, Terminate() will NOT be executed");
   fAbort->Connect("Clicked()", "TProofProgressDialog", this, "DoAbort()");
   hf3->AddFrame(fAbort, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 7, 7, 0, 0));
   height = TMath::Max(height, fAbort->GetDefaultHeight());
   width  = TMath::Max(width, fAbort->GetDefaultWidth()); ++nb;

   fClose = new TGTextButton(hf3, "&Close");
   fClose->SetToolTipText("Close this dialog");
   fClose->SetState(kButtonDisabled);
   fClose->Connect("Clicked()", "TProofProgressDialog", this, "DoClose()");
   hf3->AddFrame(fClose, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 7, 7, 0, 0));
   height = TMath::Max(height, fClose->GetDefaultHeight());
   width  = TMath::Max(width, fClose->GetDefaultWidth()); ++nb;

   fLog = new TGTextButton(hf3, "&Show Logs");
   fLog->SetToolTipText("Show query log messages");
   fLog->Connect("Clicked()", "TProofProgressDialog", this, "DoLog()");
   hf3->AddFrame(fLog, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 7, 7, 0, 0));
   height = TMath::Max(height, fLog->GetDefaultHeight());
   width  = TMath::Max(width, fLog->GetDefaultWidth()); ++nb;

   // place button frame (hf3) at the bottom
   fDialog->AddFrame(hf3, new TGLayoutHints(kLHintsBottom | kLHintsCenterX, 10, 10, 20, 10));

   // keep buttons centered and with the same width
   hf3->Resize((width + 40) * nb, height);

   // connect slot to proof progress signal
   if (fProof) {
      fProof->Connect("Progress(Long64_t,Long64_t)", "TProofProgressDialog",
                      this, "Progress(Long64_t,Long64_t)");
      fProof->Connect("StopProcess(Bool_t)", "TProofProgressDialog", this,
                      "IndicateStop(Bool_t)");
      fProof->Connect("ResetProgressDialog(const char*,Int_t,Long64_t,Long64_t)",
                      "TProofProgressDialog", this,
                      "ResetProgressDialog(const char*,Int_t,Long64_t,Long64_t)");
      fProof->Connect("CloseProgressDialog()", "TProofProgressDialog", this, "DoClose()");
   }

   // set dialog title
   fDialog->SetWindowName(Form("PROOF Query Progress: %s",
                          (fProof ? fProof->GetMaster() : "<dummy>")));

   // map all widgets and calculate size of dialog
   fDialog->MapSubwindows();

   width  = fDialog->GetDefaultWidth();
   height = fDialog->GetDefaultHeight();

   fDialog->Resize(width, height);

   // position relative to the parent window (which is the root window)
   Window_t wdum;
   int      ax, ay;
   Int_t    mw = ((TGFrame *) main)->GetWidth();
   Int_t    mh = ((TGFrame *) main)->GetHeight();

   gVirtualX->TranslateCoordinates(main->GetId(), main->GetId(),
                          (mw - width), (mh - height) >> 1, ax, ay, wdum);
   fDialog->Move(ax-5, ay - mh/4);
   fDialog->SetWMPosition(ax-5, ay - mh/4);

   // make the message box non-resizable
   fDialog->SetWMSize(width, height);
   fDialog->SetWMSizeHints(width, height, width, height, 0, 0);

   fDialog->SetMWMHints(kMWMDecorAll | kMWMDecorResizeH  | kMWMDecorMaximize |
                                       kMWMDecorMinimize | kMWMDecorMenu,
                        kMWMFuncAll  | kMWMFuncResize    | kMWMFuncMaximize |
                                       kMWMFuncMinimize,
                        kMWMInputModeless);

   // popup dialog and wait till user replies
   fDialog->MapWindow();

   fStartTime = gSystem->Now();

   //gClient->WaitFor(fDialog);
}

//______________________________________________________________________________
void TProofProgressDialog::ResetProgressDialog(const char *selec,
                                               Int_t files, Long64_t first,
                                               Long64_t entries)
{
   // Reset dialog box preparing for new query
   char buf[512];

   // Update title
   sprintf(buf, "Executing on PROOF cluster \"%s\" with %d parallel workers:",
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
   sprintf(buf,"Selector: %s", selec);
   fSelector->SetText(buf);

   // Update numbers
   sprintf(buf, "%d files, number of events %lld, starting event %lld",
           fFiles, fEntries, fFirst);
   fFilesEvents->SetText(buf);

   // Reset progress bar
   fBar->SetBarColor("green");
   fBar->Reset();

   // Reset buttons
   fStop->SetState(kButtonUp);
   fAbort->SetState(kButtonUp);
   fClose->SetState(kButtonDisabled);

   // Reconnect the slots
   if (fProof) {
      fProof->Connect("Progress(Long64_t,Long64_t)", "TProofProgressDialog",
                      this, "Progress(Long64_t,Long64_t)");
      fProof->Connect("StopProcess(Bool_t)", "TProofProgressDialog", this,
                      "IndicateStop(Bool_t)");
   }

   // Reset start time
   fStartTime = gSystem->Now();
}

//______________________________________________________________________________
void TProofProgressDialog::Progress(Long64_t total, Long64_t processed)
{
   // Update progress bar and status labels.
   // Use "processed == total" or "processed < 0" to indicate end of processing.

   char buf[256];
   static const char *cproc[] = { "running", "done",
                                  "STOPPED", "ABORTED", "***EVENTS SKIPPED***"};

   // Update title
   sprintf(buf, "Executing on PROOF cluster \"%s\" with %d parallel workers:",
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
      sprintf(buf, "%d files, number of events %lld, starting event %lld",
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
      fProcessed->SetText("Processed:");
      sprintf(buf, "%lld events in %.1f sec", total, Long_t(tdiff)/1000.);
      fTotal->SetText(buf);

      if (fProof) {
         fProof->Disconnect("Progress(Long64_t,Long64_t)", this,
                            "Progress(Long64_t,Long64_t)");
         fProof->Disconnect("StopProcess(Bool_t)", this,
                            "IndicateStop(Bool_t)");
      }

      // Set button state
      fStop->SetState(kButtonDisabled);
      fAbort->SetState(kButtonDisabled);
      fClose->SetState(kButtonUp);
      if (!fKeep)
         DoClose();
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

      if (fStatus > kDone) {
         sprintf(buf, "%.1f sec (%lld events of %lld processed) - %s",
                      eta, evproc, total, cproc[fStatus]);
      } else {
         sprintf(buf, "%.1f sec (%lld events of %lld processed)",
                      eta, evproc, total);
      }
      fTotal->SetText(buf);
      sprintf(buf, "%.1f events/sec", Float_t(evproc)/Long_t(tdiff)*1000.);
      fRate->SetText(buf);

      if (processed < 0) {
         // And we disable the buttons
         fStop->SetState(kButtonDisabled);
         fAbort->SetState(kButtonDisabled);
         fClose->SetState(kButtonUp);
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
      fProof->Disconnect("StopProcess(Bool_t)", this,
                         "IndicateStop(Bool_t)");
      fProof->Disconnect("LogMessage(const char*,Bool_t)", this,
                         "LogMessage(const char*,Bool_t)");
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
      fProof->Disconnect("StopProcess(Bool_t)", this,
                         "IndicateStop(Bool_t)");
      // These buttons are meaningless at this point
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
      } else {
         // Clear window
         fLogWindow->Clear();
      }
      fProof->Connect("LogMessage(const char*,Bool_t)", "TProofProgressDialog",
                      this, "LogMessage(const char*,Bool_t)");
      if (!fLogQuery) {
         fProof->LogMessage(0, kTRUE);
      } else {
         Int_t qry = -2;
         TString qs = fTextQuery->GetString();
         if (qs != "last" && qs.IsDigit())
            qry = qs.Atoi();
         Bool_t logonly = fProof->SendingLogToWindow();
         fProof->SendLogToWindow(kTRUE);
         fProof->ShowLog(qry);
         fProof->SendLogToWindow(logonly);
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
   fStop->SetState(kButtonDisabled);
   fAbort->SetState(kButtonDisabled);
   fClose->SetState(kButtonUp);
}
