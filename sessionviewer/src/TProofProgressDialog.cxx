// @(#)root/proof:$Name:  $:$Id: TProofProgressDialog.cxx,v 1.7 2004/04/20 19:35:17 brun Exp $
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
#include "TGLabel.h"
#include "TGButton.h"
#include "TGProgressBar.h"
#include "TProof.h"
#include "TSystem.h"
#include "TTimer.h"


Bool_t TProofProgressDialog::fgKeep = kTRUE;


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

   const TGWindow *main = gClient->GetRoot();
   fDialog = new TGTransientFrame(main, main, 10, 10);
   fDialog->Connect("CloseWindow()", "TProofProgressDialog", this, "DoClose()");
   fDialog->DontCallClose();

   // title label
   char buf[256];
   sprintf(buf, "Executing on PROOF cluster \"%s\" with %d parallel slaves:",
           fProof ? fProof->GetMaster() : "<dummy>",
           fProof ? fProof->GetParallel() : 0);
   fDialog->AddFrame(new TGLabel(fDialog, buf),
                     new TGLayoutHints(kLHintsNormal, 10, 10, 20, 0));
   fDialog->AddFrame(new TGLabel(fDialog, selector),
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

   // keep toggle button
   fKeep = new TGCheckButton(fDialog, new TGHotString("Close this dialog when processing complete"));
   if (!fgKeep) fKeep->SetState(kButtonDown);
   fKeep->Connect("Toggled(Bool_t)", "TProofProgressDialog", this, "DoKeep(Bool_t)");
   fDialog->AddFrame(fKeep, new TGLayoutHints(kLHintsNormal, 10, 10, 20, 0));

   // stop and close buttons
   TGHorizontalFrame *hf3 = new TGHorizontalFrame(fDialog, 60, 20, kFixedWidth);

   UInt_t  nb = 0, width = 0, height = 0;

   fStop = new TGTextButton(hf3, "&Stop");
   fStop->Connect("Clicked()", "TProofProgressDialog", this, "DoStop()");
   hf3->AddFrame(fStop, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 5, 10, 0, 0));
   height = fStop->GetDefaultHeight();
   width  = TMath::Max(width, fStop->GetDefaultWidth()); ++nb;

   fClose = new TGTextButton(hf3, "&Close");
   fClose->Connect("Clicked()", "TProofProgressDialog", this, "DoClose()");
   hf3->AddFrame(fClose, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 10, 5, 0, 0));
   height = fClose->GetDefaultHeight();
   width  = TMath::Max(width, fClose->GetDefaultWidth()); ++nb;

   // place button frame (hf3) at the bottom
   fDialog->AddFrame(hf3, new TGLayoutHints(kLHintsBottom | kLHintsCenterX, 10, 10, 20, 10));

   // keep buttons centered and with the same width
   hf3->Resize((width + 40) * nb, height);

   // connect slot to proof progress signal
   if (fProof)
      fProof->Connect("Progress(Long64_t,Long64_t)", "TProofProgressDialog",
                      this, "Progress(Long64_t,Long64_t)");

   // set dialog title
   fDialog->SetWindowName("PROOF Query Progress");

   // map all widgets and calculate size of dialog
   fDialog->MapSubwindows();

   width  = fDialog->GetDefaultWidth();
   height = fDialog->GetDefaultHeight();

   fDialog->Resize(width, height);

   // position relative to the parent window (which is the root window)
   Window_t wdum;
   int      ax, ay;

   gVirtualX->TranslateCoordinates(main->GetId(), main->GetId(),
                          (((TGFrame *) main)->GetWidth() - width) >> 1,
                          (((TGFrame *) main)->GetHeight() - height) >> 1,
                          ax, ay, wdum);
   fDialog->Move(ax, ay);
   fDialog->SetWMPosition(ax, ay);

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
void TProofProgressDialog::Progress(Long64_t total, Long64_t processed)
{
   // Update progress bar and status labels.

   if (fPrevProcessed == processed)
      return;

   char buf[256];
   if (fEntries != total) {
      fEntries = total;
      sprintf(buf, "%d files, number of events %lld, starting event %lld",
              fFiles, fEntries, fFirst);
      fFilesEvents->SetText(buf);
   }

   Float_t pos = Float_t(Double_t(processed * 100)/Double_t(total));
   fBar->SetPosition(pos);

   // get current time
   fEndTime = gSystem->Now();
   TTime tdiff = fEndTime - fStartTime;
   Float_t eta = 0;
   if (processed)
      eta = ((Float_t)((Long_t)tdiff)*total/Float_t(processed) - Long_t(tdiff))/1000.;

   if (processed == total) {
      fProcessed->SetText("Processed:");
      sprintf(buf, "%lld events in %.1f sec", total, Long_t(tdiff)/1000.);
      fTotal->SetText(buf);

      if (fProof)
         fProof->Disconnect("Progress(Long64_t,Long64_t)", this,
                            "Progress(Long64_t,Long64_t)");

      if (!fgKeep)
         DoClose();
   } else {
      sprintf(buf, "%.1f sec (%lld events of %lld processed)", eta, processed,
              total);
      fTotal->SetText(buf);
      sprintf(buf, "%.1f events/sec", Float_t(processed)/Long_t(tdiff)*1000.);
      fRate->SetText(buf);
   }
   fPrevProcessed = processed;

   fDialog->Layout();
}

//______________________________________________________________________________
TProofProgressDialog::~TProofProgressDialog()
{
   // Cleanup dialog.

   if (fProof)
      fProof->Disconnect("Progress(Long64_t,Long64_t)", this,
                         "Progress(Long64_t,Long64_t)");

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
void TProofProgressDialog::DoClose()
{
   // Close dialog.

   fClose->SetState(kButtonDisabled);
   TTimer::SingleShot(50, "TProofProgressDialog", this, "CloseWindow()");
}

//______________________________________________________________________________
void TProofProgressDialog::DoKeep(Bool_t)
{
   // Handle keep toggle button.

   fgKeep = !fgKeep;
}

//______________________________________________________________________________
void TProofProgressDialog::DoStop()
{
   // Handle Stop button.

   printf("DoStop\n");
}
