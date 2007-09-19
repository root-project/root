// @(#)root/peac:$Id$
// Author: Maarten Ballintijn    21/10/2004
// Author: Kris Gulbrandsen      21/10/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofStartupDialog                                                  //
//                                                                      //
// This class provides a query progress bar for data being staged.      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofStartupDialog.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TGProgressBar.h"
#include "TSystem.h"
#include "TTimer.h"
#include "TProof.h"


ClassImp(TProofStartupDialog)

//______________________________________________________________________________
TProofStartupDialog::TProofStartupDialog(TProof *proof,
                                         const Char_t *dataset,
                                         Int_t nfiles,
                                         Long64_t totalbytes)
{
   // Create data staging progress dialog.

   fProof = proof;
   fPrevStaged = 0;
   fFiles = nfiles;
   fTotalBytes = totalbytes;

   const TGWindow *main = gClient->GetRoot();
   fDialog = new TGTransientFrame(main, main, 10, 10);
   fDialog->Connect("CloseWindow()", "TProofStartupDialog", this, "DoClose()");
   fDialog->DontCallClose();

   // title label
   char buf[256];
   sprintf(buf, "Staging files for data set '%s'", dataset);
   fDialog->AddFrame(new TGLabel(fDialog, buf),
                     new TGLayoutHints(kLHintsNormal, 10, 10, 20, 0));
   sprintf(buf, "%d files, number of bytes %lld", fFiles, fTotalBytes);
   fFilesBytes = new TGLabel(fDialog, buf);
   fDialog->AddFrame(fFilesBytes, new TGLayoutHints(kLHintsNormal, 10, 10, 5, 0));

   // progress bar
   fBar = new TGHProgressBar(fDialog, TGProgressBar::kFancy, 450);
   fBar->SetBarColor("green");
   fDialog->AddFrame(fBar, new TGLayoutHints(kLHintsTop | kLHintsLeft |
                     kLHintsExpandX, 10, 10, 20, 20));

   // status labels
   TGHorizontalFrame *hf1 = new TGHorizontalFrame(fDialog, 0, 0);
   TGCompositeFrame *cf1 = new TGCompositeFrame(hf1, 110, 0, kFixedWidth);
   fStaged = new TGLabel(cf1, "Estimated time left:");
   cf1->AddFrame(fStaged);
   hf1->AddFrame(cf1);
   fTotal= new TGLabel(hf1, "- sec (- bytes of - staged)");
   hf1->AddFrame(fTotal, new TGLayoutHints(kLHintsNormal, 10, 10, 0, 0));
   fDialog->AddFrame(hf1, new TGLayoutHints(kLHintsNormal, 10, 10, 5, 0));

   TGHorizontalFrame *hf2 = new TGHorizontalFrame(fDialog, 0, 0);
   TGCompositeFrame *cf2 = new TGCompositeFrame(hf2, 110, 0, kFixedWidth);
   cf2->AddFrame(new TGLabel(cf2, "Staging rate:"));
   hf2->AddFrame(cf2);
   fRate = new TGLabel(hf2, "- bytes/sec");
   hf2->AddFrame(fRate, new TGLayoutHints(kLHintsNormal, 10, 10, 0, 0));
   fDialog->AddFrame(hf2, new TGLayoutHints(kLHintsNormal, 10, 10, 5, 0));

   // stop and close buttons
   TGHorizontalFrame *hf3 = new TGHorizontalFrame(fDialog, 60, 20, kFixedWidth);

   UInt_t  nb = 0, width = 0, height = 0;

   // place button frame (hf3) at the bottom
   fDialog->AddFrame(hf3, new TGLayoutHints(kLHintsBottom | kLHintsCenterX, 10, 10, 20, 10));

   // keep buttons centered and with the same width
   hf3->Resize((width + 40) * nb, height);

   // connect slot to proof progress signal
   if (fProof)
      fProof->Connect("IsDataReady(Long64_t,Long64_t)", "TProofStartupDialog",
                      this, "Progress(Long64_t,Long64_t)");

   // set dialog title
   fDialog->SetWindowName("Data Staging Progress");

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
}

//______________________________________________________________________________
void TProofStartupDialog::Progress(Long64_t totalbytes, Long64_t bytesready)
{
   // Update progress bar and status labels.

   if (fPrevStaged == bytesready)
      return;

   char buf[256];
   if (fTotalBytes != totalbytes) {
      fTotalBytes = totalbytes;
      sprintf(buf, "%d files, number of bytes %lld", fFiles, fTotalBytes);
      fFilesBytes->SetText(buf);
   }

   Float_t pos = Float_t(Double_t(bytesready * 100)/Double_t(totalbytes));
   fBar->SetPosition(pos);

   // get current time
   fEndTime = gSystem->Now();
   TTime tdiff = fEndTime - fStartTime;
   Float_t eta = 0;
   if (bytesready)
      eta = ((Float_t)((Long_t)tdiff)*totalbytes/Float_t(bytesready) - Long_t(tdiff))/1000.;

   if (bytesready == totalbytes) {
      fStaged->SetText("Staged:");
      sprintf(buf, "%lld bytes in %.1f sec", totalbytes, Long_t(tdiff)/1000.);
      fTotal->SetText(buf);

      if (fProof) {
         fProof->Disconnect("IsDataReady(Long64_t,Long64_t)", this,
                            "Progress(Long64_t,Long64_t)");
         fProof = 0;
      }

      DoClose();
   } else {
      sprintf(buf, "%.1f sec (%lld bytes of %lld staged)", eta, bytesready,
              totalbytes);
      fTotal->SetText(buf);
      sprintf(buf, "%.1f bytes/sec", Float_t(bytesready)/Long_t(tdiff)*1000.);
      fRate->SetText(buf);
   }
   fPrevStaged = bytesready;

   fDialog->Layout();
}

//______________________________________________________________________________
TProofStartupDialog::~TProofStartupDialog()
{
   // Cleanup dialog.

   if (fProof)
      fProof->Disconnect("IsDataReady(Long64_t,Long64_t)", this,
                         "Progress(Long64_t,Long64_t)");

   fDialog->Cleanup();
   delete fDialog;
}

//______________________________________________________________________________
void TProofStartupDialog::CloseWindow()
{
   // Called when dialog is closed.

   delete this;
}

//______________________________________________________________________________
void TProofStartupDialog::DoClose()
{
   // Close dialog.

   TTimer::SingleShot(500, "TProofStartupDialog", this, "CloseWindow()");
}
