// @(#)root/proof:$Name:  $:$Id: TProof.h,v 1.61 2005/08/15 15:57:18 rdm Exp $
// Author: G Ganis, Jul 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TError.h"
#include "TGFrame.h"
#include "TGTextView.h"
#include "TGScrollBar.h"
#include "TProof.h"
#include "TProofProgressDialog.h"
#include "TProofProgressLog.h"


ClassImp(TProofProgressLog)

//____________________________________________________________________________
TProofProgressLog::TProofProgressLog(TProofProgressDialog *d)
{
   // Create a window frame for log messages.

   fDialog = d;

   const TGWindow *main = gClient->GetRoot();
   Int_t wdt = 700;
   Int_t hgt = 300;

   fMain = new TGTransientFrame(main, main, wdt, hgt);
   fMain->Connect("CloseWindow()", "TProofProgressLog", this, "CloseWindow()");

   // use hierarchical cleaning
   fMain->SetCleanup(kDeepCleanup);

   fText = new TGTextView(fMain, wdt, hgt);
   fMain->AddFrame(fText, new TGLayoutHints(kLHintsExpandX |
                                            kLHintsExpandY, 3, 3, 3, 3));

   fClose = new TGTextButton(fMain, "  &Close  ");
   fClose->Connect("Clicked()", "TProofProgressLog", this, "DoClose()");
   fMain->AddFrame(fClose, new TGLayoutHints(kLHintsBottom |
                                             kLHintsCenterX, 0, 0, 5, 5));

   char title[256] = {0};
   strcpy(title,Form("PROOF Processing Logs: %s",
                     (fDialog->fProof ? fDialog->fProof->GetMaster() : "<dummy>")));
   fMain->SetWindowName(title);
   fMain->SetIconName(title);

   fMain->MapSubwindows();

   fMain->Resize();

   Window_t wdummy;
   int ax, ay;
   gVirtualX->TranslateCoordinates(main->GetId(), fDialog->fDialog->GetId(),
       (Int_t)(((TGFrame *)main)->GetWidth() + wdt),
       (Int_t)(((TGFrame *)main)->GetHeight()- 3*hgt/2), ax, ay, wdummy);
   fMain->Move(ax, ay);

   Popup();
}

//____________________________________________________________________________
TProofProgressLog::~TProofProgressLog()
{
   // Delete log window.

   // Detach from owner dialog
   fDialog->fLogWindow = 0;
   fDialog->fProof->Disconnect("LogMessage(const char*,Bool_t)", this,
                               "LogMessage(const char*,Bool_t)");
   delete fMain;
}

//____________________________________________________________________________
void TProofProgressLog::Popup()
{
   // Show log window.

   fMain->MapWindow();
}

//____________________________________________________________________________
void TProofProgressLog::Clear()
{
   // Clear log window.

   if (fText)
      fText->Clear();
}

//____________________________________________________________________________
void TProofProgressLog::LoadBuffer(const char *buffer)
{
   // Load a text buffer in the window.

   if (fText)
      fText->LoadBuffer(buffer);
}

//____________________________________________________________________________
void TProofProgressLog::LoadFile(const char *file)
{
   // Load a file in the window.

   if (fText)
      fText->LoadFile(file);
}

//____________________________________________________________________________
void TProofProgressLog::AddBuffer(const  char *buffer)
{
   // Add text to the window.

   if (fText) {
      TGText txt;
      txt.LoadBuffer(buffer);
      fText->AddText(&txt);
   }
}

//____________________________________________________________________________
void TProofProgressLog::DoClose()
{
   // Handle close button.

   // Detach from owner dialog
   fDialog->fLog = 0;
   fDialog->fProof->Disconnect("LogMessage(const char*,Bool_t)", this,
                               "LogMessage(const char*,Bool_t)");

   fMain->SendCloseMessage();
}

//____________________________________________________________________________
void TProofProgressLog::CloseWindow()
{
   // Called when closed via window manager action.

   delete this;
}
