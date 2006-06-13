// @(#)root/proof:$Name:  $:$Id: TProofProgressLog.cxx,v 1.1 2005/08/30 10:25:29 rdm Exp $
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
TProofProgressLog::TProofProgressLog(TProofProgressDialog *d, Int_t w, Int_t h) :
   TGTransientFrame(gClient->GetRoot(), gClient->GetRoot(), w, h)
{
   // Create a window frame for log messages.

   fDialog = d;

   // use hierarchical cleaning
   SetCleanup(kDeepCleanup);

   fText = new TGTextView(this, w, h);
   AddFrame(fText, new TGLayoutHints(kLHintsExpandX |
                                     kLHintsExpandY, 3, 3, 3, 3));

   fClose = new TGTextButton(this, "  &Close  ");
   fClose->Connect("Clicked()", "TProofProgressLog", this, "CloseWindow()");
   AddFrame(fClose, new TGLayoutHints(kLHintsBottom |
                                      kLHintsCenterX, 0, 0, 5, 5));

   char title[256] = {0};
   strcpy(title,Form("PROOF Processing Logs: %s",
                     (fDialog->fProof ? fDialog->fProof->GetMaster() : "<dummy>")));
   SetWindowName(title);
   SetIconName(title);

   MapSubwindows();

   Resize();

   Window_t wdummy;
   int ax, ay;
   gVirtualX->TranslateCoordinates(GetParent()->GetId(), fDialog->fDialog->GetId(),
       (Int_t)(((TGFrame *)GetParent())->GetWidth() + w),
       (Int_t)(((TGFrame *)GetParent())->GetHeight()- 3*h/2), ax, ay, wdummy);
   Move(ax, ay);

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
}

//____________________________________________________________________________
void TProofProgressLog::Popup()
{
   // Show log window.

   MapWindow();
}

//____________________________________________________________________________
void TProofProgressLog::Clear(Option_t *)
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
void TProofProgressLog::CloseWindow()
{
   // Handle close button or when closed via window manager action.

   DeleteWindow();
}
