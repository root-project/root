// @(#)root/sessionviewer:$Id$
// Author: Bertrand Bellenot, Gerri Ganis 15/09/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TSessionLogView.h"
#include "TSessionViewer.h"
#include "TProof.h"
#include "KeySymbols.h"

//_____________________________________________________________________________
//
// TSessionLogView
//
// Dialog used to display session logs from the session viewer
//_____________________________________________________________________________

ClassImp(TSessionLogView)

//____________________________________________________________________________
TSessionLogView::TSessionLogView(TSessionViewer *viewer, UInt_t w, UInt_t h) :
   TGTransientFrame(gClient->GetRoot(), viewer, w, h)
{
   // Create an editor in a dialog.

   fViewer = viewer;
   fTextView = new TGTextView(this, w, h, kSunkenFrame | kDoubleBorder);
   fL1 = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 3, 3, 3, 3);
   AddFrame(fTextView, fL1);

   fClose = new TGTextButton(this, "  &Close  ");
   fL2 = new TGLayoutHints(kLHintsBottom | kLHintsCenterX, 0, 0, 5, 5);
   AddFrame(fClose, fL2);

   SetTitle();
   fViewer->SetLogWindow(this);

   MapSubwindows();

   Resize(GetDefaultSize());
}

//____________________________________________________________________________
TSessionLogView::~TSessionLogView()
{
}

//____________________________________________________________________________
void TSessionLogView::SetTitle()
{
   // Set title in editor window.

   TString title;
   title.Form("PROOF Processing Logs: %s", (fViewer->GetActDesc()->fProof ?
              fViewer->GetActDesc()->fProof->GetMaster() : "<dummy>"));
   SetWindowName(title);
   SetIconName(title);
}

//____________________________________________________________________________
void TSessionLogView::Popup()
{
   // Show editor.

   MapWindow();
}

//____________________________________________________________________________
void TSessionLogView::AddBuffer(const char *buffer)
{
   // Load a text buffer in the editor.

   TGText txt;
   txt.LoadBuffer(buffer);
   fTextView->AddText(&txt);
   fTextView->ShowBottom();
}

//____________________________________________________________________________
void TSessionLogView::ClearLogView()
{
   // Clear log window.

   fTextView->Clear();
}

//____________________________________________________________________________
void TSessionLogView::LoadBuffer(const char *buffer)
{
   // Load a text buffer in the editor.

   fTextView->LoadBuffer(buffer);
   fTextView->ShowBottom();
}

//____________________________________________________________________________
void TSessionLogView::LoadFile(const char *file)
{
   // Load a file in the editor.

   fTextView->LoadFile(file);
   fTextView->ShowBottom();
}

//____________________________________________________________________________
void TSessionLogView::CloseWindow()
{
   // Called when closed via window manager action.
   if (fViewer->GetActDesc()->fProof) {
      fViewer->GetActDesc()->fProof->Disconnect(
            "LogMessage(const char*,Bool_t)", fViewer,
            "LogMessage(const char*,Bool_t)");
   }
   fViewer->SetLogWindow(0);
   delete fTextView;
   delete fClose;
   delete fL1;
   delete fL2;
   DestroyWindow();
}

//____________________________________________________________________________
Bool_t TSessionLogView::ProcessMessage(Long_t msg, Long_t, Long_t)
{
   // Process OK button.

   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_BUTTON:
               // Only one button and one action...
               CloseWindow();
               break;
            default:
               break;
         }
         break;
      default:
         break;
   }
   return kTRUE;
}

