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


/** \class TSessionLogView
    \ingroup sessionviewer

Dialog used to display session logs from the session viewer

*/


ClassImp(TSessionLogView);

////////////////////////////////////////////////////////////////////////////////
/// Create an editor in a dialog.

TSessionLogView::TSessionLogView(TSessionViewer *viewer, UInt_t w, UInt_t h) :
   TGTransientFrame(gClient->GetRoot(), viewer, w, h)
{
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

////////////////////////////////////////////////////////////////////////////////

TSessionLogView::~TSessionLogView()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Set title in editor window.

void TSessionLogView::SetTitle()
{
   TString title;
   title.Form("PROOF Processing Logs: %s", (fViewer->GetActDesc()->fProof ?
              fViewer->GetActDesc()->fProof->GetMaster() : "<dummy>"));
   SetWindowName(title);
   SetIconName(title);
}

////////////////////////////////////////////////////////////////////////////////
/// Show editor.

void TSessionLogView::Popup()
{
   MapWindow();
}

////////////////////////////////////////////////////////////////////////////////
/// Load a text buffer in the editor.

void TSessionLogView::AddBuffer(const char *buffer)
{
   TGText txt;
   txt.LoadBuffer(buffer);
   fTextView->AddText(&txt);
   fTextView->ShowBottom();
}

////////////////////////////////////////////////////////////////////////////////
/// Clear log window.

void TSessionLogView::ClearLogView()
{
   fTextView->Clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Load a text buffer in the editor.

void TSessionLogView::LoadBuffer(const char *buffer)
{
   fTextView->LoadBuffer(buffer);
   fTextView->ShowBottom();
}

////////////////////////////////////////////////////////////////////////////////
/// Load a file in the editor.

void TSessionLogView::LoadFile(const char *file)
{
   fTextView->LoadFile(file);
   fTextView->ShowBottom();
}

////////////////////////////////////////////////////////////////////////////////
/// Called when closed via window manager action.

void TSessionLogView::CloseWindow()
{
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

////////////////////////////////////////////////////////////////////////////////
/// Process OK button.

Bool_t TSessionLogView::ProcessMessage(Longptr_t msg, Longptr_t, Longptr_t)
{
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

