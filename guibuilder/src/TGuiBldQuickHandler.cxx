// @(#)root/guibuilder:$Name:  $:$Id: TGuiBldQuickHandler.cxx,v 1.3 2004/10/06 14:38:19 brun Exp $
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiBldQuickHandler                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGuiBldQuickHandler.h"
#include "TGTextEntry.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TApplication.h"
#include "TGMimeTypes.h"
#include "THashList.h"


ClassImp(TGuiBldQuickHandler)
ClassImp(TGuiBldTextDialog)

TGuiBldTextDialog *gTextDialog = 0;

////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________
TGuiBldTextDialog::TGuiBldTextDialog(const char *name, const char *setter, const char *getter) :
            TGTransientFrame(gClient->GetDefaultRoot(), gClient->GetRoot(), 1, 1)
{
   //

   delete gTextDialog;
   gTextDialog = this;

   fEditDisabled = kTRUE;

   SetCleanup(kDeepCleanup);

   TGWindow *win = fClient->GetWindowByName(name);

   if (!win) return;

   fSelected = win;
   TString cmd = "((";
   cmd += fSelected->ClassName();
   cmd += "*)";
   cmd += Form("%d)" , win);
   cmd += "->";
   cmd += getter;

   fSavedText = (const char*)gROOT->ProcessLineFast(cmd.Data());

   TString title = "Edit ";
   title += fSelected->ClassName();
   title += "::";
   title += fSelected->GetName();
   SetWindowName(title.Data());

   title = "Call ";
   title += fSelected->ClassName();
   title += "::";
   title += setter;

   TGLabel *label = new TGLabel(this, title.Data());
   AddFrame(label, new TGLayoutHints(kLHintsNormal, 5, 5, 5, 5));

   fEntry = new TGTextEntry(this);
   AddFrame(fEntry, new TGLayoutHints(kLHintsNormal | kLHintsExpandX, 5, 5, 5, 5));
   fEntry->SetText(fSavedText.Data());
   fEntry->Resize(300, fEntry->GetDefaultHeight());
   fEntry->SelectAll();

   TGHorizontalFrame *hf = new TGHorizontalFrame(this);
   AddFrame(hf, new TGLayoutHints(kLHintsRight | kLHintsBottom,5,5,5,5));

   fOK = new TGTextButton(hf, "OK");
   hf->AddFrame(fOK, new TGLayoutHints(kLHintsBottom | kLHintsExpandX, 2, 2));

   fCancel = new TGTextButton(hf, "    Cancel     ");
   hf->AddFrame(fCancel, new TGLayoutHints(kLHintsBottom | kLHintsExpandX, 2, 2));

   UInt_t w = fOK->GetDefaultWidth();
   w = TMath::Max(w, fCancel->GetDefaultWidth());
   hf->Resize(2 * (w + 10), hf->GetDefaultHeight());

   MapSubwindows();
   Resize();
   MapRaised();

   fEntry->Connect("TextChanged(char*)", fSelected->ClassName(), fSelected, setter);
   fCancel->Connect("Pressed()", "TGuiBldTextDialog", this, "DoCancel()");
   fOK->Connect("Pressed()", "TGuiBldTextDialog", this, "DoOK()");
}

//______________________________________________________________________________
TGuiBldTextDialog::~TGuiBldTextDialog()
{
   //

   gTextDialog = 0;
}

//______________________________________________________________________________
void TGuiBldTextDialog::DoCancel()
{
   //

   fEntry->SetText(fSavedText.Data());
   DeleteWindow();
}

//______________________________________________________________________________
void TGuiBldTextDialog::DoOK()
{
   //

   DeleteWindow();
}


////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________
TGuiBldQuickHandler::TGuiBldQuickHandler() :  TObject()
{
   //

   fSelected = 0;
}

//______________________________________________________________________________
TGuiBldQuickHandler::~TGuiBldQuickHandler()
{
   //

   delete gTextDialog;
   gTextDialog = 0;

   fSelected = 0;
}

//______________________________________________________________________________
Bool_t  TGuiBldQuickHandler::HandleEvent(Event_t *ev)
{
   //

   TGWindow *win  = gClient->GetWindowById(ev->fWindow);

   if (!win) return kFALSE;

   fSelected = win;
   char action[512];

   if (gClient->GetMimeTypeList()->GetAction(fSelected->ClassName(), action)) {
      TString act = action;
      act.ReplaceAll("%s", fSelected->GetName());

      if (act[0] == '!') {
         act.Remove(0, 1);
         gSystem->Exec(act.Data());
      } else {
         gApplication->ProcessLine(act.Data());
      }
   }
   return kTRUE;
}
