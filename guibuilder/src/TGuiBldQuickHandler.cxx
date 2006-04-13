// @(#)root/guibuilder:$Name:  $:$Id: TGuiBldQuickHandler.cxx,v 1.8 2006/04/07 10:05:09 antcheva Exp $
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
#include "TRootGuiBuilder.h"
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


static TGuiBldQuickHandler *gQuickHandler = 0;

////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________
TGuiBldTextDialog::TGuiBldTextDialog(const char *name, const char *setter, const char *getter) :
            TGTransientFrame(gClient->GetDefaultRoot(), gClient->GetRoot(), 1, 1)
{
   // ctor

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
   TRootGuiBuilder::PropagateBgndColor(this, TRootGuiBuilder::GetBgnd());

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
   TRootGuiBuilder::PropagateBgndColor(hf, TRootGuiBuilder::GetBgnd());

   MapSubwindows();
   Resize();
   CenterOnParent();
   MapRaised();

   fEntry->Connect("TextChanged(char*)", fSelected->ClassName(), fSelected, setter);
   fCancel->Connect("Pressed()", "TGuiBldTextDialog", this, "DoCancel()");
   fOK->Connect("Pressed()", "TGuiBldTextDialog", this, "DoOK()");

   if (gQuickHandler) gQuickHandler->fEditor = this;
}

//______________________________________________________________________________
TGuiBldTextDialog::~TGuiBldTextDialog()
{
   // dtor

}

//______________________________________________________________________________
void TGuiBldTextDialog::DoCancel()
{
   // process Cancel button press event

   fEntry->SetText(fSavedText.Data());

   DeleteWindow();
   if (gQuickHandler) gQuickHandler->fEditor = 0;
}

//______________________________________________________________________________
void TGuiBldTextDialog::DoOK()
{
   //  process OK button press event

   DeleteWindow();
   if (gQuickHandler) gQuickHandler->fEditor = 0;
}

//______________________________________________________________________________
void TGuiBldTextDialog::CloseWindow()
{
   // on window close

   DeleteWindow();
   if (gQuickHandler) gQuickHandler->fEditor = 0;
}



////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________
TGuiBldQuickHandler::TGuiBldQuickHandler() :  TObject()
{
   // ctor

   fSelected = 0;
   fEditor = 0;
   gQuickHandler = this;
}

//______________________________________________________________________________
TGuiBldQuickHandler::~TGuiBldQuickHandler()
{
   // dtor

   fSelected = 0;
   fEditor = 0;
   gQuickHandler = 0;
}

//______________________________________________________________________________
Bool_t TGuiBldQuickHandler::HandleEvent(TGWindow *win)
{
   // handle event

   if (!win) return kFALSE;

   fSelected = win;
   char action[512];
   Bool_t ret = kFALSE;

   if (gClient->GetMimeTypeList()->GetAction(fSelected->ClassName(), action)) {
      TString act = action;
      act.ReplaceAll("%s", fSelected->GetName());

      if (act[0] == '!') {
         act.Remove(0, 1);
         gSystem->Exec(act.Data());
      } else {
         gROOT->ProcessLineFast(act.Data());
      }
      ret = kTRUE;
   }
   return ret;
}
