// @(#)root/gui:$Name:  $:$Id: TGInputDialog.cxx,v 1.0 2006/07/19 11:13:18 dgmaline Exp $
// Author: David Gonzalez Maline  19/07/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Input Dialog Widget                                                   //
//                                                                       //
// An Input dialog box                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGInputDialog.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGTextEntry.h"

ClassImp(TGInputDialog)


//______________________________________________________________________________
TGInputDialog::TGInputDialog(const TGWindow *p, const TGWindow *main, 
                             const char *prompt, const char *defval, 
                             char *retstr, UInt_t options) :
      TGTransientFrame(p, main, 10, 10, options)
{
   // Create simple input dialog.

   if (!p && !main) {
      MakeZombie();
      return;
   }
   SetCleanup(kDeepCleanup);
   // create prompt label and textentry widget
   fLabel = new TGLabel(this, prompt);

   TGTextBuffer *tbuf = new TGTextBuffer(256);  //will be deleted by TGtextEntry
   tbuf->AddText(0, defval);

   fTE = new TGTextEntry(this, tbuf);
   fTE->Resize(260, fTE->GetDefaultHeight());

   AddFrame(fLabel, new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 5, 5, 0));
   AddFrame(fTE, new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 5, 5, 5));

   // create frame and layout hints for Ok and Cancel buttons
   TGHorizontalFrame *hf = new TGHorizontalFrame(this, 60, 20, kFixedWidth);
   hf->SetCleanup(kDeepCleanup);

   // create OK and Cancel buttons in their own frame (hf)
   UInt_t  width = 0, height = 0;

   fOk = new TGTextButton(hf, "&Ok", 1);
   fOk->Associate(this);
   hf->AddFrame(fOk, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 5, 5, 0, 0));
   height = fOk->GetDefaultHeight();
   width  = TMath::Max(width, fOk->GetDefaultWidth());

   fCancel = new TGTextButton(hf, "&Cancel", 2);
   fCancel->Associate(this);
   hf->AddFrame(fCancel, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 5, 5, 0, 0));
   height = fCancel->GetDefaultHeight();
   width  = TMath::Max(width, fCancel->GetDefaultWidth());

   // place button frame (hf) at the bottom
   AddFrame(hf, new TGLayoutHints(kLHintsBottom | kLHintsCenterX, 0, 0, 5, 5));

   // keep buttons centered and with the same width
   hf->Resize((width + 20) * 2, height);

   // set dialog title
   SetWindowName("Get Input");

   // map all widgets and calculate size of dialog
   MapSubwindows();

   width  = GetDefaultWidth();
   height = GetDefaultHeight();

   Resize(width, height);

   // position relative to the parent's window
   CenterOnParent();

   // make the message box non-resizable
   SetWMSize(width, height);
   SetWMSizeHints(width, height, width, height, 0, 0);

   SetMWMHints(kMWMDecorAll | kMWMDecorResizeH  | kMWMDecorMaximize |
                                       kMWMDecorMinimize | kMWMDecorMenu,
                        kMWMFuncAll  | kMWMFuncResize    | kMWMFuncMaximize |
                                       kMWMFuncMinimize,
                        kMWMInputModeless);

   // popup dialog and wait till user replies
   MapWindow();
   fTE->SetFocus();

   fRetStr = retstr;

   gClient->WaitFor(this);
}

//______________________________________________________________________________
TGInputDialog::~TGInputDialog()
{
   // Cleanup dialog.

   Cleanup();
}

//______________________________________________________________________________
Bool_t TGInputDialog::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Handle button and text enter events

   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_BUTTON:
               switch (parm1) {
                  case 1:
                     // here copy the string from text buffer to return variable
                     strcpy(fRetStr, fTE->GetBuffer()->GetString());
                     delete this;
                     break;
                  case 2:
                     fRetStr[0] = 0;
                     delete this;
                     break;
               }
               default:
                  break;
         }
         break;

      case kC_TEXTENTRY:
         switch (GET_SUBMSG(msg)) {
            case kTE_ENTER:
               // here copy the string from text buffer to return variable
               strcpy(fRetStr, fTE->GetBuffer()->GetString());
               delete this;
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
