// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   10/7/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTextEditDialogs                                                    //
//                                                                      //
// This file defines several dialogs that are used by the TGTextEdit    //
// widget via its associated context popup menu.                        //
// The following dialogs are available: TGSearchDialog, TGGotoDialog    //
// and TGPrintDialog.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGTextEditDialogs.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGTextEntry.h"
#include "TGIcon.h"

#include <stdlib.h>


ClassImp(TGSearchDialog)
ClassImp(TGPrintDialog)
ClassImp(TGGotoDialog)


//______________________________________________________________________________
TGSearchDialog::TGSearchDialog(const TGWindow *p, const TGWindow *main,
                               UInt_t w, UInt_t h, TGSearchType *sstruct,
                               Int_t *ret_code, UInt_t options) :
     TGTransientFrame(p, main, w, h, options)
{
   // Create a search dialog box. Used to get from the user the required
   // search instructions. Ret_code is kTRUE when sstruct has been set,
   // kFALSE otherwise (like when dialog was canceled).

   fRetCode = ret_code;
   fType = sstruct;

   ChangeOptions((GetOptions() & ~kVerticalFrame) | kHorizontalFrame);

   fF1 = new TGCompositeFrame(this, 60, 20, kVerticalFrame | kFixedWidth);
   fF2 = new TGCompositeFrame(this, 60, 20, kVerticalFrame);
   fF3 = new TGCompositeFrame(fF2, 60, 20, kHorizontalFrame);
   fF4 = new TGCompositeFrame(fF2, 60, 20, kHorizontalFrame);

   fSearchButton = new TGTextButton(fF1, new TGHotString("&Search"), 1);
   fCancelButton = new TGTextButton(fF1, new TGHotString("&Cancel"), 2);
   fF1->Resize(fSearchButton->GetDefaultWidth()+40, GetDefaultHeight());

   fSearchButton->Associate(this);
   fCancelButton->Associate(this);

   fL1 = new TGLayoutHints(kLHintsTop | kLHintsExpandX, 2, 2, 3, 0);
   fL2 = new TGLayoutHints(kLHintsTop | kLHintsRight | kLHintsExpandX,
                           2, 5, 0, 2);
   fL21 = new TGLayoutHints(kLHintsTop | kLHintsRight, 2, 5, 10, 0);

   fF1->AddFrame(fSearchButton, fL1);
   fF1->AddFrame(fCancelButton, fL1);
   AddFrame(fF1, fL21);

   fLSearch = new TGLabel(fF3, new TGHotString("Search &for:"));

   fBSearch = new TGTextBuffer(100);
   if (fType->fBuffer)
      fBSearch->AddText(0, fType->fBuffer);
   else
      fSearchButton->SetState(kButtonDisabled);

   fSearch = new TGTextEntry(fF3, fBSearch);
   fSearch->Associate(this);
   fSearch->Resize(220, fSearch->GetDefaultHeight());
   fSearch->SelectAll();

   fL5 = new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 3, 5, 0, 0);
   fL6 = new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 0, 2, 0, 0);

   fF3->AddFrame(fLSearch, fL5);
   fF3->AddFrame(fSearch, fL6);

   fG2 = new TGGroupFrame(fF4, new TGString("Direction"), kHorizontalFrame);

   fL3 = new TGLayoutHints(kLHintsTop | kLHintsRight, 2, 2, 2, 2);
   fL9 = new TGLayoutHints(kLHintsBottom | kLHintsLeft, 0, 0, 0, 0);
   fL4 = new TGLayoutHints(kLHintsBottom | kLHintsLeft, 0, 0, 5, 0);
   fL10 = new TGLayoutHints(kLHintsBottom | kLHintsRight, 0, 0, 5, 0);

   fCaseCheck = new TGCheckButton(fF4, new TGHotString("&Case sensitive"), 1);
   fCaseCheck->Associate(this);
   fF4->AddFrame(fCaseCheck, fL9);

   fDirectionRadio[0] = new TGRadioButton(fG2, new TGHotString("Forward"),  1);
   fDirectionRadio[1] = new TGRadioButton(fG2, new TGHotString("Backward"), 2);

   fG2->AddFrame(fDirectionRadio[0], fL4);
   fG2->AddFrame(fDirectionRadio[1], fL10);
   fDirectionRadio[0]->Associate(this);
   fDirectionRadio[1]->Associate(this);

   if (fType->fCaseSensitive == kFALSE)
      fCaseCheck->SetState(kButtonUp);
   else
      fCaseCheck->SetState(kButtonDown);

   if (fType->fDirection)
      fDirectionRadio[0]->SetState(kButtonDown);
   else
      fDirectionRadio[1]->SetState(kButtonDown);

   fF4->AddFrame(fG2, fL3);

   fF2->AddFrame(fF3, fL1);
   fF2->AddFrame(fF4, fL1);

   AddFrame(fF2, fL2);

   MapSubwindows();
   Resize(GetDefaultSize());

   // position relative to the parent's window
   Int_t ax, ay;
   if (main) {
      Window_t wdum;
      gVirtualX->TranslateCoordinates(main->GetId(), GetParent()->GetId(),
                          (((TGFrame *) main)->GetWidth() - fWidth) >> 1,
                          (((TGFrame *) main)->GetHeight() - fHeight) >> 1,
                          ax, ay, wdum);
   } else {
      UInt_t root_w, root_h;
      gVirtualX->GetWindowSize(fClient->GetRoot()->GetId(), ax, ay, root_w, root_h);
      ax = (root_w - fWidth) >> 1;
      ay = (root_h - fHeight) >> 1;
   }

   Move(ax, ay);
   SetWMPosition(ax, ay);

   SetWindowName("Search");
   SetIconName("Search");

   SetMWMHints(kMWMDecorAll | kMWMDecorMaximize | kMWMDecorMenu,
               kMWMFuncAll | kMWMFuncMaximize | kMWMFuncResize,
               kMWMInputModeless);

   MapWindow();
   fClient->WaitFor(this);
}

//______________________________________________________________________________
TGSearchDialog::~TGSearchDialog()
{
   // Clean up search dialog.

   delete fSearchButton;
   delete fCancelButton;
   delete fDirectionRadio[0]; delete fDirectionRadio[1];
   delete fCaseCheck;
   delete fSearch;
   delete fLSearch;
   delete fG2;
   delete fF1; delete fF2; delete fF3; delete fF4;
   delete fL1; delete fL2; delete fL3; delete fL4; delete fL5; delete fL6;
   delete fL21;delete fL9; delete fL10;
}

//______________________________________________________________________________
void TGSearchDialog::CloseWindow()
{
   // Close the dialog. On close the dialog will be deleted and cannot be
   // re-used.

   delete this;
}

//______________________________________________________________________________
Bool_t TGSearchDialog::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Process search dialog widget messages.

   const char *string;

   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_BUTTON:
               switch (parm1) {
                  case 1:
                     string = fBSearch->GetString();
                     fType->fBuffer = new char[strlen(string)+1];
                     strcpy(fType->fBuffer, string);
                     *fRetCode = kTRUE;
                     CloseWindow();
                     break;
                  case 2:
                     *fRetCode = kFALSE;
                     CloseWindow();
                     break;
               }
               break;

            case kCM_CHECKBUTTON:
               fType->fCaseSensitive = !fType->fCaseSensitive;
               break;

            case kCM_RADIOBUTTON:
               switch (parm1) {
                  case 1:
                     fType->fDirection = kTRUE;
                     fDirectionRadio[1]->SetState(kButtonUp);
                     break;
                  case 2:
                     fType->fDirection = kFALSE;
                     fDirectionRadio[0]->SetState(kButtonUp);
                     break;
               }
               break;

            default:
               break;
         }
         break;

      case kC_TEXTENTRY:
         switch (GET_SUBMSG(msg)) {
            case kTE_TEXTCHANGED:
               string = fBSearch->GetString();
               if (strlen(string) == 0)
                  fSearchButton->SetState(kButtonDisabled);
               else
                  fSearchButton->SetState(kButtonUp);
               break;
            case kTE_ENTER:
               string = fBSearch->GetString();
               fType->fBuffer = new char[strlen(string)+1];
               strcpy(fType->fBuffer, string);
               *fRetCode = kTRUE;
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

//______________________________________________________________________________
TGPrintDialog::TGPrintDialog(const TGWindow *p, const TGWindow *main,
                             UInt_t w, UInt_t h, char **printerName,
                             char **printProg, Int_t *ret_code,
                             UInt_t options) :
   TGTransientFrame(p, main, w, h, options)
{
   // Create the printer dialog box. Returns kTRUE in ret_code when
   // printerName and printProg have been set and cancel was not pressed,
   // kFALSE otherwise.

   fPrinter      = printerName;
   fPrintCommand = printProg;
   fRetCode      = ret_code;

   ChangeOptions((GetOptions() & ~kVerticalFrame) | kHorizontalFrame);

   fF1 = new TGCompositeFrame(this, 60, 20, kVerticalFrame | kFixedWidth);
   fF5 = new TGCompositeFrame(this, 60, 20, kHorizontalFrame);
   fF2 = new TGCompositeFrame(this, 60, 20, kVerticalFrame);
   fF3 = new TGCompositeFrame(fF2,  60, 20, kHorizontalFrame);
   fF4 = new TGCompositeFrame(fF2,  60, 20, kHorizontalFrame);

   fPrintButton  = new TGTextButton(fF1, new TGHotString("&Print"), 1);
   fCancelButton = new TGTextButton(fF1, new TGHotString("&Cancel"), 2);
   fF1->Resize(fPrintButton->GetDefaultWidth()+40, GetDefaultHeight());

   fPrintButton->Associate(this);
   fCancelButton->Associate(this);

   fL1 = new TGLayoutHints(kLHintsTop | kLHintsExpandX, 2, 2, 2, 0);
   fL2 = new TGLayoutHints(kLHintsTop | kLHintsRight | kLHintsExpandX,
                           2, 5, 0, 2);
   fL21 = new TGLayoutHints(kLHintsTop | kLHintsRight, 2, 5, 10, 0);

   fF1->AddFrame(fPrintButton, fL1);
   fF1->AddFrame(fCancelButton, fL1);
   AddFrame(fF1, fL21);


   fLPrintCommand = new TGLabel(fF3, new TGHotString("Print &command:"));
   fBPrintCommand = new TGTextBuffer(20);
   fBPrintCommand->AddText(0, *printProg);
   fPrintCommandEntry = new TGTextEntry(fF3, fBPrintCommand);
   fPrintCommandEntry->Associate(this);
   fPrintCommandEntry->Resize(100, fPrintCommandEntry->GetDefaultHeight());
   fL5 = new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 3, 5, 0, 0);
   fL6 = new TGLayoutHints(kLHintsRight | kLHintsCenterY, 0, 2, 0, 0);

   fF3->AddFrame(fLPrintCommand, fL5);
   fF3->AddFrame(fPrintCommandEntry, fL6);

   fLPrinter = new TGLabel(fF4, new TGHotString("&Printer:"));
   fBPrinter = new TGTextBuffer(20);
   fBPrinter->AddText(0, *printerName);
   fPrinterEntry = new TGTextEntry(fF4, fBPrinter);
   fPrinterEntry->Associate(this);
   fPrinterEntry->Resize(100, fPrinterEntry->GetDefaultHeight());
   fF4->AddFrame(fLPrinter, fL5);
   fF4->AddFrame(fPrinterEntry, fL6);

   fL3 = new TGLayoutHints(kLHintsTop | kLHintsRight, 2, 2, 2, 2);
   fL7 = new TGLayoutHints(kLHintsLeft, 10, 10, 10, 10);

   fF2->AddFrame(fF3, fL1);
   fF2->AddFrame(fF4, fL1);

   const TGPicture *printerPicture = fClient->GetPicture("printer_s.xpm");
   if (!printerPicture) {
      Error("TGPrintDialog", "printer_s.xpm not found");
      fPrinterIcon = 0;
   } else {
      fPrinterIcon = new TGIcon(fF5, printerPicture, 32, 32);
      fF5->AddFrame(fPrinterIcon, fL7);
   }
   fF5->AddFrame(fF2, fL21);
   AddFrame(fF5, fL21);

   MapSubwindows();
   Resize(GetDefaultSize());

   // position relative to the parent's window
   Int_t ax, ay;
   if (main) {
      Window_t wdum;
      gVirtualX->TranslateCoordinates(main->GetId(), GetParent()->GetId(),
                          (((TGFrame *) main)->GetWidth() - fWidth) >> 1,
                          (((TGFrame *) main)->GetHeight() - fHeight) >> 1,
                          ax, ay, wdum);
   } else {
      UInt_t root_w, root_h;
      gVirtualX->GetWindowSize(fClient->GetRoot()->GetId(), ax, ay, root_w, root_h);
      ax = (root_w - fWidth) >> 1;
      ay = (root_h - fHeight) >> 1;
   }

   Move(ax, ay);
   SetWMPosition(ax, ay);

   SetWindowName("Print");
   SetIconName("Print");

   SetMWMHints(kMWMDecorAll | kMWMDecorMaximize | kMWMDecorMenu,
               kMWMFuncAll | kMWMFuncMaximize | kMWMFuncResize,
               kMWMInputModeless);

   MapWindow();
   fClient->WaitFor(this);
}

//______________________________________________________________________________
TGPrintDialog::~TGPrintDialog()
{
   // Clean up print dialog.

   delete fPrinterIcon;
   delete fPrintButton;
   delete fCancelButton;
   delete fPrinterEntry;       // deletes also fBPrinter
   delete fPrintCommandEntry;  // deletes also fBPrintCommand
   delete fLPrinter; delete fLPrintCommand;
   delete fF1; delete fF2; delete fF3; delete fF4; delete fF5;
   delete fL1; delete fL2; delete fL3; delete fL5; delete fL6; delete fL7;
   delete fL21;
}

//______________________________________________________________________________
void TGPrintDialog::CloseWindow()
{
   // Close the dialog. On close the dialog will be deleted and cannot be
   // re-used.

  delete this;
}

//______________________________________________________________________________
Bool_t TGPrintDialog::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Process print dialog widget messages.

   const char *string;

   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_BUTTON:
               switch (parm1) {
                  case 1:
                     *fRetCode = kTRUE;
                     string = fBPrinter->GetString();
                     delete *fPrinter;
                     *fPrinter = new char[strlen(string)+1];
                     strcpy(*fPrinter, string);

                     string = fBPrintCommand->GetString();
                     delete *fPrintCommand;
                     *fPrintCommand = new char[strlen(string)+1];
                     strcpy(*fPrintCommand, string);
                     CloseWindow();
                     break;
                  case 2:
                     *fRetCode = kFALSE;
                     CloseWindow();
                     break;
               }
               break;
         }
         break;

      default:
         break;
   }

   return kTRUE;
}

//______________________________________________________________________________
TGGotoDialog::TGGotoDialog(const TGWindow *p, const TGWindow *main,
                           UInt_t w, UInt_t h, Long_t *ret_code,
                           UInt_t options) :
   TGTransientFrame(p, main, w, h, options)
{
   // Create a dialog to GoTo a specific line number. Returns -1 in
   // ret_code in case no valid line number was given or in case
   // cancel was pressed. If on input *ret_code is > 0 then this value
   // will be used as default value.

   fRetCode = ret_code;

   ChangeOptions((GetOptions() & ~kVerticalFrame) | kHorizontalFrame);

   fF1 = new TGCompositeFrame(this, 60, 20, kVerticalFrame | kFixedWidth);
   fF2 = new TGCompositeFrame(this, 60, 20, kHorizontalFrame);

   fGotoButton = new TGTextButton(fF1, new TGHotString("&Goto"), 1);
   fCancelButton = new TGTextButton(fF1, new TGHotString("&Cancel"), 2);
   fF1->Resize(fGotoButton->GetDefaultWidth()+40, GetDefaultHeight());

   fGotoButton->Associate(this);
   fCancelButton->Associate(this);

   fL1 = new TGLayoutHints(kLHintsTop | kLHintsExpandX, 2, 2, 3, 0);
   fL21 = new TGLayoutHints(kLHintsTop | kLHintsRight, 2, 5, 10, 0);

   fF1->AddFrame(fGotoButton, fL1);
   fF1->AddFrame(fCancelButton, fL1);
   AddFrame(fF1, fL21);

   fLGoTo = new TGLabel(fF2, new TGHotString("&Goto Line:"));

   fBGoTo = new TGTextBuffer(50);
   if (*fRetCode > 0) {
      char curline[32];
      sprintf(curline, "%ld", *fRetCode);
      fBGoTo->AddText(0, curline);
   } else
      fGotoButton->SetState(kButtonDisabled);
   fGoTo = new TGTextEntry(fF2, fBGoTo);
   fGoTo->Associate(this);
   fGoTo->Resize(220, fGoTo->GetDefaultHeight());
   fGoTo->SelectAll();

   fL5 = new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 3, 5, 0, 0);
   fL6 = new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 0, 2, 0, 0);

   fF2->AddFrame(fLGoTo, fL5);
   fF2->AddFrame(fGoTo, fL5);
   AddFrame(fF2, fL1);

   MapSubwindows();
   Resize(GetDefaultSize());

   // position relative to the parent's window
   Int_t ax, ay;
   if (main) {
      Window_t wdum;
      gVirtualX->TranslateCoordinates(main->GetId(), GetParent()->GetId(),
                          (((TGFrame *) main)->GetWidth() - fWidth) >> 1,
                          (((TGFrame *) main)->GetHeight() - fHeight) >> 1,
                          ax, ay, wdum);
   } else {
      UInt_t root_w, root_h;
      gVirtualX->GetWindowSize(fClient->GetRoot()->GetId(), ax, ay, root_w, root_h);
      ax = (root_w - fWidth) >> 1;
      ay = (root_h - fHeight) >> 1;
   }

   Move(ax, ay);
   SetWMPosition(ax, ay);

   SetWindowName("Goto");
   SetIconName("Print");

   SetMWMHints(kMWMDecorAll | kMWMDecorMaximize | kMWMDecorMenu,
               kMWMFuncAll | kMWMFuncMaximize | kMWMFuncResize,
               kMWMInputModeless);

   MapWindow();
   fClient->WaitFor(this);
}

//______________________________________________________________________________
TGGotoDialog::~TGGotoDialog()
{
   // Clean up goto dialog

   delete fGotoButton;
   delete fCancelButton;
   delete fGoTo;
   delete fLGoTo;
   delete fF1; delete fF2;
   delete fL1; delete fL5; delete fL6; delete fL21;
}

//______________________________________________________________________________
void TGGotoDialog::CloseWindow()
{
   // Close the dialog. On close the dialog will be deleted and cannot be
   // re-used.

   delete this;
}

//______________________________________________________________________________
Bool_t TGGotoDialog::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   const char *string;

   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_BUTTON:
               switch (parm1) {
                  case 1:
                     string = fBGoTo->GetString();
                     *fRetCode = (Long_t) atof(string);
                     CloseWindow();
                     break;
                  case 2:
                     *fRetCode = -1;
                     CloseWindow();
                     break;
               }
               break;

            default:
               break;
         }
         break;

      case kC_TEXTENTRY:
         switch (GET_SUBMSG(msg)) {
            case kTE_TEXTCHANGED:
               string = fBGoTo->GetString();
               if (strlen(string) == 0)
                  fGotoButton->SetState(kButtonDisabled);
               else
                  fGotoButton->SetState(kButtonUp);
               break;
            case kTE_ENTER:
               string = fBGoTo->GetString();
               *fRetCode = (Long_t) atof(string);
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

