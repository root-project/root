// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   09/01/98

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
// TMsgBox                                                              //
//                                                                      //
// A message dialog box.                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGMsgBox.h"
#include "TGButton.h"
#include "TGIcon.h"
#include "TGLabel.h"
#include "TMath.h"


ClassImp(TGMsgBox)


//______________________________________________________________________________
TGMsgBox::TGMsgBox(const TGWindow *p, const TGWindow *main,
                   const char *title, const char *msg, const TGPicture *icon,
                   Int_t buttons, Int_t *ret_code, UInt_t options) :
   TGTransientFrame(p, main, 10, 10, options)
{
   // Create a message dialog box.

   PMsgBox(main, title, msg, icon, buttons, ret_code);
}

//______________________________________________________________________________
TGMsgBox::TGMsgBox(const TGWindow *p, const TGWindow *main,
                   const char *title, const char *msg, EMsgBoxIcon icon,
                   Int_t buttons, Int_t *ret_code, UInt_t options) :
   TGTransientFrame(p, main, 10, 10, options)
{
   // Create a message dialog box.

   const TGPicture *icon_pic;

   switch (icon) {
      case kMBIconStop:
         icon_pic = fClient->GetPicture("mb_stop_s.xpm");
         if (!icon_pic) Error("TGMsgBox", "mb_stop_s.xpm not found");
         break;

      case kMBIconQuestion:
         icon_pic = fClient->GetPicture("mb_question_s.xpm");
         if (!icon_pic) Error("TGMsgBox", "mb_question_s.xpm not found");
         break;

      case kMBIconExclamation:
         icon_pic = fClient->GetPicture("mb_exclamation_s.xpm");
         if (!icon_pic) Error("TGMsgBox", "mb_exclamation_s.xpm not found");
         break;

      case kMBIconAsterisk:
         icon_pic = fClient->GetPicture("mb_asterisk_s.xpm");
         if (!icon_pic) Error("TGMsgBox", "mb_asterisk_s.xpm not found");
         break;

      default:
         icon_pic = 0;
         break;
   }

   PMsgBox(main, title, msg, icon_pic, buttons, ret_code);
}

//______________________________________________________________________________
void TGMsgBox::PMsgBox(const TGWindow *main, const char *title,
                       const char *msg, const TGPicture *icon,
                       Int_t buttons, Int_t *ret_code)
{
   // Protected, common message dialog box initialization.

   Int_t  ax, ay;
   UInt_t nb, width, height;

   fYes = fNo = fOK = fApply = fRetry = fIgnore = fCancel = fClose =
   fDismiss   = 0;
   fIcon      = 0;
   fMsgList   = new TList;
   fRetCode   = ret_code;
   nb = width = 0;

   // create the buttons

   fButtonFrame = new TGHorizontalFrame(this, 60, 20, kFixedWidth);
   fL1 = new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 3, 3, 0, 0);

   buttons &= (kMBYes | kMBNo | kMBOk | kMBApply |
               kMBRetry | kMBIgnore | kMBCancel | kMBClose | kMBDismiss);
   if (buttons == 0) buttons = kMBDismiss;

   if (buttons & kMBYes) {
      fYes = new TGTextButton(fButtonFrame, new TGHotString("&Yes"), kMBYes);
      fYes->Associate(this);
      fButtonFrame->AddFrame(fYes, fL1);
      width = TMath::Max(width, fYes->GetDefaultWidth()); ++nb;
   }

   if (buttons & kMBNo) {
      fNo = new TGTextButton(fButtonFrame, new TGHotString("&No"), kMBNo);
      fNo->Associate(this);
      fButtonFrame->AddFrame(fNo, fL1);
      width = TMath::Max(width, fNo->GetDefaultWidth()); ++nb;
   }

   if (buttons & kMBOk) {
      fOK = new TGTextButton(fButtonFrame, new TGHotString("&OK"), kMBOk);
      fOK->Associate(this);
      fButtonFrame->AddFrame(fOK, fL1);
      width = TMath::Max(width, fOK->GetDefaultWidth()); ++nb;
   }

   if (buttons & kMBApply) {
      fApply = new TGTextButton(fButtonFrame, new TGHotString("&Apply"), kMBApply);
      fApply->Associate(this);
      fButtonFrame->AddFrame(fApply, fL1);
      width = TMath::Max(width, fApply->GetDefaultWidth()); ++nb;
   }

   if (buttons & kMBRetry) {
      fRetry = new TGTextButton(fButtonFrame, new TGHotString("&Retry"), kMBRetry);
      fRetry->Associate(this);
      fButtonFrame->AddFrame(fRetry, fL1);
      width = TMath::Max(width, fRetry->GetDefaultWidth()); ++nb;
   }

   if (buttons & kMBIgnore) {
      fIgnore = new TGTextButton(fButtonFrame, new TGHotString("&Ignore"), kMBIgnore);
      fIgnore->Associate(this);
      fButtonFrame->AddFrame(fIgnore, fL1);
      width = TMath::Max(width, fIgnore->GetDefaultWidth()); ++nb;
   }

   if (buttons & kMBCancel) {
      fCancel = new TGTextButton(fButtonFrame, new TGHotString("&Cancel"), kMBCancel);
      fCancel->Associate(this);
      fButtonFrame->AddFrame(fCancel, fL1);
      width = TMath::Max(width, fCancel->GetDefaultWidth()); ++nb;
   }

   if (buttons & kMBClose) {
      fClose = new TGTextButton(fButtonFrame, new TGHotString("C&lose"), kMBClose);
      fClose->Associate(this);
      fButtonFrame->AddFrame(fClose, fL1);
      width = TMath::Max(width, fClose->GetDefaultWidth()); ++nb;
   }

   if (buttons & kMBDismiss) {
      fDismiss = new TGTextButton(fButtonFrame, new TGHotString("&Dismiss"), kMBDismiss);
      fDismiss->Associate(this);
      fButtonFrame->AddFrame(fDismiss, fL1);
      width = TMath::Max(width, fDismiss->GetDefaultWidth()); ++nb;
   }

   // place buttons at the bottom

   fL2 = new TGLayoutHints(kLHintsBottom | kLHintsCenterX, 0, 0, 5, 5);
   AddFrame(fButtonFrame, fL2);

   // keep the buttons centered and with the same width

   fButtonFrame->Resize((width + 20) * nb, GetDefaultHeight());

   fIconFrame = new TGHorizontalFrame(this, 60, 20);

   fL3 = new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 2, 2, 2, 2);

   if (icon) {
      fIcon = new TGIcon(fIconFrame, icon, icon->GetWidth(), icon->GetHeight());
      fIconFrame->AddFrame(fIcon, fL3);
   }

   fLabelFrame = new TGVerticalFrame(fIconFrame, 60, 20);

   fL4 = new TGLayoutHints(kLHintsCenterY | kLHintsLeft | kLHintsExpandX,
                           4, 2, 2, 2);
   fL5 = new TGLayoutHints(kLHintsTop | kLHintsExpandX, 10, 10, 7, 2);

   // make one label per line of the message
   TGLabel *label;

   char *line;
   char *tmpMsg, *nextLine;

   tmpMsg = new char[strlen(msg) + 1];
   nextLine = tmpMsg;

   line = tmpMsg;
   strcpy(nextLine, msg);
   while ((nextLine = strchr(line, '\n'))) {
      *nextLine = 0;
      label = new TGLabel(fLabelFrame, line);
      fMsgList->Add(label);
      fLabelFrame->AddFrame(label, fL4);
      line = nextLine + 1;
   }

   label = new TGLabel(fLabelFrame, line);
   fMsgList->Add(label);
   fLabelFrame->AddFrame(label, fL4);
   delete [] tmpMsg;
	
   fIconFrame->AddFrame(fLabelFrame, fL4);
   AddFrame(fIconFrame, fL5);

   MapSubwindows();

   width  = GetDefaultWidth();
   height = GetDefaultHeight();

   Resize(width, height);

   // position relative to the parent's window

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

   // make the message box non-resizable

   SetWMSize(width, height);
   SetWMSizeHints(width, height, width, height, 0, 0);

   // set names

   SetWindowName(title);
   SetIconName(title);
   SetClassHints("MsgBox", "MsgBox");

   SetMWMHints(kMWMDecorAll | kMWMDecorResizeH  | kMWMDecorMaximize |
                              kMWMDecorMinimize | kMWMDecorMenu,
               kMWMFuncAll  | kMWMFuncResize    | kMWMFuncMaximize |
                              kMWMFuncMinimize,
               kMWMInputModeless);

   MapWindow();
   fClient->WaitFor(this);
}

//______________________________________________________________________________
TGMsgBox::~TGMsgBox()
{
   // Destroy message dialog box.

   if (fYes)     delete fYes;
   if (fNo)      delete fNo;
   if (fOK)      delete fOK;
   if (fApply)   delete fApply;
   if (fRetry)   delete fRetry;
   if (fIgnore)  delete fIgnore;
   if (fCancel)  delete fCancel;
   if (fClose)   delete fClose;
   if (fDismiss) delete fDismiss;

   if (fIcon) delete fIcon;
   delete fButtonFrame;
   delete fIconFrame;
   delete fLabelFrame;
   fMsgList->Delete();
   delete fMsgList;
   delete fL1; delete fL2; delete fL3; delete fL4; delete fL5;
}

//______________________________________________________________________________
void TGMsgBox::CloseWindow()
{
   // Close dialog box. Before deleting itself it sets the return code
   // to kMBClose.

   if (fRetCode) *fRetCode = kMBClose;
   delete this;
}

//______________________________________________________________________________
Bool_t TGMsgBox::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Process message dialog box event.

   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_BUTTON:
               if (fRetCode) *fRetCode = (Int_t) parm1;
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
