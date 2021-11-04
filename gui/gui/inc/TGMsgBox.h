// @(#)root/gui:$Id$
// Author: Fons Rademakers   09/01/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGMsgBox
#define ROOT_TGMsgBox


#include "TGFrame.h"
#include "TGWidget.h"

//--- Icon types

enum EMsgBoxIcon {
   kMBIconStop,
   kMBIconQuestion,
   kMBIconExclamation,
   kMBIconAsterisk
};

//--- Button types and return codes

enum EMsgBoxButton {
   kMBYes     = BIT(0),
   kMBNo      = BIT(1),
   kMBOk      = BIT(2),
   kMBApply   = BIT(3),
   kMBRetry   = BIT(4),
   kMBIgnore  = BIT(5),
   kMBCancel  = BIT(6),
   kMBClose   = BIT(7),
   kMBDismiss = BIT(8),
   kMBYesAll  = BIT(9),
   kMBNoAll   = BIT(10),
   kMBAppend  = BIT(11),
   kMBNewer   = BIT(12)
};


class TGButton;
class TGIcon;
class TGLabel;


class TGMsgBox : public TGTransientFrame {

protected:
   TGButton            *fYes, *fNo, *fOK, *fApply;   ///< buttons in dialog box
   TGButton            *fRetry, *fIgnore, *fCancel;  ///< buttons in dialog box
   TGButton            *fClose, *fYesAll, *fNoAll;   ///< buttons in dialog box
   TGButton            *fNewer, *fAppend, *fDismiss; ///< buttons in dialog box
   TGIcon              *fIcon;                       ///< icon
   TGHorizontalFrame   *fButtonFrame;                ///< frame containing buttons
   TGHorizontalFrame   *fIconFrame;                  ///< frame containing icon and text
   TGVerticalFrame     *fLabelFrame;                 ///< frame containing text
   TGLayoutHints       *fL1, *fL2, *fL3, *fL4, *fL5; ///< layout hints
   TList               *fMsgList;                    ///< text (list of TGLabels)
   Int_t               *fRetCode;                    ///< address to store return code

   void PMsgBox(const char *title, const char *msg, const TGPicture *icon,
                Int_t buttons, Int_t *ret_code, Int_t text_align);

private:
   TGMsgBox(const TGMsgBox&) = delete;
   TGMsgBox& operator=(const TGMsgBox&) = delete;

public:
   TGMsgBox(const TGWindow *p = nullptr, const TGWindow *main = nullptr,
            const char *title = nullptr, const char *msg = nullptr, const TGPicture *icon = nullptr,
            Int_t buttons = kMBDismiss, Int_t *ret_code = nullptr,
            UInt_t options = kVerticalFrame,
            Int_t text_align = kTextCenterX | kTextCenterY);
   TGMsgBox(const TGWindow *p, const TGWindow *main,
            const char *title, const char *msg, EMsgBoxIcon icon,
            Int_t buttons = kMBDismiss, Int_t *ret_code = nullptr,
            UInt_t options = kVerticalFrame,
            Int_t text_align = kTextCenterX | kTextCenterY);
   virtual ~TGMsgBox();

   virtual void CloseWindow();
   virtual Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);
   virtual Bool_t HandleKey(Event_t* event);

   ClassDef(TGMsgBox,0)  // A message dialog box
};

#endif
