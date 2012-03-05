// @(#)root/gui:$Id$
// Author: Fons Rademakers   09/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGMsgBox
#define ROOT_TGMsgBox


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMsgBox                                                              //
//                                                                      //
// A message dialog box.                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGString
#include "TGString.h"
#endif
#ifndef ROOT_TGPicture
#include "TGPicture.h"
#endif
#ifndef ROOT_TGWidget
#include "TGWidget.h"
#endif


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
   TGButton            *fYes, *fNo, *fOK, *fApply;   // buttons in dialog box
   TGButton            *fRetry, *fIgnore, *fCancel;  // buttons in dialog box
   TGButton            *fClose, *fYesAll, *fNoAll;   // buttons in dialog box
   TGButton            *fNewer, *fAppend, *fDismiss; // buttons in dialog box
   TGIcon              *fIcon;                       // icon
   TGHorizontalFrame   *fButtonFrame;                // frame containing buttons
   TGHorizontalFrame   *fIconFrame;                  // frame containing icon and text
   TGVerticalFrame     *fLabelFrame;                 // frame containing text
   TGLayoutHints       *fL1, *fL2, *fL3, *fL4, *fL5; // layout hints
   TList               *fMsgList;                    // text (list of TGLabels)
   Int_t               *fRetCode;                    // address to store return code

   void PMsgBox(const char *title, const char *msg, const TGPicture *icon,
                Int_t buttons, Int_t *ret_code, Int_t text_align);

private:
   TGMsgBox(const TGMsgBox&);              // not implemented
   TGMsgBox& operator=(const TGMsgBox&);   // not implemented

public:
   TGMsgBox(const TGWindow *p = 0, const TGWindow *main = 0,
            const char *title = 0, const char *msg = 0, const TGPicture *icon = 0,
            Int_t buttons = kMBDismiss, Int_t *ret_code = 0,
            UInt_t options = kVerticalFrame,
            Int_t text_align = kTextCenterX | kTextCenterY);
   TGMsgBox(const TGWindow *p, const TGWindow *main,
            const char *title, const char *msg, EMsgBoxIcon icon,
            Int_t buttons = kMBDismiss, Int_t *ret_code = 0,
            UInt_t options = kVerticalFrame,
            Int_t text_align = kTextCenterX | kTextCenterY);
   virtual ~TGMsgBox();

   virtual void CloseWindow();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   virtual Bool_t HandleKey(Event_t* event);

   ClassDef(TGMsgBox,0)  // A message dialog box
};

#endif
