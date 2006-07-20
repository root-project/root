// @(#)root/gui:$Name:  $:$Id: TGInputDialog.h,v 1.0 2006/07/19 11:13:18 dgmaline Exp $
// Author: David Gonzalez Maline  19/07/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGInputDialog
#define ROOT_TGInputDialog

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Input Dialog Widget                                                   //
//                                                                       //
// An Input dialog box                                                   //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGLabel;
class TGTextEntry;
class TGTextButton;

class TGInputDialog : public TGTransientFrame {

private:
   TGLabel          *fLabel;   // text entry label
   TGTextEntry      *fTE;      // text entry widget
   TGTextButton     *fOk;      // ok button
   TGTextButton     *fCancel;  // cancel button
   char             *fRetStr;  // address to store return string

public:
   TGInputDialog(const TGWindow *p = 0, const TGWindow *main = 0,
                 const char *prompt =0, const char *defval = 0, 
                 char *retstr = 0, UInt_t options = kVerticalFrame);
   ~TGInputDialog();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t);

   ClassDef(TGInputDialog, 0)  // Simple input dialog

};

#endif
