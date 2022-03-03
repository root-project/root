// @(#)root/sessionviewer:$Id$
// Author: Bertrand Bellenot, Gerri Ganis 15/09/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSessionLogView
#define ROOT_TSessionLogView

#include "TGFrame.h"

#include "TGButton.h"

#include "TGTextView.h"

class TSessionViewer;

class TSessionLogView : public TGTransientFrame {

private:
   TSessionViewer       *fViewer;   // pointer on main viewer
   TGTextView           *fTextView; // Text view widget
   TGTextButton         *fClose;    // OK button
   TGLayoutHints        *fL1;       // layout of TGTextEdit
   TGLayoutHints        *fL2;       // layout of OK button

public:
   TSessionLogView(TSessionViewer *viewer, UInt_t w, UInt_t h);
   virtual ~TSessionLogView();

   void   AddBuffer(const char *buffer);
   void   LoadBuffer(const char *buffer);
   void   LoadFile(const char *file);

   Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);

   void   CloseWindow();
   void   ClearLogView();
   void   Popup();
   void   SetTitle();

   ClassDef(TSessionLogView, 0)  // PROOF progress dialog
};

#endif
