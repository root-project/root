// @(#)root/gui:$Id$
// Author: Fons Rademakers   24/02/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootHelpDialog
#define ROOT_TRootHelpDialog


#include "TGFrame.h"

class TGTextView;
class TGTextButton;


class TRootHelpDialog : public TGTransientFrame {

private:
   TGTextView       *fView;   ///< text view
   TGTextButton     *fOK;     ///< OK button
   TGLayoutHints    *fL1;     ///< layout of TGTextView
   TGLayoutHints    *fL2;     ///< layout of OK button

private:
   TRootHelpDialog(const TRootHelpDialog&) = delete;
   TRootHelpDialog& operator=(const TRootHelpDialog&) = delete;

public:
   TRootHelpDialog(const TGWindow *main = nullptr, const char *title = "ROOT Help Dialog",
                   UInt_t w = 1, UInt_t h = 1);
   virtual ~TRootHelpDialog();

   void   SetText(const char *helpText);
   void   AddText(const char *helpText);

   void   Popup();
   void   CloseWindow();
   Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TRootHelpDialog,0)  //Dialog to display help text
};

#endif
