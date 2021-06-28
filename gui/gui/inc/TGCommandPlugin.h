// @(#)root/gui:$Id$
// Author: Bertrand Bellenot   26/09/2007

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGCommandPlugin
#define ROOT_TGCommandPlugin


#include "TGFrame.h"

class TGLabel;
class TGComboBox;
class TGTextEntry;
class TGTextBuffer;
class TGTextView;
class TTimer;

class TGCommandPlugin : public TGMainFrame {

protected:
   Bool_t             fHistAdd;           ///< flag to add commands to history
   Int_t              fPid;               ///< current process id
   Int_t              fPos;               ///< current history position
   TGHorizontalFrame *fHf;                ///< horizontal frame
   TGLabel           *fLabel;             ///< "command :" label
   TGComboBox        *fComboCmd;          ///< commands combobox
   TGTextEntry       *fCommand;           ///< command text entry widget
   TGTextBuffer      *fCommandBuf;        ///< command text buffer
   TGTextView        *fStatus;            ///< output capture view
   TTimer            *fTimer;             ///< for local/remote update
   TString           fTempString;         ///< temporary command string

public:

   TGCommandPlugin(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~TGCommandPlugin();

   void           CheckRemote(const char * /*str*/);
   void           HandleArrows(Int_t keysym);
   void           HandleCommand();
   void           HandleTab();
   void           HandleTextChanged(const char *);
   Bool_t         GetHistAdd() { return fHistAdd; }
   void           SetHistAdd(Bool_t add = kTRUE);

   virtual Bool_t HandleTimer(TTimer *t);
   void           StopTimer();

   ClassDef(TGCommandPlugin, 0) // Command (I/O redirection) plugin for the new ROOT Browser
};

#endif
