// @(#)root/ged:$Id$
// Author: Denis Favre-Miville   08/09/05

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TStyleDialog
#define ROOT_TStyleDialog


#include "TGFrame.h"

class TGLabel;
class TGTextButton;
class TGTextEntry;
class TGTransientFrame;
class TList;
class TStyle;
class TStyleManager;
class TVirtualPad;

class TStyleDialog : public TGTransientFrame {

private:
   TStyleManager       *fStyleManager;    ///< parent style manager
   TGTextEntry         *fName;            ///< TStyle name text entry
   TGLabel             *fNameLabel;       ///< name label
   TGTextEntry         *fTitle;           ///< TStyle title text entry
   TGLabel             *fTitleLabel;      ///< title label
   TGLabel             *fWarnLabel;       ///< label for warnings
   TGTextButton        *fOK;              ///< save button
   TGTextButton        *fCancel;          ///< cancel button
   TStyle              *fCurStyle;        ///< style to copy or to rename
   Int_t                fMode;            ///< 1=new, 2=rename, 3=import
   TVirtualPad         *fCurPad;          ///< current pad from which to import
   TList               *fTrashListFrame;  ///< to avoid memory leak
   TList               *fTrashListLayout; ///< to avoid memory leak

public:
   TStyleDialog(TStyleManager *sm, TStyle *cur, Int_t mode,
                  TVirtualPad *currentPad = 0);
   virtual ~TStyleDialog();

   void DoCloseWindow();                  // SLOT
   void DoCancel();                       // SLOT
   void DoOK();                           // SLOT
   void DoUpdate();                       // SLOT

   ClassDef(TStyleDialog, 0) // Dialog box used by the TStyleManager class
};

#endif
