// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Definition of a dialog box used to access the main shower parameters //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef SETTINGSDIALOG_H
#define SETTINGSDIALOG_H

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

#ifndef ROOT_TGButton
#include "TGButton.h"
#endif

#ifndef ROOT_TGTextEntry
#include "TGTextEntry.h"
#endif

#ifndef ROOT_TGListBox
#include <TGListBox.h>
#endif

#ifndef ROOT_TGTab
#include <TGTab.h>
#endif

class SettingsDialog : public TGTransientFrame {

private:
    TGCompositeFrame    *fFrame1;
    TGGroupFrame        *fF1, *fF3, *fF4;
    TGButton            *fOkButton, *fCancelButton, *fHelpButton;
    TGCheckButton       *fCheck1;
    TGListBox           *fListBox;
    TGTab               *fTab;
    TGTextEntry         *fTxt4, *fTxt5;
    TGLayoutHints       *fL1, *fL2, *fL3;
    Int_t                fFirstEntry;
    Int_t                fLastEntry;

public:
    SettingsDialog(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h,
                         UInt_t options = kVerticalFrame);
    virtual ~SettingsDialog();

    // slots
    virtual void     CloseWindow();
    virtual Bool_t   ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
};

#endif
