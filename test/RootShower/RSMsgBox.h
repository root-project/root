// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Definition of a simple message box for the RootShower application    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOTSHOWERMSGBOX_H
#define ROOTSHOWERMSGBOX_H

#include "TGFrame.h"

#include "TGButton.h"

#include "TGPicture.h"

#include "TGIcon.h"

#include "TGLabel.h"


class RootShowerMsgBox : public TGTransientFrame {

private:
    TGVerticalFrame*    fVFrame;
    TGHorizontalFrame*  fHFrame;
    TGTextButton*       fCancelButton;
    TGTextButton*       fHelpButton;
    TGTextButton*       fOkButton;
    TGLabel*            fLabel1;
    TGLabel*            fLabel2;

    TGLayoutHints      *fLogoLayout;

    TGLayoutHints*      fL1;
    TGLayoutHints*      fL2;

    TGLayoutHints*      fBly;
    TGLayoutHints*      fBfly;

public:
    RootShowerMsgBox(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h,
                    UInt_t options = kMainFrame | kVerticalFrame);
    virtual ~RootShowerMsgBox();

    virtual void CloseWindow();
    virtual Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);
};

#endif // ROOTSHOWERMSGBOX_H
