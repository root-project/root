// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Definition of the "About" message box for the RootShower application //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOTSHOWERABOUT_H
#define ROOTSHOWERABOUT_H

#include "TGFrame.h"

#include "TGButton.h"

#include "TGPicture.h"

#include "TGIcon.h"

#include "TGLabel.h"


class RootShowerAbout : public TGTransientFrame {

private:
    TGVerticalFrame*    fVFrame;
    TGHorizontalFrame*  fHFrame;
    TGTextButton*       fOkButton;
    TGLabel*            fLabel1;
    TGLabel*            fLabel2;
    TGLabel*            fLabel4;

    TGLayoutHints      *fLogoLayout;

    TGLayoutHints*      fL1;
    TGLayoutHints*      fL2;

    TGLayoutHints*      fBly;
    TGLayoutHints*      fBfly;

public:
    RootShowerAbout(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h,
                   UInt_t options = kMainFrame | kVerticalFrame);
    virtual ~RootShowerAbout();

    virtual void CloseWindow();
    virtual Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);
};

#endif // ROOTSHOWERABOUT_H
