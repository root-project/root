// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GTitleFrame                                                          //
//                                                                      //
// This File contains the declaration of the GTitleFrame-class for      //
// the RootShower application                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef GTITLEFRAME_H
#define GTITLEFRAME_H

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGLabel;
class TGButton;
class TGPicture;
class TGIcon;


class GTitleFrame: public TGCompositeFrame {

private:

    TGLayoutHints *fRightLogoLayout;
    TGLayoutHints *fLeftLogoLayout;

    TGPicture *fRightIconPicture;
    TGIcon *fRightIcon;
    TGPicture *fLeftIconPicture;
    TGIcon *fLeftIcon;

    TGLayoutHints *fTextFrameLayout;
    TGCompositeFrame *fTextFrame;
    TGLayoutHints *fTextLabelLayout;
    TGLabel *fTextLabel1;

    TGLabel *fTextLabel2;

    TGGC fTextGC;

public:

    // Constructor & destructor
    GTitleFrame(const TGWindow *p, const Text_t *mainText, const Text_t *subText,
		        UInt_t w, UInt_t h, UInt_t options = kHorizontalFrame | kRaisedFrame);
    void ChangeRightLogo(Int_t frame);
    virtual ~GTitleFrame();

};


#endif // GTITLEFRAME_H
