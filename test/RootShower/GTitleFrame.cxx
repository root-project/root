// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

#include <TSystem.h>
#include <TGButton.h>
#include <TGLabel.h>
#include <TGPicture.h>
#include <TGIcon.h>
#include <TGResourcePool.h>

#include "GTitleFrame.h"

//______________________________________________________________________________
//
// GTitleFrame
//______________________________________________________________________________

//______________________________________________________________________________
GTitleFrame::GTitleFrame(const TGWindow *p,
			     const Text_t *mainText, const Text_t *subText,
			     UInt_t w, UInt_t h,
			     UInt_t options):
             TGCompositeFrame(p, w, h, options)
{
    // Create GTitleFrame object, with TGWindow parent 'p', text 'mainText'
    // with sub text 'subText'.
    const TGFont *font = fClient->GetFont("-*-times-bold-r-*-*-24-*-*-*-*-*-*-*");
    if (!font)
       font = fClient->GetResourcePool()->GetDefaultFont();
    FontStruct_t labelfont = font->GetFontStruct();
    GCValues_t   gval;
    gval.fMask = kGCForeground | kGCFont;
    gval.fFont = font->GetFontHandle();
    gClient->GetColorByName("red", gval.fForeground);
    fTextGC.SetAttributes(&gval);

    // add pictures
    TString theLeftLogoFilename = StrDup(gProgPath);
    theLeftLogoFilename.Append("/icons/left.xpm");
    fLeftIconPicture = (TGPicture *)gClient->GetPicture(theLeftLogoFilename);
    fLeftIcon = new TGIcon(this, fLeftIconPicture,
                    fLeftIconPicture->GetWidth(),
                    fLeftIconPicture->GetHeight());
    fLeftLogoLayout = new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0);
    AddFrame(fLeftIcon, fLeftLogoLayout);

    TString theRightLogoFilename = StrDup(gProgPath);
    theRightLogoFilename.Append("/anim/anim01.xpm");
    fRightIconPicture = (TGPicture *)gClient->GetPicture(theRightLogoFilename);
    fRightIcon = new TGIcon(this, fRightIconPicture,
                     fRightIconPicture->GetWidth(),
                     fRightIconPicture->GetHeight());
    fRightLogoLayout = new TGLayoutHints(kLHintsRight, 0, 0, 0, 0);
    AddFrame(fRightIcon, fRightLogoLayout);

    // add text
    fTextFrameLayout = new TGLayoutHints(kLHintsCenterX | kLHintsCenterY, 0, 0, 0, 0);
    fTextLabelLayout = new TGLayoutHints(kLHintsTop | kLHintsExpandX, 10, 10, 10, 10);
    fTextFrame = new TGCompositeFrame(this, 0, 0, kVerticalFrame);
    fTextLabel1 = new TGLabel(fTextFrame, mainText, fTextGC(), labelfont);

    fTextLabel2 = new TGLabel(fTextFrame, subText, fTextGC(), labelfont);
    fTextFrame->AddFrame(fTextLabel1, fTextLabelLayout);
    fTextFrame->AddFrame(fTextLabel2, fTextLabelLayout);

    AddFrame(fTextFrame, fTextFrameLayout);

}

//______________________________________________________________________________
void GTitleFrame::ChangeRightLogo(Int_t frame)
{
    // Change the right logo ( used for animation )
    Char_t name[12];
    TString theRightLogoFilename = StrDup(gProgPath);
    sprintf(name,"%02d.xpm",frame);
    theRightLogoFilename.Append("/anim/anim");
    theRightLogoFilename.Append(name);
    fRightIconPicture = (TGPicture *)gClient->GetPicture(theRightLogoFilename);
    fRightIcon->SetPicture(fRightIconPicture);
}

//______________________________________________________________________________
GTitleFrame::~GTitleFrame()
{
    // Destroy GTitleFrame object. Delete all created widgets
    delete fTextLabel1;

    delete fTextLabel2;

    delete fTextFrame;
    delete fTextLabelLayout;
    delete fTextFrameLayout;

    delete fLeftLogoLayout;
    delete fRightLogoLayout;
}

