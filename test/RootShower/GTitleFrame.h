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

#include "TGFrame.h"

class TGLabel;
class TGButton;
class TGPicture;
class TGIcon;

class GTitleFrame: public TGCompositeFrame {

private:
   TGLayoutHints     *fRightLogoLayout;   // Right logo layout
   TGLayoutHints     *fLeftLogoLayout;    // Left logo layout

   TGPicture         *fRightIconPicture;  // Right icon's picture
   TGIcon            *fRightIcon;         // Right icon (logo)
   TGPicture         *fLeftIconPicture;   // Left icon's picture
   TGIcon            *fLeftIcon;          // Right icon (logo)

   TGLayoutHints     *fTextFrameLayout;
   TGCompositeFrame  *fTextFrame;
   TGLayoutHints     *fTextLabelLayout;
   TGLabel           *fTextLabel1;        // First line title's label
   TGLabel           *fTextLabel2;        // Second line title's label

public:
   // Constructor & destructor
   GTitleFrame(const TGWindow *p, const char *mainText, const char *subText,
               UInt_t w, UInt_t h, UInt_t options = kHorizontalFrame | kRaisedFrame);
   void ChangeRightLogo(Int_t frame);
   virtual ~GTitleFrame();
};

#endif // GTITLEFRAME_H
