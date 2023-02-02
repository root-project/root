// @(#)root/gui:$Id$
// Author: Fons Rademakers   22/02/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGToolTip
#define ROOT_TGToolTip


#include "TGFrame.h"

class TGLabel;
class TTimer;
class TVirtualPad;
class TBox;


class TGToolTip : public TGCompositeFrame {

private:
   TGLabel           *fLabel;   ///< help text
   TGLayoutHints     *fL1;      ///< layout used to place text in frame
   TTimer            *fDelay;   ///< popup delay timer
   const TGFrame     *fWindow;  ///< frame to which tool tip is associated
   const TVirtualPad *fPad;     ///< pad to which tooltip is associated
   const TBox        *fBox;     ///< box in pad to which tooltip is associated
   Int_t              fX;       ///< X position in fWindow where to popup
   Int_t              fY;       ///< Y position in fWindow where to popup

   TGToolTip(const TGToolTip&) = delete;
   TGToolTip& operator=(const TGToolTip&) = delete;

public:
   TGToolTip(const TGWindow *p = nullptr, const TGFrame *f = nullptr, const char *text = nullptr, Long_t delayms = 350);
   TGToolTip(const TGWindow *p, const TBox *b, const char *text, Long_t delayms);
   TGToolTip(const TBox *b, const char *text, Long_t delayms);
   TGToolTip(Int_t x, Int_t y, const char *text, Long_t delayms);
   virtual ~TGToolTip();

   void DrawBorder() override;

   Bool_t HandleTimer(TTimer *t) override;
   void   Show(Int_t x, Int_t y);    //*SIGNAL*
   void   Hide();                    //*SIGNAL*
   void   Reset();                   //*SIGNAL*
   void   Reset(const TVirtualPad *parent);
   void   SetText(const char *new_text);
   void   SetPosition(Int_t x, Int_t y);
   void   SetDelay(Long_t delayms);
   const TGString *GetText() const;

   ClassDefOverride(TGToolTip,0)  //One or multiple lines help text
};

#endif
