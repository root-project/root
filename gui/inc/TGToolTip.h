// @(#)root/gui:$Name:  $:$Id: TGToolTip.h,v 1.1.1.1 2000/05/16 17:00:42 rdm Exp $
// Author: Fons Rademakers   22/02/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGToolTip
#define ROOT_TGToolTip


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGToolTip                                                            //
//                                                                      //
// A tooltip is a one line help text that is displayed in a window      //
// when the cursor rests over a widget. For an example of usage see     //
// the TGButton class.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGLabel;
class TTimer;
class TVirtualPad;
class TBox;


class TGToolTip : public TGCompositeFrame {

friend class TGClient;

private:
   TGLabel           *fLabel;   // help text
   TGLayoutHints     *fL1;      // layout used to place text in frame
   TTimer            *fDelay;   // popup delay timer
   const TGFrame     *fWindow;  // frame to which tool tip is associated
   const TVirtualPad *fPad;     // pad to which tooltip is associated
   const TBox        *fBox;     // box in pad to which tooltip is associated
   Int_t              fX;       // X position in fWindow where to popup
   Int_t              fY;       // Y position in fWindow where to popup

   static ULong_t fgLightYellowPixel;

public:
   TGToolTip(const TGWindow *p, const TGFrame *f, const char *text, Long_t delayms);
   TGToolTip(const TGWindow *p, const TBox *b, const char *text, Long_t delayms);
   TGToolTip(const TBox *b, const char *text, Long_t delayms);
   virtual ~TGToolTip();

   virtual void DrawBorder();

   Bool_t HandleTimer(TTimer *t);
   void   Show(Int_t x, Int_t y);
   void   Hide();
   void   Reset();
   void   Reset(const TVirtualPad *parent);
   void   SetText(const char *new_text);
   void   SetPosition(Int_t x, Int_t y);

   ClassDef(TGToolTip,0)  //One line help text
};

#endif
