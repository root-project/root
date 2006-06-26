// @(#)root/gui:$Name:  $:$Id: TGToolTip.h,v 1.7 2006/04/06 10:49:28 antcheva Exp $
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

private:
   TGLabel           *fLabel;   // help text
   TGLayoutHints     *fL1;      // layout used to place text in frame
   TTimer            *fDelay;   // popup delay timer
   const TGFrame     *fWindow;  // frame to which tool tip is associated
   const TVirtualPad *fPad;     // pad to which tooltip is associated
   const TBox        *fBox;     // box in pad to which tooltip is associated
   Int_t              fX;       // X position in fWindow where to popup
   Int_t              fY;       // Y position in fWindow where to popup

protected:
   TGToolTip(const TGToolTip& gtt);
   TGToolTip& operator=(const TGToolTip& gtt);

public:
   TGToolTip(const TGWindow *p = 0, const TGFrame *f = 0, const char *text = 0, Long_t delayms = 350);
   TGToolTip(const TGWindow *p, const TBox *b, const char *text, Long_t delayms);
   TGToolTip(const TBox *b, const char *text, Long_t delayms);
   virtual ~TGToolTip();

   virtual void DrawBorder();

   Bool_t HandleTimer(TTimer *t);
   void   Show(Int_t x, Int_t y);    //*SIGNAL*
   void   Hide();                    //*SIGNAL*
   void   Reset();                   //*SIGNAL*
   void   Reset(const TVirtualPad *parent);
   void   SetText(const char *new_text);
   void   SetPosition(Int_t x, Int_t y);
   void   SetDelay(Long_t delayms);
   const TGString *GetText() const;

   ClassDef(TGToolTip,0)  //One line help text
};

#endif
