// @(#)root/gui:$Name:  $:$Id: TGSlider.h,v 1.2 2000/10/22 19:28:58 rdm Exp $
// Author: Fons Rademakers   14/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGSlider
#define ROOT_TGSlider


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGSlider, TGVSlider and TGHSlider                                    //
//                                                                      //
// Slider widgets allow easy selection out of a range.                  //
// Sliders can be either horizontal or vertical oriented and there is   //
// a choice of two different slider types and three different types     //
// of tick marks.                                                       //
//                                                                      //
// TGSlider is an abstract base class. Use the concrete TGVSlider and   //
// TGHSlider.                                                           //
//                                                                      //
// Dragging the slider will generate the event:                         //
// kC_VSLIDER, kSL_POS, slider id, position  (for vertical slider)      //
// kC_HSLIDER, kSL_POS, slider id, position  (for horizontal slider)    //
//                                                                      //
// Pressing the mouse will generate the event:                          //
// kC_VSLIDER, kSL_PRESS, slider id, 0  (for vertical slider)           //
// kC_HSLIDER, kSL_PRESS, slider id, 0  (for horizontal slider)         //
//                                                                      //
// Releasing the mouse will generate the event:                         //
// kC_VSLIDER, kSL_RELEASE, slider id, 0  (for vertical slider)         //
// kC_HSLIDER, kSL_RELEASE, slider id, 0  (for horizontal slider)       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGWidget
#include "TGWidget.h"
#endif


//--- sizes for vert. and horz. sliders

enum ESliderSize {
   kSliderWidth  = 24,
   kSliderHeight = kSliderWidth
};


enum ESliderType {
   //--- slider types (type of slider picture)
   kSlider1        = BIT(0),
   kSlider2        = BIT(1),

   //--- scaling of slider
   kScaleNo        = BIT(2),
   kScaleDownRight = BIT(3),
   kScaleBoth      = BIT(4)
};


class TGSlider : public TGFrame, public TGWidget {

protected:
   Int_t            fPos;           // logical position between fVmin and fVmax
   Int_t            fRelPos;        // slider position in pixel coordinates
   Int_t            fVmin;          // logical lower limit of slider
   Int_t            fVmax;          // logical upper limit of slider
   Int_t            fType;          // slider type bits
   Int_t            fScale;         // tick mark scale
   Bool_t           fDragging;      // true if in dragging mode
   const TGPicture *fSliderPic;     // picture to draw slider

public:
   TGSlider(const TGWindow *p, UInt_t w, UInt_t h, UInt_t type, Int_t id = -1,
            UInt_t options = kChildFrame,
            ULong_t back = GetDefaultFrameBackground());

   virtual ~TGSlider() { }

   virtual Bool_t HandleButton(Event_t *event) = 0;
   virtual Bool_t HandleMotion(Event_t *event) = 0;

   virtual void  SetScale(Int_t scale) { fScale = scale; }
   virtual void  SetRange(Int_t min, Int_t max) { fVmin = min; fVmax = max; }
   virtual void  SetPosition(Int_t pos) { fPos = pos; fClient->NeedRedraw(this); }
   virtual Int_t GetPosition() const { return fPos; }
   virtual void  MapSubwindows() { TGWindow::MapSubwindows(); }

   virtual void  PositionChanged(Int_t pos) { Emit("PositionChanged(Int_t)", pos); } //*SIGNAL*
   virtual void  Pressed() { Emit("Pressed()"); }    //*SIGNAL*
   virtual void  Released() { Emit("Released()"); }  //*SIGNAL*

   ClassDef(TGSlider,0)  // Slider widget abstract base class
};


class TGVSlider : public TGSlider {

protected:
   Int_t   fYp;      // vertical slider y position in pixel coordinates

   virtual void DoRedraw();

public:
   TGVSlider(const TGWindow *p, UInt_t h, UInt_t type, Int_t id = -1,
             UInt_t options = kVerticalFrame,
             ULong_t back = GetDefaultFrameBackground());
   virtual ~TGVSlider();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual TGDimension GetDefaultSize() const
                     { return TGDimension(kSliderWidth, fHeight); }

   ClassDef(TGVSlider,0)  // Vertical slider widget
};


class TGHSlider : public TGSlider {

protected:
   Int_t       fXp;     // horizontal slider x position in pixel coordinates

   virtual void DoRedraw();

public:
   TGHSlider(const TGWindow *p, UInt_t w, UInt_t type, Int_t id = -1,
             UInt_t options = kHorizontalFrame,
             ULong_t back = GetDefaultFrameBackground());
   virtual ~TGHSlider();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual TGDimension GetDefaultSize() const
                     { return TGDimension(fWidth, kSliderHeight); }

   ClassDef(TGHSlider,0)  // Horizontal slider widget
};

#endif
