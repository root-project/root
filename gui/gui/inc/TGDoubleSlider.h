// @(#)root/gui:$Id$
// Author: Reiner Rohlfs   30/09/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGDoubleSlider
#define ROOT_TGDoubleSlider


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGDoubleSlider, TGDoubleVSlider and TGDoubleHSlider                  //
//                                                                      //
// DoubleSlider widgets allow easy selection of a min and a max value   //
// out of a range.                                                      //
// DoubleSliders can be either horizontal or vertical oriented and      //
// there is a choice of three different types of tick marks.            //
//                                                                      //
// To change the min value press the mouse near to the left / bottom    //
// edge of the slider.                                                  //
// To change the max value press the mouse near to the right / top      //
// edge of the slider.                                                  //
// To change both values simultaneously press the mouse near to the     //
// center of the slider.                                                //
//                                                                      //
// TGDoubleSlider is an abstract base class. Use the concrete           //
// TGDoubleVSlider and TGDoubleHSlider.                                 //
//                                                                      //
// Dragging the slider will generate the event:                         //
// kC_VSLIDER, kSL_POS, slider id, 0  (for vertical slider)             //
// kC_HSLIDER, kSL_POS, slider id, 0  (for horizontal slider)           //
//                                                                      //
// Pressing the mouse will generate the event:                          //
// kC_VSLIDER, kSL_PRESS, slider id, 0  (for vertical slider)           //
// kC_HSLIDER, kSL_PRESS, slider id, 0  (for horizontal slider)         //
//                                                                      //
// Releasing the mouse will generate the event:                         //
// kC_VSLIDER, kSL_RELEASE, slider id, 0  (for vertical slider)         //
// kC_HSLIDER, kSL_RELEASE, slider id, 0  (for horizontal slider)       //
//                                                                      //
// Use the functions GetMinPosition(), GetMaxPosition() and             //
// GetPosition() to retrieve the position of the slider.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGWidget
#include "TGWidget.h"
#endif

class TGPicture;

enum EDoubleSliderSize {
   //--- sizes for vert. and horz. sliders
   kDoubleSliderWidth  = 24,
   kDoubleSliderHeight = kDoubleSliderWidth
};


enum EDoubleSliderScale {
   //--- type of slider scale
   kDoubleScaleNo        = BIT(0),
   kDoubleScaleDownRight = BIT(1),
   kDoubleScaleBoth      = BIT(2)
};


class TGDoubleSlider : public TGFrame, public TGWidget {

private:
   TGDoubleSlider(const TGDoubleSlider&); // Not implemented
   TGDoubleSlider& operator=(const TGDoubleSlider&); // Not implemented

protected:
   Float_t       fPos;           // logical position between fVmin and fVmax
   Float_t       fSmin;          // logical position of min value of Slider
   Float_t       fSmax;          // logical position of max value of Slider
   Int_t         fRelPos;        // slider position in pixel coordinates
   Float_t       fVmin;          // logical lower limit of slider
   Float_t       fVmax;          // logical upper limit of slider
   Int_t         fScale;         // tick mark scale
   Int_t         fScaleType;     // tick mark scale type (no, downright, both)
   Int_t         fPressPoint;    // mouse position at button press event
   Float_t       fPressSmin;     // logical min position at button press event
   Float_t       fPressSmax;     // logical max position at button press event
   Int_t         fMove;          // 1: move min value
                                 // 2: move max value
                                 // 3: move min and max value
                                 // 0: don't move any value
   Bool_t        fReversedScale; // reverse which end is min and max
   Bool_t        fMarkEnds;      // lines marking where stretch zones begin
   const TGPicture *fSliderPic;  // picture to draw slider ends

   TString       GetSString() const; // returns scaling type as string

   static void   FixBounds(Float_t &min, Float_t &max);
   void          ChangeCursor(Event_t *event);

public:
   TGDoubleSlider(const TGWindow *p = 0, UInt_t w = 1, UInt_t h = 1, UInt_t type = 1, Int_t id = -1,
                  UInt_t options = kChildFrame,
                  Pixel_t back = GetDefaultFrameBackground(),
                  Bool_t reversed = kFALSE,
                  Bool_t mark_ends = kFALSE);

   virtual ~TGDoubleSlider() { }

   virtual Bool_t HandleButton(Event_t *event) = 0;
   virtual Bool_t HandleMotion(Event_t *event) = 0;

   virtual void  SetScale(Int_t scale) { fScale = scale; }
   virtual void  SetRange(Float_t min, Float_t max) { 
      fVmin = min; fVmax = max; 
      FixBounds(fVmin, fVmax);
   }

   virtual void SetPosition(Float_t min, Float_t max) {
      if (fReversedScale) { fSmin = fVmin+fVmax-max; fSmax = fVmin+fVmax-min; }
      else { fSmin = min; fSmax = max; }
      fClient->NeedRedraw(this);
   }

   virtual Float_t GetMinPosition() const {
      if (fReversedScale) return fVmin+fVmax-fSmax;
      else return fSmin;
   }
   virtual Float_t GetMaxPosition() const {
      if (fReversedScale) return fVmin+fVmax-fSmin;
      else return fSmax;
   }
   virtual void GetPosition(Float_t &min, Float_t &max) const {
      if (fReversedScale) { min = fVmin+fVmax-fSmax; max = fVmin+fVmax-fSmin; }
      else { min = fSmin; max = fSmax; }
   }
   virtual void GetPosition(Float_t *min, Float_t *max) const {
      if (fReversedScale) { *min = fVmin+fVmax-fSmax; *max = fVmin+fVmax-fSmin; }
      else { *min = fSmin; *max = fSmax; }
   }

   virtual void  MapSubwindows() { TGWindow::MapSubwindows(); }

   virtual void  PositionChanged() { Emit("PositionChanged()"); } //*SIGNAL*
   virtual void  Pressed() { Emit("Pressed()"); }                 //*SIGNAL*
   virtual void  Released() { Emit("Released()"); }               //*SIGNAL*

   ClassDef(TGDoubleSlider,0)  // Double slider widget abstract base class
};


class TGDoubleVSlider : public TGDoubleSlider {

protected:
   Int_t   fYp;      // vertical slider y position in pixel coordinates

   virtual void DoRedraw();

public:
   TGDoubleVSlider(const TGWindow *p = 0, UInt_t h = 1, UInt_t type = 1, Int_t id = -1,
                   UInt_t options = kVerticalFrame,
                   Pixel_t back = GetDefaultFrameBackground(),
                   Bool_t reversed = kFALSE,
                   Bool_t mark_ends = kFALSE);

   virtual ~TGDoubleVSlider();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual TGDimension GetDefaultSize() const
                     { return TGDimension(kDoubleSliderWidth, fHeight); }
   virtual void   SavePrimitive(ostream &out, Option_t *option = "");

   ClassDef(TGDoubleVSlider,0)  // Vertical double slider widget
};


class TGDoubleHSlider : public TGDoubleSlider {

protected:
   Int_t       fXp;     // horizontal slider x position in pixel coordinates

   virtual void DoRedraw();

public:
   TGDoubleHSlider(const TGWindow *p = 0, UInt_t w = 1, UInt_t type = 1, Int_t id = -1,
                   UInt_t options = kHorizontalFrame,
                   Pixel_t back = GetDefaultFrameBackground(),
                   Bool_t reversed = kFALSE,
                   Bool_t mark_ends = kFALSE);

   virtual ~TGDoubleHSlider();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual TGDimension GetDefaultSize() const
                     { return TGDimension(fWidth, kDoubleSliderHeight); }
   virtual void   SavePrimitive(ostream &out, Option_t *option = "");

   ClassDef(TGDoubleHSlider,0)  // Horizontal double slider widget
};

#endif
