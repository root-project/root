// @(#)root/gui:$Id$
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
// Slider widgets allow easy selection of a range.                      //
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
   const TGPicture *fDisabledPic;   // picture to draw disabled slider

   TString GetTypeString() const;   // used in SavePrimitive
   virtual void CreateDisabledPicture();

private:
   TGSlider(const TGSlider&);             // not implemented
   TGSlider& operator=(const TGSlider&);  // not implemented

public:
   TGSlider(const TGWindow *p = 0, UInt_t w = 1, UInt_t h = 1,
            UInt_t type = kSlider1 | kScaleBoth, Int_t id = -1,
            UInt_t options = kChildFrame,
            Pixel_t back = GetDefaultFrameBackground());

   virtual ~TGSlider() { }

   virtual Bool_t HandleButton(Event_t *event) = 0;
   virtual Bool_t HandleConfigureNotify(Event_t* event) = 0;
   virtual Bool_t HandleMotion(Event_t *event) = 0;

   virtual void  SetEnabled(Bool_t flag = kTRUE) { SetState( flag ); }              //*TOGGLE* *GETTER=IsEnabled
   virtual void  SetState(Bool_t state);
   virtual void  SetScale(Int_t scale) { fScale = scale; }                          //*MENU*
   virtual void  SetRange(Int_t min, Int_t max) { fVmin = min; fVmax = max; }       //*MENU*
   virtual void  SetPosition(Int_t pos) { fPos = pos; fClient->NeedRedraw(this); }  //*MENU*
   virtual Int_t GetPosition() const { return fPos; }
   virtual Int_t GetMinPosition() const { return fVmin; }
   virtual Int_t GetMaxPosition() const { return fVmax; }
   virtual Int_t GetScale() const { return fScale; }
   virtual void  MapSubwindows() { TGWindow::MapSubwindows(); }
   virtual void  ChangeSliderPic(const char *name) {
                    if (fSliderPic) fClient->FreePicture(fSliderPic);
                    fSliderPic = fClient->GetPicture(name);
                 }

   virtual void  PositionChanged(Int_t pos) { Emit("PositionChanged(Int_t)", pos); } // *SIGNAL*
   virtual void  Pressed() { Emit("Pressed()"); }    // *SIGNAL*
   virtual void  Released() { Emit("Released()"); }  // *SIGNAL*

   ClassDef(TGSlider,0)  // Slider widget abstract base class
};


class TGVSlider : public TGSlider {

protected:
   Int_t   fYp;      // vertical slider y position in pixel coordinates

   virtual void DoRedraw();

public:
   TGVSlider(const TGWindow *p = 0, UInt_t h = 40,
             UInt_t type = kSlider1 | kScaleBoth, Int_t id = -1,
             UInt_t options = kVerticalFrame,
             Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGVSlider();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleConfigureNotify(Event_t* event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual TGDimension GetDefaultSize() const
                     { return TGDimension(kSliderWidth, fHeight); }
   virtual void   Resize(UInt_t w, UInt_t h) { TGFrame::Resize(w, h ? h+16 : fHeight + 16); }
   virtual void   Resize(TGDimension size) { Resize(size.fWidth, size.fHeight); }
   virtual void   SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGVSlider,0)  // Vertical slider widget
};


class TGHSlider : public TGSlider {

protected:
   Int_t       fXp;     // horizontal slider x position in pixel coordinates

   virtual void DoRedraw();

public:
   TGHSlider(const TGWindow *p = 0, UInt_t w = 40,
             UInt_t type = kSlider1 | kScaleBoth, Int_t id = -1,
             UInt_t options = kHorizontalFrame,
             Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGHSlider();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleConfigureNotify(Event_t* event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual TGDimension GetDefaultSize() const
                     { return TGDimension(fWidth, kSliderHeight); }
   virtual void   Resize(UInt_t w, UInt_t h) { TGFrame::Resize(w ? w+16 : fWidth + 16, h); }
   virtual void   Resize(TGDimension size) { Resize(size.fWidth, size.fHeight); }
   virtual void   SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGHSlider,0)  // Horizontal slider widget
};

#endif
