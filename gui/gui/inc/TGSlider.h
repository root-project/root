// @(#)root/gui:$Id$
// Author: Fons Rademakers   14/01/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGSlider
#define ROOT_TGSlider


#include "TGFrame.h"
#include "TGWidget.h"


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
   Int_t            fPos;           ///< logical position between fVmin and fVmax
   Int_t            fRelPos;        ///< slider position in pixel coordinates
   Int_t            fVmin;          ///< logical lower limit of slider
   Int_t            fVmax;          ///< logical upper limit of slider
   Int_t            fType;          ///< slider type bits
   Int_t            fScale;         ///< tick mark scale
   Bool_t           fDragging;      ///< true if in dragging mode
   const TGPicture *fSliderPic;     ///< picture to draw slider
   const TGPicture *fDisabledPic;   ///< picture to draw disabled slider

   TString GetTypeString() const;   ///< used in SavePrimitive
   virtual void CreateDisabledPicture();

private:
   TGSlider(const TGSlider&) = delete;
   TGSlider& operator=(const TGSlider&) = delete;

public:
   TGSlider(const TGWindow *p = nullptr, UInt_t w = 1, UInt_t h = 1,
            UInt_t type = kSlider1 | kScaleBoth, Int_t id = -1,
            UInt_t options = kChildFrame,
            Pixel_t back = GetDefaultFrameBackground());

   virtual ~TGSlider() {}

   Bool_t HandleButton(Event_t *event) override = 0;
   Bool_t HandleConfigureNotify(Event_t* event) override = 0;
   Bool_t HandleMotion(Event_t *event) override = 0;

   virtual void  SetEnabled(Bool_t flag = kTRUE) { SetState( flag ); }              //*TOGGLE* *GETTER=IsEnabled
   virtual void  SetState(Bool_t state);
   virtual void  SetScale(Int_t scale) { fScale = scale; }                          //*MENU*
   virtual void  SetRange(Int_t min, Int_t max);                                    //*MENU*
   virtual void  SetPosition(Int_t pos);                                            //*MENU*
   virtual Int_t GetPosition() const { return fPos; }
   virtual Int_t GetMinPosition() const { return fVmin; }
   virtual Int_t GetMaxPosition() const { return fVmax; }
   virtual Int_t GetScale() const { return fScale; }
           void  MapSubwindows() override { TGWindow::MapSubwindows(); }
   virtual void  ChangeSliderPic(const char *name);

   virtual void  PositionChanged(Int_t pos) { Emit("PositionChanged(Int_t)", pos); } // *SIGNAL*
   virtual void  Pressed() { Emit("Pressed()"); }    // *SIGNAL*
   virtual void  Released() { Emit("Released()"); }  // *SIGNAL*

   ClassDefOverride(TGSlider,0)  // Slider widget abstract base class
};


class TGVSlider : public TGSlider {

protected:
   Int_t   fYp;      ///< vertical slider y position in pixel coordinates

   void DoRedraw() override;

public:
   TGVSlider(const TGWindow *p = nullptr, UInt_t h = 40,
             UInt_t type = kSlider1 | kScaleBoth, Int_t id = -1,
             UInt_t options = kVerticalFrame,
             Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGVSlider();

   Bool_t HandleButton(Event_t *event) override;
   Bool_t HandleConfigureNotify(Event_t* event) override;
   Bool_t HandleMotion(Event_t *event) override;
   TGDimension GetDefaultSize() const override
                { return TGDimension(kSliderWidth, fHeight); }
   void   Resize(UInt_t w, UInt_t h) override { TGFrame::Resize(w, h ? h+16 : fHeight + 16); }
   void   Resize(TGDimension size) override { Resize(size.fWidth, size.fHeight); }
   void   SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGVSlider,0)  // Vertical slider widget
};


class TGHSlider : public TGSlider {

protected:
   Int_t       fXp;     ///< horizontal slider x position in pixel coordinates

   void DoRedraw() override;

public:
   TGHSlider(const TGWindow *p = nullptr, UInt_t w = 40,
             UInt_t type = kSlider1 | kScaleBoth, Int_t id = -1,
             UInt_t options = kHorizontalFrame,
             Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGHSlider();

   Bool_t HandleButton(Event_t *event) override;
   Bool_t HandleConfigureNotify(Event_t* event) override;
   Bool_t HandleMotion(Event_t *event) override;
   TGDimension GetDefaultSize() const override
          { return TGDimension(fWidth, kSliderHeight); }
   void   Resize(UInt_t w, UInt_t h) override { TGFrame::Resize(w ? w+16 : fWidth + 16, h); }
   void   Resize(TGDimension size) override { Resize(size.fWidth, size.fHeight); }
   void   SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGHSlider,0)  // Horizontal slider widget
};

#endif
