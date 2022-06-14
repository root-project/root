// @(#)root/gui:$Id$
// Author: Fons Rademakers   10/01/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGScrollBar
#define ROOT_TGScrollBar


#include "TGButton.h"


//--- scrollbar types

enum EScrollBarMode {
   kSBHorizontal,
   kSBVertical
};

class TTimer;


class TGScrollBarElement : public TGFrame {

private:
   TGScrollBarElement(const TGScrollBarElement&) = delete;
   TGScrollBarElement& operator=(const TGScrollBarElement&) = delete;

protected:
   Int_t            fState;      ///< state of scrollbar element (button up or down)
   const TGPicture *fPic;        ///< picture in scrollbar element
   const TGPicture *fPicN;       ///< picture for normal state of scrollbar element
   const TGPicture *fPicD;       ///< picture for disabled state of scrollbar element
   Pixel_t          fBgndColor;  ///< background color
   Pixel_t          fHighColor;  ///< highlight color
   Int_t            fStyle;      ///< modern or classic style

public:
   TGScrollBarElement(const TGWindow *p = nullptr, const TGPicture *pic = nullptr,
                      UInt_t w = 1, UInt_t h = 1,
                      UInt_t options = kRaisedFrame | kDoubleBorder,
                      Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGScrollBarElement();

   virtual void SetState(Int_t state);
   void DrawBorder() override;
   virtual void SetEnabled(Bool_t on = kTRUE);
   virtual Bool_t IsEnabled() const { return !(fState & kButtonDisabled); }
   Bool_t HandleCrossing(Event_t *event) override;

   ClassDefOverride(TGScrollBarElement,0)  // Scrollbar element (head, tail, slider)
};


class TGScrollBar : public TGFrame, public TGWidget {

private:
   TGScrollBar(const TGScrollBar&) = delete;
   TGScrollBar& operator=(const TGScrollBar&) = delete;

protected:
   Int_t                fX0, fY0;      ///< current slider position in pixels
   Int_t                fXp, fYp;      ///< previous slider position in pixels
   Bool_t               fDragging;     ///< in dragging mode?
   Bool_t               fGrabPointer;  ///< grab pointer when dragging
   Int_t                fRange;        ///< logical upper range of scrollbar
   Int_t                fPsize;        ///< logical page size of scrollbar
   Int_t                fPos;          ///< logical current position
   Int_t                fSliderSize;   ///< logical slider size
   Int_t                fSliderRange;  ///< logical slider range
   Int_t                fSmallInc;     ///< Small Increment in the sliding algorithm
   TGScrollBarElement  *fHead;         ///< head button of scrollbar
   TGScrollBarElement  *fTail;         ///< tail button of scrollbar
   TGScrollBarElement  *fSlider;       ///< slider
   const TGPicture     *fHeadPic;      ///< picture in head (up or left arrow)
   const TGPicture     *fTailPic;      ///< picture in tail (down or right arrow)
   TTimer              *fRepeat;       ///< repeat rate timer (when mouse stays pressed)
   Window_t             fSubw;         ///< sub window in which mouse is pressed
   Bool_t               fAccelerated;  ///< kFALSE - normal, kTRUE - accelerated
   Pixel_t              fBgndColor;    ///< background color
   Pixel_t              fHighColor;    ///< highlight color

   static Pixmap_t    fgBckgndPixmap;
   static Int_t       fgScrollBarWidth;

public:
   static Pixmap_t  GetBckgndPixmap();
   static Int_t     GetScrollBarWidth();

   TGScrollBar(const TGWindow *p = nullptr, UInt_t w = 1, UInt_t h = 1,
               UInt_t options = kChildFrame,
               Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGScrollBar();

   void           GrabPointer(Bool_t grab) { fGrabPointer = grab; }

   void   DrawBorder() override { }
   Bool_t HandleButton(Event_t *event) override = 0;
   Bool_t HandleCrossing(Event_t *event) override;
   Bool_t HandleMotion(Event_t *event) override = 0;
   Bool_t HandleTimer(TTimer *t) override;
   void   Layout() override = 0;

   virtual void  SetDragging(Bool_t drag) { fDragging = drag; }
   virtual void  SetRange(Int_t range, Int_t page_size) = 0;
   virtual void  SetPosition(Int_t pos) = 0;
   virtual Int_t GetPosition() const { return fPos; }
   virtual Int_t GetPageSize() const { return fPsize; }
   virtual Int_t GetRange() const { return fRange; }
           void  Resize(UInt_t w = 0, UInt_t h = 0) override
                 { TGFrame::Resize(w, h); SetRange(fRange, fPsize); }
           void  MoveResize(Int_t x, Int_t y, UInt_t w = 0, UInt_t h = 0) override
                  { TGFrame::MoveResize(x, y, w, h); SetRange(fRange, fPsize); }
           void  Resize(TGDimension size) override { Resize(size.fWidth, size.fHeight); }
           void  ChangeBackground(Pixel_t back) override;
   virtual void  SetAccelerated(Bool_t m = kTRUE) { fAccelerated = m; }
         Bool_t  IsAccelerated() const { return fAccelerated; }

          void   MapSubwindows() override { TGWindow::MapSubwindows(); }
   TGScrollBarElement *GetHead() const { return fHead; }
   TGScrollBarElement *GetTail() const { return fTail; }
   TGScrollBarElement *GetSlider() const { return fSlider; }

   virtual void  PositionChanged(Int_t pos) { Emit("PositionChanged(Int_t)", pos); } //*SIGNAL*
   virtual void  RangeChanged(Int_t range) { Emit("RangeChanged(Int_t)", range); } //*SIGNAL*
   virtual void  PageSizeChanged(Int_t range) { Emit("PageSizeChanged(Int_t)", range); } //*SIGNAL*

   virtual Int_t GetSmallIncrement() { return fSmallInc; }
   virtual void  SetSmallIncrement(Int_t increment) { fSmallInc = increment; }

   ClassDefOverride(TGScrollBar,0)  // Scrollbar widget
};



class TGHScrollBar : public TGScrollBar {

public:
   TGHScrollBar(const TGWindow *p = nullptr, UInt_t w = 4, UInt_t h = 2,
                UInt_t options = kHorizontalFrame,
                Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGHScrollBar() { }

   Bool_t HandleButton(Event_t *event) override;
   Bool_t HandleMotion(Event_t *event) override;
   TGDimension GetDefaultSize() const override
                { return TGDimension(fWidth, GetScrollBarWidth()); }
   void Layout() override;

   void SetRange(Int_t range, Int_t page_size) override;  //*MENU*
   void SetPosition(Int_t pos) override;                  //*MENU* *GETTER=GetPosition
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGHScrollBar,0)  // Horizontal scrollbar widget
};



class TGVScrollBar : public TGScrollBar {

public:
   TGVScrollBar(const TGWindow *p = nullptr, UInt_t w = 2, UInt_t h = 4,
                UInt_t options = kVerticalFrame,
                Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGVScrollBar() { }

   Bool_t HandleButton(Event_t *event) override;
   Bool_t HandleMotion(Event_t *event) override;
   TGDimension GetDefaultSize() const override
                { return TGDimension(GetScrollBarWidth(), fHeight); }
   void Layout() override;

   void SetRange(Int_t range, Int_t page_size) override;  //*MENU*
   void SetPosition(Int_t pos) override;                  //*MENU*  *GETTER=GetPosition
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGVScrollBar,0)  // Vertical scrollbar widget
};

#endif
