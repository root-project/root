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
   virtual void DrawBorder();
   virtual void SetEnabled(Bool_t on = kTRUE);
   virtual Bool_t IsEnabled() const { return !(fState & kButtonDisabled); }
   virtual Bool_t HandleCrossing(Event_t *event);

   ClassDef(TGScrollBarElement,0)  // Scrollbar element (head, tail, slider)
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

   virtual void   DrawBorder() { }
   virtual Bool_t HandleButton(Event_t *event) = 0;
   virtual Bool_t HandleCrossing(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event) = 0;
   virtual Bool_t HandleTimer(TTimer *t);
   virtual void   Layout() = 0;

   virtual void  SetDragging(Bool_t drag) { fDragging = drag; }
   virtual void  SetRange(Int_t range, Int_t page_size) = 0;
   virtual void  SetPosition(Int_t pos) = 0;
   virtual Int_t GetPosition() const { return fPos; }
   virtual Int_t GetPageSize() const { return fPsize; }
   virtual Int_t GetRange() const { return fRange; }
   virtual void  Resize(UInt_t w = 0, UInt_t h = 0) { TGFrame::Resize(w, h); SetRange(fRange, fPsize); }
   virtual void  MoveResize(Int_t x, Int_t y, UInt_t w = 0, UInt_t h = 0)
                  { TGFrame::MoveResize(x, y, w, h); SetRange(fRange, fPsize); }
   virtual void  Resize(TGDimension size) { Resize(size.fWidth, size.fHeight); }
   virtual void  ChangeBackground(Pixel_t back);
   virtual void  SetAccelerated(Bool_t m = kTRUE) { fAccelerated = m; }
         Bool_t  IsAccelerated() const { return fAccelerated; }

   virtual void MapSubwindows() { TGWindow::MapSubwindows(); }
   TGScrollBarElement *GetHead() const { return fHead; }
   TGScrollBarElement *GetTail() const { return fTail; }
   TGScrollBarElement *GetSlider() const { return fSlider; }

   virtual void  PositionChanged(Int_t pos) { Emit("PositionChanged(Int_t)", pos); } //*SIGNAL*
   virtual void  RangeChanged(Int_t range) { Emit("RangeChanged(Int_t)", range); } //*SIGNAL*
   virtual void  PageSizeChanged(Int_t range) { Emit("PageSizeChanged(Int_t)", range); } //*SIGNAL*

   virtual Int_t GetSmallIncrement() { return fSmallInc; }
   virtual void  SetSmallIncrement(Int_t increment) { fSmallInc = increment; }

   ClassDef(TGScrollBar,0)  // Scrollbar widget
};



class TGHScrollBar : public TGScrollBar {

public:
   TGHScrollBar(const TGWindow *p = nullptr, UInt_t w = 4, UInt_t h = 2,
                UInt_t options = kHorizontalFrame,
                Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGHScrollBar() { }

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual TGDimension GetDefaultSize() const
                        { return TGDimension(fWidth, GetScrollBarWidth()); }
   virtual void Layout();

   virtual void SetRange(Int_t range, Int_t page_size);  //*MENU*
   virtual void SetPosition(Int_t pos);                  //*MENU* *GETTER=GetPosition
   virtual void SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGHScrollBar,0)  // Horizontal scrollbar widget
};



class TGVScrollBar : public TGScrollBar {

public:
   TGVScrollBar(const TGWindow *p = nullptr, UInt_t w = 2, UInt_t h = 4,
                UInt_t options = kVerticalFrame,
                Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGVScrollBar() { }

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual TGDimension GetDefaultSize() const
                        { return TGDimension(GetScrollBarWidth(), fHeight); }
   virtual void Layout();

   virtual void SetRange(Int_t range, Int_t page_size);  //*MENU*
   virtual void SetPosition(Int_t pos);                  //*MENU*  *GETTER=GetPosition
   virtual void SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGVScrollBar,0)  // Vertical scrollbar widget
};

#endif
