// @(#)root/gui:$Id$
// Author: Fons Rademakers   30/6/2000

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGView
#define ROOT_TGView


#include "TGFrame.h"
#include "TGWidget.h"

class TGViewFrame;
class TGHScrollBar;
class TGVScrollBar;

class TGView : public TGCompositeFrame, public TGWidget {

friend class TGViewFrame;

public:
   enum { kNoHSB = BIT(0), kNoVSB = BIT(1) };
   enum { kHorizontal = 0, kVertical = 1 };

protected:
   TGLongPosition    fVisible;      ///< position of visible region
   TGLongPosition    fMousePos;     ///< position of mouse
   TGLongPosition    fScrollVal;    ///< scroll value
   TGDimension       fVirtualSize;  ///< the current virtual window size
   TGRectangle       fExposedRegion;///< exposed area

   Int_t             fScrolling;    ///< scrolling direction
   Atom_t            fClipboard;    ///< clipboard property
   UInt_t            fXMargin;      ///< x margin
   UInt_t            fYMargin;      ///< y margin
   TGViewFrame      *fCanvas;       ///< frame containing the text
   TGHScrollBar     *fHsb;          ///< horizontal scrollbar
   TGVScrollBar     *fVsb;          ///< vertical scrollbar

   TGGC              fWhiteGC;      ///< graphics context used for scrolling
                                    ///< generates GraphicsExposure events

   void DoRedraw() override;
   virtual void UpdateRegion(Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual Bool_t ItemLayout() { return kFALSE; }

private:
   TGView(const TGView&) = delete;
   TGView& operator=(const TGView&) = delete;

public:
   TGView(const TGWindow *p = nullptr, UInt_t w = 1, UInt_t h = 1, Int_t id = -1,
          UInt_t xMargin = 0, UInt_t yMargin = 0,
          UInt_t options = kSunkenFrame | kDoubleBorder,
          UInt_t sboptions = 0,
          Pixel_t back = GetWhitePixel());

   virtual ~TGView();

   TGViewFrame   *GetCanvas() const { return fCanvas; }

   void           Clear(Option_t * = "") override;
   virtual void   SetVisibleStart(Int_t newTop, Int_t direction);
   virtual void   ScrollCanvas(Int_t newTop, Int_t direction);
   Bool_t         ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;
   void           DrawBorder() override;
   void           Layout() override;
   void           SetLayoutManager(TGLayoutManager*) override {}
   virtual void   DrawRegion(Int_t x, Int_t y, UInt_t width, UInt_t height);

   virtual void ScrollToPosition(TGLongPosition newPos);
   void ScrollUp(Int_t pixels)
      { ScrollToPosition(TGLongPosition(fVisible.fX, fVisible.fY + pixels)); }
   void ScrollDown(Int_t pixels)
      { ScrollToPosition(TGLongPosition(fVisible.fX, fVisible.fY - pixels)); }
   void ScrollLeft(Int_t pixels)
      { ScrollToPosition(TGLongPosition(fVisible.fX + pixels, fVisible.fY)); }
   void ScrollRight(Int_t  pixels)
      { ScrollToPosition(TGLongPosition(fVisible.fX - pixels, fVisible.fY)); }

   TGDimension    GetDefaultSize() const override { return TGDimension(fWidth, fHeight); }
   TGDimension    GetVirtualSize() const { return fVirtualSize; }
   TGLongPosition GetScrollValue() const { return fScrollVal; }
   TGLongPosition GetScrollPosition() const { return fVisible; }

   TGLongPosition ToVirtual(TGLongPosition coord)  const { return coord + fVisible; }
   TGLongPosition ToPhysical(TGLongPosition coord) const { return coord - fVisible; }

   Bool_t         HandleButton(Event_t *event) override;
   Bool_t         HandleExpose(Event_t *event) override;

   void           ChangeBackground(Pixel_t) override;
   void           SetBackgroundColor(Pixel_t) override;
   void           SetBackgroundPixmap(Pixmap_t p) override;
   virtual void   UpdateBackgroundStart();

   const TGGC &GetViewWhiteGC() { return fWhiteGC; }

   ClassDefOverride(TGView,0)  // View widget base class
};


class TGViewFrame : public TGCompositeFrame {
private:
   TGView   *fView;  // pointer back to the view

   TGViewFrame(const TGViewFrame&) = delete;
   TGViewFrame& operator=(const TGViewFrame&) = delete;

public:
   TGViewFrame(TGView *v, UInt_t w, UInt_t h, UInt_t options = 0,
               Pixel_t back = GetWhitePixel());

   Bool_t HandleSelectionRequest(Event_t *event) override
            { return fView->HandleSelectionRequest(event); }
   Bool_t HandleSelectionClear(Event_t *event) override
            { return fView->HandleSelectionClear(event); }
   Bool_t HandleSelection(Event_t *event) override
            { return fView->HandleSelection(event); }
   Bool_t HandleButton(Event_t *event) override
            { return fView->HandleButton(event); }
   Bool_t HandleExpose(Event_t *event) override
            { return fView->HandleExpose(event); }
   Bool_t HandleCrossing(Event_t *event) override
            { return fView->HandleCrossing(event); }
   Bool_t HandleMotion(Event_t *event) override
            { return fView->HandleMotion(event); }
   Bool_t HandleKey(Event_t *event) override
            { return fView->HandleKey(event); }
   Bool_t HandleDoubleClick(Event_t *event) override
            { return fView->HandleDoubleClick(event); }

   ClassDefOverride(TGViewFrame,0)  // Frame containing the actual text
};

#endif
