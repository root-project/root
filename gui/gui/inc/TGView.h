// @(#)root/gui:$Id$
// Author: Fons Rademakers   30/6/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGView
#define ROOT_TGView


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGView                                                               //
//                                                                      //
// A TGView provides the infrastructure for text viewer and editor      //
// widgets. It provides a canvas (TGViewFrame) and (optionally) a       //
// vertical and horizontal scrollbar and methods for marking and        //
// scrolling.                                                           //
//                                                                      //
// The TGView (and derivatives) will generate the following             //
// event messages:                                                      //
// kC_TEXTVIEW, kTXT_ISMARKED, widget id, [true|false]                  //
// kC_TEXTVIEW, kTXT_DATACHANGE, widget id, 0                           //
// kC_TEXTVIEW, kTXT_CLICK2, widget id, position (y << 16) | x)         //
// kC_TEXTVIEW, kTXT_CLICK3, widget id, position (y << 16) | x)         //
// kC_TEXTVIEW, kTXT_F3, widget id, true                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGWidget
#include "TGWidget.h"
#endif
#ifndef ROOT_TTimer
#include "TTimer.h"
#endif

class TGViewFrame;
class TGHScrollBar;
class TGVScrollBar;

class TGView : public TGCompositeFrame, public TGWidget {

friend class TGViewFrame;

public:
   enum { kNoHSB = BIT(0), kNoVSB = BIT(1) };
   enum { kHorizontal = 0, kVertical = 1 };

protected:
   TGLongPosition    fVisible;      // position of visible region
   TGLongPosition    fMousePos;     // position of mouse
   TGLongPosition    fScrollVal;    // scroll value
   TGDimension       fVirtualSize;  // the current virtual window size
   TGRectangle       fExposedRegion;// exposed area

   Int_t             fScrolling;    // scrolling direction
   Atom_t            fClipboard;    // clipboard property
   UInt_t            fXMargin;      // x margin
   UInt_t            fYMargin;      // y margin
   TGViewFrame      *fCanvas;       // frame containing the text
   TGHScrollBar     *fHsb;          // horizontal scrollbar
   TGVScrollBar     *fVsb;          // vertical scrollbar

   TGGC              fWhiteGC;      // graphics context used for scrolling
                                    // generates GraphicsExposure events

   virtual void DoRedraw();
   virtual void UpdateRegion(Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual Bool_t ItemLayout() { return kFALSE; }

private:
   TGView(const TGView&);              // not implemented
   TGView& operator=(const TGView&);   // not implemented

public:
   TGView(const TGWindow *p = 0, UInt_t w = 1, UInt_t h = 1, Int_t id = -1,
          UInt_t xMargin = 0, UInt_t yMargin = 0,
          UInt_t options = kSunkenFrame | kDoubleBorder,
          UInt_t sboptions = 0,
          Pixel_t back = GetWhitePixel());

   virtual ~TGView();

   TGViewFrame   *GetCanvas() const { return fCanvas; }

   virtual void   Clear(Option_t * = "");
   virtual void   SetVisibleStart(Int_t newTop, Int_t direction);
   virtual void   ScrollCanvas(Int_t newTop, Int_t direction);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   virtual void   DrawBorder();
   virtual void   Layout();
   virtual void   SetLayoutManager(TGLayoutManager*) { }
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

   virtual TGDimension GetDefaultSize() const { return TGDimension(fWidth, fHeight); }
   TGDimension GetVirtualSize() const { return fVirtualSize; }
   TGLongPosition  GetScrollValue() const { return fScrollVal; }
   TGLongPosition  GetScrollPosition() const { return fVisible; }

   TGLongPosition ToVirtual(TGLongPosition coord)  const { return coord + fVisible; }
   TGLongPosition ToPhysical(TGLongPosition coord) const { return coord - fVisible; }

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleExpose(Event_t *event);

   virtual void   ChangeBackground(Pixel_t);
   virtual void   SetBackgroundColor(Pixel_t);
   virtual void   SetBackgroundPixmap(Pixmap_t p);
   virtual void   UpdateBackgroundStart();

   const TGGC &GetViewWhiteGC() { return fWhiteGC; }

   ClassDef(TGView,0)  // View widget base class
};


class TGViewFrame : public TGCompositeFrame {
private:
   TGView   *fView;  // pointer back to the view

   TGViewFrame(const TGViewFrame&);              // not implemented
   TGViewFrame& operator=(const TGViewFrame&);   // not implemented

public:
   TGViewFrame(TGView *v, UInt_t w, UInt_t h, UInt_t options = 0,
               Pixel_t back = GetWhitePixel());

   Bool_t HandleSelectionRequest(Event_t *event)
            { return fView->HandleSelectionRequest(event); }
   Bool_t HandleSelectionClear(Event_t *event)
            { return fView->HandleSelectionClear(event); }
   Bool_t HandleSelection(Event_t *event)
            { return fView->HandleSelection(event); }
   Bool_t HandleButton(Event_t *event)
            { return fView->HandleButton(event); }
   Bool_t HandleExpose(Event_t *event)
            { return fView->HandleExpose(event); }
   Bool_t HandleCrossing(Event_t *event)
            { return fView->HandleCrossing(event); }
   Bool_t HandleMotion(Event_t *event)
            { return fView->HandleMotion(event); }
   Bool_t HandleKey(Event_t *event)
            { return fView->HandleKey(event); }
   Bool_t HandleDoubleClick(Event_t *event)
            { return fView->HandleDoubleClick(event); }

   ClassDef(TGViewFrame,0)  // Frame containing the actual text
};



#endif
