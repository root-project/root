// @(#)root/gui:$Name:  $:$Id: TGView.h,v 1.3 2000/07/06 16:47:54 rdm Exp $
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
class TViewTimer;


class TGView : public TGCompositeFrame, public TGWidget {

friend class TGClient;

public:
   enum { kNoHSB = BIT(0), kNoVSB = BIT(1) };
   enum { kHorizontal = 0, kVertical = 1 };

protected:
   TGLongPosition    fMarkedStart;  // start position of marked text
   TGLongPosition    fMarkedEnd;    // end position of marked text
   TGLongPosition    fVisible;      // position of visible region
   TGLongPosition    fMousePos;     // position of mouse
   TGLongPosition    fScrollVal;    // position of scrollbar
   Bool_t            fIsMarked;     // true if text is marked/selected
   Bool_t            fIsMarking;    // true if in marking mode
   Bool_t            fIsSaved;      // true is content is saved
   Int_t             fScrolling;    // scrolling direction
   Atom_t            fClipboard;    // clipboard property
   UInt_t            fXMargin;      // x margin
   UInt_t            fYMargin;      // y margin
   TGViewFrame      *fCanvas;       // frame containing the text
   TGHScrollBar     *fHsb;          // horizontal scrollbar
   TGVScrollBar     *fVsb;          // vertical scrollbar
   TViewTimer       *fScrollTimer;  // scrollbar timer
   GContext_t        fWhiteGC;      // graphics context used for scrolling
                                    // generates GraphicsExposure events

   static Atom_t     fgClipboard;

public:
   TGView(const TGWindow *p, UInt_t w, UInt_t h, Int_t id = -1,
          UInt_t xMargin = 0, UInt_t yMargin = 0,
          UInt_t options = kSunkenFrame | kDoubleBorder,
          UInt_t sboptions = 0,
          ULong_t back = fgWhitePixel);

   virtual ~TGView();

   virtual void   Clear(Option_t * = "");
   virtual void   SetVisibleStart(Int_t newTop, Int_t direction);
   virtual void   ScrollCanvas(Int_t newTop, Int_t direction);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   virtual void   DrawBorder();
   virtual void   Layout();
   virtual void   HLayout();
   virtual void   VLayout();
   virtual void   SetSBRange(Int_t direction);
   virtual void   SetHsbPosition(Long_t newPos);
   virtual void   SetVsbPosition(Long_t newPos);
   virtual TGDimension GetDefaultSize() const { return TGDimension(fWidth, fHeight); }

   virtual Long_t ToObjXCoord(Long_t xCoord, Long_t line);
   virtual Long_t ToObjYCoord(Long_t yCoord) { return yCoord; }
   virtual Long_t ToScrXCoord(Long_t xCoord, Long_t line);
   virtual Long_t ToScrYCoord(Long_t yCoord) { return yCoord; }

   virtual void Mark(Long_t xPos, Long_t yPos);
   virtual void UnMark() { }
   virtual void DrawRegion(Int_t x, Int_t y, UInt_t width, UInt_t height);

   virtual Long_t ReturnLineLength(Long_t line);
   virtual Long_t ReturnLineCount() { return 0; }
   virtual Long_t ReturnHeighestColHeight() { return 0; }
   virtual Long_t ReturnLongestLineWidth() { return 0; }

   Bool_t IsMarked() const { return fIsMarked; }
   Bool_t IsSaved() { return fIsSaved; }

   virtual Bool_t HandleMotion(Event_t *event);
   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleExpose(Event_t *event);
   virtual Bool_t HandleCrossing(Event_t *event);
   virtual Bool_t HandleTimer(TTimer *t);

   ClassDef(TGView,0)  // Text view widget base class
};

inline Long_t TGView::ToObjXCoord(Long_t xCoord, Long_t) { return xCoord; }
inline Long_t TGView::ToScrXCoord(Long_t xCoord, Long_t) { return xCoord; }
inline void   TGView::Mark(Long_t, Long_t) { }
inline void   TGView::DrawRegion(Int_t, Int_t, UInt_t, UInt_t) { }
inline Long_t TGView::ReturnLineLength(Long_t) { return 0; }


class TGViewFrame : public TGCompositeFrame {
private:
   TGView   *fView;  // pointer back to the view
public:
   TGViewFrame(TGView *v, UInt_t w, UInt_t h, UInt_t options = 0,
               ULong_t back = fgWhitePixel);

   Bool_t HandleSelectionRequest(Event_t *event)
        { fView->HandleSelectionRequest(event); return kTRUE; }
   Bool_t HandleSelectionClear(Event_t *event)
        { fView->HandleSelectionClear(event); return kTRUE; }
   Bool_t HandleSelection(Event_t *event)
        { fView->HandleSelection(event); return kTRUE; }
   Bool_t HandleButton(Event_t *event)
        { fView->HandleButton(event); return kTRUE; }
   Bool_t HandleExpose(Event_t *event)
        { fView->HandleExpose(event); return kTRUE; }
   Bool_t HandleCrossing(Event_t *event)
        { fView->HandleCrossing(event); return kTRUE; }
   Bool_t HandleMotion(Event_t *event)
        { fView->HandleMotion(event); return kTRUE; }
   Bool_t HandleKey(Event_t *event)
        { fView->HandleKey(event); return kTRUE; }

   ClassDef(TGViewFrame,0)  // Frame containing the actual text
};


class TViewTimer : public TTimer {
private:
   TGView   *fView;
public:
   TViewTimer(TGView *t, Long_t ms) : TTimer(ms, kTRUE) { fView = t; }
   Bool_t Notify();
};

#endif
