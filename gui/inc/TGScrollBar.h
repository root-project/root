// @(#)root/gui:$Name:  $:$Id: TGScrollBar.h,v 1.4 2000/10/08 14:27:54 rdm Exp $
// Author: Fons Rademakers   10/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGScrollBar
#define ROOT_TGScrollBar


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGScrollBar and TGScrollBarElement                                   //
//                                                                      //
// The classes in this file implement scrollbars. Scrollbars can be     //
// either placed horizontal or vertical. A scrollbar contains three     //
// TGScrollBarElements: The "head", "tail" and "slider". The head and   //
// tail are fixed at either end and have the typical arrows in them.    //
//                                                                      //
// The TGHScrollBar will generate the following event messages:         //
// kC_HSCROLL, kSB_SLIDERPOS, position, 0                               //
// kC_HSCROLL, kSB_SLIDERTRACK, position, 0                             //
//                                                                      //
// The TGVScrollBar will generate the following event messages:         //
// kC_VSCROLL, kSB_SLIDERPOS, position, 0                               //
// kC_VSCROLL, kSB_SLIDERTRACK, position, 0                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGButton.h"
#endif


//--- scrollbar types

enum EScrollBarMode {
   kSBHorizontal,
   kSBVertical
};

class TTimer;


class TGScrollBarElement : public TGFrame {

protected:
   Int_t            fState;  // state of scrollbar element (button up or down)
   const TGPicture *fPic;    // picture in scrollbar element

public:
   TGScrollBarElement(const TGWindow *p, const TGPicture *pic, UInt_t w, UInt_t h,
              UInt_t options = kRaisedFrame | kDoubleBorder,
              ULong_t back = GetDefaultFrameBackground()) :
      TGFrame(p, w, h, options | kOwnBackground, back)
      { fPic = pic; fState = kButtonUp; }

   virtual void SetState(Int_t state);
   virtual void DrawBorder();

   ClassDef(TGScrollBarElement,0)  // Scrollbar element (head, tail, slider)
};


class TGScrollBar : public TGFrame, public TGWidget {

friend class TGClient;

protected:
   Int_t                fX0, fY0;      // current slider position in pixels
   Int_t                fXp, fYp;      // previous slider position in pixels
   Bool_t               fDragging;     // in dragging mode?
   Bool_t               fGrabPointer;  // grab pointer when dragging
   Int_t                fRange;        // logical upper range of scrollbar
   Int_t                fPsize;        // logical page size of scrollbar
   Int_t                fPos;          // logical current position
   Int_t                fSliderSize;   // logical slider size
   Int_t                fSliderRange;  // logical slider range
   TGScrollBarElement  *fHead;         // head button of scrollbar
   TGScrollBarElement  *fTail;         // tail button of scrollbar
   TGScrollBarElement  *fSlider;       // slider
   const TGPicture     *fHeadPic;      // picture in head (up or left arrow)
   const TGPicture     *fTailPic;      // picture in tail (down or right arrow)
   TTimer              *fRepeat;       // repeat rate timer (when mouse stays pressed)
   Window_t             fSubw;         // sub window in which mouse is pressed

   static Pixmap_t    fgBckgndPixmap;
   static Int_t       fgScrollBarWidth;

public:
   static Pixmap_t  GetBckgndPixmap();
   static Int_t     GetScrollBarWidth();

   TGScrollBar(const TGWindow *p, UInt_t w, UInt_t h,
               UInt_t options = kChildFrame,
               ULong_t back = GetDefaultFrameBackground()) :
      TGFrame(p, w, h, options | kOwnBackground, back)
            { fMsgWindow = p; fRepeat = 0; fGrabPointer = kTRUE;
              SetBackgroundPixmap(GetBckgndPixmap()); }
   virtual ~TGScrollBar();

   void           GrabPointer(Bool_t grab) { fGrabPointer = grab; }

   virtual void   DrawBorder() { }
   virtual Bool_t HandleButton(Event_t *event) = 0;
   virtual Bool_t HandleMotion(Event_t *event) = 0;
   virtual Bool_t HandleTimer(TTimer *t);
   virtual void   Layout() = 0;

   virtual void  SetRange(Int_t range, Int_t page_size) = 0;
   virtual void  SetPosition(Int_t pos) = 0;
   virtual Int_t GetPosition() const { return fPos; }

   virtual void MapSubwindows() { TGWindow::MapSubwindows(); }

   ClassDef(TGScrollBar,0)  // Scrollbar widget
};



class TGHScrollBar : public TGScrollBar {

public:
   TGHScrollBar(const TGWindow *p, UInt_t w, UInt_t h,
                UInt_t options = kHorizontalFrame,
                ULong_t back = GetDefaultFrameBackground());
   virtual ~TGHScrollBar() { }

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual TGDimension GetDefaultSize() const
                     { return TGDimension(fWidth, GetScrollBarWidth()); }
   virtual void Layout();

   virtual void SetRange(Int_t range, Int_t page_size);
   virtual void SetPosition(Int_t pos);

   ClassDef(TGHScrollBar,0)  // Horizontal scrollbar widget
};



class TGVScrollBar : public TGScrollBar {

public:
   TGVScrollBar(const TGWindow *p, UInt_t w, UInt_t h,
                UInt_t options = kVerticalFrame,
                ULong_t back = GetDefaultFrameBackground());
   virtual ~TGVScrollBar() { }

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual TGDimension GetDefaultSize() const
                      { return TGDimension(GetScrollBarWidth(), fHeight); }
   virtual void Layout();

   virtual void SetRange(Int_t range, Int_t page_size);
   virtual void SetPosition(Int_t pos);

   ClassDef(TGVScrollBar,0)  // Vertical scrollbar widget
};

#endif
