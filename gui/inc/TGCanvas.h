// @(#)root/gui:$Name:  $:$Id: TGCanvas.h,v 1.4 2000/10/11 16:12:18 rdm Exp $
// Author: Fons Rademakers   11/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGCanvas
#define ROOT_TGCanvas


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGCanvas and TGViewPort                                              //
//                                                                      //
// A TGCanvas is a frame containing two scrollbars (horizontal and      //
// vertical) and a viewport. The viewport acts as the window through    //
// which we look at the contents of the container frame.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGHScrollBar;
class TGVScrollBar;


class TGViewPort : public TGCompositeFrame {

protected:
   Int_t       fX0, fY0;     // position of container frame in viewport
   TGFrame    *fContainer;   // container frame (must inherit from TGCompositeFrame)

public:
   TGViewPort(const TGWindow *p, UInt_t w, UInt_t h,
              UInt_t options = kChildFrame,
              ULong_t back = GetDefaultFrameBackground());

   TGFrame *GetContainer() const { return fContainer; }
   void SetContainer(TGFrame *f);

   virtual void DrawBorder() { };
   virtual void Layout() { }
   virtual TGDimension GetDefaultSize() const { return TGDimension(fWidth, fHeight); }

   void SetHPos(Int_t xpos) { if (fContainer) fContainer->Move(fX0 = xpos, fY0); }
   void SetVPos(Int_t ypos) { if (fContainer) fContainer->Move(fX0, fY0 = ypos); }
   void SetPos(Int_t xpos, Int_t ypos) { if (fContainer) fContainer->Move(fX0 = xpos, fY0 = ypos); }

   ClassDef(TGViewPort,0)  // Viewport through which to look at a frame
};


class TGCanvas : public TGFrame {

protected:
   TGViewPort      *fVport;        // viewport through which we look at contents
   TGHScrollBar    *fHScrollbar;   // horizontal scrollbar
   TGVScrollBar    *fVScrollbar;   // vertical scrollbar
   Int_t            fScrolling;    // flag which srolling modes are allowed

public:
   enum { kCanvasNoScroll         = 0,
          kCanvasScrollHorizontal = BIT(0),
          kCanvasScrollVertical   = BIT(1),
          kCanvasScrollBoth       = (kCanvasScrollHorizontal | kCanvasScrollVertical)
   };

   TGCanvas(const TGWindow *p, UInt_t w, UInt_t h,
            UInt_t options = kSunkenFrame | kDoubleBorder,
            ULong_t back = GetDefaultFrameBackground());
   virtual ~TGCanvas();

   TGFrame      *GetContainer() const { return fVport->GetContainer(); }
   TGViewPort   *GetViewPort() const { return fVport; }
   virtual void  AddFrame(TGFrame *f, TGLayoutHints *l = 0);
   virtual void  SetContainer(TGFrame *f) { fVport->SetContainer(f); }
   virtual void  MapSubwindows();
   virtual void  DrawBorder();
   virtual void  Layout();
   virtual Int_t GetHsbPosition() const;
   virtual Int_t GetVsbPosition() const;
   virtual void  SetHsbPosition(Int_t newPos);
   virtual void  SetVsbPosition(Int_t newPos);
   void          SetScrolling(Int_t scrolling);
   Int_t         GetScrolling() const { return fScrolling; }

   virtual TGDimension GetDefaultSize() const { return TGDimension(fWidth, fHeight); }

   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TGCanvas,0)  // A canvas with two scrollbars and a viewport
};


#endif
