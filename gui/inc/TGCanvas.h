// @(#)root/gui:$Name:  $:$Id: TGCanvas.h,v 1.7 2002/06/12 17:56:25 rdm Exp $
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
// TGCanvas and TGViewPort and TGContainer                              //
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
class TGClient;
class TGViewPort;
class TGCanvas;
class TGFrameElement;
class TGContainerTimer;
class TGSearchType;
class TGContainerKeyboardTimer;
class TGContainerScrollTimer;


class TGContainer : public TGCompositeFrame {

friend class TGClient;
friend class TGViewPort;
friend class TGCanvas;
friend class TGContainerKeyboardTimer;
friend class TGContainerScrollTimer;

protected:
   TGViewPort        *fViewPort;      // container viewport
   TGCanvas          *fCanvas;        // pointer to canvas
   const TGWindow    *fMsgWindow;     // window handling container messages
   TGFrameElement    *fLastActiveEl;  // last active  item
   Int_t              fXp, fYp;       // previous pointer position
   Int_t              fX0, fY0;       // corner of rubber band box
   Int_t              fXf, fYf;       // other corner of rubber band box
   Bool_t             fDragging;      // true if in dragging mode
   Int_t              fTotal;         // total items
   Int_t              fSelected;      // number of selected items
   TTimer            *fScrollTimer;   // autoscroll timer
   Bool_t             fMapSubwindows; // kTRUE - map subwindows
   Bool_t             fOnMouseOver;   // kTRUE when mouse pointer is over entry
   TGSearchType      *fSearch;        // structure used by search dialog
   Bool_t             fLastDir;       // direction of last search
   Bool_t             fLastCase;      // case-sensetivity of last search
   TString            fLastName;      // the name of object of last search
   TTimer            *fKeyTimer;      // keyboard timer
   TString            fKeyInput;      // keyboard input (buffer)
   Bool_t             fKeyTimerActive;// kTRUE - keyboard timer is active
   Bool_t             fScrolling;     // kTRUE - when scrolling is ON

   static TGGC        fgLineGC;

   virtual void DoRedraw();
   virtual void ClearViewPort();
   virtual void ActivateItem(TGFrameElement* el);
   virtual void LineUp(Bool_t select = kFALSE);
   virtual void LineDown(Bool_t select = kFALSE);
   virtual void LineLeft(Bool_t select = kFALSE);
   virtual void LineRight(Bool_t select = kFALSE);
   virtual void PageUp(Bool_t select = kFALSE);
   virtual void PageDown(Bool_t select = kFALSE);
   virtual void Home(Bool_t select = kFALSE);
   virtual void End(Bool_t select = kFALSE);
   virtual void Search();
   virtual void RepeatSearch();
   virtual void SearchPattern();
   virtual void OnAutoScroll();

   virtual TGFrameElement *FindFrame(const TString& name,
                                     Bool_t direction = kTRUE,
                                     Bool_t caseSensitive = kTRUE,
                                     Bool_t beginWith = kFALSE);
public:
   TGContainer(const TGWindow *p, UInt_t w, UInt_t h,
                 UInt_t options = kSunkenFrame,
                 ULong_t back = GetDefaultFrameBackground());
   virtual ~TGContainer();

   virtual void MapSubwindows();
   virtual void DrawRegion(Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual void Associate(const TGWindow *w) { fMsgWindow = w; }
   virtual void AdjustPosition();
   virtual void SetPagePosition(const TGPosition& pos);
   virtual void SetPagePosition(Int_t x, Int_t y);
   virtual void SetPageDimension(const TGDimension& dim);
   virtual void SetPageDimension(UInt_t w, UInt_t h);
   virtual void RemoveAll();
   virtual void RemoveItem(TGFrame *item);
   virtual void Layout();

   const TGWindow   *GetMessageWindow() const { return fMsgWindow; }
   TGPosition        GetPagePosition() const;
   TGDimension       GetPageDimension() const;

   virtual Bool_t IsMapSubwindows() const { return fMapSubwindows; }
   virtual Int_t NumSelected() const { return fSelected; }
   virtual Int_t NumItems() const { return fTotal; }
   virtual TGFrameElement *FindFrame(Int_t x,Int_t y,Bool_t exclude=kTRUE);

   virtual const TGFrame *GetNextSelected(void **current);
   virtual TGFrame *GetLastActive() const { return fLastActiveEl->fFrame; }

   virtual Bool_t HandleExpose(Event_t *event);
   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleDoubleClick(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual Bool_t HandleKey(Event_t *event);

   virtual void SelectAll();                    //*SIGNAL*
   virtual void UnSelectAll();                  //*SIGNAL*
   virtual void InvertSelection();              //*SIGNAL*
   virtual void ReturnPressed(TGFrame*);        //*SIGNAL*
   virtual void SpacePressed(TGFrame*);         //*SIGNAL*
   virtual void OnMouseOver(TGFrame*);          //*SIGNAL*
   virtual void CurrentChanged(Int_t x,Int_t y);//*SIGNAL*
   virtual void CurrentChanged(TGFrame* f);     //*SIGNAL*
   virtual void Clicked(TGFrame *f, Int_t btn); //*SIGNAL*
   virtual void DoubleClicked(TGFrame *f, Int_t btn);  //*SIGNAL*
   virtual void DoubleClicked(TGFrame *f, Int_t btn, Int_t x, Int_t y); //*SIGNAL*
   virtual void Clicked(TGFrame *f, Int_t btn, Int_t x, Int_t y);       //*SIGNAL*

   ClassDef(TGContainer,0)  // canvas container
};


class TGViewPort : public TGCompositeFrame {

protected:
   Int_t       fX0, fY0;     // position of container frame in viewport
   TGFrame         *fContainer;       // container frame

public:
   TGViewPort(const TGWindow *p, UInt_t w, UInt_t h,
              UInt_t options = kChildFrame,
              ULong_t back = GetDefaultFrameBackground());

   TGFrame *GetContainer() const { return fContainer; }
   void SetContainer(TGFrame *f);

   virtual void DrawBorder() { };
   virtual void Layout() { }
   virtual TGDimension GetDefaultSize() const { return TGDimension(fWidth, fHeight); }

   virtual void SetHPos(Int_t xpos);
   virtual void SetVPos(Int_t ypos);
   void SetPos(Int_t xpos, Int_t ypos);

   Int_t GetHPos() const { return fX0; }
   Int_t GetVPos() const { return fY0; }
   virtual Bool_t HandleConfigureNotify(Event_t *event);

   ClassDef(TGViewPort,0)  // Viewport through which to look at a container frame
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
   TGHScrollBar *GetHScrollbar() const { return fHScrollbar; }
   TGVScrollBar *GetVScrollbar() const { return fVScrollbar; }

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
