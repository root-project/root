// @(#)root/gui:$Id$
// Author: Fons Rademakers   6/09/2000

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGSplitter
#define ROOT_TGSplitter


#include "TGFrame.h"


class TGSplitter : public TGFrame {

protected:
   Cursor_t    fSplitCursor;      ///< split cursor
   Bool_t      fDragging;         ///< true if in dragging mode
   Bool_t      fExternalHandler;  ///< true when splitter movement is handled externally
   const TGPicture *fSplitterPic; ///< picture to draw splitter

private:
   TGSplitter(const TGSplitter&) = delete;
   TGSplitter& operator=(const TGSplitter&) = delete;

public:
   TGSplitter(const TGWindow *p = nullptr, UInt_t w = 2, UInt_t h = 4,
              UInt_t options = kChildFrame,
              Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGSplitter() { }

   virtual void   SetFrame(TGFrame *frame, Bool_t prev) = 0;

   virtual Bool_t HandleButton(Event_t *event) override = 0;
   virtual Bool_t HandleMotion(Event_t *event) override = 0;
   virtual Bool_t HandleCrossing(Event_t *event) override = 0;

   void DragStarted();      // *SIGNAL*
   void Moved(Int_t delta); // *SIGNAL*

   Bool_t GetExternalHandler() const { return fExternalHandler; }
   void SetExternalHandler(Bool_t x) { fExternalHandler = x; }

   ClassDefOverride(TGSplitter,0)  //A frame splitter abstract base class
};


class TGVSplitter : public TGSplitter {

private:
   TGVSplitter(const TGVSplitter&) = delete;
   TGVSplitter& operator=(const TGVSplitter&) = delete;

protected:
   Int_t       fStartX;         ///< x position when dragging starts
   UInt_t      fFrameWidth;     ///< width of frame to be resized
   UInt_t      fFrameHeight;    ///< height of frame to be resized
   Int_t       fMin;            ///< min x position frame can be resized to
   Int_t       fMax;            ///< max x position frame can be resized to
   TGFrame    *fFrame;          ///< frame that should be resized
   Bool_t      fLeft;           ///< true if frame is on the left of splitter

public:
   TGVSplitter(const TGWindow *p = nullptr, UInt_t w = 4, UInt_t h = 4,
               UInt_t options = kChildFrame,
               Pixel_t back = GetDefaultFrameBackground());
   TGVSplitter(const TGWindow *p, UInt_t w, UInt_t h, Bool_t external);
   virtual ~TGVSplitter();

   void           DrawBorder() override;
   void           SetFrame(TGFrame *frame, Bool_t left) override;
   const TGFrame *GetFrame() const { return fFrame; }
   Bool_t         GetLeft() const { return fLeft; }
   Bool_t         IsLeft() const { return fLeft; }
   void           SavePrimitive(std::ostream &out, Option_t *option = "") override;

   Bool_t HandleButton(Event_t *event) override;
   Bool_t HandleMotion(Event_t *event) override;
   Bool_t HandleCrossing(Event_t *event) override;

   ClassDefOverride(TGVSplitter,0)  //A vertical frame splitter
};


class TGHSplitter : public TGSplitter {

private:
   TGHSplitter(const TGHSplitter&) = delete;
   TGHSplitter& operator=(const TGHSplitter&) = delete;

protected:
   Int_t       fStartY;         ///< y position when dragging starts
   UInt_t      fFrameWidth;     ///< width of frame to be resized
   UInt_t      fFrameHeight;    ///< height of frame to be resized
   Int_t       fMin;            ///< min y position frame can be resized to
   Int_t       fMax;            ///< max y position frame can be resized to
   TGFrame    *fFrame;          ///< frame that should be resized
   Bool_t      fAbove;          ///< true if frame is above the splitter

public:
   TGHSplitter(const TGWindow *p = nullptr, UInt_t w = 4, UInt_t h = 4,
               UInt_t options = kChildFrame,
               Pixel_t back = GetDefaultFrameBackground());
   TGHSplitter(const TGWindow *p, UInt_t w, UInt_t h, Bool_t external);
   virtual ~TGHSplitter();

   void           DrawBorder() override;
   void           SetFrame(TGFrame *frame, Bool_t above) override;
   const TGFrame *GetFrame() const { return fFrame; }
   Bool_t         GetAbove() const { return fAbove; }
   Bool_t         IsAbove() const { return fAbove; }
   void   SavePrimitive(std::ostream &out, Option_t *option = "") override;

   Bool_t HandleButton(Event_t *event) override;
   Bool_t HandleMotion(Event_t *event) override;
   Bool_t HandleCrossing(Event_t *event) override;

   ClassDefOverride(TGHSplitter,0)  //A horizontal frame splitter
};

class TGVFileSplitter : public TGVSplitter {

private:
   TGVFileSplitter(const TGVFileSplitter&) = delete;
   TGVFileSplitter& operator=(const TGVFileSplitter&) = delete;

public:
   TGVFileSplitter(const TGWindow *p = nullptr, UInt_t w = 4, UInt_t h = 4,
               UInt_t options = kChildFrame,
               Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGVFileSplitter();

   Bool_t HandleDoubleClick(Event_t *) override;
   Bool_t HandleButton(Event_t *event) override;
   Bool_t HandleMotion(Event_t *event) override;
   void   SavePrimitive(std::ostream &out, Option_t *option = "") override;

   void LayoutHeader(TGFrame *f);  //*SIGNAL*
   void LayoutListView();  //*SIGNAL*
   void ButtonPressed();   //*SIGNAL*
   void ButtonReleased();  //*SIGNAL*
   void DoubleClicked(TGVFileSplitter* frame);  //*SIGNAL*

   ClassDefOverride(TGVFileSplitter,0)  //A vertical file frame splitter
};


#endif
