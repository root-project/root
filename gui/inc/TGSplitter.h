// @(#)root/gui:$Name:$:$Id:$
// Author: Fons Rademakers   6/09/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGSplitter
#define ROOT_TGSplitter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGSplitter, TGVSplitter, TGHorizontal3DLine and TGVertical3DLine     //
//                                                                      //
// A splitter allows the frames left and right or above and below of    //
// it to be resized.                                                    //
// A horizontal 3D line is a line that typically separates a toolbar    //
// from the menubar.                                                    //
// A vertical 3D line is a line that can be used to separate groups of  //
// widgets.                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif


class TGSplitter : public TGFrame {

protected:
   Cursor_t    fSplitCursor;      // split cursor
   Bool_t      fDragging;         // true if in dragging mode

public:
   TGSplitter(const TGWindow *p, UInt_t w = 2, UInt_t h = 4,
              UInt_t options = kChildFrame,
              ULong_t back = fgDefaultFrameBackground);
   virtual ~TGSplitter() { }

   virtual Bool_t HandleButton(Event_t *event) = 0;
   virtual Bool_t HandleMotion(Event_t *event) = 0;
   virtual Bool_t HandleCrossing(Event_t *event) = 0;

   ClassDef(TGSplitter,0)  //A frame splitter abstract base class
};


class TGVSplitter : public TGSplitter {

protected:
   Int_t       fStartX;         // x position when dragging starts
   Int_t       fDelta;          // pixels with which to resize selected frame
   UInt_t      fWidth;          // width of frame to be resized
   UInt_t      fHeight;         // height of frame to be resized
   TGFrame    *fFrame;          // frame that should be resized

public:
   TGVSplitter(const TGWindow *p, UInt_t w = 2, UInt_t h = 4,
               UInt_t options = kChildFrame,
               ULong_t back = fgDefaultFrameBackground);
   virtual ~TGVSplitter() { }

   virtual void DrawBorder();
   virtual void SetFrame(TGFrame *frame);

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual Bool_t HandleCrossing(Event_t *event);

   ClassDef(TGVSplitter,0)  //A vertical frame splitter
};


class TGHorizontal3DLine : public TGFrame {

public:
   TGHorizontal3DLine(const TGWindow *p, UInt_t w = 4, UInt_t h = 2,
                      UInt_t options = kChildFrame,
                      ULong_t back = fgDefaultFrameBackground) :
      TGFrame(p, w, h, options, back) { }

   virtual void DrawBorder() {
      gVirtualX->DrawLine(fId, fgShadowGC,  0, 0, fWidth-2, 0);
      gVirtualX->DrawLine(fId, fgHilightGC, 0, 1, fWidth-1, 1);
      gVirtualX->DrawLine(fId, fgHilightGC, fWidth-1, 0, fWidth-1, 1);
   }

   ClassDef(TGHorizontal3DLine,0)  //A horizontal 3D separator line
};


class TGVertical3DLine : public TGFrame {

public:
   TGVertical3DLine(const TGWindow *p, UInt_t w = 2, UInt_t h = 4,
                    UInt_t options = kChildFrame,
                    ULong_t back = fgDefaultFrameBackground) :
      TGFrame(p, w, h, options, back) { }

   virtual void DrawBorder() {
      gVirtualX->DrawLine(fId, fgShadowGC,  0, 0, 0, fHeight-2);
      gVirtualX->DrawLine(fId, fgHilightGC, 1, 0, 1, fHeight-1);
      gVirtualX->DrawLine(fId, fgHilightGC, 1, fHeight-1, 0, fHeight-1);
   }

   ClassDef(TGVertical3DLine,0)  //A vertical 3D separator line
};

#endif
