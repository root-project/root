// @(#)root/gui:$Name:  $:$Id: TGMdiMainFrame.h,v 1.2 2004/09/03 16:19:37 rdm Exp $
// Author: Bertrand Bellenot   20/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**************************************************************************

    This file is part of TGMdi, an extension to the xclass toolkit.
    Copyright (C) 1998-2002 by Harald Radke, Hector Peraza.

    This application is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This application is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

**************************************************************************/

#ifndef ROOT_TGMdiMainFrame
#define ROOT_TGMdiMainFrame

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGMdiMainFrame.                                                      //
//                                                                      //
// This file contains the TGMdiMainFrame class.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGCanvas
#include "TGCanvas.h"
#endif
#ifndef ROOT_TGMenu
#include "TGMenu.h"
#endif
#ifndef ROOT_TGFont
#include "TGFont.h"
#endif


// MDI resizing modes
enum EMdiResizingModes {
   kMdiOpaque            = 1,
   kMdiNonOpaque         = 2,
   kMdiDefaultResizeMode = kMdiOpaque
};

// MDI hints, also used to identify titlebar buttons
enum EMdiHints {
   kMdiClose         = 4,
   kMdiRestore       = 8,
   kMdiMove          = 16,
   kMdiSize          = 32,
   kMdiMinimize      = 64,
   kMdiMaximize      = 128,
   kMdiHelp          = 256,
   kMdiMenu          = 512,
   kMdiDefaultHints  = kMdiMenu | kMdiMinimize | kMdiRestore |
                       kMdiMaximize | kMdiSize | kMdiClose
};

// window arrangement modes
enum EMdiArrangementModes {
   kMdiTileHorizontal = 1,
   kMdiTileVertical   = 2,
   kMdiCascade        = 3
};

// geometry value masks for ConfigureWindow() call
enum EMdiGeometryMask {
   kMdiClientGeometry = BIT(0),
   kMdiDecorGeometry  = BIT(1),
   kMdiIconGeometry   = BIT(2)
};


class TGGC;
class TGMdiMenuBar;
class TGMdiContainer;
class TGMdiDecorFrame;
class TGMdiFrame;

//----------------------------------------------------------------------

class TGMdiFrameList {

friend class TGMdiMainFrame;

protected:
   UInt_t            fFrameId;
   TGMdiDecorFrame  *fDecor;
   TGMdiFrameList   *fPrev, *fNext;
   TGMdiFrameList   *fCyclePrev, *fCycleNext;

public:
   UInt_t            GetFrameId() const { return fFrameId; }
   TGMdiDecorFrame  *GetDecorFrame() const { return fDecor; }
   TGMdiFrameList   *GetPrev() const { return fPrev; }
   TGMdiFrameList   *GetNext() const { return fNext; }
   TGMdiFrameList   *GetCyclePrev() const { return fCyclePrev; }
   TGMdiFrameList   *GetCycleNext() const { return fCycleNext; }

   void              SetFrameId(UInt_t id) { fFrameId = id; }
   void              SetDecorFrame(TGMdiDecorFrame *decor) { fDecor = decor; }
   void              SetPrev(TGMdiFrameList *prev) { fPrev = prev; }
   void              SetNext(TGMdiFrameList *next) { fNext = next; }
   void              SetCyclePrev(TGMdiFrameList *prev) { fCyclePrev = prev; }
   void              SetCycleNext(TGMdiFrameList *next) { fCycleNext = next; }

   ClassDef(TGMdiFrameList, 0)
};


class TGMdiGeometry {

public:
   Int_t            fValueMask;
   TGRectangle      fClient, fDecoration, fIcon;

   virtual ~TGMdiGeometry() { }

   ClassDef(TGMdiGeometry, 0)
};


//----------------------------------------------------------------------

class TGMdiMainFrame : public TGCanvas {

friend class TGMdiFrame;

protected:
   enum {
      // the width of minimized windows, in "height" units
      kMinimizedWidth = 5
   };

   Int_t            fCurrentX, fCurrentY, fResizeMode;
   TGFont          *fFontCurrent, *fFontNotCurrent;
   Pixel_t          fBackCurrent, fForeCurrent;
   Pixel_t          fBackNotCurrent, fForeNotCurrent;

   TGGC            *fBoxGC;

   Long_t           fNumberOfFrames;
   TGMdiMenuBar    *fMenuBar;
   TGFrame         *fContainer;
   TGPopupMenu     *fWinListMenu;
   TGMdiFrameList  *fChildren;
   TGMdiFrameList  *fCurrent;

   void             AddMdiFrame(TGMdiFrame *f);
   Bool_t           RemoveMdiFrame(TGMdiFrame *f);

   Bool_t           SetCurrent(TGMdiFrameList *newcurrent);
   TGMdiDecorFrame *GetDecorFrame(UInt_t id) const;
   TGMdiDecorFrame *GetDecorFrame(TGMdiFrame *frame) const;

   void             UpdateWinListMenu();

public:
   TGMdiMainFrame(const TGWindow *p, TGMdiMenuBar *menu, Int_t w, Int_t h,
                  UInt_t options = 0,
                  Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGMdiMainFrame();

   virtual Bool_t   HandleKey(Event_t *event);
   virtual Bool_t   ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   virtual void     Layout();

   void             FreeMove(TGMdiFrame *frame);
   void             FreeSize(TGMdiFrame *frame);
   void             Restore(TGMdiFrame *frame);
   void             Maximize(TGMdiFrame *frame);
   void             Minimize(TGMdiFrame *frame);
   Int_t            Close(TGMdiFrame *frame);
   Int_t            ContextHelp(TGMdiFrame *frame);

   void             Cascade() { ArrangeFrames(kMdiCascade); }
   void             TileHorizontal() { ArrangeFrames(kMdiTileHorizontal); }
   void             TileVertical() { ArrangeFrames(kMdiTileVertical); }

   void             ArrangeFrames(Int_t mode);
   void             ArrangeMinimized();

   void             CirculateUp();
   void             CirculateDown();

   TGMdiFrame      *GetCurrent() const;
   TGMdiFrame      *GetMdiFrame(UInt_t id) const;
   Bool_t           SetCurrent(UInt_t newcurrent);
   Bool_t           SetCurrent(TGMdiFrame *f);

   TGPopupMenu     *GetWinListMenu() const { return fWinListMenu; }
   TGMdiMenuBar    *GetMenu() const { return fMenuBar; }

   TGMdiFrameList  *GetWindowList(Int_t current = kFALSE) const
                       { return current ? fCurrent : fChildren; }
   Long_t           GetNumberOfFrames() const { return fNumberOfFrames; }

   void             SetResizeMode(Int_t mode = kMdiDefaultResizeMode);

   TGRectangle      GetBBox() const;
   TGRectangle      GetMinimizedBBox() const;

   TGMdiGeometry    GetWindowGeometry(TGMdiFrame *f) const;
   void             ConfigureWindow(TGMdiFrame *f, TGMdiGeometry &geom);

   Bool_t           IsMaximized(TGMdiFrame *f);
   Bool_t           IsMinimized(TGMdiFrame *f);

   virtual void     FrameCreated(Int_t id) { Emit("FrameCreated(Int_t)", id); } //*SIGNAL*
   virtual void     FrameClosed(Int_t id) { Emit("FrameClosed(Int_t)", id); } //*SIGNAL*
   virtual void     FrameMaximized(Int_t id) { Emit("FrameMaximized(Int_t)", id); } //*SIGNAL*
   virtual void     FrameMinimized(Int_t id) { Emit("FrameMinimized(Int_t)", id); } //*SIGNAL*
   virtual void     FrameRestored(Int_t id) { Emit("FrameRestored(Int_t)", id); } //*SIGNAL*
   virtual void     FramesArranged(Int_t mode) { Emit("FramesArranged(Int_t)", mode); } //*SIGNAL*

   ClassDef(TGMdiMainFrame, 0)
};


//----------------------------------------------------------------------

class TGMdiContainer : public TGFrame {

protected:
   const TGMdiMainFrame *fMain;

public:
   TGMdiContainer(const TGMdiMainFrame *p, Int_t w, Int_t h,
                  UInt_t options = 0,
                  ULong_t back = GetDefaultFrameBackground());

   virtual Bool_t HandleConfigureNotify(Event_t *event);
   virtual TGDimension GetDefaultSize() const;

   ClassDef(TGMdiContainer, 0)
};

#endif
