// @(#)root/gui:$Id$
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


#include "TGCanvas.h"
#include "TGMenu.h"
#include "TGFont.h"


/// MDI resizing modes
enum EMdiResizingModes {
   kMdiOpaque            = 1,
   kMdiNonOpaque         = 2,
   kMdiDefaultResizeMode = kMdiOpaque
};

/// MDI hints, also used to identify titlebar buttons
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

/// window arrangement modes
enum EMdiArrangementModes {
   kMdiTileHorizontal = 1,
   kMdiTileVertical   = 2,
   kMdiCascade        = 3
};

/// geometry value masks for ConfigureWindow() call
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
   UInt_t            fFrameId;                  ///< TGMdiFrameList Id
   TGMdiDecorFrame  *fDecor;                    ///< MDI decor frame
   TGMdiFrameList   *fPrev, *fNext;             ///< pointers on previous and next TGMdiFrameList
   TGMdiFrameList   *fCyclePrev, *fCycleNext;   ///< pointers on previous and next TGMdiFrameList

public:
   virtual ~TGMdiFrameList() { }

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

   ClassDef(TGMdiFrameList, 0) // MDI Frame list
};


class TGMdiGeometry {

public:
   Int_t            fValueMask;                    ///< MDI hints mask
   TGRectangle      fClient, fDecoration, fIcon;   ///< client, decoration and icon rectangles

   virtual ~TGMdiGeometry() {}

   ClassDef(TGMdiGeometry, 0) // MDI Geometry
};


//----------------------------------------------------------------------

class TGMdiMainFrame : public TGCanvas {

friend class TGMdiFrame;

protected:
   enum {
      // the width of minimized windows, in "height" units
      kMinimizedWidth = 5
   };

   Int_t            fCurrentX, fCurrentY, fResizeMode;   ///< current MDI child XY position and resize mode
   Int_t            fArrangementMode;                    ///< MDI children arrangement mode
   TGFont          *fFontCurrent, *fFontNotCurrent;      ///< fonts for active and inactive MDI children
   Pixel_t          fBackCurrent, fForeCurrent;          ///< back and fore colors for active MDI children
   Pixel_t          fBackNotCurrent, fForeNotCurrent;    ///< back and fore colors for inactive MDI children

   TGGC            *fBoxGC;                              ///< GC used to draw resizing box (rectangle)

   Long_t           fNumberOfFrames;                     ///< number of MDI child windows
   TGMdiMenuBar    *fMenuBar;                            ///< menu bar
   TGFrame         *fContainer;                          ///< MDI container
   TGPopupMenu     *fWinListMenu;                        ///< popup menu with list of MDI child windows
   TGMdiFrameList  *fChildren;                           ///< list of MDI child windows
   TGMdiFrameList  *fCurrent;                            ///< current list of MDI child windows

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

   Bool_t           HandleKey(Event_t *event) override;
   Bool_t           ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;

   void             Layout() override;

   virtual void     FreeMove(TGMdiFrame *frame);
   virtual void     FreeSize(TGMdiFrame *frame);
   virtual void     Restore(TGMdiFrame *frame);
   virtual void     Maximize(TGMdiFrame *frame);
   virtual void     Minimize(TGMdiFrame *frame);
   virtual Int_t    Close(TGMdiFrame *frame);
   virtual Int_t    ContextHelp(TGMdiFrame *frame);
   virtual void     CloseAll();

   virtual void     Cascade() { ArrangeFrames(kMdiCascade); }
   virtual void     TileHorizontal() { ArrangeFrames(kMdiTileHorizontal); }
   virtual void     TileVertical() { ArrangeFrames(kMdiTileVertical); }

   virtual void     ArrangeFrames(Int_t mode);
   virtual void     ArrangeMinimized();

   virtual void     CirculateUp();
   virtual void     CirculateDown();

   TGMdiFrame      *GetCurrent() const;
   TGMdiFrame      *GetMdiFrame(UInt_t id) const;
   TGFrame         *GetContainer() const { return fContainer; }
   Bool_t           SetCurrent(UInt_t newcurrent);
   Bool_t           SetCurrent(TGMdiFrame *f);  //*SIGNAL*

   TGPopupMenu     *GetWinListMenu() const { return fWinListMenu; }
   TGMdiMenuBar    *GetMenu() const { return fMenuBar; }

   TGMdiFrameList  *GetWindowList(Int_t current = kFALSE) const
                     { return current ? fCurrent : fChildren; }
   Long_t           GetNumberOfFrames() const { return fNumberOfFrames; }

   void             SetResizeMode(Int_t mode = kMdiDefaultResizeMode);
   void             UpdateMdiButtons();

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

   void             SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGMdiMainFrame, 0) // MDI main frame
};


//----------------------------------------------------------------------

class TGMdiContainer : public TGFrame {

protected:
   const TGMdiMainFrame *fMain;     // pointer to MDI main frame

public:
   TGMdiContainer(const TGMdiMainFrame *p, Int_t w, Int_t h,
                  UInt_t options = 0,
                  ULong_t back = GetDefaultFrameBackground());

   Bool_t HandleConfigureNotify(Event_t *event) override;
   TGDimension GetDefaultSize() const override;

   ClassDefOverride(TGMdiContainer, 0) // MDI container
};

#endif
