// @(#)root/win32gdk:$Id$
// Author: Valeriy Onuchin  05/08/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGWin32GL
#define ROOT_TGWin32GL


#include "TVirtualGL.h"
#include "TVirtualViewer3D.h"


class TGWin32GLManager : public TGLManager {
private:
   class TGWin32GLImpl;
   TGWin32GLImpl *fPimpl;

public:
   TGWin32GLManager();
   ~TGWin32GLManager() override;

   //All public functions are TGLManager's final-overriders

   //index returned can be used as a result of gVirtualX->InitWindow
   Int_t    InitGLWindow(Window_t winID) override;
   //winInd is the index, returned by InitGLWindow
   Int_t    CreateGLContext(Int_t winInd) override;

   //[            Off-screen rendering part
   //create DIB section to read GL buffer into it,
   //ctxInd is the index, returned by CreateGLContext
   Bool_t   AttachOffScreenDevice(Int_t ctxInd, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   Bool_t   ResizeOffScreenDevice(Int_t devInd, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   //analog of gVirtualX->SelectWindow(fPixmapID) => gVirtualGL->SelectOffScreenDevice(fPixmapID)
   void     SelectOffScreenDevice(Int_t devInd) override;
   //Index of DIB, valid for gVirtualX
   Int_t    GetVirtualXInd(Int_t devInd) override;
   //copy DIB into window directly/by pad
   void     MarkForDirectCopy(Int_t devInd, Bool_t) override;
   //Off-screen device holds sizes for glViewport
   void     ExtractViewport(Int_t devInd, Int_t *vp) override;
   //Read GL buffer into DIB
   void     ReadGLBuffer(Int_t devInd) override;
   //]

   //Make the gl context current
   Bool_t   MakeCurrent(Int_t devInd) override;
   //Swap buffers or "blits" DIB
   void     Flush(Int_t ctxInd) override;
   //Generic function for gl context and off-screen device deletion
   void     DeleteGLContext(Int_t devInd) override;

   //functions to switch between threads in win32
   //used by viewer
   Bool_t   SelectManip(TVirtualGLManip *manip, const TGLCamera *camera, const TGLRect *rect, const TGLBoundingBox *sceneBox) override;

   Bool_t   PlotSelected(TVirtualGLPainter *plot, Int_t px, Int_t py) override;
   char    *GetPlotInfo(TVirtualGLPainter *plot, Int_t px, Int_t py) override;

   void     PaintSingleObject(TVirtualGLPainter *) override;
   void     PanObject(TVirtualGLPainter *o, Int_t x, Int_t y) override;
   void     PrintViewer(TVirtualViewer3D *vv) override;

   Bool_t   HighColorFormat(Int_t ctx) override;

private:
   struct TGLContext;
   Bool_t   CreateDIB(TGLContext &ctx)const;

   TGWin32GLManager(const TGWin32GLManager &);
   TGWin32GLManager &operator = (const TGWin32GLManager &);

   ClassDefOverride(TGWin32GLManager, 0)
};

#endif
