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
   ~TGWin32GLManager();

   //All public functions are TGLManager's final-overriders

   //index returned can be used as a result of gVirtualX->InitWindow
   Int_t    InitGLWindow(Window_t winID);
   //winInd is the index, returned by InitGLWindow
   Int_t    CreateGLContext(Int_t winInd);

   //[            Off-screen rendering part
   //create DIB section to read GL buffer into it,
   //ctxInd is the index, returned by CreateGLContext
   Bool_t   AttachOffScreenDevice(Int_t ctxInd, Int_t x, Int_t y, UInt_t w, UInt_t h);
   Bool_t   ResizeOffScreenDevice(Int_t devInd, Int_t x, Int_t y, UInt_t w, UInt_t h);
   //analog of gVirtualX->SelectWindow(fPixmapID) => gVirtualGL->SelectOffScreenDevice(fPixmapID)
   void     SelectOffScreenDevice(Int_t devInd);
   //Index of DIB, valid for gVirtualX
   Int_t    GetVirtualXInd(Int_t devInd);
   //copy DIB into window directly/by pad
   void     MarkForDirectCopy(Int_t devInd, Bool_t);
   //Off-screen device holds sizes for glViewport
   void     ExtractViewport(Int_t devInd, Int_t *vp);
   //Read GL buffer into DIB
   void     ReadGLBuffer(Int_t devInd);
   //]

   //Make the gl context current
   Bool_t   MakeCurrent(Int_t devInd);
   //Swap buffers or "blits" DIB
   void     Flush(Int_t ctxInd);
   //Generic function for gl context and off-screen device deletion
   void     DeleteGLContext(Int_t devInd);

   //functions to switch between threads in win32
   //used by viewer
   Bool_t   SelectManip(TVirtualGLManip *manip, const TGLCamera *camera, const TGLRect *rect, const TGLBoundingBox *sceneBox);

   Bool_t   PlotSelected(TVirtualGLPainter *plot, Int_t px, Int_t py);
   char    *GetPlotInfo(TVirtualGLPainter *plot, Int_t px, Int_t py);

   void     PaintSingleObject(TVirtualGLPainter *);
   void     PanObject(TVirtualGLPainter *o, Int_t x, Int_t y);
   void     PrintViewer(TVirtualViewer3D *vv);

   Bool_t   HighColorFormat(Int_t ctx);

private:
   struct TGLContext;
   Bool_t   CreateDIB(TGLContext &ctx)const;

   TGWin32GLManager(const TGWin32GLManager &);
   TGWin32GLManager &operator = (const TGWin32GLManager &);

   ClassDef(TGWin32GLManager, 0)
};

#endif
