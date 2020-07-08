// @(#)root/win32gdk:$Id$
// Author: Valeriy Onuchin   05/08/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGWin32VirtualGLProxy
#define ROOT_TGWin32VirtualGLProxy


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32VirtualGLProxy                                                //
//                                                                      //
// The TGWin32VirtualGLProxy proxy class to TVirtualGL                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualGL.h"

#include "TGWin32ProxyBase.h"


class TGWin32GLManagerProxy : public TGLManager, public TGWin32ProxyBase
{
public:
   TGWin32GLManagerProxy();

   Int_t    InitGLWindow(Window_t winID) override;
   Int_t    CreateGLContext(Int_t winInd) override;
   Bool_t   AttachOffScreenDevice(Int_t ctxInd, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   Bool_t   ResizeOffScreenDevice(Int_t devInd, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   void     SelectOffScreenDevice(Int_t devInd) override;
   Int_t    GetVirtualXInd(Int_t devInd) override;
   void     MarkForDirectCopy(Int_t devInd, Bool_t) override;
   void     ExtractViewport(Int_t devInd, Int_t *vp) override;
   void     ReadGLBuffer(Int_t devInd) override;
   Bool_t   MakeCurrent(Int_t devInd) override;
   void     Flush(Int_t ctxInd) override;
   void     DeleteGLContext(Int_t devInd) override;
   Bool_t   SelectManip(TVirtualGLManip *manip, const TGLCamera *camera, const TGLRect *rect, const TGLBoundingBox *sceneBox) override;
   void     PaintSingleObject(TVirtualGLPainter *) override;
   void     PanObject(TVirtualGLPainter *o, Int_t x, Int_t y) override;
   void     PrintViewer(TVirtualViewer3D *vv) override;
   Bool_t   PlotSelected(TVirtualGLPainter *plot, Int_t px, Int_t py) override;
   char    *GetPlotInfo(TVirtualGLPainter *plot, Int_t px, Int_t py) override;
   Bool_t   HighColorFormat(Int_t ctx) override;

   static TGLManager *ProxyObject();
   static TGLManager *RealObject();
};
#endif
