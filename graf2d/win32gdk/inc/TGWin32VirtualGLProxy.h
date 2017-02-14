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

   Int_t    InitGLWindow(Window_t winID);
   Int_t    CreateGLContext(Int_t winInd);
   Bool_t   AttachOffScreenDevice(Int_t ctxInd, Int_t x, Int_t y, UInt_t w, UInt_t h);
   Bool_t   ResizeOffScreenDevice(Int_t devInd, Int_t x, Int_t y, UInt_t w, UInt_t h);
   void     SelectOffScreenDevice(Int_t devInd);
   Int_t    GetVirtualXInd(Int_t devInd);
   void     MarkForDirectCopy(Int_t devInd, Bool_t);
   void     ExtractViewport(Int_t devInd, Int_t *vp);
   void     ReadGLBuffer(Int_t devInd);
   Bool_t   MakeCurrent(Int_t devInd);
   void     Flush(Int_t ctxInd);
   void     DeleteGLContext(Int_t devInd);
   Bool_t   SelectManip(TVirtualGLManip *manip, const TGLCamera *camera, const TGLRect *rect, const TGLBoundingBox *sceneBox);
   void     PaintSingleObject(TVirtualGLPainter *);
   void     PanObject(TVirtualGLPainter *o, Int_t x, Int_t y);
   void     PrintViewer(TVirtualViewer3D *vv);
   Bool_t   PlotSelected(TVirtualGLPainter *plot, Int_t px, Int_t py);
   char    *GetPlotInfo(TVirtualGLPainter *plot, Int_t px, Int_t py);
   Bool_t   HighColorFormat(Int_t ctx);

   static TGLManager *ProxyObject();
   static TGLManager *RealObject();
};
#endif
