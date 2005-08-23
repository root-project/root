// @(#)root/x11:$Name:  $:$Id: TX11GL.h,v 1.4 2005/08/18 11:12:58 brun Exp $
// Author: Timur Pocheptsov 09/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TX11GL
#define ROOT_TX11GL


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TX11GL                                                               //
//                                                                      //
// The TX11GL is X11 implementation of TVirtualGLImp class.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualGL
#include "TVirtualGL.h"
#endif

#if !defined(__CINT__)
#include <GL/glx.h>
#else
struct Display;
struct XVisualInfo;
#endif


class TX11GL : public TVirtualGLImp {

private:
   Display     *fDpy;
   XVisualInfo *fVisInfo;

public:
   TX11GL();

   Window_t CreateGLWindow(Window_t wind);
   ULong_t  CreateContext(Window_t wind);
   void     DeleteContext(ULong_t ctx);
   void     MakeCurrent(Window_t wind, ULong_t ctx);
   void     SwapBuffers(Window_t wind);

   ClassDef(TX11GL, 0);
};

class TX11GLManager : public TGLManager {
private:
   class TX11GLImpl;
   TX11GLImpl *fPimpl;
   
public:
   TX11GLManager();
   ~TX11GLManager();
   
   Int_t    InitGLWindow(Window_t winId, Bool_t isOffScreen);
   Int_t    CreateGLContext(Int_t winInd);
   Int_t    OpenGLPixmap(Int_t winInd, Int_t x, Int_t y, UInt_t w, UInt_t h);
   void     ResizeGLPixmap(Int_t pixInd, Int_t x, Int_t y, UInt_t w, UInt_t h);
   void     SelectGLPixmap(Int_t pixInd);
   void     MarkForDirectCopy(Int_t pixInd, Bool_t pix);
   Int_t    GetVirtualXInd(Int_t glPix);

   //The same for direct-rendering and offscreen
   Bool_t   MakeCurrent(Int_t deviceInd);
   //swaps buffers or copies pixmap.
   void     Flush(Int_t deviceInd, Int_t x, Int_t y);
   //deletes context or pixmap and context
   void     DeletePaintDevice(Int_t deviceInd);
   void     ExtractViewport(Int_t pixId, Int_t *viewport);

   void     DrawViewer(TVirtualViewer3D *v);
   TObject *Select(TVirtualViewer3D *v, Int_t x, Int_t y);
   
private:
   //Used internally by OpenPixmap and ResizePixmap
   Bool_t CreateGLPixmap(Int_t winId, Int_t x, Int_t y, UInt_t w, UInt_t h, Int_t preferInd = -1);
   
   //implicit copy-ctor/assignment generation
   // was already disabled by base class, but to be explicit ...
   TX11GLManager(const TX11GLManager &);
   TX11GLManager &operator = (const TX11GLManager &);

   ClassDef(TX11GLManager, 0);
};


#endif
