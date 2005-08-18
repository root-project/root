// @(#)root/win32gdk:$Name:  $:$Id: TGWin32GL.cxx,v 1.5 2005/08/17 09:10:44 brun Exp $
// Author: Valeriy Onuchin  05/08/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32GL                                                            //
//                                                                      //
// The TGWin32GL is win32gdk implementation of TVirtualGLImp class.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <deque>

#include "TGWin32GL.h"
#include "TGWin32VirtualGLProxy.h"
#include "TVirtualViewer3D.h"
#include "TVirtualX.h"
#include "TError.h"
#include "TROOT.h"

#include "Windows4root.h"
#include "gdk/gdk.h"
#include "gdk/win32/gdkwin32.h"

#include <GL/gl.h>
#include <GL/glu.h>


//______________________________________________________________________________
TGWin32GL::TGWin32GL()
{
   // Ctor.

   gPtr2VirtualGL = &TGWin32VirtualGLProxy::ProxyObject;
}

//______________________________________________________________________________
TGWin32GL::~TGWin32GL()
{
   //

   gPtr2VirtualGL = 0;
}

//______________________________________________________________________________
Window_t TGWin32GL::CreateGLWindow(Window_t wind)
{
   // Win32gdk specific code to initialize GL window.

   GdkColormap *cmap;
   GdkWindow *GLWin;
   GdkWindowAttr xattr;
   int xval, yval;
   int wval, hval;
   ULong_t mask;
   int pixelformat;
   int depth;
   HDC hdc;
   static PIXELFORMATDESCRIPTOR pfd =
   {
      sizeof(PIXELFORMATDESCRIPTOR),  // size of this pfd
      1,                              // version number
      PFD_DRAW_TO_WINDOW |            // support window
      PFD_SUPPORT_OPENGL |            // support OpenGL
      PFD_DOUBLEBUFFER,               // double buffered
      PFD_TYPE_RGBA,                  // RGBA type
      24,                             // 24-bit color depth
      0, 0, 0, 0, 0, 0,               // color bits ignored
      0,                              // no alpha buffer
      0,                              // shift bit ignored
      0,                              // no accumulation buffer
      0, 0, 0, 0,                     // accum bits ignored
      32,                             // 32-bit z-buffer
      0,                              // no stencil buffer
      0,                              // no auxiliary buffer
      PFD_MAIN_PLANE,                 // main layer
      0,                              // reserved
      0, 0, 0                         // layer masks ignored
   };

   gdk_window_get_geometry((GdkDrawable *) wind, &xval, &yval, &wval, &hval, &depth);
   cmap = gdk_colormap_get_system();

   // window attributes
   xattr.width = wval;
   xattr.height = hval;
   xattr.x = xval;
   xattr.y = yval;
   xattr.wclass = GDK_INPUT_OUTPUT;
   xattr.event_mask = 0L; //GDK_ALL_EVENTS_MASK;
   xattr.event_mask |= GDK_EXPOSURE_MASK | GDK_STRUCTURE_MASK |
                       GDK_KEY_PRESS_MASK | GDK_KEY_RELEASE_MASK;
   xattr.colormap = cmap;
   mask = GDK_WA_X | GDK_WA_Y | GDK_WA_COLORMAP | GDK_WA_WMCLASS | GDK_WA_NOREDIR;
   xattr.window_type = GDK_WINDOW_CHILD;

   GLWin = gdk_window_new((GdkWindow *) wind, &xattr, mask);
   gdk_window_set_events(GLWin, (GdkEventMask)0);
   gdk_window_show(GLWin);
   hdc = GetWindowDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)GLWin));

   if ( (pixelformat = ChoosePixelFormat(hdc,&pfd)) == 0 ) {
       Error("InitGLWindow", "Barf! ChoosePixelFormat Failed");
   }
   if ( (SetPixelFormat(hdc, pixelformat,&pfd)) == 0 ) {
      Error("InitGLWindow", "Barf! SetPixelFormat Failed");
   }

   return (Window_t)GLWin;
}

//______________________________________________________________________________
ULong_t TGWin32GL::CreateContext(Window_t wind)
{
   //

   HDC hdc = GetWindowDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)wind));
   ULong_t retVal = (ULong_t)wglCreateContext(hdc);
   ReleaseDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)wind), hdc);
   return retVal;
}

//______________________________________________________________________________
void TGWin32GL::DeleteContext(ULong_t ctx)
{
   //

   ::wglDeleteContext((HGLRC)ctx);
}

//______________________________________________________________________________
void TGWin32GL::MakeCurrent(Window_t wind, ULong_t ctx)
{
   //

   HDC hdc = GetWindowDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)wind));
   ::wglMakeCurrent(hdc,(HGLRC) ctx);
   ReleaseDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)wind), hdc);
}

//______________________________________________________________________________
void TGWin32GL::SwapBuffers(Window_t wind)
{
   //

   HDC hdc = GetWindowDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)wind));
   ::wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
   ::glFinish();
   ReleaseDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)wind), hdc);
}

///////////////////////////////////////////////////////////////
//New Win32 GL stuff
//////////////////////////////

namespace {
   struct PaintDevice {
      Int_t fWindowIndex;
      Int_t fPixmapIndex;//-1 for double buffered gl
      //
      HDC fDC;
      HBITMAP fHBitmap;
      HGLRC fGLContext;
      //
      UInt_t fRealW;
      UInt_t fRealH;
      //
      UInt_t fCurrW;
      UInt_t fCurrH;
      //
      Int_t fX;
      Int_t fY;
      //
      Bool_t fDirect;
      HBITMAP fOldBitmap;
      PaintDevice *fNextFreeDevice;
   };

   PaintDevice emptyDev;

   class CDCGuard {
   private:
      HDC fHDC;
      CDCGuard(const CDCGuard &);
      CDCGuard &operator = (const CDCGuard &);
   public:
      explicit CDCGuard(HDC hDC) : fHDC(hDC)
      {}
      ~CDCGuard()
      {
         if (fHDC)
            DeleteDC(fHDC);
      }
      void Stop()
      {
         fHDC = 0;
      }   
   };

   class WDCGuard {
   private:
      HDC fHDC;
      Window_t fWinID;
      WDCGuard(const WDCGuard &);
      WDCGuard &operator = (const WDCGuard &);
   public:
      WDCGuard(HDC hDC, Window_t winID) : fHDC(hDC), fWinID(winID)
      {}
      ~WDCGuard()
      {
         if (fHDC)
            ReleaseDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)fWinID), fHDC);
      }
      void Stop()
      {
         fHDC = 0;
      }
   };

   class BMPGuard {
   private:
      HBITMAP fBMP;
      BMPGuard(const BMPGuard &);
      BMPGuard &operator = (const BMPGuard &);
   public:
      explicit BMPGuard(HBITMAP bmp) : fBMP(bmp)
      {}
      ~BMPGuard()
      {
         if (fBMP)
            DeleteObject(fBMP);
      }
      void Stop()
      {
         fBMP = 0;
      }
   };

   class WGLGuard {
   private:
      HGLRC fCtx;
      WGLGuard(const WGLGuard &);
      WGLGuard &operator = (const WGLGuard &);

   public:
      explicit WGLGuard(HGLRC glrc) : fCtx(glrc)
      {}
      ~WGLGuard()
      {
         if (fCtx) 
            wglDeleteContext(fCtx);
      }
      void Stop()
      {
         fCtx = 0;
      }
   };
}

const PIXELFORMATDESCRIPTOR
doubleBufferDesc = {
   sizeof doubleBufferDesc,	   // size of this pfd
   1,                              // version number
   PFD_DRAW_TO_WINDOW |            // support window
   PFD_SUPPORT_OPENGL |            // support OpenGL
   PFD_DOUBLEBUFFER,               // double buffered
   PFD_TYPE_RGBA,                  // RGBA type
   24,                             // 24-bit color depth
   0, 0, 0, 0, 0, 0,               // color bits ignored
   0,                              // no alpha buffer
   0,                              // shift bit ignored
   0,                              // no accumulation buffer
   0, 0, 0, 0,                     // accum bits ignored
   32,                             // 32-bit z-buffer
   0,                              // no stencil buffer
   0,                              // no auxiliary buffer
   PFD_MAIN_PLANE                  // main layer
};

const PIXELFORMATDESCRIPTOR
offScreenDesc = {
   sizeof offScreenDesc,	   // size of this pfd
   1,                              // version number
   PFD_DRAW_TO_BITMAP |	           // draw into bitmap
   PFD_SUPPORT_OPENGL,             // support OpenGL
   PFD_TYPE_RGBA,                  // RGBA type
   24,                             // 24-bit color depth
   0, 0, 0, 0, 0, 0,               // color bits ignored
   0,                              // no alpha buffer
   0,                              // shift bit ignored
   0,                              // no accumulation buffer
   0, 0, 0, 0,                     // accum bits ignored
   32,                             // 32-bit z-buffer
   0,                              // no stencil buffer
   0,                              // no auxiliary buffer
   PFD_MAIN_PLANE                  // main layer
};

class TGWin32GLManager::TGWin32GLImpl {
public:
   TGWin32GLImpl() : fNextFreeDevice(0)
   {}
   ~TGWin32GLImpl();
   std::deque<PaintDevice> fPaintDevices;
   PaintDevice *fNextFreeDevice;
};

TGWin32GLManager::TGWin32GLImpl::~TGWin32GLImpl()
{
   //all devices should be destroyed at this moment
   std::deque<PaintDevice>::size_type i = 0;

   for (; i < fPaintDevices.size(); ++i) {
      PaintDevice &currDev = fPaintDevices[i];      

      if (currDev.fGLContext) {
         //gl context (+pixmap, if exists) must be destroyed from outside, by pad.
         ::Warning("TGWin32GLManager::~TGLWin32GLManager", " you forget to destroy gl-device %d\n", i);

         //destroy hdc and glrc, pixmap will be destroyed by TVirtualX (?)
         if (currDev.fPixmapIndex != -1) {
            gVirtualX->SelectWindow(currDev.fPixmapIndex);
            gVirtualX->ClosePixmap();
         }

         wglDeleteContext(currDev.fGLContext);
         ReleaseDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)gVirtualX->GetWindowID(currDev.fWindowIndex)),
                   currDev.fDC);
      }
   }
}

ClassImp(TGWin32GLManager)

//______________________________________________________________________________
TGWin32GLManager::TGWin32GLManager() : fPimpl(new TGWin32GLImpl)
{
   gPtr2GLManager = &TGWin32GLManagerProxy::ProxyObject;
   gROOT->GetListOfSpecials()->AddLast(this);
   gGLManager = this;
}

//______________________________________________________________________________
TGWin32GLManager::~TGWin32GLManager()
{
   delete fPimpl;
}

//______________________________________________________________________________
Int_t TGWin32GLManager::InitGLWindow(Window_t winId, Bool_t)
{
   return gVirtualX->InitWindow(winId);
}

//______________________________________________________________________________
Int_t TGWin32GLManager::CreateGLContext(Int_t winInd)
{
   Window_t winID = gVirtualX->GetWindowID(winInd);
   HDC hDC = GetWindowDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)winID));
   
   if (!hDC) {
      Error("CreateGLContext", "GetWindowDC failed\n");
      return -1; 
   }

   WDCGuard dcGuard(hDC, winID);

   if (Int_t pixFormat = ChoosePixelFormat(hDC, &doubleBufferDesc)) {
      if (SetPixelFormat(hDC, pixFormat, &doubleBufferDesc)) {
         HGLRC glCtx = wglCreateContext(hDC);

         if (glCtx) {
            PaintDevice newDevice = {winInd, -1, hDC, 0, glCtx};

            if (PaintDevice *dev = fPimpl->fNextFreeDevice) {
               Int_t ind = dev->fWindowIndex;
               *dev = newDevice;
               dcGuard.Stop();
               
               return ind;
            } else {
               WGLGuard wglGuard(glCtx);
               fPimpl->fPaintDevices.push_back(newDevice);
               wglGuard.Stop();
               dcGuard.Stop();

               return fPimpl->fPaintDevices.size() - 1;
            }
         } else
            Error("CreateGLContext", "wglCreateContext failed\n");
      } else
         Error("CreateGLContext", "SetPixelFormat failed\n");
   } else
      Error("CreateGLContext", "ChoosePixelFormat failed\n");

   return -1;
}

//______________________________________________________________________________
Bool_t TGWin32GLManager::CreateGLPixmap(Int_t winInd, Int_t x, Int_t y, UInt_t w, UInt_t h, Int_t prevInd)
{
   HDC dibDC = CreateCompatibleDC(0);// new DC in memory

   if (!dibDC) {
      Error("CreateGLPixmap", "CreateCompatibleDC failed\n");
      return kFALSE;
   }

   CDCGuard dcGuard(dibDC);
	
   BITMAPINFOHEADER bmpHeader = {sizeof bmpHeader, w, h, 1, 24, BI_RGB};
   void *bmpCnt = 0;
   HBITMAP hDIB = CreateDIBSection(dibDC, (BITMAPINFO*)&bmpHeader, DIB_RGB_COLORS, &bmpCnt, 0, 0);
   
   if (!hDIB) {
      Error("CreateGLPixmap", "CreateDIBSection failed\n");
      return kFALSE;
   }

   BMPGuard bmpGuard(hDIB);
   HBITMAP hOldDIB = (HBITMAP)SelectObject(dibDC, hDIB);

   if (Int_t pixelFormat = ChoosePixelFormat(dibDC, &offScreenDesc)) {
      if (SetPixelFormat(dibDC, pixelFormat, &offScreenDesc)) {
         HGLRC glrc = wglCreateContext(dibDC);

         if (glrc) {
            PaintDevice newDev = {winInd, -1, dibDC, hDIB, glrc, w, h, w, h, x, y, kFALSE, hOldDIB, 0};

            if (prevInd == -1) {
               WGLGuard wglGuard(glrc);
               newDev.fPixmapIndex = gVirtualX->AddPixmap((ULong_t)hDIB, w, h);
               fPimpl->fPaintDevices.push_back(newDev);
               wglGuard.Stop();
            } else if (fPimpl->fPaintDevices[prevInd].fGLContext){
               //resize existing pixmap
               gVirtualX->AddPixmap((ULong_t)hDIB, w, h, fPimpl->fPaintDevices[prevInd].fPixmapIndex);
               newDev.fPixmapIndex = fPimpl->fPaintDevices[prevInd].fPixmapIndex;
               wglDeleteContext(fPimpl->fPaintDevices[prevInd].fGLContext);
               DeleteDC(fPimpl->fPaintDevices[prevInd].fDC);
               fPimpl->fPaintDevices[prevInd] = newDev;
            } else {
               //reuse existing place in fPaintDevices, add new pixmap to TVirtualX
               newDev.fPixmapIndex = gVirtualX->AddPixmap((ULong_t)hDIB, w, h);
               fPimpl->fPaintDevices[prevInd] = newDev;
            }

            bmpGuard.Stop();
            dcGuard.Stop();
               
            return kTRUE; 
         } else
            Error("OpenGLPixmap", "wglCreateContext failed");
      } else
         Error("OpenGLPixmap", "SetPixelFormat failed\n");
   } else
      Error("OpenGLPixmap", "ChoosePixelFormat Failed");


   return kFALSE;
}

//______________________________________________________________________________
Int_t TGWin32GLManager::OpenGLPixmap(Int_t winInd, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   if (PaintDevice *dev = fPimpl->fNextFreeDevice) {
      //reuse existing place in fPaintDevices
      Int_t prevInd = dev->fWindowIndex; //obscure usage of fWindowIndex

      if (CreateGLPixmap(winInd, x, y, w, h, prevInd)) {
         fPimpl->fNextFreeDevice = fPimpl->fNextFreeDevice->fNextFreeDevice;

         return prevInd;
      }
   } else if (CreateGLPixmap(winInd, x, y, w, h))
      return Int_t(fPimpl->fPaintDevices.size()) - 1;

   return -1;
}

//______________________________________________________________________________
void TGWin32GLManager::ResizeGLPixmap(Int_t pixInd, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   PaintDevice &dev = fPimpl->fPaintDevices[pixInd];

   if (w - dev.fRealW > 1 || h - dev.fRealH > 1) {
      //destroy old DIB with such index and create new in place
      CreateGLPixmap(dev.fWindowIndex, x, y, w, h, pixInd);
   } else {
      //simply change size-description
      dev.fCurrW = w;
      dev.fCurrH = h;
      gVirtualX->AddPixmap(0, w, h, dev.fPixmapIndex);
   }

   dev.fX = x;
   dev.fY = y;
}

//______________________________________________________________________________
void TGWin32GLManager::SelectGLPixmap(Int_t pixInd)
{
   gVirtualX->SelectWindow(fPimpl->fPaintDevices[pixInd].fPixmapIndex);
}

//______________________________________________________________________________
void TGWin32GLManager::MarkForDirectCopy(Int_t pixInd, Bool_t isDirect)
{
   if (fPimpl->fPaintDevices[pixInd].fPixmapIndex != -1) {
      //pixmap and context, not simply context
      fPimpl->fPaintDevices[pixInd].fDirect = isDirect;
   }
}

//______________________________________________________________________________
Int_t TGWin32GLManager::GetVirtualXInd(Int_t pixInd)
{
   //this HBITMAP will be used outside of gl code
   //but HBITMAP can can be selected only into
   //one dc at a time, so deselect from curr dc first
   PaintDevice &currDev = fPimpl->fPaintDevices[pixInd];

   if (currDev.fOldBitmap != currDev.fHBitmap) {
      currDev.fOldBitmap = (HBITMAP)SelectObject(currDev.fDC, currDev.fOldBitmap);
   }

   return fPimpl->fPaintDevices[pixInd].fPixmapIndex;
}

//______________________________________________________________________________
Bool_t TGWin32GLManager::MakeCurrent(Int_t devInd)
{
   PaintDevice &currDev = fPimpl->fPaintDevices[devInd];
   //fDC can be HDC obtained by GetWindowDC or CreateCompatibleDC (the later
   //is for gl-to-bitmap mode)
   if (currDev.fPixmapIndex != -1) {
      //select HBITMAP into dc
      if (currDev.fOldBitmap == currDev.fHBitmap) {
         currDev.fOldBitmap = (HBITMAP)SelectObject(currDev.fDC, currDev.fHBitmap);
       }
   }
   return (Bool_t)wglMakeCurrent(currDev.fDC, currDev.fGLContext);
}

//______________________________________________________________________________
void TGWin32GLManager::Flush(Int_t devInd, Int_t, Int_t)
{
   PaintDevice &currDev = fPimpl->fPaintDevices[devInd];

   if (currDev.fPixmapIndex == -1) {
      //doube-buffered OpenGL
      wglSwapLayerBuffers(currDev.fDC, WGL_SWAP_MAIN_PLANE);
   } else if (currDev.fDirect) {
      //DIB is flushed by viewer directly
      Window_t winID = gVirtualX->GetWindowID(currDev.fWindowIndex);
      HDC hDC = GetWindowDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)winID));

      if (!hDC) {
         Error("Flush", " GetWindowDC failed\n");
         return;
      }
  
      if (!BitBlt(hDC, currDev.fX, currDev.fY, currDev.fCurrW, 
                  currDev.fCurrH, currDev.fDC, 0, 0, SRCCOPY))
      {
         currDev.fDirect = kFALSE;
      }

      ReleaseDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)winID), hDC);
   }
   //nothing done for non-direct DIB, it will
   //be copied by pad.
}

//______________________________________________________________________________
void TGWin32GLManager::DeletePaintDevice(Int_t devInd)
{
   PaintDevice &currDev = fPimpl->fPaintDevices[devInd];

   if (currDev.fPixmapIndex != -1) {
      gVirtualX->SelectWindow(currDev.fPixmapIndex);
      gVirtualX->ClosePixmap();
      currDev.fPixmapIndex = -1;
   }

   wglDeleteContext(currDev.fGLContext);
   currDev.fGLContext = 0;
   ReleaseDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)gVirtualX->GetWindowID(currDev.fWindowIndex)), 
             currDev.fDC);

   //now, save its own index before putting into list of free devices
   currDev.fWindowIndex = devInd;
   currDev.fNextFreeDevice = fPimpl->fNextFreeDevice;
   fPimpl->fNextFreeDevice = &currDev;
}

//______________________________________________________________________________
void TGWin32GLManager::ExtractViewport(Int_t devInd, Int_t *viewport)
{
   PaintDevice &dev = fPimpl->fPaintDevices[devInd];

   if (dev.fPixmapIndex != -1) {
      viewport[0] = 0;
      viewport[1] = dev.fRealH - dev.fCurrH;
      viewport[2] = dev.fCurrW;
      viewport[3] = dev.fCurrH;
   }
}

//______________________________________________________________________________
void TGWin32GLManager::DrawViewer(TVirtualViewer3D *vv)
{
   vv->DrawViewer();
}

//______________________________________________________________________________
TObject *TGWin32GLManager::Select(TVirtualViewer3D *vv, Int_t x, Int_t y)
{
   return vv->SelectObject(x, y);
}
