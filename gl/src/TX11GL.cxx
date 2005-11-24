// @(#)root/gx11:$Name:  $:$Id: TX11GL.cxx,v 1.10 2005/11/17 14:43:17 couet Exp $
// Author: Timur Pocheptsov 09/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TX11GL                                                               //
//                                                                      //
// The TX11GL is X11 implementation of TVirtualGLImp class.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include <deque>
#include <map>

#include "TVirtualViewer3D.h"
#include "TVirtualX.h"
#include "TX11GL.h"
#include "TError.h"
#include "TROOT.h"

ClassImp(TX11GL)

//______________________________________________________________________________
TX11GL::TX11GL() : fDpy(0), fVisInfo(0)
{
}

//______________________________________________________________________________
Window_t TX11GL::CreateGLWindow(Window_t wind)
{
   if(!fDpy)
      fDpy = (Display *)gVirtualX->GetDisplay();

   static int dblBuf[] = {
                           GLX_DOUBLEBUFFER,
#ifdef STEREO_GL
                           GLX_STEREO,
#endif
                           GLX_RGBA, GLX_DEPTH_SIZE, 16,
                           GLX_RED_SIZE, 1, GLX_GREEN_SIZE, 1,
                           GLX_BLUE_SIZE, 1,None
                           };
   static int * snglBuf = dblBuf + 1;

   if(!fVisInfo){
      fVisInfo = glXChooseVisual(fDpy, DefaultScreen(fDpy), dblBuf);

      if(!fVisInfo)
         fVisInfo = glXChooseVisual(fDpy, DefaultScreen(fDpy), snglBuf);

      if(!fVisInfo){
         ::Error("TX11GL::CreateGLWindow", "no good visual found");
         return 0;
      }
   }

   Int_t  xval = 0, yval = 0;
   UInt_t wval = 0, hval = 0, border = 0, d = 0;
   Window root;

   XGetGeometry(fDpy, wind, &root, &xval, &yval, &wval, &hval, &border, &d);
   ULong_t mask = 0;
   XSetWindowAttributes attr;

   attr.background_pixel = 0;
   attr.border_pixel = 0;
   attr.colormap = XCreateColormap(fDpy, root, fVisInfo->visual, AllocNone);
   attr.event_mask = NoEventMask;
   attr.backing_store = Always;
   attr.bit_gravity = NorthWestGravity;
   mask = CWBackPixel | CWBorderPixel | CWColormap | CWEventMask |
          CWBackingStore | CWBitGravity;

   Window glWin = XCreateWindow(fDpy, wind, xval, yval, wval, hval,
                                0, fVisInfo->depth, InputOutput,
                                fVisInfo->visual, mask, &attr);
   XMapWindow(fDpy, glWin);
   return (Window_t)glWin;
}

//______________________________________________________________________________
ULong_t TX11GL::CreateContext(Window_t)
{
   return (ULong_t)glXCreateContext(fDpy, fVisInfo, None, GL_TRUE);
}

//______________________________________________________________________________
void TX11GL::DeleteContext(ULong_t ctx)
{
   glXDestroyContext(fDpy, (GLXContext)ctx);
}

//______________________________________________________________________________
void TX11GL::MakeCurrent(Window_t wind, ULong_t ctx)
{
   glXMakeCurrent(fDpy, (GLXDrawable)wind, (GLXContext)ctx);
}

//______________________________________________________________________________
void TX11GL::SwapBuffers(Window_t wind)
{
   glXSwapBuffers(fDpy, (GLXDrawable)wind);
}

////////////////////////////////////////GL Manager stuff///////////////////////////////////////

namespace {

   struct PaintDevice {
      //these are numbers returned by gVirtualX->AddWindow and gVirtualX->AddPixmap
      //need both, I can have a pixmap, which is always created for certain window
      Int_t        fWindowIndex;
      Int_t        fPixmapIndex;
      //Pixmap info, not used for double buffered windows
      Pixmap       fX11Pixmap; //required by TX11GLManager::DirectCopy and explicit destruction in resize
      GLXPixmap    fGLXPixmap;
      //Pixmap parameters
      UInt_t       fRealW;//to check, if we really need new pixmap during resize
      UInt_t       fRealH;
      UInt_t       fCurrW;//used by DirectCopy
      UInt_t       fCurrH;
      //Where to XCopyArea pixmap
      Int_t        fX;//used by DirectCopy
      Int_t        fY;
      //ctx, used for off-screen and double-buffered gl painting
      GLXContext   fGLXContext;
      Bool_t       fDirect;
      //
      PaintDevice *fNextFreeDevice;
   };
    
   typedef std::deque<PaintDevice> DeviceTable_t;
   typedef DeviceTable_t::size_type size_type;
   typedef std::map<Int_t, XVisualInfo *> WinTable_t;
   
   XSetWindowAttributes dummyAttr;  
   PaintDevice          dummyDevice; 
   
   const Int_t dblBuff[] = {
                            GLX_DOUBLEBUFFER,
                            GLX_RGBA, 
                            GLX_DEPTH_SIZE, 16,
                            GLX_RED_SIZE, 1, 
                            GLX_GREEN_SIZE, 1,
                            GLX_BLUE_SIZE, 1,
                            None
                           };

   const Int_t *snglBuff = dblBuff + 1;
   
   //Here I can have one universal guard class, but :
   //I'm not shure about X11/GLX function linkage (are they always extern "C" ???)
   //man does not show return types of these functions, in old K&R C it's int
   //in such cases, but in C++ there are no implicit int. So... :))
   
   class X11PixGuard {
   private:
      Display *fDpy;
      Pixmap   fPix;

   public:
      X11PixGuard(Display *dpy, Pixmap pix) : fDpy(dpy), fPix(pix) {}
      ~X11PixGuard(){if (fPix) XFreePixmap(fDpy, fPix);}
      void Stop(){fPix = 0;}
   
   private:
      X11PixGuard(const X11PixGuard &);
      X11PixGuard &operator = (const X11PixGuard &);
   };

   class GLXPixGuard {
   private:
      Display    *fDpy;
      GLXPixmap   fPix;

   public:
      GLXPixGuard(Display *dpy, GLXPixmap pix) : fDpy(dpy), fPix(pix) {}
      ~GLXPixGuard(){if (fPix) glXDestroyGLXPixmap(fDpy, fPix);}
      void Stop(){fPix = 0;}
   
   private:
      GLXPixGuard(const GLXPixGuard &);
      GLXPixGuard &operator = (const GLXPixGuard &);
   };
   
   class GLXCtxGuard {
   private:
      Display    *fDpy;
      GLXContext  fCtx;

   public:
      GLXCtxGuard(Display *dpy, GLXContext ctx) : fDpy(dpy), fCtx(ctx) {}
      ~GLXCtxGuard(){if (fCtx) glXDestroyContext(fDpy, fCtx);}
      void Stop(){fCtx = 0;}
   
   private:
      GLXCtxGuard(const GLXCtxGuard &);
      GLXCtxGuard &operator = (const GLXCtxGuard &);
   };
   
   Int_t length(PaintDevice *p)
   {
      Int_t rez = 0;
      while(p && ++rez, p = p->fNextFreeDevice);
      return rez;
   }
}


class TX11GLManager::TX11GLImpl {
public:
   TX11GLImpl();
   ~TX11GLImpl();

   WinTable_t      fGLWindows;
   DeviceTable_t   fPaintDevices;
   Display        *fDpy;
   PaintDevice    *fNextFreeDevice;
   
private:
   TX11GLImpl(const TX11GLImpl &);
   TX11GLImpl &operator = (const TX11GLImpl &);
};


ClassImp(TX11GLManager)

//______________________________________________________________________________
TX11GLManager::TX11GLImpl::TX11GLImpl() : fDpy(0), fNextFreeDevice(0)
{
   fDpy = reinterpret_cast<Display *>(gVirtualX->GetDisplay());
}

//______________________________________________________________________________
TX11GLManager::TX11GLImpl::~TX11GLImpl()
{
   //destroys only gl contexts and GLXPixmap,
   //pixmaps and windows must be
   //closed through gVirtualX
   for (size_type i = 0,  e = fPaintDevices.size(); i < e; ++i) {
      PaintDevice &currDev = fPaintDevices[i];

      if (GLXContext ctx = currDev.fGLXContext) {
         ::Warning("TX11GLManager::~TX11GLManager", "opengl device with index %d was not destroyed", i);
         glXDestroyContext(fDpy, ctx);
         
         if (currDev.fPixmapIndex != -1) {
            gVirtualX->SelectWindow(currDev.fPixmapIndex);
            gVirtualX->ClosePixmap();
            glXDestroyGLXPixmap(fDpy, currDev.fGLXPixmap);
         }
      }
   }
}

//______________________________________________________________________________
TX11GLManager::TX11GLManager() : fPimpl(new TX11GLImpl)
{
   gGLManager = this;
   gROOT->GetListOfSpecials()->Add(this);
}

//______________________________________________________________________________
TX11GLManager::~TX11GLManager()
{
   delete fPimpl;
}

//______________________________________________________________________________
Int_t TX11GLManager::InitGLWindow(Window_t winID, Bool_t isOffScreen)
{
   //Try to find correct visual
   XVisualInfo *visInfo = glXChooseVisual(
                                          fPimpl->fDpy, DefaultScreen(fPimpl->fDpy), 
                                          const_cast<Int_t *>(isOffScreen ? snglBuff : dblBuff)
                                         );

   if (!visInfo) {
      Error("InitWindow", "No good visual found!\n");
      return -1;
   }

   Int_t  xVal = 0, yVal = 0;
   UInt_t wVal = 0, hVal = 0, border = 0, d = 0;
   Window root = 0;
   XGetGeometry(fPimpl->fDpy, winID, &root, &xVal, &yVal, &wVal, &hVal, &border, &d);

   XSetWindowAttributes attr(dummyAttr);
   attr.colormap = XCreateColormap(fPimpl->fDpy, root, visInfo->visual, AllocNone);
   attr.event_mask = NoEventMask;
   attr.backing_store = Always;
   attr.bit_gravity = NorthWestGravity;

   ULong_t mask = CWBackPixel | CWBorderPixel | CWColormap | CWEventMask | CWBackingStore | CWBitGravity;
   //Create window with specific visual
   Window glWin = XCreateWindow(
                                fPimpl->fDpy, winID, 
                                xVal, yVal, wVal, hVal,
                                0, visInfo->depth, InputOutput,
                                visInfo->visual, mask, &attr
                               );
                               
   XMapWindow(fPimpl->fDpy, glWin);
   //register window for gVirtualX
   Int_t x11Ind = gVirtualX->AddWindow(glWin,  wVal, hVal);
   
   //register this window for gl manager
   fPimpl->fGLWindows[x11Ind] = visInfo;
   
   return x11Ind;
}

//______________________________________________________________________________
Int_t TX11GLManager::CreateGLContext(Int_t winInd)
{
   //context creation requires Display * and XVisualInfo (was saved for such winInd)
   GLXContext glxCtx = glXCreateContext(fPimpl->fDpy, fPimpl->fGLWindows[winInd], None, True);
   
   if (!glxCtx) {
      Error("CreateContext", "glXCreateContext failed\n");
      return -1;
   }

   //register new context now
   if (PaintDevice *dev = fPimpl->fNextFreeDevice) {
      Int_t ind = dev->fWindowIndex;
      dev->fWindowIndex = winInd;
      dev->fGLXContext = glxCtx;
      fPimpl->fNextFreeDevice = fPimpl->fNextFreeDevice->fNextFreeDevice;
      
      return ind;
   } else {
      GLXCtxGuard glxCtxGuard(fPimpl->fDpy, glxCtx);
      PaintDevice newDev(dummyDevice);

      newDev.fWindowIndex = winInd;
      newDev.fPixmapIndex = -1;//on-screen rendering device
      newDev.fGLXContext = glxCtx;
   
      fPimpl->fPaintDevices.push_back(newDev);
      glxCtxGuard.Stop();
      
      return Int_t(fPimpl->fPaintDevices.size()) - 1;      
   }
}

//______________________________________________________________________________
Bool_t TX11GLManager::MakeCurrent(Int_t devInd)
{
   //Make gl context current
   PaintDevice &currDev = fPimpl->fPaintDevices[devInd];
   
   if (currDev.fPixmapIndex != -1) {
      //off-screen rendering into pixmap
      return glXMakeCurrent(fPimpl->fDpy, currDev.fGLXPixmap, currDev.fGLXContext);
   } else {
      Window winID = gVirtualX->GetWindowID(currDev.fWindowIndex);
      return glXMakeCurrent(fPimpl->fDpy, winID, currDev.fGLXContext);
   }
}

//______________________________________________________________________________
void TX11GLManager::Flush(Int_t devInd, Int_t, Int_t)
{
   //swaps buffers for window or copy pixmap
   PaintDevice &currDev = fPimpl->fPaintDevices[devInd];  
   Window winID = gVirtualX->GetWindowID(currDev.fWindowIndex);
   
   if (currDev.fPixmapIndex != -1) {
      if (!currDev.fDirect) return;
      
      GC gc = XCreateGC(fPimpl->fDpy, winID, 0, 0);

      if (!gc) {
         Error("Flush", "XCreateGC failed\n");
         currDev.fDirect = kFALSE;
         return;
      }

      XCopyArea(fPimpl->fDpy, currDev.fX11Pixmap, winID, gc, 0, 0, currDev.fCurrW,
                currDev.fCurrH, currDev.fX, currDev.fY);
      XFreeGC(fPimpl->fDpy, gc); 
   } else {
      glXSwapBuffers(fPimpl->fDpy, winID);
   }
}

//______________________________________________________________________________
Bool_t TX11GLManager::CreateGLPixmap(Int_t winInd, Int_t x, Int_t y, UInt_t w, UInt_t h, Int_t prevInd)
{
   //creates new x11 pixmap, gl pixmap, gl context
   Pixmap x11Pix = XCreatePixmap(fPimpl->fDpy, gVirtualX->GetWindowID(winInd), w, 
                                 h, fPimpl->fGLWindows[winInd]->depth);
   
   if (!x11Pix) {
      Error("CreateGLPixmap", "XCreatePixmap failed\n");
      return kFALSE;
   }

   X11PixGuard x11PixGuard(fPimpl->fDpy, x11Pix);
   GLXPixmap glxPix = glXCreateGLXPixmap(fPimpl->fDpy, fPimpl->fGLWindows[winInd], x11Pix);
   
   if (!glxPix) {
      Error("CreateGLPixmap", "glXCreateGLXPixmap failed\n");
      return kFALSE;
   }
   
   GLXPixGuard glxPixGuard(fPimpl->fDpy, glxPix);
   
   if (prevInd == -1 || !fPimpl->fPaintDevices[prevInd].fGLXContext) {
      GLXContext glxCtx = glXCreateContext(fPimpl->fDpy, fPimpl->fGLWindows[winInd],
                                           None, False);

      if (!glxCtx) {
         Error("CreateGLPixmap", "glXCreateContext failed\n");
         return kFALSE;
      }

      GLXCtxGuard glxCtxGuard(fPimpl->fDpy, glxCtx);
      PaintDevice newDev = {winInd, gVirtualX->AddPixmap(x11Pix, w, h), x11Pix,
                            glxPix, w, h, w, h, x, y, glxCtx, kFALSE, 0};

      if (prevInd == -1) {
         fPimpl->fPaintDevices.push_back(newDev);
      } else {
         newDev.fNextFreeDevice = fPimpl->fPaintDevices[prevInd].fNextFreeDevice;
         fPimpl->fPaintDevices[prevInd] = newDev;
      }
         
      glxCtxGuard.Stop();
   } else {
      PaintDevice &dev = fPimpl->fPaintDevices[prevInd];
      
      XFreePixmap(fPimpl->fDpy, dev.fX11Pixmap);
      gVirtualX->AddPixmap(x11Pix, w, h, dev.fPixmapIndex);
      dev.fX11Pixmap = x11Pix;
      dev.fGLXPixmap = glxPix;
      dev.fCurrW = w, dev.fRealW = w;
      dev.fCurrH = h, dev.fRealH = h;
      dev.fX = x, dev.fY = y;
   }
   
   glxPixGuard.Stop();
   x11PixGuard.Stop();
   
   return kTRUE;
}

//______________________________________________________________________________
Int_t TX11GLManager::OpenGLPixmap(Int_t winInd, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   //create new X11 pixmap and GLX pixmap and gl context for
   //off-screen drawing - new device
   if (PaintDevice *dev = fPimpl->fNextFreeDevice) {
      //reuse existing place in fPaintDevices
      Int_t prevInd = dev->fWindowIndex; //obscure usage of fWindowIndex
      
      if (CreateGLPixmap(winInd, x, y, w, h, prevInd)) {
         fPimpl->fNextFreeDevice = fPimpl->fNextFreeDevice->fNextFreeDevice;

         return prevInd;
      }
   } else if (CreateGLPixmap(winInd, x, y, w, h)) {
      return Int_t(fPimpl->fPaintDevices.size()) - 1;
   }

   return -1;
}

//______________________________________________________________________________
void TX11GLManager::ResizeGLPixmap(Int_t pixInd, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   PaintDevice &dev = fPimpl->fPaintDevices[pixInd];
   
   if (w > dev.fRealW || h > dev.fRealH) {
      //destroy old pixmap with such index and create
      //new in place
      CreateGLPixmap(dev.fWindowIndex, x, y, w, h, pixInd);
   } else {
      //simply change size-description
      dev.fX = x;
      dev.fY = y;
      dev.fCurrW = w;
      dev.fCurrH = h;
      gVirtualX->AddPixmap(0, w, h, dev.fPixmapIndex);
   }
}

//______________________________________________________________________________
void TX11GLManager::SelectGLPixmap(Int_t /*pixInd*/)
{
   //selects off-screen device to make it
   //accessible by gVirtualX
   //gVirtualX->SelectWindow(fPimpl->fPaintDevices[pixInd].fPixmapIndex);
}

//______________________________________________________________________________
void TX11GLManager::MarkForDirectCopy(Int_t pixInd, Bool_t dir)
{
   if (fPimpl->fPaintDevices[pixInd].fPixmapIndex != -1)
      fPimpl->fPaintDevices[pixInd].fDirect = dir;
   //Part of
   //selection-rotation support for TPad/TCanvas
}

//______________________________________________________________________________
void TX11GLManager::DeletePaintDevice(Int_t devInd)
{
   PaintDevice &currDev = fPimpl->fPaintDevices[devInd];
   //free gl context
   glXDestroyContext(fPimpl->fDpy, currDev.fGLXContext);
   currDev.fGLXContext = 0;
   //if pixmap - destroy
   if (currDev.fPixmapIndex != -1) {
      gVirtualX->SelectWindow(currDev.fPixmapIndex);
      gVirtualX->ClosePixmap();
      glXDestroyGLXPixmap(fPimpl->fDpy, currDev.fGLXPixmap);
   }

   currDev.fNextFreeDevice = fPimpl->fNextFreeDevice;
   fPimpl->fNextFreeDevice = &currDev;
   currDev.fWindowIndex = devInd;
}

Int_t TX11GLManager::GetVirtualXInd(Int_t glPix)
{
   return fPimpl->fPaintDevices[glPix].fPixmapIndex;
}

void TX11GLManager::ExtractViewport(Int_t pixId, Int_t *viewport)
{
   PaintDevice &dev = fPimpl->fPaintDevices[pixId];
   
   if (dev.fPixmapIndex != -1) {
      viewport[0] = 0;
      viewport[1] = dev.fRealH - dev.fCurrH;//dev.fY;
      viewport[2] = dev.fCurrW;
      viewport[3] = dev.fCurrH;
   }
}

//______________________________________________________________________________
void TX11GLManager::DrawViewer(TVirtualViewer3D *vv)
{
   vv->DrawViewer();
}

//______________________________________________________________________________
TObject *TX11GLManager::Select(TVirtualViewer3D *vv, Int_t x, Int_t y)
{
   return vv->SelectObject(x, y);
}

//______________________________________________________________________________
void TX11GLManager::PaintSingleObject(TVirtualGLPainter *p)
{
   p->Paint();
}
