// @(#)root/gx11:$Name:  $:$Id: TX11GL.cxx,v 1.6 2005/08/17 09:10:44 brun Exp $
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

   Window GLWin = XCreateWindow(fDpy, wind, xval, yval, wval, hval,
                                0, fVisInfo->depth, InputOutput,
                                fVisInfo->visual, mask, &attr);
   XMapWindow(fDpy, GLWin);
   return (Window_t)GLWin;
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

namespace {

   struct PaintDevice {
      //these are numbers returned by gVirtualX->AddWindow and gVirtualX->AddPixmap
      //need both, I can have a pixmap, which is always created for certain window
      Int_t       fWinIndex;
      Int_t       fPixmapIndex;
      //Pixmap info, not used for double buffered windows
      Pixmap      fX11Pixmap; //required by TX11GLManager::DirectCopy and explicit destruction in resize
      GLXPixmap   fGLXPixmap;
      //Pixmap parameters
      UInt_t      fRealW;//to check, if we really need new pixmap during resize
      UInt_t      fRealH;
      UInt_t      fCurrW;//used by DirectCopy
      UInt_t      fCurrH;
      //Where to XCopyArea pixmap
      Int_t       fX;//used by DirectCopy
      Int_t       fY;
      //ctx, used for off-screen and double-buffered gl painting
      GLXContext  fGLXContext;
      Bool_t      fDirect;
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

}


class TX11GLManager::TX11GLPimpl {
public:
   TX11GLPimpl();
   ~TX11GLPimpl();

   WinTable_t     fGLWindows;
   DeviceTable_t  fPaintDevices;
   Display        *fDpy;
   
private:
   TX11GLPimpl(const TX11GLPimpl &);
   TX11GLPimpl &operator = (const TX11GLPimpl &);
};


ClassImp(TX11GLManager)

//______________________________________________________________________________
TX11GLManager::TX11GLPimpl::TX11GLPimpl() : fDpy(0)
{
   fDpy = reinterpret_cast<Display *>(gVirtualX->GetDisplay());
}

//______________________________________________________________________________
TX11GLManager::TX11GLPimpl::~TX11GLPimpl()
{
   //destroys only gl contexts,
   //pixmaps and windows must be
   //closed through gVirtualX
   for (size_type i = 0,  e = fPaintDevices.size(); i < e; ++i) {
      if (GLXContext ctx = fPaintDevices[i].fGLXContext)
         glXDestroyContext(fDpy, ctx);
   }
}

//______________________________________________________________________________
TX11GLManager::TX11GLManager() : fPimpl(new TX11GLPimpl)
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
Int_t TX11GLManager::InitGLWindow(Window_t winId, Bool_t isOffScreen)
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
   XGetGeometry(fPimpl->fDpy, winId, &root, &xVal, &yVal, &wVal, &hVal, &border, &d);

   XSetWindowAttributes attr(dummyAttr);
   attr.colormap = XCreateColormap(fPimpl->fDpy, root, visInfo->visual, AllocNone);
   attr.event_mask = NoEventMask;
   attr.backing_store = Always;
   attr.bit_gravity = NorthWestGravity;

   ULong_t mask = CWBackPixel | CWBorderPixel | CWColormap | CWEventMask | CWBackingStore | CWBitGravity;
   //Create window with specific visual
   Window glWin = XCreateWindow(
                                fPimpl->fDpy, winId, 
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
   WinTable_t::iterator win = fPimpl->fGLWindows.find(winInd);
   
   if (win == fPimpl->fGLWindows.end()) {
      Error("CreateContext", "No window with an index %d exists\n", winInd);
      
      return -1;
   }
   
   GLXContext glCtx = glXCreateContext(fPimpl->fDpy, win->second, None, True);
   
   if (!glCtx) {
      Error("CreateContext", "glXCreateContext failed\n");
   
      return -1;
   }

   //register new context now
   PaintDevice newDev(dummyDevice);
   newDev.fWinIndex = winInd;
   newDev.fPixmapIndex = -1;//on-screen rendering device
   newDev.fGLXContext = glCtx;
   
   fPimpl->fPaintDevices.push_back(newDev); //can throw

   return fPimpl->fPaintDevices.size() - 1;
}

//______________________________________________________________________________
Bool_t TX11GLManager::MakeCurrent(Int_t deviceInd)
{
   if (deviceInd < 0 || deviceInd >= Int_t(fPimpl->fPaintDevices.size())) {
      Error("MakeCurrent", "Bad device index %d\n", deviceInd);
      
      return kFALSE;
   }
   
   PaintDevice &currDev = fPimpl->fPaintDevices[deviceInd];
   
   if (currDev.fPixmapIndex != -1) {
      //off-screen rendering into pixmap
      return glXMakeCurrent(fPimpl->fDpy, currDev.fGLXPixmap, currDev.fGLXContext);
   } else {
      Window winId = gVirtualX->GetWindowID(currDev.fWinIndex);
      return glXMakeCurrent(fPimpl->fDpy, winId, currDev.fGLXContext);
   }
}

//______________________________________________________________________________
void TX11GLManager::Flush(Int_t deviceInd, Int_t, Int_t)
{
   //swaps buffers for window or copy pixmap
   if (deviceInd < 0 || deviceInd >= Int_t(fPimpl->fPaintDevices.size())) {
      Error("Flush", "Bad device index %d\n", deviceInd);
      
      return;
   }
   
   PaintDevice &currDev = fPimpl->fPaintDevices[deviceInd];  
   Window winId = gVirtualX->GetWindowID(currDev.fWinIndex);
   
   if (currDev.fPixmapIndex != -1) {
      if (!currDev.fDirect) return;
      
      GC gc = XCreateGC(fPimpl->fDpy, winId, 0, 0);

      if (!gc) {
         Error("Flush", "Problem with context for direct copying creation\n");
      
         return;
      }

      XCopyArea(
                fPimpl->fDpy, currDev.fX11Pixmap, 
                winId, gc, 0, 0, currDev.fCurrW, currDev.fCurrH, currDev.fX, currDev.fY
               );

      XFreeGC(fPimpl->fDpy, gc); 
   } else {
      glXSwapBuffers(fPimpl->fDpy, winId);
   }
}

//______________________________________________________________________________
Bool_t TX11GLManager::CreateGLPixmap(Int_t winInd, UInt_t w, UInt_t h, Int_t preferInd)
{
   //creates new pixmap and gl-pixmap, saving info in new place
   //or in place of old pixmap (preferInd)
   Window wid = gVirtualX->GetWindowID(winInd);
   Pixmap x11Pix = XCreatePixmap(fPimpl->fDpy, wid, w, h, fPimpl->fGLWindows[winInd]->depth);
   
   if (!x11Pix) 
      return kFALSE;
   
   //specific Mesa function exists, but now I call this
   GLXPixmap glxPix = glXCreateGLXPixmap(fPimpl->fDpy, fPimpl->fGLWindows[winInd], x11Pix);
   
   if (!glxPix) {
      XFreePixmap(fPimpl->fDpy, x11Pix);

      return kFALSE;
   }

   if (preferInd != -1) {
      //use existing device (gl context), just reset pixmaps (and re-register it for gVirtualX)
      PaintDevice &dev = fPimpl->fPaintDevices[preferInd];
      //replace old pixmaps with new
      XFreePixmap(fPimpl->fDpy, dev.fX11Pixmap);
      dev.fX11Pixmap = x11Pix;
      glXDestroyGLXPixmap(fPimpl->fDpy, dev.fGLXPixmap);
      dev.fGLXPixmap = glxPix;
      
      dev.fCurrW = dev.fRealW = w;
      dev.fCurrH = dev.fRealH = h;
      gVirtualX->AddPixmap(x11Pix, w, h, dev.fPixmapIndex);//change sizes
   } else {
      //new device
      GLXContext glCtx = glXCreateContext(fPimpl->fDpy, fPimpl->fGLWindows[winInd], 0, False);
      
      if (!glCtx) {
         XFreePixmap(fPimpl->fDpy, x11Pix);
         
         return kFALSE;
      }

      Int_t pixInd = gVirtualX->AddPixmap(x11Pix, w, h);
      
      PaintDevice newDev = {
                            winInd, pixInd, x11Pix, glxPix, 
                            w, h, w, h, 0, 0, //instead of 0, 0 OpenGLPixmap or ResizeGLPixmap will specify real x and y
                            glCtx, kFALSE
                           };
      fPimpl->fPaintDevices.push_back(newDev);
   }
   
   return kTRUE;
}

//______________________________________________________________________________
Int_t TX11GLManager::OpenGLPixmap(Int_t winInd, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   //create new X11 pixmap and GLX pixmap and gl context for
   //off-screen drawing - new device
   if (fPimpl->fGLWindows.find(winInd) == fPimpl->fGLWindows.end()) {
      Error("OpenGLPixmap", "No window with an index %d exists\n", winInd);
      
      return -1;
   }
   
   if (CreateGLPixmap(winInd, w, h)) {
      fPimpl->fPaintDevices.back().fX = x;
      fPimpl->fPaintDevices.back().fY = y;
      return fPimpl->fPaintDevices.size() - 1;
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
      CreateGLPixmap(dev.fWinIndex, w, h, pixInd);
      dev.fX = x;
      dev.fY = y;
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
void TX11GLManager::SelectGLPixmap(Int_t pixInd)
{
   //selects off-screen device to make it
   //accessible by gVirtualX
   gVirtualX->SelectWindow(fPimpl->fPaintDevices[pixInd].fPixmapIndex);
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
void TX11GLManager::DeletePaintDevice(Int_t /*deviceInd*/)
{
   //these device should be removed or marked as free
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

