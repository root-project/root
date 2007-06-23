// @(#)root/gl:$Name:  $:$Id: TGLWidget.cxx,v 1.6 2007/06/18 14:24:57 brun Exp $
// Author:  Timur Pocheptsov, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdexcept>
#include <vector>

#ifndef WIN32
#include <GL/glx.h>
#endif

#include "TVirtualX.h"
#include "TGClient.h"
#include "TError.h"
#include "TROOT.h"

#include "TGLWidget.h"
#include "TGLIncludes.h"
#include "TGLUtil.h"

ClassImp(TGLWidgetContainer)

//______________________________________________________________________________
TGLWidgetContainer::TGLWidgetContainer(TGLWidget *owner, Window_t id, const TGWindow *parent)
                     : TGCompositeFrame(gClient, id, parent),
                       fOwner(owner)
{
   //Constructor.
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleButton(Event_t *ev)
{
   //Delegate call to the owner.
   return fOwner->HandleButton(ev);
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleDoubleClick(Event_t *ev)
{
   //Delegate call to the owner.
   return fOwner->HandleDoubleClick(ev);
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleConfigureNotify(Event_t *ev)
{
   //Delegate call to the owner.
   return fOwner->HandleConfigureNotify(ev);
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleKey(Event_t *ev)
{
   //Delegate call to the owner.
   return fOwner->HandleKey(ev);
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleMotion(Event_t *ev)
{
   //Delegate call to the owner.
   return fOwner->HandleMotion(ev);
}

/*
//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleExpose(Event_t *ev)
{
   //Delegate call to the owner.
   return fOwner->HandleExpose(ev);
}
*/

//______________________________________________________________________________
void TGLWidgetContainer::DoRedraw()
{
   //Delegate call to the owner.
   return fOwner->Repaint();
}


ClassImp(TGLWidget)

//______________________________________________________________________________
TGLWidget::TGLWidget(const TGWindow &p, Bool_t select,
                     const TGLPaintDevice *shareDevice,
                     UInt_t w, UInt_t h, UInt_t opt, Pixel_t back)
              : TGCanvas(&p, w, h, opt, back),
                fWindowIndex(-1),
                fFromCtor(kTRUE)
{
   //Creates widget with default pixel format.
   CreateWidget(shareDevice);

   if (select) {
      gVirtualX->GrabButton(
                            fContainer->GetId(), kAnyButton, kAnyModifier,
                            kButtonPressMask | kButtonReleaseMask, kNone, kNone
                           );
      gVirtualX->SelectInput(
                             fContainer->GetId(), kKeyPressMask | kExposureMask | kPointerMotionMask
                             | kStructureNotifyMask | kFocusChangeMask
                            );
      gVirtualX->SetInputFocus(fContainer->GetId());
   }

   fFromCtor = kFALSE;
}

//______________________________________________________________________________
TGLWidget::TGLWidget(const TGLFormat &format, const TGWindow &p, Bool_t select,
                     const TGLPaintDevice *shareDevice,
                     UInt_t w, UInt_t h, UInt_t opt, Pixel_t back)
              : TGCanvas(&p, w, h, opt, back),
                fWindowIndex(-1),
                fGLFormat(format),
                fFromCtor(kTRUE)
{
   //Create widget with the requested pixel format.
   CreateWidget(shareDevice);

   if (select) {
      gVirtualX->GrabButton(
                            fContainer->GetId(), kAnyButton, kAnyModifier,
                            kButtonPressMask | kButtonReleaseMask, kNone, kNone
                           );
      gVirtualX->SelectInput(
                             fContainer->GetId(), kKeyPressMask | kExposureMask | kPointerMotionMask
                             | kStructureNotifyMask | kFocusChangeMask
                            );
      gVirtualX->SetInputFocus(fContainer->GetId());
   }

   fFromCtor = kFALSE;
}

//______________________________________________________________________________
TGLWidget::TGLWidget(const TGWindow &p, Bool_t select,
                     UInt_t w, UInt_t h, UInt_t opt, Pixel_t back)
              : TGCanvas(&p, w, h, opt, back),
                fWindowIndex(-1),
                fFromCtor(kTRUE)
{
   //Creates widget with default pixel format, default shareList.
   CreateWidget();

   if (select) {
      gVirtualX->GrabButton(
                            fContainer->GetId(), kAnyButton, kAnyModifier,
                            kButtonPressMask | kButtonReleaseMask, kNone, kNone
                           );
      gVirtualX->SelectInput(
                             fContainer->GetId(), kKeyPressMask | kExposureMask | kPointerMotionMask
                             | kStructureNotifyMask | kFocusChangeMask
                            );
      gVirtualX->SetInputFocus(fContainer->GetId());
   }

   fFromCtor = kFALSE;
}

//______________________________________________________________________________
TGLWidget::TGLWidget(const TGLFormat &format, const TGWindow &p, Bool_t select,
                     UInt_t w, UInt_t h, UInt_t opt, Pixel_t back)
              : TGCanvas(&p, w, h, opt, back),
                fWindowIndex(-1),
                fGLFormat(format),
                fFromCtor(kTRUE)
{
   //Create widget with the requested pixel format, default shareList.
   CreateWidget();

   if (select) {
      gVirtualX->GrabButton(
                            fContainer->GetId(), kAnyButton, kAnyModifier,
                            kButtonPressMask | kButtonReleaseMask, kNone, kNone
                           );
      gVirtualX->SelectInput(
                             fContainer->GetId(), kKeyPressMask | kExposureMask | kPointerMotionMask
                             | kStructureNotifyMask | kFocusChangeMask
                            );
      gVirtualX->SetInputFocus(fContainer->GetId());
   }

   fFromCtor = kFALSE;
}

//______________________________________________________________________________
TGLWidget::~TGLWidget()
{
   //Destructor. Deletes window ???? and XVisualInfo
   gVirtualX->SelectWindow(fWindowIndex);
   gVirtualX->CloseWindow();
#ifndef WIN32
   XFree(fInnerData.second);//free XVisualInfo
#endif
   if (fValidContexts.size() > 1u) {
      Warning("~TGLWidget", "There are some gl-contexts connected to this gl device"
                            "which have longer lifetime than lifetime of gl-device");
   }

   std::set<TGLContext *>::iterator it = fValidContexts.begin();
   for (; it != fValidContexts.end(); ++it) {
      (*it)->Release();
   }
}

//______________________________________________________________________________
void TGLWidget::InitGL()
{
   //Call glEnable(... in overrider of InitGL.
}

//______________________________________________________________________________
void TGLWidget::PaintGL()
{
   //Do actual drawing in overrider of PaintGL.
}

//______________________________________________________________________________
Bool_t TGLWidget::MakeCurrent()
{
   //Make the gl-context current.
   return fGLContext->MakeCurrent();
}

//______________________________________________________________________________
void TGLWidget::SwapBuffers()
{
   //Swap buffers.
   fGLContext->SwapBuffers();
}

//______________________________________________________________________________
const TGLContext *TGLWidget::GetContext()const
{
   //Get gl context.
   return fGLContext.get();
}

//______________________________________________________________________________
Bool_t TGLWidget::HandleButton(Event_t *ev)
{
   //Signal. Under Win32 I have to switch between
   //threads to let direct usage of gl code.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%x)->HandleButton((Event_t *)0x%x)", this, ev));
      return kTRUE;
   }

   Emit("HandleButton(Event_t*)", (Long_t)ev);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLWidget::HandleDoubleClick(Event_t *ev)
{
   //Signal. Under Win32 I have to switch between
   //threads to let direct usage of gl code.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%x)->HandleDoubleClick((Event_t *)0x%x)", this, ev));
      return kTRUE;
   }

   Emit("HandleDoubleClick(Event_t*)", (Long_t)ev);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLWidget::HandleConfigureNotify(Event_t *ev)
{
   //Signal. Under Win32 I have to switch between
   //threads to let direct usage of gl code.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%x)->HandleConfigureNotify((Event_t *)0x%x)", this, ev));
      return kTRUE;
   }

   Emit("HandleConfigureNotify(Event_t*)", (Long_t)ev);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLWidget::HandleKey(Event_t *ev)
{
   //Signal. Under Win32 I have to switch between
   //threads to let direct usage of gl code.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%x)->HandleKey((Event_t *)0x%x)", this, ev));
      return kTRUE;
   }

   Emit("HandleKey(Event_t*)", (Long_t)ev);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLWidget::HandleMotion(Event_t *ev)
{
   //Signal. Under Win32 I have to switch between
   //threads to let direct usage of gl code.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%x)->HandleMotion((Event_t *)0x%x)", this, ev));
      return kTRUE;
   }

   Emit("HandleMotion(Event_t*)", (Long_t)ev);

   return kTRUE;
}

/*
//______________________________________________________________________________
Bool_t TGLWidget::HandleExpose(Event_t *ev)
{
   //Signal. Under Win32 I have to switch between
   //threads to let direct usage of gl code.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%x)->HandleExpose((Event_t *)0x%x)", this, ev));
      return kTRUE;
   }

   MakeCurrent();
   InitGL();
   PaintGL();
   SwapBuffers();

   Emit("HandleExpose(Event_t*)", (Long_t)ev);

   return kTRUE;
}
*/

//______________________________________________________________________________
Int_t TGLWidget::GetWindowIndex()const
{
   //Index of window, registered by gVirtualX.
   return fWindowIndex;
}

//______________________________________________________________________________
const TGLFormat *TGLWidget::GetPixelFormat()const
{
   //Pixel format.
   return &fGLFormat;
}

//______________________________________________________________________________
Int_t TGLWidget::GetContId()const
{
   //Id of container.
   return fContainer->GetId();
}

//______________________________________________________________________________
std::pair<void *, void *> TGLWidget::GetInnerData()const
{
   //Dpy*, XVisualInfo *
   return fInnerData;
}

#ifdef WIN32

namespace {

   struct LayoutCompatible_t {
      void          *fDummy0;
      void          *fDummy1;
      HWND          *fPHwnd;
      unsigned char  fDummy2;
      unsigned       fDummy3;
      unsigned short fDummy4;
      unsigned short fDummy5;
      void          *fDummy6;
      unsigned       fDummy7:2;
   };

   void fill_pfd(PIXELFORMATDESCRIPTOR *pfd, const TGLFormat &request)
   {
      pfd->nSize = sizeof(PIXELFORMATDESCRIPTOR);
      pfd->nVersion = 1;
      pfd->dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
      if (request.IsDoubleBuffered())
         pfd->dwFlags |= PFD_DOUBLEBUFFER;
      pfd->iPixelType = PFD_TYPE_RGBA;
      pfd->cColorBits = 24;
      if (UInt_t acc = request.GetAccumSize())
         pfd->cAccumBits = acc;
      if (UInt_t depth = request.GetDepthSize())
         pfd->cDepthBits = depth;
      if (UInt_t stencil = request.GetStencilSize())
         pfd->cStencilBits = stencil;
   }

   void check_pixel_format(Int_t pixIndex, HDC hDC, TGLFormat &request)
   {
      PIXELFORMATDESCRIPTOR pfd = {};

      if (!DescribePixelFormat(hDC, pixIndex, sizeof pfd, &pfd)) {
         Warning("TGLContext::SetContext", "DescribePixelFormat failed");
         return;
      }

      if (pfd.cAccumBits)
         request.SetAccumSize(pfd.cAccumBits);

      if (pfd.cDepthBits)
         request.SetDepthSize(pfd.cDepthBits);

      if (pfd.cStencilBits)
         request.SetStencilSize(pfd.cStencilBits);
   }

}

//______________________________________________________________________________
void TGLWidget::CreateWidget(const TGLPaintDevice *shareDevice)
{
   //CreateWidget. This function called only during construction,
   //I've extracted this code from ctors to make WIN32/X11
   //separation simpler and because of gInterpreter usage.
   //new, TGLContext can throw
   //std::bad_alloc and std::runtime_error. Before try block, the only
   //resource allocated is pointed by fWindowIndex (InitWindow cannot throw).
   //In try block (and after successful constraction)
   //resources are controlled by std::auto_ptrs and dtor.
   fWindowIndex = gVirtualX->InitWindow((ULong_t)GetViewPort()->GetId());

   try {
      if (!gVirtualX->IsCmdThread())
         gROOT->ProcessLineFast(Form("((TGLWidget *)0x%x)->SetFormat()", this));
      else
         SetFormat();
      fGLContext.reset(new TGLContext(this, shareDevice ? shareDevice->GetContext() : 0));
      fContainer.reset(new TGLWidgetContainer(this, gVirtualX->GetWindowID(fWindowIndex), GetViewPort()));
      SetContainer(fContainer.get());
   } catch (std::exception &) {
      gVirtualX->SelectWindow(fWindowIndex);
      gVirtualX->CloseWindow();
      throw;
   }
}

//______________________________________________________________________________
void TGLWidget::CreateWidget()
{
   //CreateWidget. Copy of the above for default shareList.
   fWindowIndex = gVirtualX->InitWindow((ULong_t)GetViewPort()->GetId());

   try {
      if (!gVirtualX->IsCmdThread())
         gROOT->ProcessLineFast(Form("((TGLWidget *)0x%x)->SetFormat()", this));
      else
         SetFormat();
      fGLContext.reset(new TGLContext(this));
      fContainer.reset(new TGLWidgetContainer(this, gVirtualX->GetWindowID(fWindowIndex), GetViewPort()));
      SetContainer(fContainer.get());
   } catch (std::exception &) {
      gVirtualX->SelectWindow(fWindowIndex);
      gVirtualX->CloseWindow();
      throw;
   }
}

//______________________________________________________________________________
void TGLWidget::SetFormat()
{
   //Set pixel format.
   //Resource - hDC, owned and freed by guard object.
   if (!fFromCtor) {
      Error("TGLWidget::SetFormat", "Sorry, you should not call this function");
      return;
   }

   LayoutCompatible_t *trick =
      reinterpret_cast<LayoutCompatible_t *>(gVirtualX->GetWindowID(GetWindowIndex()));
   HWND hWND = *trick->fPHwnd;
   HDC  hDC  = GetWindowDC(hWND);

   if (!hDC) {
      Error("TGLWidget::SetFormat", "GetWindowDC failed");
      throw std::runtime_error("GetWindowDC failed");
   }

   const Rgl::TGuardBase &dcGuard = Rgl::make_guard(ReleaseDC, hWND, hDC);
   PIXELFORMATDESCRIPTOR pfd = {};
   fill_pfd(&pfd, fGLFormat);

   if (const Int_t pixIndex = ChoosePixelFormat(hDC, &pfd)) {
      check_pixel_format(pixIndex, hDC, fGLFormat);

      if (!SetPixelFormat(hDC, pixIndex, &pfd)) {
         Error("TGLWidget::SetFormat", "SetPixelFormat failed");
         throw std::runtime_error("SetPixelFormat failed");
      }
   } else {
      Error("TGLWidget::SetFormat", "ChoosePixelFormat failed");
      throw std::runtime_error("ChoosePixelFormat failed");
   }
}

#else

namespace {

   void fill_format(std::vector<Int_t> &format, const TGLFormat &request)
   {
      format.push_back(GLX_RGBA);
      format.push_back(GLX_RED_SIZE);
      format.push_back(1);
      format.push_back(GLX_GREEN_SIZE);
      format.push_back(1);
      format.push_back(GLX_BLUE_SIZE);
      format.push_back(1);

      if (request.IsDoubleBuffered())
         format.push_back(GLX_DOUBLEBUFFER);

      if (request.HasDepth()) {
         format.push_back(GLX_DEPTH_SIZE);
         format.push_back(request.GetDepthSize());
      }

      if (request.HasStencil()) {
         format.push_back(GLX_STENCIL_SIZE);
         format.push_back(request.GetStencilSize());
      }

      if (request.HasAccumBuffer()) {
         format.push_back(GLX_ACCUM_RED_SIZE);
         format.push_back(1);
         format.push_back(GLX_ACCUM_GREEN_SIZE);
         format.push_back(1);
         format.push_back(GLX_ACCUM_BLUE_SIZE);
         format.push_back(1);
      }

      format.push_back(None);
   }

   XSetWindowAttributes dummyAttr;

}

//______________________________________________________________________________
void TGLWidget::CreateWidget(const TGLPaintDevice *shareDevice)
{
   //CreateWidget - X11 version.
   //This function called only during construction,
   //This function, new, TGLContext ctor can throw
   //std::bad_alloc and std::runtime_error.
   //This version is bad - I do not check the results of
   //X11 calls.
   std::vector<Int_t> format;
   fill_format(format, fGLFormat);

   Window_t winID = GetViewPort()->GetId();
   Display *dpy = reinterpret_cast<Display *>(gVirtualX->GetDisplay());
   XVisualInfo *visInfo = glXChooseVisual(dpy, DefaultScreen(dpy), &format[0]);

   if (!visInfo) {
      Error("CreateGLContainer", "No good visual found!");
      throw std::runtime_error("No good visual found!");
   }

   Int_t  x = 0, y = 0;
   UInt_t w = 0, h = 0, b = 0, d = 0;
   Window root = 0;
   XGetGeometry(dpy, winID, &root, &x, &y, &w, &h, &b, &d);

   XSetWindowAttributes attr(dummyAttr);
   attr.colormap = XCreateColormap(dpy, root, visInfo->visual, AllocNone); // Can fail?
   attr.event_mask = NoEventMask;
   attr.backing_store = Always;
   attr.bit_gravity = NorthWestGravity;

   ULong_t mask = CWBackPixel | CWBorderPixel | CWColormap | CWEventMask | CWBackingStore | CWBitGravity;
   Window glWin = XCreateWindow(dpy, winID, x, y, w, h, 0, visInfo->depth,
                                InputOutput, visInfo->visual, mask, &attr);
   XMapWindow(dpy, glWin);
   // Register window for gVirtualX.
   fWindowIndex = gVirtualX->AddWindow(glWin, w, h);
   fInnerData.first  = dpy;
   fInnerData.second = visInfo;

   try {
      fGLContext.reset(new TGLContext(this, shareDevice ? shareDevice->GetContext() : 0));
      fContainer.reset(new TGLWidgetContainer(this, gVirtualX->GetWindowID(fWindowIndex), GetViewPort()));
      SetContainer(fContainer.get());
   } catch (const std::exception &) {
      gVirtualX->SelectWindow(fWindowIndex);
      gVirtualX->CloseWindow();
      XFree(fInnerData.second);
      throw;
   }
}

//______________________________________________________________________________
void TGLWidget::CreateWidget()
{
   //CreateWidget - X11 version. Copy of above for default context.
   std::vector<Int_t> format;
   fill_format(format, fGLFormat);

   Window_t winID = GetViewPort()->GetId();
   Display *dpy = reinterpret_cast<Display *>(gVirtualX->GetDisplay());
   XVisualInfo *visInfo = glXChooseVisual(dpy, DefaultScreen(dpy), &format[0]);

   if (!visInfo) {
      Error("CreateGLContainer", "No good visual found!");
      throw std::runtime_error("No good visual found!");
   }

   Int_t  x = 0, y = 0;
   UInt_t w = 0, h = 0, b = 0, d = 0;
   Window root = 0;
   XGetGeometry(dpy, winID, &root, &x, &y, &w, &h, &b, &d);

   XSetWindowAttributes attr(dummyAttr);
   attr.colormap = XCreateColormap(dpy, root, visInfo->visual, AllocNone); // Can fail?
   attr.event_mask = NoEventMask;
   attr.backing_store = Always;
   attr.bit_gravity = NorthWestGravity;

   ULong_t mask = CWBackPixel | CWBorderPixel | CWColormap | CWEventMask | CWBackingStore | CWBitGravity;
   Window glWin = XCreateWindow(dpy, winID, x, y, w, h, 0, visInfo->depth,
                                InputOutput, visInfo->visual, mask, &attr);
   XMapWindow(dpy, glWin);
   // Register window for gVirtualX.
   fWindowIndex = gVirtualX->AddWindow(glWin, w, h);
   fInnerData.first  = dpy;
   fInnerData.second = visInfo;

   try {
      fGLContext.reset(new TGLContext(this));
      fContainer.reset(new TGLWidgetContainer(this, gVirtualX->GetWindowID(fWindowIndex), GetViewPort()));
      SetContainer(fContainer.get());
   } catch (const std::exception &) {
      gVirtualX->SelectWindow(fWindowIndex);
      gVirtualX->CloseWindow();
      XFree(fInnerData.second);
      throw;
   }
}

//______________________________________________________________________________
void TGLWidget::SetFormat()
{
   //Set pixel format.
   //Empty version for X11.
}

#endif

//______________________________________________________________________________
void TGLWidget::Repaint()
{
   //Forse re-drawing of gl-widget.
/*   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%x)->Repaint()", this));
      return;
   }

   MakeCurrent();
   InitGL();
   PaintGL();
   SwapBuffers();*/
   
   Emit("Repaint()");
}

//______________________________________________________________________________
void TGLWidget::AddContext(TGLContext *ctx)
{
   //Register gl-context created for this window.
   fValidContexts.insert(ctx);
}

//______________________________________________________________________________
void TGLWidget::RemoveContext(TGLContext *ctx)
{
   //Remove context (no real deletion, done by TGLContex dtor).
   std::set<TGLContext *>::iterator it = fValidContexts.find(ctx);
   if (it != fValidContexts.end())
      fValidContexts.erase(it);
}

//______________________________________________________________________________
void TGLWidget::ExtractViewport(Int_t *vp)const
{
   //For camera.
   vp[0] = 0;
   vp[1] = 0;
   vp[2] = GetWidth();
   vp[3] = GetHeight();
}
