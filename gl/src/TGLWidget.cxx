// @(#)root/gl:$Id$
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

#include "TGLEventHandler.h"

/******************************************************************************/
//TGLWidgetContainer
/******************************************************************************/

//______________________________________________________________________________
//
// Auxiliary "widget container" class.
// Does not throw (base classe can throw?).
// Immutable - after constructed, fOwner is
// invariant, cannot change.
// Non-copyable.

ClassImp(TGLWidgetContainer)

//______________________________________________________________________________
TGLWidgetContainer::TGLWidgetContainer(TGLWidget *owner, Window_t id, const TGWindow *parent)
                     : TGCompositeFrame(gClient, id, parent),
                       fOwner(owner), fEventHandler(0)
{
   //Constructor.
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleCrossing(Event_t *ev)
{
   // Handle mouse crossing event.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidgetContainer *)0x%lx)->HandleCrossing((Event_t *)0x%lx)", this, ev));
      return kTRUE;
   }
   if ((ev->fType == kEnterNotify) &&
       (!gVirtualX->InheritsFrom("TGX11")) &&
       (gVirtualX->GetInputFocus() != GetId())) {
      gVirtualX->SetInputFocus(GetId());
   }
   if (fEventHandler)
      return fEventHandler->HandleCrossing(ev);
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleButton(Event_t *ev)
{
   //Delegate call to the owner.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidgetContainer *)0x%lx)->HandleButton((Event_t *)0x%lx)", this, ev));
      return kTRUE;
   }
   if (fEventHandler)
      return fEventHandler->HandleButton(ev);
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleDoubleClick(Event_t *ev)
{
   //Delegate call to the owner.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidgetContainer *)0x%lx)->HandleDoubleClick((Event_t *)0x%lx)", this, ev));
      return kTRUE;
   }
   if (fEventHandler)
      return fEventHandler->HandleDoubleClick(ev);
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleConfigureNotify(Event_t *ev)
{
   //Delegate call to the owner.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidgetContainer *)0x%lx)->HandleConfigureNotify((Event_t *)0x%lx)", this, ev));
      return kTRUE;
   }
   if (fEventHandler)
      return fEventHandler->HandleConfigureNotify(ev);
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleFocusChange(Event_t *ev)
{
   //Delegate call to the owner.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidgetContainer *)0x%lx)->HandleFocusChange((Event_t *)0x%lx)", this, ev));
      return kTRUE;
   }
   if (fEventHandler)
      return fEventHandler->HandleFocusChange(ev);
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleKey(Event_t *ev)
{
   //Delegate call to the owner.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidgetContainer *)0x%lx)->HandleKey((Event_t *)0x%lx)", this, ev));
      return kTRUE;
   }
   if (fEventHandler)
      return fEventHandler->HandleKey(ev);
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleMotion(Event_t *ev)
{
   //Delegate call to the owner.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidgetContainer *)0x%lx)->HandleMotion((Event_t *)0x%lx)", this, ev));
      return kTRUE;
   }
   if (fEventHandler)
      return fEventHandler->HandleMotion(ev);
   return kFALSE;
}

//______________________________________________________________________________
void TGLWidgetContainer::DoRedraw()
{
   //Delegate call to the owner.
//   if (!gVirtualX->IsCmdThread()) {
//      gROOT->ProcessLineFast(Form("((TGLWidgetContainer *)0x%lx)->DoRedraw()", this));
//      return;
//   }
   if (fEventHandler)
      return fEventHandler->Repaint();
}


/******************************************************************************/
// TGLWidget
/******************************************************************************/

//______________________________________________________________________________
//
// GL window with context. _Must_ _have_ a parent window
// (the 'parent' parameter of ctors). The current version inherits
// TGCanvas (I'm not sure about future versions), probably, in future
// multiple inheritance will be added - the second
// base class will be TGLPaintDevice or something like this.
//
// Usage:
// - Simply create TGLWidget as an embedded widget, and
//   connect your slots to signals you need: HandleExpose, HandleConfigureNotify, etc.
//   In your slots you can use gl API directly - under Win32 TGLWidget switches
//   between threads internally (look TGLPShapeObjEditor for such usage).
// - You can write your own class, derived from TGLWidget, with PaintGL and InitGL
//   overriden.
//
// Resources (and invariants):
// -fContainer (TGLWidgetContainer) - controlled by std::auto_ptr
// -fWindowIndex - controlled manually (see CreateWidget and dtor)
// -fGLContext - controlled by std::auto_ptr
// -visual info for X11 version, controlled manually (see CreateGLContainer and dtor)
//
// Exceptions:
// -can be thrown only during construction.
// -under win32 class does not throw itself (but some internal operations can throw)
// -under X11 can throw std::runtime_error (from CreateGLContext).
// -In case of exceptions resources will be freed.
//
// TGLWidget object is immutable as far as it was created.
//
// Boolean parameter defines, if you want to grab user's input or not.
// By default you want, but for example when not - see TGLPShapeObjEditor.
//
// Non-copyable.

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
                             | kEnterWindowMask | kLeaveWindowMask
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
                             | kEnterWindowMask | kLeaveWindowMask
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
                             | kEnterWindowMask | kLeaveWindowMask
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
                             | kEnterWindowMask | kLeaveWindowMask
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
         gROOT->ProcessLineFast(Form("((TGLWidget *)0x%lx)->SetFormat()", this));
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
         gROOT->ProcessLineFast(Form("((TGLWidget *)0x%lx)->SetFormat()", this));
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

//______________________________________________________________________________
void TGLWidget::SetEventHandler(TGEventHandler *eh)
{
   //Set event-handler. All events are passed to this object.
   fEventHandler = eh;
   fContainer->SetEventHandler(eh);
}
