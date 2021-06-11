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

#include "TVirtualX.h"
#include "TGClient.h"
#include "TError.h"
#include "TROOT.h"
#include "TVirtualMutex.h"

#include "TGLWidget.h"
#include "TGLIncludes.h"
#include "TGLWSIncludes.h"
#include "TGLUtil.h"
#include "TGLEventHandler.h"
#include "RConfigure.h"

/** \class TGLWidget
\ingroup opengl
GL window with context. _Must_ _have_ a parent window
(the 'parent' parameter of ctors). The current version inherits
TGCanvas (I'm not sure about future versions), probably, in future
multiple inheritance will be added - the second
base class will be TGLPaintDevice or something like this.

Usage:
  - Simply create TGLWidget as an embedded widget, and
    connect your slots to signals you need: HandleExpose, HandleConfigureNotify, etc.
    In your slots you can use gl API directly - under Win32 TGLWidget switches
    between threads internally (look TGLPShapeObjEditor for such usage).
  - You can write your own class, derived from TGLWidget, with PaintGL and InitGL
    overriden.

Resources (and invariants):
  - fContainer (TGLWidgetContainer) - controlled by std::auto_ptr
  - fWindowIndex - controlled manually (see CreateWidget and dtor)
  - fGLContext - controlled manually (see CreateWidget and dtor)
  - visual info for X11 version, controlled manually (see CreateGLContainer and dtor)

Exceptions:
  - can be thrown only during construction.
  - under win32 class does not throw itself (but some internal operations can throw)
  - under X11 can throw std::runtime_error (from CreateGLContext).
  - In case of exceptions resources will be freed.

TGLWidget object is immutable as far as it was created.

Boolean parameter defines, if you want to grab user's input or not.
By default you want, but for example when not - see TGLPShapeObjEditor.

Non-copyable.
*/

ClassImp(TGLWidget);

//==============================================================================
// TGLWidget - system-independent methods
//==============================================================================

////////////////////////////////////////////////////////////////////////////////
/// Static constructor for creating widget with default pixel format.

TGLWidget* TGLWidget::CreateDummy()
{
   TGLFormat format(Rgl::kNone);

   return Create(format, gClient->GetDefaultRoot(), kFALSE, kFALSE, 0, 1, 1);
}

////////////////////////////////////////////////////////////////////////////////
/// Static constructor for creating widget with default pixel format.

TGLWidget* TGLWidget::Create(const TGWindow* parent, Bool_t selectInput,
              Bool_t shareDefault, const TGLPaintDevice *shareDevice,
              UInt_t width, UInt_t height)
{
   TGLFormat format;

   return Create(format, parent, selectInput, shareDefault, shareDevice,
                 width, height);
}

////////////////////////////////////////////////////////////////////////////////
/// Static constructor for creating widget with given pixel format.

TGLWidget* TGLWidget::Create(const TGLFormat &format,
             const TGWindow* parent, Bool_t selectInput,
             Bool_t shareDefault, const TGLPaintDevice *shareDevice,
             UInt_t width, UInt_t height)
{
   // Make sure window-system dependent part of GL-util is initialized.
   TGLUtil::InitializeIfNeeded();

   std::pair<void *, void *> innerData;

   Window_t wid = CreateWindow(parent, format, width, height, innerData);

   TGLWidget* glw = new TGLWidget(wid, parent, selectInput);

#ifdef WIN32
   glw->fWindowIndex = (Int_t)(Longptr_t)innerData.second;
#elif defined(R__HAS_COCOA)
   glw->fWindowIndex = wid;
#else
   glw->fWindowIndex = gVirtualX->AddWindow(wid, width, height);
   glw->fInnerData   = innerData;
#endif
   glw->fGLFormat  = format;

   try
   {
      glw->SetFormat();
      glw->fGLContext = new TGLContext
         (glw, shareDefault, shareDevice && !shareDefault ? shareDevice->GetContext() : 0);
   }
   catch (const std::exception &)
   {
      delete glw;
      throw;
   }

   glw->fFromInit = kFALSE;

   return glw;
}

////////////////////////////////////////////////////////////////////////////////
/// Creates widget with default pixel format.

TGLWidget::TGLWidget(Window_t glw, const TGWindow* p, Bool_t selectInput)
   : TGFrame(gClient, glw, p),
     fGLContext(0),
     fWindowIndex(-1),
     fGLFormat(Rgl::kNone),
     fFromInit(kTRUE),
     fEventHandler(0)
{
   if (selectInput)
   {
      gVirtualX->GrabButton(GetId(), kAnyButton, kAnyModifier,
                            kButtonPressMask | kButtonReleaseMask, kNone, kNone);
      gVirtualX->SelectInput(GetId(), kKeyPressMask | kKeyReleaseMask | kExposureMask |
                             kPointerMotionMask | kStructureNotifyMask | kFocusChangeMask |
                             kEnterWindowMask | kLeaveWindowMask);
      gVirtualX->SetInputFocus(GetId());
   }
}

////////////////////////////////////////////////////////////////////////////////
///Destructor. Deletes window ???? and XVisualInfo

TGLWidget::~TGLWidget()
{
#ifndef WIN32
#ifndef R__HAS_COCOA
   XFree(fInnerData.second);//free XVisualInfo
#endif
#endif
   if (fValidContexts.size() > 1u) {
      Warning("~TGLWidget", "There are some gl-contexts connected to this gl device"
                            "which have longer lifetime than lifetime of gl-device");
   }

   std::set<TGLContext *>::iterator it = fValidContexts.begin();
   for (; it != fValidContexts.end(); ++it) {
      (*it)->Release();
   }
   delete fGLContext;

   gVirtualX->SelectWindow(fWindowIndex);
   gVirtualX->CloseWindow();
}

////////////////////////////////////////////////////////////////////////////////
///Call glEnable(... in overrider of InitGL.

void TGLWidget::InitGL()
{
}

////////////////////////////////////////////////////////////////////////////////
///Do actual drawing in overrider of PaintGL.

void TGLWidget::PaintGL()
{
}

////////////////////////////////////////////////////////////////////////////////
///Make the gl-context current.

Bool_t TGLWidget::MakeCurrent()
{
   return fGLContext->MakeCurrent();
}

////////////////////////////////////////////////////////////////////////////////
///Clear the current gl-context.

Bool_t TGLWidget::ClearCurrent()
{
   return fGLContext->ClearCurrent();
}

////////////////////////////////////////////////////////////////////////////////
///Swap buffers.

void TGLWidget::SwapBuffers()
{
   fGLContext->SwapBuffers();
}

////////////////////////////////////////////////////////////////////////////////
///Get gl context.

const TGLContext *TGLWidget::GetContext()const
{
   return fGLContext;
}

////////////////////////////////////////////////////////////////////////////////
///Pixel format.

const TGLFormat *TGLWidget::GetPixelFormat()const
{
   return &fGLFormat;
}

////////////////////////////////////////////////////////////////////////////////
///Dpy*, XVisualInfo *

std::pair<void *, void *> TGLWidget::GetInnerData()const
{
   return fInnerData;
}

////////////////////////////////////////////////////////////////////////////////
///Register gl-context created for this window.

void TGLWidget::AddContext(TGLContext *ctx)
{
   fValidContexts.insert(ctx);
}

////////////////////////////////////////////////////////////////////////////////
///Remove context (no real deletion, done by TGLContex dtor).

void TGLWidget::RemoveContext(TGLContext *ctx)
{
   std::set<TGLContext *>::iterator it = fValidContexts.find(ctx);
   if (it != fValidContexts.end())
      fValidContexts.erase(it);
}

////////////////////////////////////////////////////////////////////////////////
///For camera.

void TGLWidget::ExtractViewport(Int_t *vp)const
{
   vp[0] = 0;
   vp[1] = 0;
   vp[2] = GetWidth();
   vp[3] = GetHeight();
}

//==============================================================================
// System specific methods and helper functions
//==============================================================================

//==============================================================================
#ifdef WIN32
//==============================================================================

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

////////////////////////////////////////////////////////////////////////////////
/// CreateWidget.
/// Static function called prior to widget construction,
/// I've extracted this code from ctors to make WIN32/X11
/// separation simpler and because of gInterpreter usage.
/// new, TGLContext can throw
/// std::bad_alloc and std::runtime_error. Before try block, the only
/// resource allocated is pointed by fWindowIndex (InitWindow cannot throw).
/// In try block (and after successful constraction)
/// resources are controlled by std::auto_ptrs and dtor.

Window_t TGLWidget::CreateWindow(const TGWindow* parent, const TGLFormat& /*format*/,
                                 UInt_t width, UInt_t  height,
                                 std::pair<void *, void *>& innerData)
{
   Int_t widx = gVirtualX->InitWindow((ULongptr_t)parent->GetId());
   innerData.second = (void*)(Longptr_t)widx;
   Window_t win = gVirtualX->GetWindowID(widx);
   gVirtualX->ResizeWindow(win, width, height);
   return win;
}

////////////////////////////////////////////////////////////////////////////////
/// Set pixel format.
/// Resource - hDC, owned and freed by guard object.

void TGLWidget::SetFormat()
{
   if (!fFromInit) {
      Error("TGLWidget::SetFormat", "Sorry, you should not call this function");
      return;
   }
   if (!gVirtualX->IsCmdThread())
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%zx)->SetFormat()", (size_t)this));

   R__LOCKGUARD(gROOTMutex);

   LayoutCompatible_t *trick =
      reinterpret_cast<LayoutCompatible_t *>(GetId());
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
//==============================================================================
#elif defined(R__HAS_COCOA) //MacOSX with Cocoa enabled.
//==============================================================================

////////////////////////////////////////////////////////////////////////////////
/// CreateWidget - MacOSX/Cocoa version.
/// Static function called prior to construction.

Window_t TGLWidget::CreateWindow(const TGWindow* parent, const TGLFormat &format,
                                 UInt_t width, UInt_t height,
                                 std::pair<void *, void *>& /*internalData*/)
{
   typedef std::pair<UInt_t, Int_t> component_type;

   std::vector<component_type>formatComponents;

   if (format.HasDepth())
      formatComponents.push_back(component_type(Rgl::kDepth, format.GetDepthSize()));
   if (format.HasStencil())
      formatComponents.push_back(component_type(Rgl::kStencil, format.GetStencilSize()));
   if (format.HasAccumBuffer())
      formatComponents.push_back(component_type(Rgl::kAccum, format.GetAccumSize()));
   if (format.IsDoubleBuffered())
      formatComponents.push_back(component_type(Rgl::kDoubleBuffer, 0));
   if (format.IsStereo())
      formatComponents.push_back(component_type(Rgl::kStereo, 0));
   if (format.HasMultiSampling())
      formatComponents.push_back(component_type(Rgl::kMultiSample, format.GetSamples()));

   return gVirtualX->CreateOpenGLWindow(parent->GetId(), width, height, formatComponents);
}

////////////////////////////////////////////////////////////////////////////////
/// Set pixel format.
/// Empty version for X11.

void TGLWidget::SetFormat()
{
}

//==============================================================================
#else // X11
//==============================================================================

namespace
{
   void fill_format(std::vector<Int_t> &format, const TGLFormat &request)
   {
      format.push_back(GLX_RGBA);
      format.push_back(GLX_RED_SIZE);
      format.push_back(8);
      format.push_back(GLX_GREEN_SIZE);
      format.push_back(8);
      format.push_back(GLX_BLUE_SIZE);
      format.push_back(8);

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
         format.push_back(8);
         format.push_back(GLX_ACCUM_GREEN_SIZE);
         format.push_back(8);
         format.push_back(GLX_ACCUM_BLUE_SIZE);
         format.push_back(8);
      }

      if (request.IsStereo()) {
         format.push_back(GLX_STEREO);
      }

      if (request.HasMultiSampling())
      {
         format.push_back(GLX_SAMPLE_BUFFERS_ARB);
         format.push_back(1);
         format.push_back(GLX_SAMPLES_ARB);
         format.push_back(request.GetSamples());
      }

      format.push_back(None);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// CreateWidget - X11 version.
/// Static function called prior to construction.
/// Can throw std::bad_alloc and std::runtime_error.
/// This version is bad - I do not check the results of
/// X11 calls.

Window_t TGLWidget::CreateWindow(const TGWindow* parent, const TGLFormat &format,
                                 UInt_t width, UInt_t height,
                                 std::pair<void *, void *>& innerData)
{
   std::vector<Int_t> glxfmt;
   fill_format(glxfmt, format);

   Display *dpy = reinterpret_cast<Display *>(gVirtualX->GetDisplay());
   if (!dpy) {
      ::Error("TGLWidget::CreateWindow", "Display is not set!");
      throw std::runtime_error("Display is not set!");
   }
   XVisualInfo *visInfo = glXChooseVisual(dpy, DefaultScreen(dpy), &glxfmt[0]);

   if (!visInfo) {
      ::Error("TGLWidget::CreateWindow", "No good OpenGL visual found!");
      throw std::runtime_error("No good OpenGL visual found!");
   }

   Window_t winID = parent->GetId();

   XSetWindowAttributes attr;
   attr.colormap         = XCreateColormap(dpy, winID, visInfo->visual, AllocNone); // Can fail?
   attr.background_pixel = 0;
   attr.event_mask       = NoEventMask;
   attr.backing_store    = Always;
   attr.bit_gravity      = NorthWestGravity;

   ULong_t mask = CWBackPixel | CWColormap | CWEventMask | CWBackingStore | CWBitGravity;
   Window glWin = XCreateWindow(dpy, winID, 0, 0, width, height, 0,
                                visInfo->depth,
                                InputOutput, visInfo->visual, mask, &attr);

   innerData.first  = dpy;
   innerData.second = visInfo;

   return glWin;
}

////////////////////////////////////////////////////////////////////////////////
/// Set pixel format.
/// Empty version for X11.

void TGLWidget::SetFormat()
{
}

//==============================================================================
#endif
//==============================================================================


//==============================================================================
// Event handling
//==============================================================================

////////////////////////////////////////////////////////////////////////////////
///Set event-handler. All events are passed to this object.

void TGLWidget::SetEventHandler(TGEventHandler *eh)
{
   fEventHandler = eh;
}

////////////////////////////////////////////////////////////////////////////////
/// Handle mouse crossing event.

Bool_t TGLWidget::HandleCrossing(Event_t *ev)
{
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%zx)->HandleCrossing((Event_t *)0x%zx)", (size_t)this, (size_t)ev));
      return kTRUE;
   }
   R__LOCKGUARD(gROOTMutex);

   if ((ev->fType == kEnterNotify) &&
       (!gVirtualX->InheritsFrom("TGX11")) &&
       (gVirtualX->GetInputFocus() != GetId())) {
      gVirtualX->SetInputFocus(GetId());
   }
   if (fEventHandler)
      return fEventHandler->HandleCrossing(ev);
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
///Delegate call to the owner.

Bool_t TGLWidget::HandleButton(Event_t *ev)
{
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%zx)->HandleButton((Event_t *)0x%zx)", (size_t)this, (size_t)ev));
      return kTRUE;
   }
   R__LOCKGUARD(gROOTMutex);

   if (fEventHandler)
      return fEventHandler->HandleButton(ev);
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
///Delegate call to the owner.

Bool_t TGLWidget::HandleDoubleClick(Event_t *ev)
{
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%zx)->HandleDoubleClick((Event_t *)0x%zx)", (size_t)this, (size_t)ev));
      return kTRUE;
   }
   R__LOCKGUARD(gROOTMutex);

   if (fEventHandler)
      return fEventHandler->HandleDoubleClick(ev);
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
///Delegate call to the owner.

Bool_t TGLWidget::HandleConfigureNotify(Event_t *ev)
{
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%zx)->HandleConfigureNotify((Event_t *)0x%zx)", (size_t)this, (size_t)ev));
      return kTRUE;
   }
   R__LOCKGUARD(gROOTMutex);

   if (fEventHandler && fEventHandler->HandleConfigureNotify(ev))
   {
      TGFrame::HandleConfigureNotify(ev);
      return kTRUE;
   }
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
///Delegate call to the owner.

Bool_t TGLWidget::HandleFocusChange(Event_t *ev)
{
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%zx)->HandleFocusChange((Event_t *)0x%zx)", (size_t)this, (size_t)ev));
      return kTRUE;
   }
   R__LOCKGUARD(gROOTMutex);

   if (fEventHandler)
      return fEventHandler->HandleFocusChange(ev);
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
///Delegate call to the owner.

Bool_t TGLWidget::HandleKey(Event_t *ev)
{
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%zx)->HandleKey((Event_t *)0x%zx)", (size_t)this, (size_t)ev));
      return kTRUE;
   }
   R__LOCKGUARD(gROOTMutex);

   if (fEventHandler)
      return fEventHandler->HandleKey(ev);
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
///Delegate call to the owner.

Bool_t TGLWidget::HandleMotion(Event_t *ev)
{
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%zx)->HandleMotion((Event_t *)0x%zx)", (size_t)this, (size_t)ev));
      return kTRUE;
   }
   R__LOCKGUARD(gROOTMutex);

   if (fEventHandler)
      return fEventHandler->HandleMotion(ev);
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
///Delegate call to the owner.

void TGLWidget::DoRedraw()
{
   if (fEventHandler)
      return fEventHandler->Repaint();
}
