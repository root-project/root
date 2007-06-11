#include <stdexcept>
#include <vector>

#ifndef WIN32
#include <GL/glx.h>
#endif

#include "TVirtualX.h"
#include "TGClient.h"
#include "TROOT.h"

#include "TGLWidget.h"
#include "TGLIncludes.h"

ClassImp(TGLWidgetContainer)

//______________________________________________________________________________
TGLWidgetContainer::TGLWidgetContainer(TGLWidget *owner, Window_t id, const TGWindow *parent)
                     : TGCompositeFrame(gClient, id, parent),
                       fOwner(owner)
{
   // Constructor.
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleButton(Event_t *ev)
{
   // Delegate call to the owner.
   return fOwner->HandleButton(ev);
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleDoubleClick(Event_t *ev)
{
   // Delegate call to the owner.
   return fOwner->HandleDoubleClick(ev);
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleConfigureNotify(Event_t *ev)
{
   // Delegate call to the owner.
   return fOwner->HandleConfigureNotify(ev);
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleKey(Event_t *ev)
{
   // Delegate call to the owner.
   return fOwner->HandleKey(ev);
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleMotion(Event_t *ev)
{
   // Delegate call to the owner.
   return fOwner->HandleMotion(ev);
}

//______________________________________________________________________________
Bool_t TGLWidgetContainer::HandleExpose(Event_t *ev)
{
   // Delegate call to the owner.
   return fOwner->HandleExpose(ev);
}

ClassImp(TGLWidget)

//______________________________________________________________________________
TGLWidget::TGLWidget(const TGWindow &p, Bool_t select, UInt_t w, UInt_t h, UInt_t opt, Pixel_t back)
              : TGCanvas(&p, w, h, opt, back),
                fWindowIndex(-1)
{
   // Creates widget with default pixel format.
   TGLFormat defaultFormat;//std::bad_alloc
   CreateGLContainer(defaultFormat);//std::bad_alloc, std::runtime_error

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
}

//______________________________________________________________________________
TGLWidget::TGLWidget(const TGLFormat &format, const TGWindow &p, Bool_t select,
                     UInt_t w, UInt_t h, UInt_t opt, Pixel_t back)
              : TGCanvas(&p, w, h, opt, back),
                fWindowIndex(-1)
{
   // Create widget with the requested pixel format.
   CreateGLContainer(format);//std::bad_alloc, std::runtime_error

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
}

//______________________________________________________________________________
TGLWidget::~TGLWidget()
{
   // Destructor. Deletes window ???? and XVisualInfo
   gVirtualX->SelectWindow(fWindowIndex);
   gVirtualX->CloseWindow();
#ifndef WIN32
   XFree(fInnerData.second);
#endif
}

//______________________________________________________________________________
void TGLWidget::InitGL()
{
   // Call glEnable(... in overrider of InitGL.
}

//______________________________________________________________________________
void TGLWidget::PaintGL()
{
   // Do actual drawing in overrider of PaintGL.
}

//______________________________________________________________________________
Bool_t TGLWidget::MakeCurrent()
{
   // Make the gl-context current.
   return fGLContext->MakeCurrent();
}

//______________________________________________________________________________
void TGLWidget::SwapBuffers()
{
   // Swap buffers.
   fGLContext->SwapBuffers();
}

//______________________________________________________________________________
const TGLContext *TGLWidget::GetContext()const
{
   // Get gl context.
   return fGLContext.get();
}

//______________________________________________________________________________
Bool_t TGLWidget::HandleButton(Event_t *ev)
{
   // Signal. Under Win32 I have to switch between threads to let
   // direct usage of gl code.
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
   // Signal. Under Win32 I have to switch between threads to let
   // direct usage of gl code.
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
   // Signal. Under Win32 I have to switch between threads to let
   // direct usage of gl code.
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
   // Signal. Under Win32 I have to switch between threads to let
   // direct usage of gl code.
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
   // Signal. Under Win32 I have to switch between threads to let
   // direct usage of gl code.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLWidget *)0x%x)->HandleMotion((Event_t *)0x%x)", this, ev));
      return kTRUE;
   }

   Emit("HandleMotion(Event_t*)", (Long_t)ev);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLWidget::HandleExpose(Event_t *ev)
{
   // Signal. Under Win32 I have to switch between threads to let
   // direct usage of gl code.
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

//______________________________________________________________________________
Int_t TGLWidget::GetWindowIndex()const
{
   // Index of window, registered by gVirtualX.
   return fWindowIndex;
}

//______________________________________________________________________________
const TGLFormat &TGLWidget::GetPixelFormat()const
{
   // Pixel format.
   return fGLContext->GetPixelFormat();
}

//______________________________________________________________________________
Int_t TGLWidget::GetContId()const
{
   // Id of container.
   return fContainer->GetId();
}

//______________________________________________________________________________
std::pair<void *, void *> TGLWidget::GetInnerData()const
{
   // Dpy*, XVisualInfo *
   return fInnerData;
}

#ifdef WIN32

//______________________________________________________________________________
void TGLWidget::CreateGLContainer(const TGLFormat &format)
{
   // CreateGLContainer.
   //
   // This function called only during construction, I've extracted
   // this code from ctors only to make WIN32/X11 parts simpler.
   //
   // new, TGLContext, TGLFormat copy ctors can throw std::bad_alloc
   // and std::runtime_error. Before try block, the only resource
   // allocated is pointed by fWindowIndex.
   // 
   // In try block (and after successful constraction) resources are
   // controlled by std::auto_ptrs and dtor.

   fWindowIndex = gVirtualX->InitWindow((ULong_t)GetViewPort()->GetId());
   try {
      fGLContext.reset(new TGLContext(this, format));
      fContainer.reset(new TGLWidgetContainer(this, gVirtualX->GetWindowID(fWindowIndex), GetViewPort()));
      SetContainer(fContainer.get());
   } catch (std::exception &) {
      gVirtualX->SelectWindow(fWindowIndex);
      gVirtualX->CloseWindow();
      throw;
   }
}

#else

namespace {

   void fill_format(std::vector<Int_t> &format, const TGLFormat &request)
   {
      //This version ignores accum buffer FIXFIX

      if (request.IsRGBA()) {
         format.push_back(GLX_RGBA);
         format.push_back(GLX_RED_SIZE);
         format.push_back(1);
         format.push_back(GLX_GREEN_SIZE);
         format.push_back(1);
         format.push_back(GLX_BLUE_SIZE);
         format.push_back(1);
      } else {
         format.push_back(GLX_BUFFER_SIZE);
         format.push_back(request.GetColorIndexSize());
      }

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

      format.push_back(None);
   }

   XSetWindowAttributes dummyAttr;

}

//______________________________________________________________________________
void TGLWidget::CreateGLContainer(const TGLFormat &request)
{
   // CreateGLContainer - X11 version.
   //
   // This function called only during construction,
   // This function, new, TGLContext, TGLFormat copy ctors can throw
   // std::bad_alloc and std::runtime_error.
   //
   //  This version is bad - I do not check the results of X11 calls.
   std::vector<Int_t> format;
   fill_format(format, request);

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
      fGLContext.reset(new TGLContext(this, request));
      fContainer.reset(new TGLWidgetContainer(this, gVirtualX->GetWindowID(fWindowIndex), GetViewPort()));
      SetContainer(fContainer.get());
   } catch (const std::exception &) {
      gVirtualX->SelectWindow(fWindowIndex);
      gVirtualX->CloseWindow();
      XFree(fInnerData.second);
      throw;
   }
}

#endif
