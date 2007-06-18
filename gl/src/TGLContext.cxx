// @(#)root/gl:$Name:  $:$Id: TGLContext.cxx,v 1.3 2007/06/12 20:29:00 rdm Exp $
// Author:  Timur Pocheptsov, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdexcept>
#include <memory>

#ifndef WIN32
#include <GL/glx.h>
#endif

#include "TVirtualX.h"
#include "GuiTypes.h"
#include "TString.h"
#include "TError.h"

#include "TROOT.h"

//#include "TGLPBufferPrivate.h"
#include "TGLContextPrivate.h"
#include "TGLIncludes.h"
//#include "TGLPBuffer.h"
#include "TGLContext.h"
#include "TGLWidget.h"
#include "TGLFormat.h"
#include "TGLUtil.h"

ClassImp(TGLContext)

//______________________________________________________________________________
TGLContext::TGLContext(TGLWidget *wid, const TGLContext *shareList)
               : fDevice(wid),
                 fPimpl(0),
                 fFromCtor(kTRUE),
                 fValid(kFALSE),
                 fIdentity(0)
{
   //TGLContext ctor "from" TGLWidget.
   //Makes thread switching.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLContext *)0x%x)->SetContext((TGLWidget *)0x%x, (TGLContext *)0x%x)",
                                  this, wid, shareList));
   } else
      SetContext(wid, shareList);

   fFromCtor = kFALSE;
//   fValid = kTRUE;

}

//______________________________________________________________________________
/*
TGLContext::TGLContext(TGLPBuffer *pbuff, const TGLContext *shareList)
               : fDevice(pbuff),
                 fPimpl(0),
                 fFromCtor(kTRUE),
                 fValid(kFALSE)
{
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLContext *)0x%x)->SetContextPB((TGLPBuffer *)0x%x, (TGLContext *)0x%x)",
                                  this, pbuff, shareList));
   } else
      SetContextPB(pbuff, shareList);

   fFromCtor = kFALSE;
   fValid = kTRUE;
}
*/

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

}

//______________________________________________________________________________
void TGLContext::SetContext(TGLWidget *widget, const TGLContext *shareList)
{
   //WIN32 gl-context creation. Defined as a member-function (this code removed from ctor)
   //to make WIN32/X11 separation cleaner.
   //This function is public only for calls via gROOT and called from ctor.
   if (!fFromCtor) {
      Error("TGLContext::SetContext", "SetContext must be called only from ctor");
      return;
   }

   std::auto_ptr<TGLContextPrivate> safe_ptr(fPimpl = new TGLContextPrivate);
   LayoutCompatible_t *trick =
      reinterpret_cast<LayoutCompatible_t *>(gVirtualX->GetWindowID(widget->GetWindowIndex()));
   HWND hWND = *trick->fPHwnd;
   HDC  hDC  = GetWindowDC(hWND);

   if (!hDC) {
      Error("TGLContext::SetContext", "GetWindowDC failed");
      throw std::runtime_error("GetWindowDC failed");
   }

   const Rgl::TGuardBase &dcGuard = Rgl::make_guard(ReleaseDC, hWND, hDC);
   if (HGLRC glContext = wglCreateContext(hDC)) {
      fPimpl->fHWND = hWND;
      fPimpl->fHDC = hDC;
      fPimpl->fGLContext = glContext;
      if (shareList && !wglShareLists(shareList->fPimpl->fGLContext, glContext))
         Error("TGLContext::SetContext", "Cannot share lists");
   } else {
      Error("TGLContext::SetContext", "wglCreateContext failed");
      throw std::runtime_error("wglCreateContext failed");
   }

   //Register context for "parent" gl-device.
   fValid = kTRUE;
   fDevice->AddContext(this);
   TGLContextPrivate::RegisterContext(this);

   if (shareList) {
      fIdentity = shareList->GetIdentity();
      fIdentity->AddRef();
   } else {
      fIdentity = new TGLContextIdentity;
   }

   dcGuard.Stop();
   safe_ptr.release();
}

/*
//______________________________________________________________________________
void TGLContext::SetContextPB(TGLPBuffer *pbuff, const TGLContext *shareList)
{
   if (!fFromCtor) {
      Error("TGLContext::SetContextPB", "SetContextPB must be called only from ctor");
      return;
   }

   std::auto_ptr<TGLContextPrivate> safe_ptr(fPimpl = new TGLContextPrivate);
   fPimpl->fHDC = pbuff->fPimpl->fHDC;
   fPimpl->fGLContext = pbuff->fPimpl->fGLRC;

   if (shareList && !wglShareLists(shareList->fPimpl->fGLContext, fPimpl->fGLContext))
      Error("TGLContext::SetContextPV", "Cannot share lists");

   fDevice->AddContext(this);

   safe_ptr.release();
}
*/

//______________________________________________________________________________
Bool_t TGLContext::MakeCurrent()
{
   //If context is valid (TGLPaintDevice, for which context was created still exists),
   //make it current.
   if (!fValid) {
      Error("TGLContext::MakeCurrent", "This context is invalid.");
      return kFALSE;
   }

   if (!gVirtualX->IsCmdThread())
      return Bool_t(gROOT->ProcessLineFast(Form("((TGLContext *)0x%x)->MakeCurrent()", this)));
   else {
      Bool_t rez = wglMakeCurrent(fPimpl->fHDC, fPimpl->fGLContext);
      if (rez)
         fIdentity->DeleteDisplayLists();
      return rez;
   }
}

//______________________________________________________________________________
void TGLContext::SwapBuffers()
{
   //If context is valid (TGLPaintDevice, for which context was created still exists),
   //swap buffers (in case of P-buffer call glFinish()).
   if (!fValid) {
      Error("TGLContext::SwapBuffers", "This context is invalid.");
      return;
   }

   if (!gVirtualX->IsCmdThread())
      gROOT->ProcessLineFast(Form("((TGLContext *)0x%x)->SwapBuffers()", this));
   else {
      if (fPimpl->fHWND)
         wglSwapLayerBuffers(fPimpl->fHDC, WGL_SWAP_MAIN_PLANE);
      else
         glFinish();
   }
}

//______________________________________________________________________________
void TGLContext::Release()
{
   //Make the context invalid and (do thread switch, if needed)
   //free resources.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLContext *)0x%x)->Release()", this));
      return;
   }

   if (fPimpl->fHWND)
      ReleaseDC(fPimpl->fHWND, fPimpl->fHDC);

   TGLContextPrivate::RemoveContext(this);
   wglDeleteContext(fPimpl->fGLContext);
   fValid = kFALSE;
}

#else

//______________________________________________________________________________
void TGLContext::SetContext(TGLWidget *widget, const TGLContext *shareList)
{
   //X11 gl-context creation. Defined as a member-function (this code removed from ctor)
   //to make WIN32/X11 separation cleaner.
   //This function is public only for calls via gROOT and called from ctor.

   if (!fFromCtor) {
      Error("TGLContext::SetContext", "SetContext must be called only from ctor");
      return;
   }

   std::auto_ptr<TGLContextPrivate> safe_ptr(fPimpl = new TGLContextPrivate);
   Display *dpy = static_cast<Display *>(widget->GetInnerData().first);
   XVisualInfo *visInfo = static_cast<XVisualInfo *>(widget->GetInnerData().second);

   GLXContext glCtx = shareList ? glXCreateContext(dpy, visInfo, shareList->fPimpl->fGLContext, True)
                                : glXCreateContext(dpy, visInfo, None, True);

   if (!glCtx) {
      Error("TGLContext::SetContext", "glXCreateContext failed!");
      throw std::runtime_error("glXCreateContext failed!");
   }

   fPimpl->fDpy = dpy;
   fPimpl->fVisualInfo = visInfo;
   fPimpl->fGLContext = glCtx;
   fPimpl->fWindowIndex = widget->GetWindowIndex();

   if (shareList) {
      fIdentity = shareList->GetIdentity();
      fIdentity->AddRef();
   } else {
      fIdentity = new TGLContextIdentity;
   }

   fValid = kTRUE;
   fDevice->AddContext(this);
   TGLContextPrivate::RegisterContext(this);

   safe_ptr.release();
}

/*
//______________________________________________________________________________
void TGLContext::SetContextPB(TGLPBuffer *pbuff, const TGLContext *shareList)
{
   std::auto_ptr<TGLContextPrivate> safe_ptr(fPimpl = new TGLContextPrivate);
   fPimpl->fDpy = pbuff->fPimpl->fDpy;
   fPimpl->fGLContext = pbuff->fPimpl->fPBRC;
   fPimpl->fPBDC = pbuff->fPimpl->fPBDC;

   fDevice->AddContext(this);
   safe_ptr.release();
}
*/

//______________________________________________________________________________
Bool_t TGLContext::MakeCurrent()
{
   //If context is valid (TGLPaintDevice, for which context was created still exists),
   //make it current.

   if (!fValid) {
      Error("TGLContext::MakeCurrent", "This context is invalid.");
      return kFALSE;
   }

   if (fPimpl->fWindowIndex != -1) {
      const Bool_t rez = glXMakeCurrent(fPimpl->fDpy,
                                        gVirtualX->GetWindowID(fPimpl->fWindowIndex),
                                        fPimpl->fGLContext);
      if (rez)
         fIdentity->DeleteDisplayLists();
      return rez;
   }

   return kFALSE;//NO pbuffer part yet.
   //return glXMakeCurrent(fPimpl->fDpy, fPimpl->fPBDC, fPimpl->fGLContext);
}

//______________________________________________________________________________
void TGLContext::SwapBuffers()
{
   //If context is valid (TGLPaintDevice, for which context was created still exists),
   //swap buffers (in case of P-buffer call glFinish()).

   if (!fValid) {
      Error("TGLContext::SwapCurrent", "This context is invalid.");
      return;
   }

   if (fPimpl->fWindowIndex != -1)
      glXSwapBuffers(fPimpl->fDpy, gVirtualX->GetWindowID(fPimpl->fWindowIndex));
   else
      glFinish();
}

//______________________________________________________________________________
void TGLContext::Release()
{
   //Make the context invalid and (do thread switch, if needed)
   //free resources.
   TGLContextPrivate::RemoveContext(this);
   glXDestroyContext(fPimpl->fDpy, fPimpl->fGLContext);
   fValid = kFALSE;
}

#endif

//______________________________________________________________________________
TGLContext::~TGLContext()
{
   //TGLContext dtor. If it's called before TGLPaintDevice's dtor (context is valid)
   //resource will be freed and context un-registered.
   if (fValid) {
      Release();
      fDevice->RemoveContext(this);
   }

   fIdentity->Release();

   delete fPimpl;
}

//______________________________________________________________________________
TGLContextIdentity *TGLContext::GetIdentity()const
{
   //We can have several shared contexts,
   //and gl-scene wants to know, if some context
   //(defined by its identity) can be used.
   return fIdentity;
}

//______________________________________________________________________________
TGLContext *TGLContext::GetCurrent()
{
   //Ask TGLContextPrivate to lookup context in its internal map.
   return TGLContextPrivate::GetCurrentContext();
}


//______________________________________________________________________________
// TGLContextIdentity
//
// Identifier of a shared GL-context.
// Objects shared among GL-contexts include:
// display-list definitions, texture objects, shader programs.

ClassImp(TGLContextIdentity)

//______________________________________________________________________________
void TGLContextIdentity::RegisterDLNameRangeToWipe(UInt_t base, Int_t size)
{
   //Remember dl range for deletion in next MakeCurrent or dtor execution.
   fDLTrash.push_back(DLRange_t(base, size));
}

//______________________________________________________________________________
void TGLContextIdentity::DeleteDisplayLists()
{
   //Delete display-list objects registered for destruction.
   if (fDLTrash.empty()) return;

   for (DLTrashIt_t it = fDLTrash.begin(), e = fDLTrash.end(); it != e; ++it)
      glDeleteLists(it->first, it->second);
   fDLTrash.clear();
}

//______________________________________________________________________________
TGLContextIdentity *TGLContextIdentity::GetCurrent()
{
   //Find identitfy of current context.
   TGLContext* ctx = TGLContext::GetCurrent();
   return ctx ? ctx->GetIdentity() : 0;
}
