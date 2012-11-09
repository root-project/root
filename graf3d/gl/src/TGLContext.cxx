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
#include <algorithm>
#include <memory>

#include "TVirtualX.h"
#include "GuiTypes.h"
#include "TString.h"
#include "TError.h"

#include "TROOT.h"

#include "TGLContextPrivate.h"
#include "RConfigure.h"
#include "TGLIncludes.h"
#include "TGLContext.h"
#include "TGLWidget.h"
#include "TGLFormat.h"
#include "TGLUtil.h"

#include "TGLFontManager.h"

//______________________________________________________________________________
//
// This class encapsulates window-system specific information about a
// GL-context and alows their proper management in ROOT.

ClassImp(TGLContext);

Bool_t TGLContext::fgGlewInitDone = kFALSE;

//______________________________________________________________________________
TGLContext::TGLContext(TGLWidget *wid, Bool_t shareDefault,
                       const TGLContext *shareList)
   : fDevice(wid),
     fPimpl(0),
     fFromCtor(kTRUE),
     fValid(kFALSE),
     fIdentity(0)
{
   // TGLContext ctor "from" TGLWidget.
   // Is shareDefault is true, the shareList is set from default
   // context-identity. Otherwise the given shareList is used (can be
   // null).
   // Makes thread switching.

   if (shareDefault)
      shareList = TGLContextIdentity::GetDefaultContextAny();

   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLContext *)0x%lx)->SetContext((TGLWidget *)0x%lx, (TGLContext *)0x%lx)",
                                  (ULong_t)this, (ULong_t)wid, (ULong_t)shareList));
   } else {
      SetContext(wid, shareList);
   }

   if (shareDefault)
      fIdentity = TGLContextIdentity::GetDefaultIdentity();
   else
      fIdentity = shareList ? shareList->GetIdentity() : new TGLContextIdentity;

   fIdentity->AddRef(this);

   fFromCtor = kFALSE;
}

//______________________________________________________________________________
void TGLContext::GlewInit()
{
   // Initialize GLEW - static private function.
   // Called immediately after creation of the first GL context.

   if (!fgGlewInitDone)
   {
      GLenum status = glewInit();
      if (status != GLEW_OK)
         Warning("TGLContext::GlewInit", "GLEW initalization failed.");
      else if (gDebug > 0)
         Info("TGLContext::GlewInit", "GLEW initalization successful.");
      fgGlewInitDone = kTRUE;
   }
}

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
      reinterpret_cast<LayoutCompatible_t *>(widget->GetId());
   HWND hWND = *trick->fPHwnd;
   HDC  hDC  = GetWindowDC(hWND);

   if (!hDC) {
      Error("TGLContext::SetContext", "GetWindowDC failed");
      throw std::runtime_error("GetWindowDC failed");
   }

   const Rgl::TGuardBase &dcGuard = Rgl::make_guard(ReleaseDC, hWND, hDC);
   if (HGLRC glContext = wglCreateContext(hDC)) {
      if (shareList && !wglShareLists(shareList->fPimpl->fGLContext, glContext)) {
         wglDeleteContext(glContext);
         Error("TGLContext::SetContext", "Context sharing failed!");
         throw std::runtime_error("Context sharing failed");
      }
      fPimpl->fHWND = hWND;
      fPimpl->fHDC = hDC;
      fPimpl->fGLContext = glContext;
   } else {
      Error("TGLContext::SetContext", "wglCreateContext failed");
      throw std::runtime_error("wglCreateContext failed");
   }

   //Register context for "parent" gl-device.
   fValid = kTRUE;
   fDevice->AddContext(this);
   TGLContextPrivate::RegisterContext(this);

   dcGuard.Stop();
   safe_ptr.release();
}

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
      return Bool_t(gROOT->ProcessLineFast(Form("((TGLContext *)0x%lx)->MakeCurrent()", this)));
   else {
      Bool_t rez = wglMakeCurrent(fPimpl->fHDC, fPimpl->fGLContext);
      if (rez) {
         if (!fgGlewInitDone)
            GlewInit();
         fIdentity->DeleteGLResources();
      }
      return rez;
   }
}

//______________________________________________________________________________
Bool_t TGLContext::ClearCurrent()
{
   //Reset current context.
   return wglMakeCurrent(0, 0);
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
      gROOT->ProcessLineFast(Form("((TGLContext *)0x%lx)->SwapBuffers()", this));
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
      gROOT->ProcessLineFast(Form("((TGLContext *)0x%lx)->Release()", this));
      return;
   }

   if (fPimpl->fHWND)
      ReleaseDC(fPimpl->fHWND, fPimpl->fHDC);

   TGLContextPrivate::RemoveContext(this);
   wglDeleteContext(fPimpl->fGLContext);
   fValid = kFALSE;
}

#elif defined(R__HAS_COCOA)

//______________________________________________________________________________
void TGLContext::SetContext(TGLWidget *widget, const TGLContext *shareList)
{
   //This function is public only for calls via gROOT and called from ctor.
   if (!fFromCtor) {
      Error("TGLContext::SetContext", "SetContext must be called only from ctor");
      return;
   }

   std::auto_ptr<TGLContextPrivate> safe_ptr(fPimpl = new TGLContextPrivate);

   fPimpl->fGLContext = gVirtualX->CreateOpenGLContext(widget->GetId(), shareList ? shareList->fPimpl->fGLContext : 0);
   fPimpl->fWindowID = widget->GetId();

   fValid = kTRUE;
   fDevice->AddContext(this);
   TGLContextPrivate::RegisterContext(this);

   safe_ptr.release();
}

//______________________________________________________________________________
Bool_t TGLContext::MakeCurrent()
{
   //If context is valid (TGLPaintDevice, for which context was created still exists),
   //make it current.

   if (!fValid) {
      Error("TGLContext::MakeCurrent", "This context is invalid.");
      return kFALSE;
   }

   const Bool_t rez = gVirtualX->MakeOpenGLContextCurrent(fPimpl->fGLContext, fPimpl->fWindowID);
   if (rez) {
      if (!fgGlewInitDone)
         GlewInit();
      fIdentity->DeleteGLResources();
      
   }

   return rez;
}

//______________________________________________________________________________
Bool_t TGLContext::ClearCurrent()
{
   //Reset current context.
   return kFALSE;
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

   gVirtualX->FlushOpenGLBuffer(fPimpl->fGLContext);  
}

//______________________________________________________________________________
void TGLContext::Release()
{
   //Make the context invalid and free resources.

   TGLContextPrivate::RemoveContext(this);
   gVirtualX->DeleteOpenGLContext(fPimpl->fGLContext);
   fValid = kFALSE;
}

//==============================================================================
#else // X11
//==============================================================================

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
   fPimpl->fWindowID = widget->GetId();

   fValid = kTRUE;
   fDevice->AddContext(this);
   TGLContextPrivate::RegisterContext(this);

   safe_ptr.release();
}

//______________________________________________________________________________
Bool_t TGLContext::MakeCurrent()
{
   //If context is valid (TGLPaintDevice, for which context was created still exists),
   //make it current.

   if (!fValid) {
      Error("TGLContext::MakeCurrent", "This context is invalid.");
      return kFALSE;
   }

   if (fPimpl->fWindowID != 0) {
      const Bool_t rez = glXMakeCurrent(fPimpl->fDpy, fPimpl->fWindowID,
                                        fPimpl->fGLContext);
      if (rez) {
         if (!fgGlewInitDone)
            GlewInit();
         fIdentity->DeleteGLResources();
      }
      return rez;
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGLContext::ClearCurrent()
{
   //Reset current context.
   return glXMakeCurrent(fPimpl->fDpy, None, 0);
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

   if (fPimpl->fWindowID != 0)
      glXSwapBuffers(fPimpl->fDpy, fPimpl->fWindowID);
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

//==============================================================================
#endif
//==============================================================================

//______________________________________________________________________________
TGLContext::~TGLContext()
{
   //TGLContext dtor. If it's called before TGLPaintDevice's dtor
   //(context is valid) resource will be freed and context
   //un-registered.
   if (fValid) {
      Release();
      fDevice->RemoveContext(this);
   }

   fIdentity->Release(this);

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
//
// Identifier of a shared GL-context.
// Objects shared among GL-contexts include:
// display-list definitions, texture objects and shader programs.

ClassImp(TGLContextIdentity)

TGLContextIdentity* TGLContextIdentity::fgDefaultIdentity = new TGLContextIdentity;

//______________________________________________________________________________
TGLContextIdentity::TGLContextIdentity():
fFontManager(0), fCnt(0), fClientCnt(0)
{
   // Constructor.

}

//______________________________________________________________________________
TGLContextIdentity::~TGLContextIdentity()
{
   // Destructor.

   if (fFontManager) delete fFontManager;
}

//______________________________________________________________________________
void TGLContextIdentity::AddRef(TGLContext* ctx)
{
   //Add context ctx to the list of references.
   ++fCnt;
   fCtxs.push_back(ctx);
}

//______________________________________________________________________________
void TGLContextIdentity::Release(TGLContext* ctx)
{
   //Remove context ctx from the list of references.
   CtxList_t::iterator i = std::find(fCtxs.begin(), fCtxs.end(), ctx);
   if (i != fCtxs.end())
   {
      fCtxs.erase(i);
      --fCnt;
      CheckDestroy();
   }
   else
   {
      Error("TGLContextIdentity::Release", "unregistered context.");
   }
}

//______________________________________________________________________________
void TGLContextIdentity::RegisterDLNameRangeToWipe(UInt_t base, Int_t size)
{
   //Remember dl range for deletion in next MakeCurrent or dtor execution.
   fDLTrash.push_back(DLRange_t(base, size));
}

//______________________________________________________________________________
void TGLContextIdentity::DeleteGLResources()
{
   //Delete GL resources registered for destruction.

   if (!fDLTrash.empty())
   {
      for (DLTrashIt_t it = fDLTrash.begin(), e = fDLTrash.end(); it != e; ++it)
         glDeleteLists(it->first, it->second);
      fDLTrash.clear();
   }

   if (fFontManager)
      fFontManager->ClearFontTrash();
}

//______________________________________________________________________________
TGLContextIdentity* TGLContextIdentity::GetCurrent()
{
   //Find identitfy of current context. Static.
   TGLContext* ctx = TGLContext::GetCurrent();
   return ctx ? ctx->GetIdentity() : 0;
}

//______________________________________________________________________________
TGLContextIdentity* TGLContextIdentity::GetDefaultIdentity()
{
   //Get identity of a default Gl context. Static.
   if (fgDefaultIdentity == 0)
      fgDefaultIdentity = new TGLContextIdentity;
   return fgDefaultIdentity;
}

//______________________________________________________________________________
TGLContext* TGLContextIdentity::GetDefaultContextAny()
{
   //Get the first GL context with the default identity.
   //Can return zero, but that's OK, too. Static.
   if (fgDefaultIdentity == 0 || fgDefaultIdentity->fCtxs.empty())
      return 0;
   return fgDefaultIdentity->fCtxs.front();
}

//______________________________________________________________________________
TGLFontManager* TGLContextIdentity::GetFontManager()
{
   //Get the free-type font-manager associated with this context-identity.
   if(!fFontManager) fFontManager = new TGLFontManager();
   return fFontManager;
}

//______________________________________________________________________________
void TGLContextIdentity::CheckDestroy()
{
   //Private function called when reference count is reduced.
   if (fCnt <= 0 && fClientCnt <= 0)
   {
      if (this == fgDefaultIdentity)
         fgDefaultIdentity = 0;
      delete this;
   }
}
