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

#include "TGLIncludes.h"
#include "TGLContext.h"
#include "TGLWidget.h"
#include "TGLFormat.h"

ClassImp(TGLContext)

#ifdef WIN32

class TGLContext::TGLContextPrivate {
public:
   HWND  fHWND;
   HDC   fHDC;
   HGLRC fGLContext;
   TGLContextPrivate()
      : fHWND(0),
        fHDC(0),
        fGLContext(0)
   {
   }

private:
   TGLContextPrivate(const TGLContextPrivate &);
   TGLContextPrivate &operator = (const TGLContextPrivate &);
};

#else

class TGLContext::TGLContextPrivate {
public:
   Display     *fDpy;
   XVisualInfo *fVisualInfo;
   GLXContext   fGLContext;
   Int_t        fWindowIndex;

   TGLContextPrivate()
      : fDpy(0),
        fVisualInfo(0),
        fGLContext(0),
        fWindowIndex(-1)
   {
   }

private:
   TGLContextPrivate(const TGLContextPrivate &);
   TGLContextPrivate &operator = (const TGLContextPrivate &);
};

#endif

//______________________________________________________________________________
TGLContext::TGLContext(const TGLWidget *wid, const TGLFormat &request)
               : fGLFormat(request),
                 fPimpl(new TGLContextPrivate)
{
   // Constructor.

   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLContext *)0x%x)->SetContext((TGLWidget *)0x%x)", this, wid));
   } else
      SetContext(wid);
}

//______________________________________________________________________________
TGLContext::~TGLContext()
{
   // Destructor.

#ifdef WIN32
   ReleaseDC(fPimpl->fHWND, fPimpl->fHDC);
   wglDeleteContext(fPimpl->fGLContext);
#else
   glXDestroyContext(fPimpl->fDpy, fPimpl->fGLContext);
#endif
   delete fPimpl;
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

   class WDCGuard_t {
   private:
      HWND fHWND;
      HDC  fHDC;

      WDCGuard_t(const WDCGuard_t &);
      WDCGuard_t &operator = (const WDCGuard_t &);

   public:
      WDCGuard_t(HWND hWND, HDC hDC)
         : fHWND(hWND),
           fHDC(hDC)
      {
      }
      ~WDCGuard_t()
      {
         if (fHDC)
            ReleaseDC(fHWND, fHDC);
      }
      void Stop()
      {
         fHDC = 0;
      }
   };

   void fill_pfd(PIXELFORMATDESCRIPTOR *pfd, const TGLFormat &request)
   {
      pfd->nSize = sizeof(PIXELFORMATDESCRIPTOR);
      pfd->nVersion = 1;
      pfd->dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
      if (request.IsDoubleBuffered())
         pfd->dwFlags |= PFD_DOUBLEBUFFER;
      pfd->iPixelType = request.IsRGBA() ? PFD_TYPE_RGBA : PFD_TYPE_COLORINDEX;
      pfd->cColorBits = request.IsRGBA() ? request.GetRGBASize() : request.GetColorIndexSize();
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

      if (pfd.iPixelType == PFD_TYPE_RGBA)
         request.SetRGBASize(pfd.cColorBits);
      else
         request.SetColorIndexSize(pfd.cColorBits);

      if (pfd.cAccumBits)
         request.SetAccumSize(pfd.cAccumBits);

      if (pfd.cDepthBits)
         request.SetDepthSize(pfd.cDepthBits);

      if (pfd.cStencilBits)
         request.SetStencilSize(pfd.cStencilBits);
   }

}

//______________________________________________________________________________
void TGLContext::SetContext(const TGLWidget *widget)
{
   std::auto_ptr<TGLContextPrivate> safe_ptr(fPimpl);
   LayoutCompatible_t *trick =
      reinterpret_cast<LayoutCompatible_t *>(gVirtualX->GetWindowID(widget->GetWindowIndex()));
   HWND hWND = *trick->fPHwnd;
   HDC  hDC  = GetWindowDC(hWND);

   if (!hDC) {
      Error("TGLContext::SetContext", "GetWindowDC failed");
      throw std::runtime_error("GetWindowDC failed");
   }

   WDCGuard_t dcGuard(hWND, hDC);
   PIXELFORMATDESCRIPTOR pfd = {};
   fill_pfd(&pfd, fGLFormat);

   if (const Int_t pixIndex = ChoosePixelFormat(hDC, &pfd)) {
      check_pixel_format(pixIndex, hDC, fGLFormat);

      if (!SetPixelFormat(hDC, pixIndex, &pfd)) {
         Error("TGLContext::SetContext", "SetPixelFormat failed");
         throw std::runtime_error("SetPixelFormat failed");
      }

      if (HGLRC glContext = wglCreateContext(hDC)) {
         fPimpl->fHWND = hWND;
         fPimpl->fHDC = hDC;
         fPimpl->fGLContext = glContext;
      } else {
         Error("TGLContext::SetContext", "wglCreateContext failed");
         throw std::runtime_error("wglCreateContext failed");
      }

   } else {
      Error("TGLContext::SetContext", "ChoosePixelFormat failed");
      throw std::runtime_error("ChoosePixelFormat failed");
   }

   dcGuard.Stop();
   safe_ptr.release();
}

//______________________________________________________________________________
Bool_t TGLContext::MakeCurrent()
{
   Bool_t rez = kFALSE;

   if (!gVirtualX->IsCmdThread())
      rez = Bool_t(gROOT->ProcessLineFast(Form("((TGLContext *)0x%x)->MakeCurrent()", this)));
   else
      return Bool_t(wglMakeCurrent(fPimpl->fHDC, fPimpl->fGLContext));

   return rez;
}

//______________________________________________________________________________
void TGLContext::SwapBuffers()
{
   if (!gVirtualX->IsCmdThread())
      gROOT->ProcessLineFast(Form("((TGLContext *)0x%x)->SwapBuffers()", this));
   else
      wglSwapLayerBuffers(fPimpl->fHDC, WGL_SWAP_MAIN_PLANE);
}

//______________________________________________________________________________
const TGLFormat &TGLContext::GetPixelFormat()const
{
   return fGLFormat;
}

#else

//______________________________________________________________________________
void TGLContext::SetContext(const TGLWidget *widget)
{
   std::auto_ptr<TGLContextPrivate> safe_ptr(fPimpl);
   Display *dpy = static_cast<Display *>(widget->GetInnerData().first);
   XVisualInfo *visInfo = static_cast<XVisualInfo *>(widget->GetInnerData().second);
   GLXContext glCtx = glXCreateContext(dpy, visInfo, None, True);//

   if (!glCtx) {
      Error("TGLContext::SetContext", "glXCreateContext failed!");
      throw std::runtime_error("glXCreateContext failed!");
   }

   fPimpl->fDpy = dpy;
   fPimpl->fVisualInfo = visInfo;
   fPimpl->fGLContext = glCtx;
   fPimpl->fWindowIndex = widget->GetWindowIndex();

   safe_ptr.release();
}

//______________________________________________________________________________
Bool_t TGLContext::MakeCurrent()
{
   return glXMakeCurrent(fPimpl->fDpy, gVirtualX->GetWindowID(fPimpl->fWindowIndex), fPimpl->fGLContext);
}

//______________________________________________________________________________
void TGLContext::SwapBuffers()
{
   glXSwapBuffers(fPimpl->fDpy, gVirtualX->GetWindowID(fPimpl->fWindowIndex));
}

//______________________________________________________________________________
const TGLFormat &TGLContext::GetPixelFormat()const
{
   return fGLFormat;
}

#endif
