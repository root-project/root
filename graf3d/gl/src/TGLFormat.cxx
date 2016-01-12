// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <cassert>
#include <algorithm>
#include <set>

#include "TGLFormat.h"
#include "TGLWSIncludes.h"
#include "TGLWidget.h"

#include "TEnv.h"
#include "TError.h"
#include "TVirtualX.h"
#include "RConfigure.h"

/** \class TGLFormat
\ingroup opengl
Encapsulation of format / contents of an OpenGL buffer.
*/

ClassImp(TGLFormat);

std::vector<Int_t> TGLFormat::fgAvailableSamples;

////////////////////////////////////////////////////////////////////////////////

TGLFormat::TGLFormat() :
   fDoubleBuffered(kTRUE),
   fStereo(kFALSE),
#ifdef WIN32
   fDepthSize(32),
#else
   // 16-bits needed for some virtual machines (VirtualBox) and Xming-mesa
   // (when running ssh from windows to linux).
   // All others seem to have 24-bit depth-buffers only and use this anyway.
   fDepthSize(16),
#endif
   fAccumSize(0),
   fStencilSize(8),
   fSamples(GetDefaultSamples())
{
   //Default ctor. Default surface is:
   //-double buffered
   //-RGBA
   //-with depth buffer
   //-no accumulation buffer
   //-with stencil
   //-multi-sampling depends on setting of "OpenGL.Framebuffer.Multisample"
}

////////////////////////////////////////////////////////////////////////////////
///Define surface using options.

TGLFormat::TGLFormat(Rgl::EFormatOptions opt) :
   fDoubleBuffered(opt & Rgl::kDoubleBuffer),
   fStereo(kFALSE),
#ifdef WIN32
   fDepthSize(opt & Rgl::kDepth ? 32 : 0),
#else
   fDepthSize(opt & Rgl::kDepth ? 16 : 0),//FIXFIX
#endif
   fAccumSize(opt & Rgl::kAccum ? 8 : 0),     //I've never tested accumulation buffer size.
   fStencilSize(opt & Rgl::kStencil ? 8 : 0), //I've never tested stencil buffer size.
   fSamples(opt & Rgl::kMultiSample ? GetDefaultSamples() : 0)
{
}

////////////////////////////////////////////////////////////////////////////////
///Destructor.

TGLFormat::~TGLFormat()
{
}

////////////////////////////////////////////////////////////////////////////////
///Check if two formats are equal.

Bool_t TGLFormat::operator == (const TGLFormat &rhs)const
{
   return fDoubleBuffered == rhs.fDoubleBuffered && fDepthSize == rhs.fDepthSize &&
          fAccumSize == rhs.fAccumSize && fStencilSize == rhs.fStencilSize;
}

////////////////////////////////////////////////////////////////////////////////
///Check for non-equality.

Bool_t TGLFormat::operator != (const TGLFormat &rhs)const
{
   return !(*this == rhs);
}

////////////////////////////////////////////////////////////////////////////////
///Get the size of depth buffer.

Int_t TGLFormat::GetDepthSize()const
{
   return fDepthSize;
}

////////////////////////////////////////////////////////////////////////////////
///Set the size of color buffer.

void TGLFormat::SetDepthSize(Int_t depth)
{
   assert(depth);
   fDepthSize = depth;
}

////////////////////////////////////////////////////////////////////////////////
///Check, if this surface has depth buffer.

Bool_t TGLFormat::HasDepth()const
{
   return GetDepthSize() != 0;
}

////////////////////////////////////////////////////////////////////////////////
///Get the size of stencil buffer.

Int_t TGLFormat::GetStencilSize()const
{
   return fStencilSize;
}

////////////////////////////////////////////////////////////////////////////////
///Set the size of stencil buffer.

void TGLFormat::SetStencilSize(Int_t stencil)
{
   assert(stencil);
   fStencilSize = stencil;
}

////////////////////////////////////////////////////////////////////////////////
///Check, if this surface has stencil buffer.

Bool_t TGLFormat::HasStencil()const
{
   return GetStencilSize() != 0;
}

////////////////////////////////////////////////////////////////////////////////
///Get the size of accum buffer.

Int_t TGLFormat::GetAccumSize()const
{
   return fAccumSize;
}

////////////////////////////////////////////////////////////////////////////////
///Set the size of accum buffer.

void TGLFormat::SetAccumSize(Int_t accum)
{
   assert(accum);
   fAccumSize = accum;
}

////////////////////////////////////////////////////////////////////////////////
///Check, if this surface has accumulation buffer.

Bool_t TGLFormat::HasAccumBuffer()const
{
   return GetAccumSize() != 0;
}

////////////////////////////////////////////////////////////////////////////////
///Check, if the surface is double buffered.

Bool_t TGLFormat::IsDoubleBuffered()const
{
   return fDoubleBuffered;
}

////////////////////////////////////////////////////////////////////////////////
///Set the surface as double/single buffered.

void TGLFormat::SetDoubleBuffered(Bool_t db)
{
   fDoubleBuffered = db;
}

////////////////////////////////////////////////////////////////////////////////
///Check, if the surface is stereo buffered.

Bool_t TGLFormat::IsStereo()const
{
   return fStereo;
}

////////////////////////////////////////////////////////////////////////////////
///Set the surface as stereo/non-stereo buffered.

void TGLFormat::SetStereo(Bool_t db)
{
   fStereo = db;
}

////////////////////////////////////////////////////////////////////////////////
///Get the number of samples for multi-sampling.

Int_t TGLFormat::GetSamples()const
{
   return fSamples;
}

////////////////////////////////////////////////////////////////////////////////
///Set the number of samples for multi-sampling.

void TGLFormat::SetSamples(Int_t samples)
{
   fSamples = samples;
}

////////////////////////////////////////////////////////////////////////////////
///Check, if multi-sampling is required.

Bool_t TGLFormat::HasMultiSampling()const
{
   return fSamples != 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return default number of samples for multi-sampling.

Int_t TGLFormat::GetDefaultSamples()
{
   Int_t req = gEnv->GetValue("OpenGL.Framebuffer.Multisample", 0);

   // Avoid query of available multi-sample modes when not required.
   // Over ssh, SLC5 lies about supporting the GLX_SAMPLES_ARB
   // extension and then dies horribly when the query is made.
   if (req == 0) {
      return 0;
   }

   if (fgAvailableSamples.empty())
      InitAvailableSamples();

   std::vector<Int_t>::iterator i = fgAvailableSamples.begin();
   while (i != fgAvailableSamples.end() - 1 && *i < req)
      ++i;

   if (*i != req) {
      Info("TGLFormat::GetDefaultSamples", "Requested multi-sampling %d not available, using %d. Adjusting default.", req, *i);
      gEnv->SetValue("OpenGL.Framebuffer.Multisample", *i);
   }

   return *i;
}

////////////////////////////////////////////////////////////////////////////////

void TGLFormat::InitAvailableSamples()
{
   std::set<Int_t> ns_set;
   ns_set.insert(0);

#ifdef WIN32

   // Missing implementation.
#elif defined(R__HAS_COCOA)
   ns_set.insert(8);
   ns_set.insert(16);
#else
   TGLWidget *widget = TGLWidget::CreateDummy();
   widget->MakeCurrent();

   if (GLXEW_ARB_multisample)
   {
      Display *dpy  = (Display*) gVirtualX->GetDisplay();
      XVisualInfo tmpl; tmpl.screen = gVirtualX->GetScreen();
      long mask = VisualScreenMask;
      int  numVisuals, use_gl, ms_ns;
      XVisualInfo *vis = XGetVisualInfo(dpy, mask, &tmpl, &numVisuals);
      for (int i = 0; i < numVisuals; i++)
      {
         if (glXGetConfig(dpy, &vis[i], GLX_USE_GL, &use_gl) == 0)
         {
            glXGetConfig(dpy, &vis[i], GLX_SAMPLES_ARB, &ms_ns);
            ns_set.insert(ms_ns);
         }
      }
      XFree(vis);
   }

   delete widget;
#endif

   fgAvailableSamples.reserve(ns_set.size());
   for (std::set<Int_t>::iterator i = ns_set.begin(); i != ns_set.end(); ++i)
   {
      fgAvailableSamples.push_back(*i);
   }
}
