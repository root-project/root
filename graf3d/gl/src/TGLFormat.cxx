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

//______________________________________________________________________________
//
// Encapsulation of format / contents of an OpenGL buffer.

ClassImp(TGLFormat);

std::vector<Int_t> TGLFormat::fgAvailableSamples;

//______________________________________________________________________________
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
   //-multi-sampling depends on seeting of "OpenGL.Framebuffer.Multisample"
}

//______________________________________________________________________________
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
   //Define surface using options.
}

//______________________________________________________________________________
TGLFormat::~TGLFormat()
{
   //Destructor.
}

//______________________________________________________________________________
Bool_t TGLFormat::operator == (const TGLFormat &rhs)const
{
   //Check if two formats are equal.
   return fDoubleBuffered == rhs.fDoubleBuffered && fDepthSize == rhs.fDepthSize &&
          fAccumSize == rhs.fAccumSize && fStencilSize == rhs.fStencilSize;
}

//______________________________________________________________________________
Bool_t TGLFormat::operator != (const TGLFormat &rhs)const
{
   //Check for non-equality.
   return !(*this == rhs);
}

//______________________________________________________________________________
Int_t TGLFormat::GetDepthSize()const
{
   //Get the size of depth buffer.
   return fDepthSize;
}

//______________________________________________________________________________
void TGLFormat::SetDepthSize(Int_t depth)
{
   //Set the size of color buffer.
   assert(depth);
   fDepthSize = depth;
}

//______________________________________________________________________________
Bool_t TGLFormat::HasDepth()const
{
   //Check, if this surface has depth buffer.
   return GetDepthSize() != 0;
}

//______________________________________________________________________________
Int_t TGLFormat::GetStencilSize()const
{
   //Get the size of stencil buffer.
   return fStencilSize;
}

//______________________________________________________________________________
void TGLFormat::SetStencilSize(Int_t stencil)
{
   //Set the size of stencil buffer.
   assert(stencil);
   fStencilSize = stencil;
}

//______________________________________________________________________________
Bool_t TGLFormat::HasStencil()const
{
   //Check, if this surface has stencil buffer.
   return GetStencilSize() != 0;
}

//______________________________________________________________________________
Int_t TGLFormat::GetAccumSize()const
{
   //Get the size of accum buffer.
   return fAccumSize;
}

//______________________________________________________________________________
void TGLFormat::SetAccumSize(Int_t accum)
{
   //Set the size of accum buffer.
   assert(accum);
   fAccumSize = accum;
}

//______________________________________________________________________________
Bool_t TGLFormat::HasAccumBuffer()const
{
   //Check, if this surface has accumulation buffer.
   return GetAccumSize() != 0;
}

//______________________________________________________________________________
Bool_t TGLFormat::IsDoubleBuffered()const
{
   //Check, if the surface is double buffered.
   return fDoubleBuffered;
}

//______________________________________________________________________________
void TGLFormat::SetDoubleBuffered(Bool_t db)
{
   //Set the surface as double/single buffered.
   fDoubleBuffered = db;
}

//______________________________________________________________________________
Bool_t TGLFormat::IsStereo()const
{
   //Check, if the surface is stereo buffered.
   return fStereo;
}

//______________________________________________________________________________
void TGLFormat::SetStereo(Bool_t db)
{
   //Set the surface as stereo/non-stereo buffered.
   fStereo = db;
}

//______________________________________________________________________________
Int_t TGLFormat::GetSamples()const
{
   //Get the number of samples for multi-sampling.
   return fSamples;
}

//______________________________________________________________________________
void TGLFormat::SetSamples(Int_t samples)
{
   //Set the number of samples for multi-sampling.
   fSamples = samples;
}

//______________________________________________________________________________
Bool_t TGLFormat::HasMultiSampling()const
{
   //Check, if multi-sampling is requred.
   return fSamples != 0;
}

//______________________________________________________________________________
Int_t TGLFormat::GetDefaultSamples()
{
   // Return default number of samples for multi-sampling.

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

//______________________________________________________________________________
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
