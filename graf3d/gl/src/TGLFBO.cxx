// @(#)root/gl:$Id$
// Author: Matevz Tadel, Aug 2009

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLFBO.h"
#include <TMath.h>
#include <TString.h>
#include <TError.h>

#include <GL/glew.h>

#include <stdexcept>

//______________________________________________________________________________
//
// Frame-buffer object.
//
// Requires GL-1.5.
//
// Taken from Gled project, see:
//   http://www.gled.org/cgi-bin/viewcvs.cgi/trunk/libsets/GledCore/Pupils/
// See also:
//   http://www.opengl.org/registry/specs/EXT/framebuffer_object.txt

ClassImp(TGLFBO);

Bool_t TGLFBO::fgRescaleToPow2       = kTRUE; // For ATI.
Bool_t TGLFBO::fgMultiSampleNAWarned = kFALSE;

//______________________________________________________________________________
TGLFBO::TGLFBO() :
   fFrameBuffer  (0),
   fColorTexture (0),
   fDepthBuffer  (0),
   fMSFrameBuffer(0),
   fMSColorBuffer(0),
   fW (-1),
   fH (-1),
   fReqW (-1),
   fReqH (-1),
   fMSSamples  (0),
   fMSCoverageSamples (0),
   fWScale     (1),
   fHScale     (1),
   fIsRescaled (kFALSE)
{
   // Constructor.
}

//______________________________________________________________________________
TGLFBO::~TGLFBO()
{
   // Destructor.

   Release();
}

//______________________________________________________________________________
void TGLFBO::Init(int w, int h, int ms_samples)
{
   // Acquire GL resources for given width, height and number of
   // multi-sampling samples.

   static const std::string eh("TGLFBO::Init ");

   // Should be replaced with ARB_framebuffer_object (SLC6).
   if (!GLEW_EXT_framebuffer_object)
   {
      throw std::runtime_error(eh + "GL_EXT_framebuffer_object extension required for FBO.");
   }

   fReqW = w; fReqH = h;

   fIsRescaled = kFALSE;
   if (fgRescaleToPow2)
   {
      Int_t nw = 1 << TMath::CeilNint(TMath::Log2(w));
      Int_t nh = 1 << TMath::CeilNint(TMath::Log2(h));
      if (nw != w || nh != h)
      {
         fWScale = ((Float_t)w) / nw;
         fHScale = ((Float_t)h) / nh;
         w = nw; h = nh;
         fIsRescaled = kTRUE;
      }
   }

   if (ms_samples > 0 && ! GLEW_EXT_framebuffer_multisample)
   {
      if (!fgMultiSampleNAWarned)
      {
         Info(eh.c_str(), "GL implementation does not support multi-sampling for FBOs.");
         fgMultiSampleNAWarned = kTRUE;
      }
      ms_samples = 0;
   }

   if (fFrameBuffer != 0)
   {
      if (fW == w && fH == h && fMSSamples == ms_samples)
         return;
      Release();
   }

   Int_t maxSize;
   glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE_EXT, &maxSize);
   if (w > maxSize || h > maxSize)
   {
      throw std::runtime_error(eh + Form("maximum size supported by GL implementation is %d.", maxSize));
   }

   fW = w; fH = h; fMSSamples = ms_samples;

   if (fMSSamples > 0)
   {
      if (GLEW_NV_framebuffer_multisample_coverage)
      {
         GLint n_modes;
         glGetIntegerv(GL_MAX_MULTISAMPLE_COVERAGE_MODES_NV, &n_modes);
         GLint *modes = new GLint[2*n_modes];
         glGetIntegerv(GL_MULTISAMPLE_COVERAGE_MODES_NV, modes);

         for (int i = 0; i < n_modes; ++i)
         {
            if (modes[i*2+1] == fMSSamples && modes[i*2] > fMSCoverageSamples)
               fMSCoverageSamples = modes[i*2];
         }

         delete [] modes;
      }
      if (gDebug > 0) {
         Info(eh.c_str(), "InitMultiSample coverage_samples=%d, color_samples=%d.", fMSCoverageSamples, fMSSamples);
      }
      InitMultiSample();
   }
   else
   {
      if (gDebug > 0) {
         Info(eh.c_str(), "InitStandard (no multi-sampling).");
      }
      InitStandard();
   }

   GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);

   glBindFramebufferEXT (GL_FRAMEBUFFER_EXT,  0);
   glBindTexture        (GL_TEXTURE_2D,       0);

   switch (status)
   {
      case GL_FRAMEBUFFER_COMPLETE_EXT:
         if (gDebug > 0)
            printf("%sConstructed TGLFBO ... all fine.\n", eh.c_str());
         break;
      case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
         Release();
         throw std::runtime_error(eh + "Constructed TGLFBO not supported, choose different formats.");
         break;
      default:
         Release();
         throw std::runtime_error(eh + "Constructed TGLFBO is not complete, unexpected error.");
         break;
   }
}

//______________________________________________________________________________
void TGLFBO::Release()
{
   // Release the allocated GL resources.

   glDeleteFramebuffersEXT (1, &fFrameBuffer);
   glDeleteRenderbuffersEXT(1, &fDepthBuffer);

   if (fMSFrameBuffer) glDeleteFramebuffersEXT (1, &fMSFrameBuffer);
   if (fMSColorBuffer) glDeleteRenderbuffersEXT(1, &fMSColorBuffer);
   if (fColorTexture)  glDeleteTextures        (1, &fColorTexture);

   fW = fH = -1; fMSSamples = fMSCoverageSamples = 0;
   fFrameBuffer = fColorTexture = fDepthBuffer = fMSFrameBuffer = fMSColorBuffer = 0;

}

//______________________________________________________________________________
void TGLFBO::Bind()
{
   // Bind the frame-buffer object.

   if (fMSSamples > 0) {
      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fMSFrameBuffer);
      // On by default
      //   glEnable(GL_MULTISAMPLE);
      // Experimenting:
      //   glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE_ARB);
      //   glEnable(GL_SAMPLE_COVERAGE_ARB);
   } else {
      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fFrameBuffer);
   }
}

//______________________________________________________________________________
void TGLFBO::Unbind()
{
   // Unbind the frame-buffer object.

   if (fMSSamples > 0)
   {
      glBindFramebufferEXT(GL_READ_FRAMEBUFFER_EXT, fMSFrameBuffer);
      glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER_EXT, fFrameBuffer);
      glBlitFramebufferEXT(0, 0, fW, fH, 0, 0, fW, fH, GL_COLOR_BUFFER_BIT, GL_NEAREST);
   }

   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

//______________________________________________________________________________
void TGLFBO::BindTexture()
{
   // Bind texture.

   glPushAttrib(GL_TEXTURE_BIT);
   glBindTexture(GL_TEXTURE_2D, fColorTexture);
   glEnable(GL_TEXTURE_2D);

   if (fIsRescaled)
   {
      glMatrixMode(GL_TEXTURE);
      glPushMatrix();
      glScalef(fWScale, fHScale, 1);
      glMatrixMode(GL_MODELVIEW);
   }
}

//______________________________________________________________________________
void TGLFBO::UnbindTexture()
{
   // Unbind texture.

   if (fIsRescaled)
   {
      glMatrixMode(GL_TEXTURE);
      glPopMatrix();
      glMatrixMode(GL_MODELVIEW);
   }

   glPopAttrib();
}

//______________________________________________________________________________
void TGLFBO::SetAsReadBuffer()
{
   glBindFramebufferEXT(GL_READ_FRAMEBUFFER_EXT, fFrameBuffer);
}

//==============================================================================

//______________________________________________________________________________
void TGLFBO::InitStandard()
{
   glGenFramebuffersEXT(1, &fFrameBuffer);
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fFrameBuffer);

   fDepthBuffer  = CreateAndAttachRenderBuffer(GL_DEPTH_COMPONENT24, GL_DEPTH_ATTACHMENT);
   fColorTexture = CreateAndAttachColorTexture();
}

//______________________________________________________________________________
void TGLFBO::InitMultiSample()
{
   glGenFramebuffersEXT(1, &fMSFrameBuffer);
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fMSFrameBuffer);

   fMSColorBuffer = CreateAndAttachRenderBuffer(GL_RGBA8,             GL_COLOR_ATTACHMENT0);
   fDepthBuffer   = CreateAndAttachRenderBuffer(GL_DEPTH_COMPONENT24, GL_DEPTH_ATTACHMENT);
   // fDepthBuffer   = CreateAndAttachRenderBuffer(GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL_ATTACHMENT);

   glGenFramebuffersEXT(1, &fFrameBuffer);
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fFrameBuffer);

   fColorTexture = CreateAndAttachColorTexture();
}

//______________________________________________________________________________
UInt_t TGLFBO::CreateAndAttachRenderBuffer(Int_t format, Int_t type)
{
   UInt_t id = 0;

   glGenRenderbuffersEXT(1, &id);
   glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, id);

   if (fMSSamples > 0)
   {
      if (fMSCoverageSamples > 0)
         glRenderbufferStorageMultisampleCoverageNV(GL_RENDERBUFFER_EXT, fMSCoverageSamples, fMSSamples, format, fW, fH);
      else
         glRenderbufferStorageMultisampleEXT(GL_RENDERBUFFER_EXT, fMSSamples, format, fW, fH);
   }
   else
   {
      glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, format, fW, fH);
   }

   glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, type, GL_RENDERBUFFER_EXT, id);

   return id;
}

//______________________________________________________________________________
UInt_t TGLFBO::CreateAndAttachColorTexture()
{
   // Initialize color-texture and attach it to current FB.

   UInt_t id = 0;

   glGenTextures(1, &id);

   glBindTexture(GL_TEXTURE_2D, id);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, fW, fH, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, NULL);

   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                             GL_TEXTURE_2D, id, 0);

   return id;
}

//______________________________________________________________________________
Bool_t TGLFBO::GetRescaleToPow2()
{
   // Return state of fgRescaleToPow2 static member.

   return fgRescaleToPow2;
}

//______________________________________________________________________________
void TGLFBO::SetRescaleToPow2(Bool_t r)
{
   // Set state of fgRescaleToPow2 static member.
   // Default is kTRUE as this works better on older hardware, especially ATI.

   fgRescaleToPow2 = r;
}
