// @(#)root/eve:$Id$
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

Bool_t TGLFBO::fgRescaleToPow2 = kTRUE; // For ATI.

TGLFBO::TGLFBO() :
   fFrameBuffer  (0),
   fColorTexture (0),
   fDepthBuffer  (0),
   fW (-1),
   fH (-1),
   fIsRescaled (kFALSE),
   fWScale     (1),
   fHScale     (1)
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
void TGLFBO::Init(int w, int h)
{
   // Acquire GL resources for given width and height.

   static const std::string eh("TGLFBO::Init ");

   if (!GLEW_VERSION_1_5)
   {
      throw std::runtime_error(eh + "GL version 1.5 required for FBO.");
   }

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

   if (fFrameBuffer != 0)
   {
      if (fW == w || fH == h)
         return;
      Release();
   }

   Int_t maxSize;
   glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE_EXT, &maxSize);
   if (w > maxSize || h > maxSize)
   {
      throw std::runtime_error(eh + Form("maximum size supported by GL implementation is %d.", maxSize));
   }

   fW = w; fH = h;

   glGenFramebuffersEXT (1, &fFrameBuffer);
   glGenTextures        (1, &fColorTexture);
   glGenRenderbuffersEXT(1, &fDepthBuffer);
   // glGenRenderbuffersEXT(1, &fStencilBuffer);

   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fFrameBuffer);

   // initialize color texture
   glBindTexture(GL_TEXTURE_2D, fColorTexture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, fW, fH, 0, GL_RGB,
                GL_UNSIGNED_BYTE, NULL);

   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                             GL_TEXTURE_2D, fColorTexture, 0);

   // initialize depth renderbuffer
   glBindRenderbufferEXT   (GL_RENDERBUFFER_EXT, fDepthBuffer);
   glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, fW, fH);

   glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                                GL_RENDERBUFFER_EXT, fDepthBuffer);

   /*
   // initialize stencil renderbuffer
   glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, fStencilBuffer);
   glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_STENCIL_INDEX, fW, fH);

   glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT,
   GL_RENDERBUFFER_EXT, fStencilBuffer);
   */

   //-------------------------

   GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);

   glBindFramebufferEXT (GL_FRAMEBUFFER_EXT,  0);
   glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0); // ? is needed
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
         throw std::runtime_error(eh + "Constructed TGLFBO is crap, fix code in TGLFBO class.");
         break;
   }
}

//______________________________________________________________________________
void TGLFBO::Release()
{
   // Release the allocated GL resources.

   glDeleteFramebuffersEXT (1, &fFrameBuffer);
   glDeleteTextures        (1, &fColorTexture);
   glDeleteRenderbuffersEXT(1, &fDepthBuffer);
   //glDeleteRenderbuffersEXT(1, &fStencilBuffer);

   fColorTexture = fFrameBuffer = fDepthBuffer = 0;
   fW = fH = -1;
}

//______________________________________________________________________________
void TGLFBO::Bind()
{
   // Bind the frame-buffer object.

   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fFrameBuffer);
}

//______________________________________________________________________________
void TGLFBO::Unbind()
{
   // Unbind the frame-buffer object.

   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

//______________________________________________________________________________
void TGLFBO::BindTexture()
{
   // Bind texture.

   glPushAttrib(GL_TEXTURE_BIT);
   glBindTexture(GL_TEXTURE_2D, fColorTexture);
   glEnable(GL_TEXTURE_2D);

   glMatrixMode(GL_TEXTURE);
   glPushMatrix();
   glScalef(fWScale, fHScale, 1);
   glMatrixMode(GL_MODELVIEW);
}

//______________________________________________________________________________
void TGLFBO::UnbindTexture()
{
   // Unbind texture.

   glMatrixMode(GL_TEXTURE);
   glPopMatrix();
   glMatrixMode(GL_MODELVIEW);

   glPopAttrib();
}
