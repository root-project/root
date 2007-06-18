// @(#)root/gl:$Name:  $:$Id: TGLContextPrivate.cxx,v 1.1 2007/06/18 07:02:16 brun Exp $
// Author:  Timur Pocheptsov, Jun 2007

#ifndef WIN32
#include <GL/glx.h>
#endif

#include "TGLContextPrivate.h"

//______________________________________________________________________________
void TGLContextPrivate::RegisterContext(TGLContext *ctx)
{
   //Register gl-context to find it later as current (GetCurrentContext)
   if (ctx->IsValid())
      fContexts[ctx->fPimpl->fGLContext] = ctx;
}

//______________________________________________________________________________
void TGLContextPrivate::RemoveContext(TGLContext *ctx)
{
   //Un-register deleted context.
   if (ctx->IsValid())
      fContexts.erase(ctx->fPimpl->fGLContext);
}

#ifdef WIN32

std::map<HGLRC, TGLContext *> TGLContextPrivate::fContexts;

//______________________________________________________________________________
TGLContext *TGLContextPrivate::GetCurrentContext()
{
   //Ask wgl what HGLRC is current and look up corresponding TGLContext.
   HGLRC glContext = wglGetCurrentContext();
   std::map<HGLRC, TGLContext *>::const_iterator it = fContexts.find(glContext);

   if (it != fContexts.end())
      return it->second;

   return 0;
}

#else

std::map<GLXContext, TGLContext *> TGLContextPrivate::fContexts;

//______________________________________________________________________________
TGLContext *TGLContextPrivate::GetCurrentContext()
{
   //Ask wgl what HGLRC is current and look up corresponding TGLContext.
   GLXContext glContext = glXGetCurrentContext();
   std::map<GLXContext, TGLContext *>::const_iterator it = fContexts.find(glContext);

   if (it != fContexts.end())
      return it->second;

   return 0;
}

#endif
