// @(#)root/gl:$Name:  $:$Id: TGLContextPrivate.cxx,v 1.2 2007/06/18 10:58:34 brun Exp $
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
      fgContexts[ctx->fPimpl->fGLContext] = ctx;
}

//______________________________________________________________________________
void TGLContextPrivate::RemoveContext(TGLContext *ctx)
{
   //Un-register deleted context.
   if (ctx->IsValid())
      fgContexts.erase(ctx->fPimpl->fGLContext);
}

#ifdef WIN32

std::map<HGLRC, TGLContext *> TGLContextPrivate::fgContexts;

//______________________________________________________________________________
TGLContext *TGLContextPrivate::GetCurrentContext()
{
   //Ask wgl what HGLRC is current and look up corresponding TGLContext.
   HGLRC glContext = wglGetCurrentContext();
   std::map<HGLRC, TGLContext *>::const_iterator it = fgContexts.find(glContext);

   if (it != fgContexts.end())
      return it->second;

   return 0;
}

#else

std::map<GLXContext, TGLContext *> TGLContextPrivate::fgContexts;

//______________________________________________________________________________
TGLContext *TGLContextPrivate::GetCurrentContext()
{
   //Ask wgl what HGLRC is current and look up corresponding TGLContext.
   GLXContext glContext = glXGetCurrentContext();
   std::map<GLXContext, TGLContext *>::const_iterator it = fgContexts.find(glContext);

   if (it != fgContexts.end())
      return it->second;

   return 0;
}

#endif
