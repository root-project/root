// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov, Jun 2007

#include "TGLContextPrivate.h"

#ifdef R__HAS_COCOA
#include <cassert>

#include "TVirtualX.h"
#include "TError.h"
#endif

////////////////////////////////////////////////////////////////////////////////
///Register gl-context to find it later as current (GetCurrentContext)

void TGLContextPrivate::RegisterContext(TGLContext *ctx)
{
   if (ctx->IsValid())
      fgContexts[ctx->fPimpl->fGLContext] = ctx;
}

////////////////////////////////////////////////////////////////////////////////
///Un-register deleted context.

void TGLContextPrivate::RemoveContext(TGLContext *ctx)
{
   if (ctx->IsValid())
      fgContexts.erase(ctx->fPimpl->fGLContext);
}

#ifdef WIN32

std::map<HGLRC, TGLContext *> TGLContextPrivate::fgContexts;

////////////////////////////////////////////////////////////////////////////////
///Ask wgl what HGLRC is current and look up corresponding TGLContext.

TGLContext *TGLContextPrivate::GetCurrentContext()
{
   HGLRC glContext = wglGetCurrentContext();
   std::map<HGLRC, TGLContext *>::const_iterator it = fgContexts.find(glContext);

   if (it != fgContexts.end())
      return it->second;

   return 0;
}

#elif defined(R__HAS_COCOA)

std::map<Handle_t, TGLContext *> TGLContextPrivate::fgContexts;

////////////////////////////////////////////////////////////////////////////////

TGLContext *TGLContextPrivate::GetCurrentContext()
{
   const Handle_t ctxID = gVirtualX->GetCurrentOpenGLContext();
   if (ctxID) {
      assert(fgContexts.find(ctxID) != fgContexts.end() && "GetCurrentContext, context id is unknown");
      return fgContexts[ctxID];
   }

   //Else part - error message was issued already by TGCocoa.
   return 0;
}

#else

std::map<GLXContext, TGLContext *> TGLContextPrivate::fgContexts;

////////////////////////////////////////////////////////////////////////////////
///Ask wgl what HGLRC is current and look up corresponding TGLContext.

TGLContext *TGLContextPrivate::GetCurrentContext()
{
   GLXContext glContext = glXGetCurrentContext();
   std::map<GLXContext, TGLContext *>::const_iterator it = fgContexts.find(glContext);

   if (it != fgContexts.end())
      return it->second;

   return 0;
}

#endif
