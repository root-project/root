// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov, Matevz Tadel, June 2007

#ifndef ROOT_TGLContextPrivate
#define ROOT_TGLContextPrivate

#include <map>

#include "TGLIncludes.h"
#include "TGLWSIncludes.h"
#include "TGLContext.h"
#include "RConfigure.h"
#include "GuiTypes.h"

#ifdef WIN32

class TGLContextPrivate {
public:
   HWND        fHWND;
   HDC         fHDC;
   HGLRC       fGLContext;

   TGLContextPrivate()
      : fHWND(0),
        fHDC(0),
        fGLContext(0)
   {
   }
   static void RegisterContext(TGLContext *ctx);
   static void RemoveContext(TGLContext *ctx);
   static TGLContext *GetCurrentContext();


private:
   TGLContextPrivate(const TGLContextPrivate &);
   TGLContextPrivate &operator = (const TGLContextPrivate &);

   static std::map<HGLRC, TGLContext *> fgContexts;
};

#elif defined(R__HAS_COCOA)

class TGLContextPrivate {
public:
   Handle_t fWindowID;
   Handle_t fGLContext;


   TGLContextPrivate()
      : fGLContext(0)
   {
   }

   static void RegisterContext(TGLContext *ctx);
   static void RemoveContext(TGLContext *ctx);
   static TGLContext *GetCurrentContext();

private:
   TGLContextPrivate(const TGLContextPrivate &);
   TGLContextPrivate &operator = (const TGLContextPrivate &);

   static std::map<Handle_t, TGLContext *> fgContexts;
};

#else

class TGLContextPrivate {
public:
   Display     *fDpy;
   XVisualInfo *fVisualInfo;
   GLXContext   fGLContext;
   Window       fWindowID;
   //GLXPbuffer   fPBDC;

   TGLContextPrivate()
      : fDpy(0),
        fVisualInfo(0),
        fGLContext(0),
        fWindowID(0)
   {
   }

   static void RegisterContext(TGLContext *ctx);
   static void RemoveContext(TGLContext *ctx);
   static TGLContext *GetCurrentContext();

private:
   TGLContextPrivate(const TGLContextPrivate &);
   TGLContextPrivate &operator = (const TGLContextPrivate &);

   static std::map<GLXContext, TGLContext *> fgContexts;
};

#endif
#endif
