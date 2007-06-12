#ifndef ROOT_TGLContextPrivate
#define ROOT_TGLContextPrivate

#ifndef ROOT_TGLIncludes
#include "TGLIncludes.h"
#endif
#ifndef ROOT_TGLContext
#include "TGLContext.h"
#endif

#ifdef WIN32

class TGLContext::TGLContextPrivate {
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
   GLXPbuffer   fPBDC;
   
   TGLContextPrivate()
      : fDpy(0),
        fVisualInfo(0),
        fGLContext(0),
        fWindowIndex(-1),
        fPBDC(0)
   {
   }
   
private:
   TGLContextPrivate(const TGLContextPrivate &);
   TGLContextPrivate &operator = (const TGLContextPrivate &);
};

#endif

#endif
