#ifndef ROOT_TGLContext
#define ROOT_TGLContext

#ifndef ROOT_TGLFormat
#include "TGLFormat.h"
#endif
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TGLWidget;

class TGLContext {
private:
   TGLFormat  fGLFormat;

   class TGLContextPrivate;
   TGLContextPrivate *fPimpl;

public:
   TGLContext(const TGLWidget *glWidget, const TGLFormat &request);//2

   virtual ~TGLContext();

   Bool_t           MakeCurrent();
   void             SwapBuffers();

   const TGLFormat &GetPixelFormat()const;

   void             SetContext(const TGLWidget *);

private:
   TGLContext(const TGLContext &);
   TGLContext &operator = (const TGLContext &);

   ClassDef(TGLContext, 0) // ROOT wrapper for OpenGL context.
};

#endif
