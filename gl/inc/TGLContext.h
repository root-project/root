#ifndef ROOT_TGLContext
#define ROOT_TGLContext

#ifndef ROOT_TGLFormat
#include "TGLFormat.h"
#endif
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TGLPaintDevice;
//class TGLPBuffer;
class TGLWidget;

class TGLContext {
   friend class TGLWidget;
//   friend class TGLPBuffer;
private:
   TGLPaintDevice *fDevice;
   class TGLContextPrivate;
   TGLContextPrivate *fPimpl;

   Bool_t fFromCtor;//To prohibit user's calls of SetContext.
   Bool_t fValid;

public:
   TGLContext(TGLWidget *glWidget, const TGLContext *shareList = 0);//2
//   TGLContext(TGLPBuffer *glPbuf, const TGLContext *shareList = 0);//2

   virtual ~TGLContext();

   Bool_t           MakeCurrent();
   void             SwapBuffers();

   //This functions are public _ONLY_ for calls via
   //gROOT under win32. Please, DO NOT CALL IT DIRECTLY.
   void             SetContext(TGLWidget *widget, const TGLContext *shareList);
//   void             SetContextPB(TGLPBuffer *pbuff, const TGLContext *shareList);
   void             Release();

private:
   TGLContext(const TGLContext &);
   TGLContext &operator = (const TGLContext &);

   ClassDef(TGLContext, 0)//This class controls internal gl-context resources.
};

#endif
