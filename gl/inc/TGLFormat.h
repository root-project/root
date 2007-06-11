#ifndef ROOT_TGLFormat
#define ROOT_TGLFormat

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

/*
   TGLFormat class describes the pixel format of a drawing surface.
   It's a generic analog of PIXELFORMATDESCRIPTOR (win32) or
   array of integer constants array for glXChooseVisual (X11).
   This class is in a very preliminary state, different
   options have not been tested yet, only defaults.

   Operations are exception-safe, only ctors can throw std::bad_alloc.
   Surface can be:
   -RGBA/color-index
   -with/without depth buffer
   -with/without stencil buffer
   -with/without accum buffer
   -double/single buffered
*/

class TGLFormat {
public:
   enum EFormatOptions {
      kRGBA = 1,
      kColorIndex = 2,
      kDoubleBuffer = 4,
      kDepth = 16,
      kAccum = 32,
      kStencil = 64
   };

private:
   Bool_t fDoubleBuffered;

   class TGLFormatPrivate;
   TGLFormatPrivate *fPimpl;

public:

   TGLFormat();//can throw std::bad_alloc
   TGLFormat(EFormatOptions options);//can throw std::bad_alloc
   TGLFormat(const TGLFormat &rhs);//can throw std::bad_alloc

   virtual ~TGLFormat();

   TGLFormat &operator = (const TGLFormat &rhs);

   Bool_t operator == (const TGLFormat &rhs)const;
   Bool_t operator != (const TGLFormat &rhs)const;

   void   SetRGBASize(UInt_t rgba);
   UInt_t GetRGBASize()const;
   Bool_t IsRGBA()const;

   UInt_t GetColorIndexSize()const;
   void   SetColorIndexSize(UInt_t colorIndex);
   Bool_t IsColorIndex()const;

   UInt_t GetDepthSize()const;
   void   SetDepthSize(UInt_t depth);
   Bool_t HasDepth()const;

   UInt_t GetStencilSize()const;
   void   SetStencilSize(UInt_t stencil);
   Bool_t HasStencil()const;

   UInt_t GetAccumSize()const;
   void   SetAccumSize(UInt_t accum);
   Bool_t HasAccumBuffer()const;

   Bool_t IsDoubleBuffered()const;
   void   SetDoubleBuffered(Bool_t db);

   ClassDef(TGLFormat, 0) // Encapsulates OpenGL buffer selection.
};

#endif
