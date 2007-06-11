#include <cassert>

#include "TGLFormat.h"

/*
   Exception-safe class.
*/
class TGLFormat::TGLFormatPrivate {
public:
   UInt_t fDepthSize;
   UInt_t fRGBASize;
   UInt_t fColorIndexSize;
   UInt_t fAccumSize;
   UInt_t fStencilSize;

   TGLFormatPrivate();
   TGLFormatPrivate(EFormatOptions opt);

   Bool_t operator == (const TGLFormatPrivate &rhs)const
   {
      return fDepthSize      == rhs.fDepthSize      &&
             fRGBASize       == rhs.fRGBASize       &&
             fColorIndexSize == rhs.fColorIndexSize &&
             fAccumSize      == rhs.fAccumSize      &&
             fStencilSize    == rhs.fStencilSize;
   }
};

//______________________________________________________________________________
TGLFormat::TGLFormatPrivate::TGLFormatPrivate():
#ifdef WIN32
                     fDepthSize(32),
#else
                     fDepthSize(16),//FIXFIX
#endif
                     fRGBASize(24),//FIXFIX
                     fColorIndexSize(0),
                     fAccumSize(0),
                     fStencilSize(0)
{
   //Default ctor for internal data.
}

//______________________________________________________________________________
TGLFormat::TGLFormatPrivate::TGLFormatPrivate(EFormatOptions opt):
#ifdef WIN32
                     fDepthSize(opt & kDepth ? 32 : 0),
#else
                     fDepthSize(opt & kDepth ? 16 : 0),//FIXFIX
#endif
                     fRGBASize(opt & kRGBA ? 24 : 0),//FIXFIX
                     fColorIndexSize(opt & kColorIndex ? 24 : 0),//I've never tested color-index buffer.
                     fAccumSize(opt & kAccum ? 24 : 0), //I've never tested accumulation buffer size.
                     fStencilSize(opt & kStencil ? 24 : 0) //I've never tested stencil buffer size.
{
   //Ctor from specified options.
}

ClassImp(TGLFormat)

//______________________________________________________________________________
TGLFormat::TGLFormat()
              : fDoubleBuffered(kTRUE),
                fPimpl(new TGLFormatPrivate)
{
   //Default ctor. Default surface is:
   //-double buffered
   //-RGBA
   //-with depth buffer
}

//______________________________________________________________________________
TGLFormat::TGLFormat(EFormatOptions opt)
              : fDoubleBuffered(opt & kDoubleBuffer),
                fPimpl(new TGLFormatPrivate(opt))
{
   //Define surface using options.
}

//______________________________________________________________________________
TGLFormat::TGLFormat(const TGLFormat &rhs)
              : fDoubleBuffered(rhs.fDoubleBuffered),
                fPimpl(new TGLFormatPrivate(*rhs.fPimpl))
{
   //Copy ctor.
}

//______________________________________________________________________________
TGLFormat::~TGLFormat()
{
   //Destructor.
   delete fPimpl;
}

//______________________________________________________________________________
TGLFormat &TGLFormat::operator = (const TGLFormat &rhs)
{
   //Copy assignment operator.
   if (this != &rhs) {
      fDoubleBuffered = rhs.fDoubleBuffered;
      *fPimpl = *rhs.fPimpl;
   }

   return *this;
}

//______________________________________________________________________________
Bool_t TGLFormat::operator == (const TGLFormat &rhs)const
{
   //Check if two formats are equal.
   return fDoubleBuffered == rhs.fDoubleBuffered && *fPimpl == *rhs.fPimpl;
}

//______________________________________________________________________________
Bool_t TGLFormat::operator != (const TGLFormat &rhs)const
{
   //Check for non-equality.
   return !(*this == rhs);
}

//______________________________________________________________________________
UInt_t TGLFormat::GetRGBASize()const
{
   //Get the size of color buffer.
   return fPimpl->fRGBASize;
}

//______________________________________________________________________________
void TGLFormat::SetRGBASize(UInt_t rgba)
{
   //Set the size of color buffer and switch off
   //color index mode.
   assert(rgba);
   fPimpl->fRGBASize = rgba;
   fPimpl->fColorIndexSize = 0;
}

//______________________________________________________________________________
Bool_t TGLFormat::IsRGBA()const
{
   //Check, if it's a RGBA surface.
   return GetRGBASize() != 0;
}

//______________________________________________________________________________
UInt_t TGLFormat::GetColorIndexSize()const
{
   //Get the size of color buffer.
   return fPimpl->fColorIndexSize;
}

//______________________________________________________________________________
void TGLFormat::SetColorIndexSize(UInt_t colorIndex)
{
   //Set the size of color buffer in color-index mode
   //and switch off RGBA mode.
   assert(colorIndex);
   fPimpl->fColorIndexSize = colorIndex;
   fPimpl->fRGBASize = 0;
}

//______________________________________________________________________________
Bool_t TGLFormat::IsColorIndex()const
{
   //Check, if it's a color-index surface.
   return GetColorIndexSize() != 0;
}

//______________________________________________________________________________
UInt_t TGLFormat::GetDepthSize()const
{
   //Get the size of depth buffer.
   return fPimpl->fDepthSize;
}

//______________________________________________________________________________
void TGLFormat::SetDepthSize(UInt_t depth)
{
   //Set the size of color buffer.
   assert(depth);
   fPimpl->fDepthSize = depth;
}

//______________________________________________________________________________
Bool_t TGLFormat::HasDepth()const
{
   //Check, if this surface has depth buffer.
   return GetDepthSize() != 0;
}

//______________________________________________________________________________
UInt_t TGLFormat::GetStencilSize()const
{
   //Get the size of stencil buffer.
   return fPimpl->fStencilSize;
}

//______________________________________________________________________________
void TGLFormat::SetStencilSize(UInt_t stencil)
{
   //Set the size of stencil buffer.
   assert(stencil);
   fPimpl->fStencilSize = stencil;
}

//______________________________________________________________________________
Bool_t TGLFormat::HasStencil()const
{
   //Check, if this surface has stencil buffer.
   return GetStencilSize() != 0;
}

//______________________________________________________________________________
UInt_t TGLFormat::GetAccumSize()const
{
   //Get the size of accum buffer.
   return fPimpl->fAccumSize;
}

//______________________________________________________________________________
void TGLFormat::SetAccumSize(UInt_t accum)
{
   //Set the size of accum buffer.
   assert(accum);
   fPimpl->fAccumSize = accum;
}

//______________________________________________________________________________
Bool_t TGLFormat::HasAccumBuffer()const
{
   //Check, if this surface has accumulation buffer.
   return GetAccumSize() != 0;
}

//______________________________________________________________________________
Bool_t TGLFormat::IsDoubleBuffered()const
{
   //Check, if the surface is double buffered.
   return fDoubleBuffered;
}

//______________________________________________________________________________
void TGLFormat::SetDoubleBuffered(Bool_t db)
{
   //Set the surface as double/single buffered.
   fDoubleBuffered = db;
}
