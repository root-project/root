// @(#)root/gl:$Name:  $:$Id: TGLFormat.cxx,v 1.3 2007/06/12 20:29:00 rdm Exp $
// Author:  Timur Pocheptsov, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <cassert>

#include "TGLFormat.h"

ClassImp(TGLFormat)

//______________________________________________________________________________
TGLFormat::TGLFormat() :
   fDoubleBuffered(kTRUE),
#ifdef WIN32
   fDepthSize(32),
#else
   fDepthSize(16),//FIXFIX
#endif
   fAccumSize(0),
   fStencilSize(8)
{
   //Default ctor. Default surface is:
   //-double buffered
   //-RGBA
   //-with depth buffer
}

//______________________________________________________________________________
TGLFormat::TGLFormat(EFormatOptions opt) :
   fDoubleBuffered(opt & kDoubleBuffer),
#ifdef WIN32
   fDepthSize(opt & kDepth ? 32 : 0),
#else
   fDepthSize(opt & kDepth ? 16 : 0),//FIXFIX
#endif
   fAccumSize(opt & kAccum ? 8 : 0),    //I've never tested accumulation buffer size.
   fStencilSize(opt & kStencil ? 8 : 0) //I've never tested stencil buffer size.
{
   //Define surface using options.
}

//______________________________________________________________________________
TGLFormat::~TGLFormat()
{
   //Destructor.
}

//______________________________________________________________________________
Bool_t TGLFormat::operator == (const TGLFormat &rhs)const
{
   //Check if two formats are equal.
   return fDoubleBuffered == rhs.fDoubleBuffered && fDepthSize == rhs.fDepthSize &&
          fAccumSize == rhs.fAccumSize && fStencilSize == rhs.fStencilSize;
}

//______________________________________________________________________________
Bool_t TGLFormat::operator != (const TGLFormat &rhs)const
{
   //Check for non-equality.
   return !(*this == rhs);
}

//______________________________________________________________________________
UInt_t TGLFormat::GetDepthSize()const
{
   //Get the size of depth buffer.
   return fDepthSize;
}

//______________________________________________________________________________
void TGLFormat::SetDepthSize(UInt_t depth)
{
   //Set the size of color buffer.
   assert(depth);
   fDepthSize = depth;
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
   return fStencilSize;
}

//______________________________________________________________________________
void TGLFormat::SetStencilSize(UInt_t stencil)
{
   //Set the size of stencil buffer.
   assert(stencil);
   fStencilSize = stencil;
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
   return fAccumSize;
}

//______________________________________________________________________________
void TGLFormat::SetAccumSize(UInt_t accum)
{
   //Set the size of accum buffer.
   assert(accum);
   fAccumSize = accum;
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
