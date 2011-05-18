// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLFormat
#define ROOT_TGLFormat

#include "Rtypes.h"

#include <vector>

/*
   TGLFormat class describes the pixel format of a drawing surface.
   It's a generic analog of PIXELFORMATDESCRIPTOR (win32) or
   array of integer constants array for glXChooseVisual (X11).
   This class is in a very preliminary state, different
   options have not been tested yet, only defaults.

   Surface can be:
   -RGBA
   -with/without depth buffer
   -with/without stencil buffer
   -with/without accum buffer
   -double/single buffered
*/

class TGLFormat
{
public:
   enum EFormatOptions
   {
      kNone         = 0,
      kDoubleBuffer = 1,
      kDepth        = 2,
      kAccum        = 4,
      kStencil      = 8,
      kStereo       = 16,
      kMultiSample  = 32
   };

private:
   Bool_t fDoubleBuffered;
   Bool_t fStereo;
   Int_t  fDepthSize;
   Int_t  fAccumSize;
   Int_t  fStencilSize;
   Int_t  fSamples;

   static std::vector<Int_t> fgAvailableSamples;

   static Int_t GetDefaultSamples();
   static void  InitAvailableSamples();

public:
   TGLFormat();
   TGLFormat(EFormatOptions options);

   //Virtual dtor only to supress warnings from g++ -
   //ClassDef adds virtual functions, so g++ wants virtual dtor.
   virtual ~TGLFormat();

   Bool_t operator == (const TGLFormat &rhs)const;
   Bool_t operator != (const TGLFormat &rhs)const;

   Int_t  GetDepthSize()const;
   void   SetDepthSize(Int_t depth);
   Bool_t HasDepth()const;

   Int_t  GetStencilSize()const;
   void   SetStencilSize(Int_t stencil);
   Bool_t HasStencil()const;

   Int_t  GetAccumSize()const;
   void   SetAccumSize(Int_t accum);
   Bool_t HasAccumBuffer()const;

   Bool_t IsDoubleBuffered()const;
   void   SetDoubleBuffered(Bool_t db);

   Bool_t IsStereo()const;
   void   SetStereo(Bool_t db);

   Int_t  GetSamples()const;
   void   SetSamples(Int_t samples);
   Bool_t HasMultiSampling()const;

   ClassDef(TGLFormat, 0); // Describes GL buffer format.
};

#endif
