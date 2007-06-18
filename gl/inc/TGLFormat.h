// @(#)root/gl:$Name:  $:$Id: TGLFormat.h,v 1.3 2007/06/12 20:29:00 rdm Exp $
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

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

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

class TGLFormat {
public:
   enum EFormatOptions {
      kDoubleBuffer = 1,
      kDepth = 2,
      kAccum = 4,
      kStencil = 8
   };

private:
   Bool_t fDoubleBuffered;
   UInt_t fDepthSize;
   UInt_t fAccumSize;
   UInt_t fStencilSize;

public:
   TGLFormat();
   TGLFormat(EFormatOptions options);

   //Virtual dtor only to supress warnings from g++ -
   //ClassDef adds virtual functions, so g++ wants virtual dtor.
   virtual ~TGLFormat();

   Bool_t operator == (const TGLFormat &rhs)const;
   Bool_t operator != (const TGLFormat &rhs)const;

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

   ClassDef(TGLFormat, 0)//Describes gl buffer format
};

#endif
