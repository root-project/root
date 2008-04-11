// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLAdapter
#define ROOT_TGLAdapter

#ifndef ROOT_TVirtualGL
#include "TVirtualGL.h"
#endif

class TGLAdapter : public TGLPaintDevice {
private:
   Int_t fGLDevice;

public:
   explicit TGLAdapter(Int_t glDevice = -1);

   Bool_t            MakeCurrent();
   void              SwapBuffers();
   const TGLFormat  *GetPixelFormat()const{return 0;}
   const TGLContext *GetContext()const{return 0;}

   void SetGLDevice(Int_t glDevice)
   {
      fGLDevice = glDevice;
   }

   void ReadGLBuffer();
   void SelectOffScreenDevice();
   void MarkForDirectCopy(Bool_t isDirect);
   void ExtractViewport(Int_t *vp)const;

private:
   TGLAdapter(const TGLAdapter &);
   TGLAdapter &operator = (const TGLAdapter &);

   void AddContext(TGLContext *){}
   void RemoveContext(TGLContext *){}

   ClassDef(TGLAdapter, 0) // Allow plot-painters to be used for gl-inpad and gl-viewer.
};

#endif
