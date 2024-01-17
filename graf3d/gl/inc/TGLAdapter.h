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

#include "TVirtualGL.h"

class TGLAdapter : public TGLPaintDevice {
private:
   Int_t fGLDevice;

public:
   explicit TGLAdapter(Int_t glDevice = -1);

   Bool_t            MakeCurrent() override;
   void              SwapBuffers() override;
   const TGLFormat  *GetPixelFormat()const override{return nullptr;}
   const TGLContext *GetContext()const override{return nullptr;}

   void SetGLDevice(Int_t glDevice)
   {
      fGLDevice = glDevice;
   }

   void ReadGLBuffer();
   void SelectOffScreenDevice();
   void MarkForDirectCopy(Bool_t isDirect);
   void ExtractViewport(Int_t *vp)const override;

private:
   TGLAdapter(const TGLAdapter &);
   TGLAdapter &operator = (const TGLAdapter &);

   void AddContext(TGLContext *) override{}
   void RemoveContext(TGLContext *) override{}

   ClassDefOverride(TGLAdapter, 0) // Allow plot-painters to be used for gl-inpad and gl-viewer.
};

#endif
