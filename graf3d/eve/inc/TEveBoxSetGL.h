// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveBoxSetGL
#define ROOT_TEveBoxSetGL

#include "TEveDigitSetGL.h"
#include "TEveBoxSet.h"

class TEveBoxSetGL : public TEveDigitSetGL
{
   TEveBoxSetGL(const TEveBoxSetGL&);            // Not implemented
   TEveBoxSetGL& operator=(const TEveBoxSetGL&); // Not implemented

protected:
   TEveBoxSet     *fM;       // Model object.

   mutable UInt_t  fBoxDL;   // Display-list id for a box atom.

   Int_t  PrimitiveType() const;
   void   MakeOriginBox(Float_t p[8][3], Float_t dx, Float_t dy, Float_t dz) const;
   void   RenderBoxStdNorm(const Float_t p[8][3]) const;
   void   RenderBoxAutoNorm(const Float_t p[8][3]) const;
   void   MakeDisplayList() const;

   void   RenderBoxes(TGLRnrCtx& rnrCtx) const;

public:
   TEveBoxSetGL();
   virtual ~TEveBoxSetGL();

   virtual Bool_t ShouldDLCache(const TGLRnrCtx& rnrCtx) const;
   virtual void   DLCacheDrop();
   virtual void   DLCachePurge();

   virtual Bool_t SetModel(TObject* obj, const Option_t *opt = nullptr);
   virtual void   DirectDraw(TGLRnrCtx& rnrCtx) const;

   virtual void Render(TGLRnrCtx& rnrCtx);

   ClassDef(TEveBoxSetGL, 0); // GL-renderer for TEveBoxSet class.
};

#endif
