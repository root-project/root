// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveQuadSetGL
#define ROOT_TEveQuadSetGL

#include "TEveDigitSetGL.h"
#include "TEveQuadSet.h"

class TEveQuadSetGL : public TEveDigitSetGL
{
   TEveQuadSetGL(const TEveQuadSetGL&);            // Not implemented
   TEveQuadSetGL& operator=(const TEveQuadSetGL&); // Not implemented

protected:
   TEveQuadSet     *fM;

   void   RenderQuads(TGLRnrCtx & rnrCtx) const;
   void   RenderLines(TGLRnrCtx & rnrCtx) const;
   void   RenderHexagons(TGLRnrCtx & rnrCtx) const;

public:
   TEveQuadSetGL();
   ~TEveQuadSetGL() override {}

   Bool_t SetModel(TObject* obj, const Option_t *opt = nullptr) override;
   void   DirectDraw(TGLRnrCtx& rnrCtx) const override;

   Bool_t IgnoreSizeForOfInterest() const override { return kTRUE; }

   ClassDefOverride(TEveQuadSetGL, 0); // GL-renderer for TEveQuadSet class.
};

#endif
