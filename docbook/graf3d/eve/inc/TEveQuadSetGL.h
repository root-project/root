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
   virtual ~TEveQuadSetGL() {}

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   DirectDraw(TGLRnrCtx& rnrCtx) const;

   virtual Bool_t IgnoreSizeForOfInterest() const { return kTRUE; }

   ClassDef(TEveQuadSetGL, 0); // GL-renderer for TEveQuadSet class.
};

#endif
