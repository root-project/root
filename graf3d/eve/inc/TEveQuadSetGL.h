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

#include "TGLObject.h"
#include "TEveQuadSet.h"

class TEveQuadSetGL : public TGLObject
{
   TEveQuadSetGL(const TEveQuadSetGL&);            // Not implemented
   TEveQuadSetGL& operator=(const TEveQuadSetGL&); // Not implemented

protected:
   TEveQuadSet* fM;

   Bool_t SetupColor(const TEveDigitSet::DigitBase_t& q) const;

   void   RenderQuads(TGLRnrCtx & rnrCtx) const;
   void   RenderLines(TGLRnrCtx & rnrCtx) const;
   void   RenderHexagons(TGLRnrCtx & rnrCtx) const;

public:
   TEveQuadSetGL();
   virtual ~TEveQuadSetGL() {}

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   SetBBox();
   virtual void   DirectDraw(TGLRnrCtx & rnrCtx) const;

   virtual Bool_t IgnoreSizeForOfInterest() const { return kTRUE; }

   virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   virtual void   ProcessSelection(TGLRnrCtx & rnrCtx, TGLSelectRecord & rec);

   ClassDef(TEveQuadSetGL, 0); // GL-renderer for TEveQuadSet class.
};

#endif
