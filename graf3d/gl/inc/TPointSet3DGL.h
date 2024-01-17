// @(#)root/gl:$Id$
// Author: Matevz Tadel  7/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TPointSet3DGL
#define ROOT_TPointSet3DGL

#include "TGLObject.h"

class TGLRnrCtx;

class TPointSet3DGL : public TGLObject
{
public:
   TPointSet3DGL() : TGLObject() {}

   Bool_t SetModel(TObject* obj, const Option_t *opt = nullptr) override;
   void   SetBBox() override;
   void   DirectDraw(TGLRnrCtx & rnrCtx) const override;

   Bool_t IgnoreSizeForOfInterest() const override { return kTRUE; }

   Bool_t ShouldDLCache(const TGLRnrCtx & rnrCtx) const override;

   void   Draw(TGLRnrCtx & rnrCtx) const override;

   Bool_t SupportsSecondarySelect() const override { return kTRUE; }
   void   ProcessSelection(TGLRnrCtx & rnrCtx, TGLSelectRecord & rec) override;

   ClassDefOverride(TPointSet3DGL,1)  // GL renderer for TPointSet3D
};

#endif
