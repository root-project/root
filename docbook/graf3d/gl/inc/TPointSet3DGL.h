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

#ifndef ROOT_TGLObject
#include "TGLObject.h"
#endif

class TGLRnrCtx;

class TPointSet3DGL : public TGLObject
{
public:
   TPointSet3DGL() : TGLObject() {}

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   SetBBox();
   virtual void   DirectDraw(TGLRnrCtx & rnrCtx) const;

   virtual Bool_t IgnoreSizeForOfInterest() const { return kTRUE; }

   virtual Bool_t ShouldDLCache(const TGLRnrCtx & rnrCtx) const;

   virtual void   Draw(TGLRnrCtx & rnrCtx) const;

   virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   virtual void   ProcessSelection(TGLRnrCtx & rnrCtx, TGLSelectRecord & rec);

   ClassDef(TPointSet3DGL,1)  // GL renderer for TPointSet3D
};

#endif
