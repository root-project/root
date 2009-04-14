// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEvePolygonSetProjectedGL
#define ROOT_TEvePolygonSetProjectedGL

#include "TGLObject.h"

class TEvePolygonSetProjectedGL : public TGLObject
{
public:
   TEvePolygonSetProjectedGL();
   virtual  ~TEvePolygonSetProjectedGL() {}

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   SetBBox();
   virtual void   Draw(TGLRnrCtx& rnrCtx) const;
   virtual void   DirectDraw(TGLRnrCtx & rnrCtx) const;

   virtual Bool_t IgnoreSizeForOfInterest() const { return kTRUE; }

   ClassDef(TEvePolygonSetProjectedGL,0);  // GL-renderer for TEvePolygonSetProjected class.
};

#endif
