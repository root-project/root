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

class TEvePolygonSetProjected;

class TEvePolygonSetProjectedGL : public TGLObject
{
protected:
   struct Edge_t
   {
      Int_t fI, fJ;
      Edge_t(Int_t i, Int_t j)
      {
         if (i <= j) { fI = i; fJ = j; }
         else        { fI = j; fJ = i; }
      }

      bool operator<(const Edge_t& e) const
      {
         if (fI == e.fI)
            return fJ < e.fJ;
         else
            return fI < e.fI;
      }
   };

   TEvePolygonSetProjected *fM;

public:
   TEvePolygonSetProjectedGL();
   virtual  ~TEvePolygonSetProjectedGL() {}

   virtual Bool_t SetModel(TObject* obj, const Option_t *opt = nullptr);
   virtual void   SetBBox();
   virtual void   Draw(TGLRnrCtx& rnrCtx) const;
   virtual void   DirectDraw(TGLRnrCtx& rnrCtx) const;

   virtual void   DrawHighlight(TGLRnrCtx& rnrCtx, const TGLPhysicalShape* pshp, Int_t lvl=-1) const;

   virtual Bool_t IgnoreSizeForOfInterest() const { return kTRUE; }

private:
   void DrawOutline() const;

   ClassDef(TEvePolygonSetProjectedGL,0);  // GL-renderer for TEvePolygonSetProjected class.
};

#endif
