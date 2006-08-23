// @(#)root/gl:$Name:  $:$Id: TPointSet3DGL.h,v 1.4 2006/05/09 19:08:44 brun Exp $
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


class TPointSet3DGL : public TGLObject
{
protected:
   virtual void DirectDraw(const TGLDrawFlags & flags) const;
   void RenderPoints(const TGLDrawFlags & flags) const;
   void RenderCrosses(const TGLDrawFlags & flags) const;

public:
   TPointSet3DGL() : TGLObject() {}

   virtual Bool_t SetModel(TObject* obj);
   virtual void   SetBBox();

   virtual Bool_t IgnoreSizeForOfInterest() const { return kTRUE; }

   virtual Bool_t ShouldCache(const TGLDrawFlags & flags) const;
   virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   virtual void   ProcessSelection(UInt_t* ptr, TGLViewer*, TGLScene*);

   ClassDef(TPointSet3DGL,1)  // GL renderer for TPointSet3D
};

#endif
