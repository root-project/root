// @(#)root/gl:$Name:  $:$Id: TPointSet3DGL.h,v 1.3 2006/04/07 09:20:43 rdm Exp $
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

public:
   TPointSet3DGL();

   virtual Bool_t SetModel(TObject* obj);
   virtual void   SetBBox();

   virtual Bool_t ShouldCache(const TGLDrawFlags & /*flags*/) const { return false; }

  ClassDef(TPointSet3DGL,1)  // GL renderer for TPointSet3D
};

#endif
