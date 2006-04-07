// @(#)root/gl:$Name:  $:$Id: TSocket.h,v 1.20 2005/07/29 14:26:51 rdm Exp $
// Author: Matevz Tadel  7/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TGLObject
#define ROOT_TGLObject

#ifndef ROOT_TGLLogicalShape
#include "TGLLogicalShape.h"
#endif


class TGLObject : public TGLLogicalShape
{
protected:
   // Abstract method from TGLDrawable:
   // virtual void DirectDraw(const TGLDrawFlags & flags) const;

   Bool_t set_model(TObject* obj, const Text_t* classname);

   void   set_axis_aligned_bbox(Float_t xmin, Float_t xmax,
                                Float_t ymin, Float_t ymax,
                                Float_t zmin, Float_t zmax);
   void   set_axis_aligned_bbox(const Float_t* p);

public:
   TGLObject() : TGLLogicalShape(0) {}
   virtual ~TGLObject() {}

   virtual ELODAxes SupportedLODAxes() const { return kLODAxesNone; }

   virtual Bool_t KeepDuringSmartRefresh() const { return true; }

   virtual Bool_t SetModel(TObject* obj) = 0;
   virtual void   SetBBox() = 0;

   ClassDef(TGLObject, 1); // Base-class for direct OpenGL renderers
};

#endif
