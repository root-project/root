// @(#)root/gl:$Name:  $:$Id: TGLObject.h,v 1.5 2007/06/11 19:56:33 brun Exp $
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
   // Abstract method from TGLLogicalShape:
   // virtual void DirectDraw(TGLRnrCtx & rnrCtx) const = 0;

   Bool_t SetModelCheckClass(TObject* obj, TClass* cls);

   void   SetAxisAlignedBBox(Float_t xmin, Float_t xmax,
                             Float_t ymin, Float_t ymax,
                             Float_t zmin, Float_t zmax);
   void   SetAxisAlignedBBox(const Float_t* p);

public:
   TGLObject() : TGLLogicalShape(0) {}
   virtual ~TGLObject() {}

   // Kept from TGLLogicalShape
   // virtual ELODAxes SupportedLODAxes() const { return kLODAxesNone; }

   // Changed from TGLLogicalShape
   virtual Bool_t KeepDuringSmartRefresh() const { return kTRUE; }
   virtual void   UpdateBoundingBox();

   // TGLObject virtuals
   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0) = 0;
   virtual void   SetBBox() = 0;

   ClassDef(TGLObject, 1); // Base-class for direct OpenGL renderers
};

#endif
