// @(#)root/gl:$Id$
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

#include "TGLLogicalShape.h"
#include <TMap.h>

class TClass;

class TGLObject : public TGLLogicalShape
{
private:
   static TMap    fgGLClassMap;
   static TClass* SearchGLRenderer(TClass* cls);

protected:
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
   // Abstract method from TGLLogicalShape:
   // virtual void DirectDraw(TGLRnrCtx & rnrCtx) const = 0;

   // Interface to class .vs. classGL map.
   static TClass* GetGLRenderer(TClass* isa);

   ClassDef(TGLObject, 0); // Base-class for direct OpenGL renderers
};

#endif
