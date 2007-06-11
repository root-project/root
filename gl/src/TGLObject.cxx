// @(#)root/gl:$Name:  $:$Id: TGLObject.cxx,v 1.1.1.1 2007/04/04 16:01:44 mtadel Exp $
// Author: Matevz Tadel  7/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TGLObject.h"
#include "TObject.h"
#include "TString.h"
#include "TAtt3D.h"
#include "TAttBBox.h"

//______________________________________________________________________
// TGLObject
//
// Base-class for direct OpenGL renderers.
// This allows classes to circumvent passing of TBuffer3D and
// use user-provided OpenGL code.
// By convention, if you want class TFoo : public TObject to have direct rendering
// you should also provide TFooGL : public TGLObject and implement
// abstract functions SetModel() and SetBBox().
// TAttBBox can be used to facilitate calculation of bounding-boxes.
// See TPointSet3D and TPointSet3DGL.

ClassImp(TGLObject)

//______________________________________________________________________________
void TGLObject::UpdateBoundingBox()
{
   // Update bounding box from external source.
   // We call abstract SetBBox() and propagate the change to all
   // attached physicals.

   SetBBox();
   UpdateBoundingBoxesOfPhysicals();
}

//______________________________________________________________________________
Bool_t TGLObject::SetModelCheckClass(TObject* obj, TClass* cls)
{
   // Checks if obj is of proper class and sets the model.
   // Protected helper for subclasses.

   if(obj->InheritsFrom(cls) == kFALSE) {
      Warning("TGLObject::SetModelCheckClass", "object of wrong class passed.");
      return kFALSE;
   }
   fExternalObj = obj;

   return kTRUE;
}

//______________________________________________________________________________
void TGLObject::SetAxisAlignedBBox(Float_t xmin, Float_t xmax,
                                   Float_t ymin, Float_t ymax,
                                   Float_t zmin, Float_t zmax)
{
   // Set axis-aligned bounding-box.
   // Protected helper for subclasses.

   fBoundingBox.SetAligned(TGLVertex3(xmin, ymin, zmin),
                           TGLVertex3(xmax, ymax, zmax));
}

//______________________________________________________________________________
void TGLObject::SetAxisAlignedBBox(const Float_t* p)
{
   // Set axis-aligned bounding-box.
   // Protected helper for subclasses.

   SetAxisAlignedBBox(p[0], p[1], p[2], p[3], p[4], p[5]);
}
