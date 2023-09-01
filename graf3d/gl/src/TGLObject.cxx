// @(#)root/gl:$Id$
// Author: Matevz Tadel  7/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TGLObject.h"
#include "TGLRnrCtx.h"
#include "TObject.h"
#include "TClass.h"
#include "TBaseClass.h"
#include "TList.h"
#include "TString.h"

/** \class TGLObject
\ingroup opengl
Base-class for direct OpenGL renderers.
This allows classes to circumvent passing of TBuffer3D and
use user-provided OpenGL code.
By convention, if you want class TFoo : public TObject to have direct rendering
you should also provide TFooGL : public TGLObject and implement
abstract functions SetModel() and SetBBox().
TAttBBox can be used to facilitate calculation of bounding-boxes.
See TPointSet3D and TPointSet3DGL.
*/

ClassImp(TGLObject);

TMap TGLObject::fgGLClassMap;

////////////////////////////////////////////////////////////////////////////////
/// Decide if display-list should be used for this pass rendering,
/// as determined by rnrCtx.

Bool_t TGLObject::ShouldDLCache(const TGLRnrCtx& rnrCtx) const
{
   if (!fDLCache ||
       !fScene   ||
       (rnrCtx.SecSelection() && SupportsSecondarySelect()) ||
       (fMultiColor && (rnrCtx.Highlight() || rnrCtx.IsDrawPassOutlineLine())) ||
       (AlwaysSecondarySelect() && rnrCtx.Highlight()))
   {
      return kFALSE;
   }

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Update bounding box from external source.
/// We call abstract SetBBox() and propagate the change to all
/// attached physicals.

void TGLObject::UpdateBoundingBox()
{
   SetBBox();
   UpdateBoundingBoxesOfPhysicals();
}

////////////////////////////////////////////////////////////////////////////////
/// Checks if obj is of proper class and sets the model.
/// Protected helper for subclasses.
/// Most sub-classes use exception-throwing SetModelDynCast() instead.

Bool_t TGLObject::SetModelCheckClass(TObject* obj, TClass* cls)
{
   if(obj->InheritsFrom(cls) == kFALSE) {
      Warning("TGLObject::SetModelCheckClass", "object of wrong class passed.");
      return kFALSE;
   }
   fExternalObj = obj;

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set axis-aligned bounding-box.
/// Protected helper for subclasses.

void TGLObject::SetAxisAlignedBBox(Float_t xmin, Float_t xmax,
                                   Float_t ymin, Float_t ymax,
                                   Float_t zmin, Float_t zmax)
{
   fBoundingBox.SetAligned(TGLVertex3(xmin, ymin, zmin),
                           TGLVertex3(xmax, ymax, zmax));
}

////////////////////////////////////////////////////////////////////////////////
/// Set axis-aligned bounding-box.
/// Protected helper for subclasses.

void TGLObject::SetAxisAlignedBBox(const Float_t* p)
{
   SetAxisAlignedBBox(p[0], p[1], p[2], p[3], p[4], p[5]);
}


////////////////////////////////////////////////////////////////////////////////
/// Recursively search cls and its base classes for a GL-renderer
/// class.

TClass* TGLObject::SearchGLRenderer(TClass* cls)
{
   TString rnr( cls->GetName() );
   rnr += "GL";
   TClass* c = TClass::GetClass(rnr);
   if (c != 0)
      return c;

   TList* bases = cls->GetListOfBases();
   if (bases == 0 || bases->IsEmpty())
      return 0;

   TIter  next_base(bases);
   TBaseClass* bc;
   while ((bc = (TBaseClass*) next_base()) != 0) {
      cls = bc->GetClassPointer();
      if ((c = SearchGLRenderer(cls)) != 0) {
         return c;
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return direct-rendering GL class for class isa.
/// Zero is a valid response.

TClass* TGLObject::GetGLRenderer(TClass* isa)
{
   TPair* p = (TPair*) fgGLClassMap.FindObject(isa);
   TClass* cls;
   if (p != 0) {
      cls = (TClass*) p->Value();
   } else {
      cls = SearchGLRenderer(isa);
      fgGLClassMap.Add(isa, cls);
   }
   return cls;
}
