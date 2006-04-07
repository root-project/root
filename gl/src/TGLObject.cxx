// $Header$

#include <TGLObject.h>
#include <TObject.h>
#include <TString.h>
#include <TAtt3D.h>
#include <TAttBBox.h>

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

Bool_t TGLObject::set_model(TObject* obj, const Text_t* classname)
{
   if(obj->InheritsFrom(classname) == false) {
      Warning("TGLObject::set_model", "object of wrong class passed.");
      return false;
   }
   fExternalObj = obj;
   fID          = reinterpret_cast<ULong_t>(obj);

   return true;
}

void TGLObject::set_axis_aligned_bbox(Float_t xmin, Float_t xmax,
				      Float_t ymin, Float_t ymax,
				      Float_t zmin, Float_t zmax)
{
   fBoundingBox.SetAligned(TGLVertex3(xmin, ymin, zmin),
			   TGLVertex3(xmax, ymax, zmax));
}

void TGLObject::set_axis_aligned_bbox(const Float_t* p)
{
   set_axis_aligned_bbox(p[0], p[1], p[2], p[3], p[4], p[5]);
}
