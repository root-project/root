#include "TGeoVGShape.h"
////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoVGShape - bridge class for using a VecGeom solid as TGeoShape.             //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TGeoVGShape.h"

#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedRootVolume.h"
#include "management/RootGeoManager.h"
#include "TError.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoVolume.h"

//_____________________________________________________________________________
TGeoVGShape::TGeoVGShape(TGeoShape *shape,  VPlacedVolume *vgshape)
           :TGeoBBox(shape->GetName(), 0, 0, 0), fVGShape(vgshape), fShape(shape)
{
// Default constructor
   // Copy box parameters from the original ROOT shape
   const TGeoBBox *box = (const TGeoBBox*)shape;
   TGeoBBox::SetBoxDimensions(box->GetDX(), box->GetDY(), box->GetDZ());   
   memcpy(fOrigin, box->GetOrigin(), 3*sizeof(Double_t));
}

//_____________________________________________________________________________
TGeoVGShape::~TGeoVGShape()
{
// Destructor
   // Cleanup only the VecGeom solid, the ROOT shape is cleaned by TGeoManager 
   delete fVGShape;
}   

//_____________________________________________________________________________
TGeoVGShape *TGeoVGShape::Create(TGeoShape *shape)
{
// Factory creating TGeoVGShape from a Root shape. Returns nullptr if the 
// shape cannot be converted
   VPlacedVolume *vgshape = TGeoVGShape::CreateVecGeomSolid(shape);
   if (!vgshape) return nullptr;
   return ( new TGeoVGShape(shape, vgshape) );
}

//_____________________________________________________________________________
VPlacedVolume *TGeoVGShape::CreateVecGeomSolid(TGeoShape *shape)
{
// Conversion method to create VecGeom solid corresponding to TGeoShape
   // Call VecGeom TGeoShape->UnplacedSolid converter
   VUnplacedVolume *unplaced = RootGeoManager::Instance().Convert(shape);
   if (!unplaced) {
      ::Warning("CreateVecGeomSolid", "Cannot convert shape type %s", shape->ClassName());
      return nullptr;
   }   
   // We have to create a placed volume from the unplaced one to have access
   // to the navigation interface
   LogicalVolume *lvol = new LogicalVolume("", unplaced);
   return ( lvol->Place() );
}

//_____________________________________________________________________________
void TGeoVGShape::ComputeBBox()
{
// Compute bounding box.
  fShape->ComputeBBox();
}   

//_____________________________________________________________________________
Double_t TGeoVGShape::Capacity() const
{
// Returns analytic capacity of the solid
   return fVGShape->Capacity();
}

//_____________________________________________________________________________
void TGeoVGShape::ComputeNormal(const Double_t *point, const Double_t */*dir*/, Double_t *norm)
{
// Normal computation.
   Vector3D<Double_t> vnorm;
   fVGShape->Normal(Vector3D<Double_t>(point[0], point[1], point[2]), vnorm);
   norm[0] = vnorm.x(); norm[1] = vnorm.y(), norm[2] = vnorm.z();
}
   
//_____________________________________________________________________________
Bool_t TGeoVGShape::Contains(const Double_t *point) const
{
// Test if point is inside this shape.
   return ( fVGShape->Contains(Vector3D<Double_t>(point[0], point[1], point[2])) );
}

//_____________________________________________________________________________
Double_t TGeoVGShape::DistFromInside(const Double_t *point, const Double_t *dir, Int_t /*iact*/, 
                                   Double_t step, Double_t * /*safe*/) const
{
   Double_t dist = fVGShape->DistanceToOut(Vector3D<Double_t>(point[0], point[1], point[2]),
                                    Vector3D<Double_t>(dir[0], dir[1], dir[2]), step);
   return ( (dist < 0.)? 0. : dist );
}

//_____________________________________________________________________________
Double_t TGeoVGShape::DistFromOutside(const Double_t *point, const Double_t *dir, Int_t /*iact*/, 
                                   Double_t step, Double_t * /*safe*/) const
{
   Double_t dist = fVGShape->DistanceToIn(Vector3D<Double_t>(point[0], point[1], point[2]),
                                    Vector3D<Double_t>(dir[0], dir[1], dir[2]), step);
   return ( (dist < 0.)? 0. : dist );
}

//_____________________________________________________________________________
Double_t TGeoVGShape::Safety(const Double_t *point, Bool_t in) const
{
   Double_t safety =  (in) ? fVGShape->SafetyToOut(Vector3D<Double_t>(point[0], point[1], point[2])) 
                           : fVGShape->SafetyToIn(Vector3D<Double_t>(point[0], point[1], point[2]));
   return ( (safety < 0.)? 0. : safety );
   
}
