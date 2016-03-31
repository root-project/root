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
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoVolume.h"

//ClassImp(TGeoVGShape)

//_____________________________________________________________________________
TGeoVGShape::TGeoVGShape(TGeoShape *shape)
           :TGeoBBox(shape->GetName(), 0, 0, 0), fVGShape(nullptr), fShape(shape)
{
// Default constructor
   fVGShape = CreateVecGeomSolid(shape);
   const TGeoBBox *box = (const TGeoBBox*)shape;
   TGeoBBox::SetBoxDimensions(box->GetDX(), box->GetDY(), box->GetDZ());   
   memcpy(fOrigin, box->GetOrigin(), 3*sizeof(Double_t));
}

//_____________________________________________________________________________
TGeoVGShape::~TGeoVGShape()
{
// Destructor
   delete fVGShape;
}   

//_____________________________________________________________________________
VPlacedVolume *TGeoVGShape::CreateVecGeomSolid(TGeoShape *shape) const
{
// Create VecGeom solid corresponding to TGeoShape
   RootGeoManager::Instance().set_verbose(1);
   VUnplacedVolume *unplaced = RootGeoManager::Instance().Convert(shape);
   if (dynamic_cast<vecgeom::UnplacedRootVolume*>(unplaced)) {
      delete unplaced;
      return nullptr;
   }
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
   return ( fVGShape->Contains(Vector3D<double>(point[0], point[1], point[2])) );
}

//_____________________________________________________________________________
Double_t TGeoVGShape::DistFromInside(const Double_t *point, const Double_t *dir, Int_t /*iact*/, 
                                   Double_t step, Double_t * /*safe*/) const
{
   return ( fVGShape->DistanceToOut(Vector3D<double>(point[0], point[1], point[2]),
                                    Vector3D<double>(dir[0], dir[1], dir[2]),
                                    step) );
}

//_____________________________________________________________________________
Double_t TGeoVGShape::DistFromOutside(const Double_t *point, const Double_t *dir, Int_t /*iact*/, 
                                   Double_t step, Double_t * /*safe*/) const
{
   return ( fVGShape->DistanceToIn(Vector3D<double>(point[0], point[1], point[2]),
                                    Vector3D<double>(dir[0], dir[1], dir[2]),
                                    step) );
}

//_____________________________________________________________________________
Double_t TGeoVGShape::Safety(const Double_t *point, Bool_t in) const
{
   return ( (in) ? fVGShape->SafetyToOut(Vector3D<double>(point[0], point[1], point[2])) 
                 : fVGShape->SafetyToIn(Vector3D<double>(point[0], point[1], point[2])));
}
