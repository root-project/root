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
   // Convert TGeoShape to PlacedVolume (navigation interface is there)
   fVGShape = CreateVecGeomSolid(shape);
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
VPlacedVolume *TGeoVGShape::CreateVecGeomSolid(TGeoShape *shape) const
{
// Conversion method to create VecGeom solid corresponding to TGeoShape
   // Initialize verbosity to catch possible error messages from VecGeom converter
   RootGeoManager::Instance().set_verbose(1);
   // Call VecGeom TGeoShape->UnplacedSolid converter
   VUnplacedVolume *unplaced = RootGeoManager::Instance().Convert(shape);
   // If ROOT shape is not supported, VecGeom will return a UnplacedRootVolume object
   // which is the inverse bridge UnplacedVolume->TGeoShape
   if (dynamic_cast<vecgeom::UnplacedRootVolume*>(unplaced)) {
      // We prefer a null pointer in order to use the original ROOT shape
      // so we have to cleanup what VecGeom just created
      delete unplaced;
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
   return ( fVGShape->DistanceToOut(Vector3D<Double_t>(point[0], point[1], point[2]),
                                    Vector3D<Double_t>(dir[0], dir[1], dir[2]),
                                    step) );
}

//_____________________________________________________________________________
Double_t TGeoVGShape::DistFromOutside(const Double_t *point, const Double_t *dir, Int_t /*iact*/, 
                                   Double_t step, Double_t * /*safe*/) const
{
   return ( fVGShape->DistanceToIn(Vector3D<Double_t>(point[0], point[1], point[2]),
                                    Vector3D<Double_t>(dir[0], dir[1], dir[2]),
                                    step) );
}

//_____________________________________________________________________________
Double_t TGeoVGShape::Safety(const Double_t *point, Bool_t in) const
{
   return ( (in) ? fVGShape->SafetyToOut(Vector3D<Double_t>(point[0], point[1], point[2])) 
                 : fVGShape->SafetyToIn(Vector3D<Double_t>(point[0], point[1], point[2])));
}
