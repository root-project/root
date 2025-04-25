// Author: Mihaela Gheata   30/03/16

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoVGShape
#define ROOT_TGeoVGShape

#include "TGeoBBox.h"

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoVGShape - bridge class for using a VecGeom solid as TGeoShape.     //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

namespace vecgeom {
inline namespace cxx {
class Transformation3D;
class VPlacedVolume;
class VUnplacedVolume;
} // namespace cxx
} // namespace vecgeom

class TGeoVGShape : public TGeoBBox {
private:
   vecgeom::cxx::VPlacedVolume *fVGShape; //! VecGeom placed solid
   TGeoShape *fShape;                     // ROOT shape

   static vecgeom::cxx::VPlacedVolume *CreateVecGeomSolid(TGeoShape *shape);
   TGeoVGShape(TGeoShape *shape, vecgeom::cxx::VPlacedVolume *vgshape);

public:
   TGeoVGShape() : TGeoBBox(), fVGShape(nullptr), fShape(nullptr) {}
   ~TGeoVGShape() override;
   static vecgeom::cxx::Transformation3D *Convert(TGeoMatrix const *const geomatrix);
   static vecgeom::cxx::VUnplacedVolume *Convert(TGeoShape const *const shape);
   static TGeoVGShape *Create(TGeoShape *shape);
   Double_t Capacity() const override;
   void ComputeBBox() override;
   void ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm) const override;
   Bool_t Contains(const Double_t *point) const override;
   Bool_t CouldBeCrossed(const Double_t *point, const Double_t *dir) const override
   {
      return fShape->CouldBeCrossed(point, dir);
   }
   Int_t DistancetoPrimitive(Int_t px, Int_t py) override { return fShape->DistancetoPrimitive(px, py); }
   Double_t DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact = 1, Double_t step = TGeoShape::Big(),
                           Double_t *safe = nullptr) const override;
   Double_t DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact = 1,
                            Double_t step = TGeoShape::Big(), Double_t *safe = nullptr) const override;
   TGeoVolume *Divide(TGeoVolume *, const char *, Int_t, Int_t, Double_t, Double_t) override { return nullptr; }
   void Draw(Option_t *option = "") override { fShape->Draw(option); } // *MENU*
   const char *GetAxisName(Int_t iaxis) const override { return (fShape->GetAxisName(iaxis)); }
   Double_t GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const override
   {
      return (fShape->GetAxisRange(iaxis, xlo, xhi));
   }
   void GetBoundingCylinder(Double_t *param) const override { return (fShape->GetBoundingCylinder(param)); }
   const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const override
   {
      return (fShape->GetBuffer3D(reqSections, localFrame));
   }
   Int_t GetByteCount() const override { return (fShape->GetByteCount()); }
   Double_t Safety(const Double_t *point, Bool_t in = kTRUE) const override;
   Bool_t GetPointsOnSegments(Int_t npoints, Double_t *array) const override
   {
      return (fShape->GetPointsOnSegments(npoints, array));
   }
   Int_t
   GetFittingBox(const TGeoBBox *parambox, TGeoMatrix *mat, Double_t &dx, Double_t &dy, Double_t &dz) const override
   {
      return (fShape->GetFittingBox(parambox, mat, dx, dy, dz));
   }
   TGeoShape *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const override
   {
      return (fShape->GetMakeRuntimeShape(mother, mat));
   }
   void GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const override
   {
      fShape->GetMeshNumbers(nvert, nsegs, npols);
   }
   const char *GetName() const override { return (fShape->GetName()); }
   Int_t GetNmeshVertices() const override { return (fShape->GetNmeshVertices()); }
   Bool_t IsAssembly() const override { return (fShape->IsAssembly()); }
   Bool_t IsComposite() const override { return (fShape->IsComposite()); }
   Bool_t IsCylType() const override { return (fShape->IsCylType()); }
   Bool_t IsReflected() const override { return (fShape->IsReflected()); }
   Bool_t IsValidBox() const override { return (fShape->IsValidBox()); }
   Bool_t IsVecGeom() const override { return kTRUE; }
   void InspectShape() const override;
   TBuffer3D *MakeBuffer3D() const override { return (fShape->MakeBuffer3D()); }
   void Paint(Option_t *option = "") override { fShape->Paint(option); }
   void SetDimensions(Double_t *param) override { fShape->SetDimensions(param); }
   void SetPoints(Double_t *points) const override { fShape->SetPoints(points); }
   void SetPoints(Float_t *points) const override { fShape->SetPoints(points); }
   void SetSegsAndPols(TBuffer3D &buff) const override { fShape->SetSegsAndPols(buff); }
   void Sizeof3D() const override { fShape->Sizeof3D(); }

   TGeoShape *GetShape() const { return fShape; }
   vecgeom::cxx::VPlacedVolume *GetVGShape() const { return fVGShape; }

   ClassDefOverride(TGeoVGShape, 1) // Adapter for a VecGeom shape
};
#endif
