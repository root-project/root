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
  }
}

class TGeoVGShape : public TGeoBBox
{
private:
   vecgeom::cxx::VPlacedVolume *fVGShape;      // VecGeom placed solid
   TGeoShape            *fShape;        // ROOT shape

   static vecgeom::cxx::VPlacedVolume *CreateVecGeomSolid(TGeoShape *shape);
   TGeoVGShape(TGeoShape *shape, vecgeom::cxx::VPlacedVolume *vgshape);

public:
   TGeoVGShape() : TGeoBBox(), fVGShape(nullptr), fShape(nullptr) {}
   virtual ~TGeoVGShape();
   static vecgeom::cxx::Transformation3D *
                         Convert(TGeoMatrix const *const geomatrix);
   static vecgeom::cxx::VUnplacedVolume *
                         Convert(TGeoShape const *const shape);
   static TGeoVGShape   *Create(TGeoShape *shape);
   virtual Double_t      Capacity() const;
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(const Double_t *point) const;
   virtual Bool_t        CouldBeCrossed(const Double_t *point, const Double_t *dir) const
                            { return fShape->CouldBeCrossed(point,dir); }
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py)
                            { return fShape->DistancetoPrimitive(px, py); }
   virtual Double_t      DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual TGeoVolume   *Divide(TGeoVolume *, const char *, Int_t, Int_t, Double_t, Double_t)
                            { return nullptr; }
   virtual void          Draw(Option_t *option="") { fShape->Draw(option); } // *MENU*
   virtual const char   *GetAxisName(Int_t iaxis) const
                            { return ( fShape->GetAxisName(iaxis) ); }
   virtual Double_t      GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const
                            { return ( fShape->GetAxisRange(iaxis, xlo, xhi) ); }
   virtual void          GetBoundingCylinder(Double_t *param) const
                            { return ( fShape->GetBoundingCylinder(param) ); }
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const
                            { return ( fShape->GetBuffer3D(reqSections, localFrame) ); }
   virtual Int_t         GetByteCount() const { return ( fShape->GetByteCount() ); }
   virtual Double_t      Safety(const Double_t *point, Bool_t in=kTRUE) const;
   virtual Bool_t        GetPointsOnSegments(Int_t npoints, Double_t *array) const
                            { return ( fShape->GetPointsOnSegments(npoints, array) ); }
   virtual Int_t         GetFittingBox(const TGeoBBox *parambox, TGeoMatrix *mat, Double_t &dx, Double_t &dy, Double_t &dz) const
                            { return ( fShape->GetFittingBox(parambox, mat, dx, dy, dz) ); }
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const
                            { return ( fShape->GetMakeRuntimeShape(mother, mat) ); }
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const
                            { fShape->GetMeshNumbers(nvert, nsegs, npols); }
   virtual const char   *GetName() const
                            { return ( fShape->GetName() ); }
   virtual Int_t         GetNmeshVertices() const
                            {return ( fShape->GetNmeshVertices() );}
   virtual Bool_t        IsAssembly() const { return ( fShape->IsAssembly() ); }
   virtual Bool_t        IsComposite() const { return ( fShape->IsComposite() ); }
   virtual Bool_t        IsCylType() const { return ( fShape->IsCylType() ); }
   virtual Bool_t        IsReflected() const { return ( fShape->IsReflected() ); }
   virtual Bool_t        IsValidBox() const  { return ( fShape->IsValidBox() ); }
   virtual Bool_t        IsVecGeom() const {return kTRUE;}
   virtual void          InspectShape() const;
   virtual TBuffer3D    *MakeBuffer3D() const { return ( fShape->MakeBuffer3D() );}
   virtual void          Paint(Option_t *option="") { fShape->Paint(option); }
   virtual void          SetDimensions(Double_t *param) { fShape->SetDimensions(param); }
   virtual void          SetPoints(Double_t *points) const { fShape->SetPoints(points); }
   virtual void          SetPoints(Float_t *points) const { fShape->SetPoints(points); }
   virtual void          SetSegsAndPols(TBuffer3D &buff) const { fShape->SetSegsAndPols(buff); }
   virtual void          Sizeof3D() const { fShape->Sizeof3D(); }

   TGeoShape            *GetShape() const { return fShape; }
   vecgeom::cxx::VPlacedVolume *GetVGShape() const { return fVGShape; }

   ClassDef(TGeoVGShape, 0) // Adapter for a VecGeom shape
};
#endif
