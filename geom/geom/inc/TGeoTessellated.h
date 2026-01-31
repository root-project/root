// @(#)root/geom:$Id$// Author: Andrei Gheata   20/12/19

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoTessellated
#define ROOT_TGeoTessellated

#include <memory> // for unique_ptr
#include <vector> // for vector

#include "Rtypes.h"         // for THashConsistencyHolder, ClassDefOverride
#include "RtypesCore.h"     // for Double_t, Bool_t, Int_t, UInt_t, kTRUE
#include "TGeoBBox.h"       // for TGeoBBox
#include "TStopwatch.h"     // for TStopwatch

#include "Tessellated/TPartitioningI.h" // for TPartitioningI
#include "Tessellated/TGeoTriangleMesh.h"  // for TGeoTriangleMesh

class TBuffer3D;
class TBuffer;
class TClass;
class TGeoMatrix;
class TGeoShape;
class TMemberInspector;

class TGeoTessellated : public TGeoBBox {

   using TPartitioningI = Tessellated::TPartitioningI;
   using TGeoTriangleMesh = Tessellated::TGeoTriangleMesh;
   using TGeoTriangle = Tessellated::TGeoTriangle;

private:
   std::unique_ptr<TGeoTriangleMesh> fMesh{nullptr};                ///<  triangle mesh
   std::unique_ptr<TPartitioningI> fPartitioningStruct{nullptr}; ///<  partitioning structure
   std::vector<UInt_t> fUsedTriangles{};                         ///<! vector of indices of valid triangles
   mutable TStopwatch fTimer{}; ///<! timer to help determine timeconsuming TGeoTessellated instances
   Bool_t fPrintTime{kFALSE};
private:
   void FillBuffer3DWithPoints(TBuffer3D &b) const;
   void FillBuffer3DWithSegmentsAndPols(TBuffer3D &b, const std::vector<UInt_t> &indices) const;

protected:
   virtual void FillBuffer3D(TBuffer3D &b, Int_t reqSections, Bool_t localFrame) const override;

public:
   TGeoTessellated();
   TGeoTessellated(const char *);
   virtual ~TGeoTessellated() override;

   virtual Bool_t Contains(const Double_t *pointa) const override;
   virtual Double_t DistFromInside(const Double_t *pointa, const Double_t *dira, Int_t iact, Double_t step,
                                   Double_t *safe) const override;
   virtual Double_t DistFromOutside(const Double_t *pointa, const Double_t *dira, Int_t iact, Double_t step,
                                    Double_t *safe) const override;
   virtual Double_t Safety(const Double_t *pointa, bool inside) const override;

   virtual void ComputeBBox() override;
   virtual void ComputeNormal(const Double_t *pointa, const Double_t *dira, Double_t *norm) const override;
   virtual void GetBoundingCylinder(Double_t *param) const override;
   virtual void InspectShape() const override;
   virtual Int_t GetByteCount() const override;
   virtual Double_t Capacity() const override;
   virtual TGeoShape *GetMakeRuntimeShape(TGeoShape *shape, TGeoMatrix *matrix) const override;
   virtual Bool_t IsCylType() const override;
   virtual Bool_t IsValidBox() const override;
   virtual Bool_t GetPointsOnSegments(Int_t intl, Double_t *list) const override;
   virtual TBuffer3D *MakeBuffer3D() const override;
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const override;
   int DistancetoPrimitive(int, int) override { return 99999; }
   void SavePrimitive(std::ostream &, Option_t *) override {}
   void Sizeof3D() const override {}

   void SetPoints(Double_t *points) const override;
   void SetPoints(Float_t *points) const override;
   void GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const override;
   Int_t GetNmeshVertices() const override { return GetTriangleMesh()->Points().size(); }

   virtual void SetMesh(std::unique_ptr<TGeoTriangleMesh> mesh);
   TGeoTriangleMesh const *GetTriangleMesh() const { return fMesh.get(); }
   const std::vector<UInt_t> &GetUsedTriangleIndices() const { return fUsedTriangles; }

   void SetPartitioningStruct(std::unique_ptr<TPartitioningI> &partitioningStruct)
   {
      fPartitioningStruct.reset(partitioningStruct.release());
   }
   TPartitioningI const *GetPartitioningStruct() const { return fPartitioningStruct.get(); }

   void ResizeCenter(Double_t maxsize);
   void PrintTime(Bool_t flag) {fPrintTime = flag;}
   ClassDefOverride(TGeoTessellated, 1)
};

#endif /*ROOT_TGeoTessellated*/
