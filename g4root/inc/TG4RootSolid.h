// @(#)root/g4root:$Name:  $:$Id: TG4RootSolid.h,v 1.1 2006/11/22 17:14:01 brun Exp $
// Author: Andrei Gheata   07/08/06

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TG4RootSolid
#define ROOT_TG4RootSolid


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TG4RootSolid                                                         //
//                                                                      //
// GEANT4 solid implemented by a ROOT shape.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef G4VSOLID_HH
#include "G4VSolid.hh"
#endif

class TGeoShape;

class TG4RootSolid : public G4VSolid {

protected:
   TGeoShape            *fShape;      // TGeo associated shape

public:
   TG4RootSolid() : G4VSolid(""), fShape(0) {}
   TG4RootSolid(TGeoShape *shape);
   virtual ~TG4RootSolid() {}

   virtual G4bool CalculateExtent(const EAxis pAxis,
			   const G4VoxelLimits& pVoxelLimit,
			   const G4AffineTransform& pTransform,
			   G4double& pMin, G4double& pMax) const;
   virtual EInside Inside(const G4ThreeVector& p) const;
   virtual G4ThreeVector SurfaceNormal(const G4ThreeVector& p) const;
   virtual G4double DistanceToIn(const G4ThreeVector& p,
                                 const G4ThreeVector& v) const;
   virtual G4double DistanceToIn(const G4ThreeVector& p) const;
   virtual G4double DistanceToOut(const G4ThreeVector& p,
			   const G4ThreeVector& v,
			   const G4bool calcNorm=false,
			   G4bool *validNorm=0,
			   G4ThreeVector *n=0) const;
   virtual G4double DistanceToOut(const G4ThreeVector& p) const;
   virtual void ComputeDimensions(G4VPVParameterisation* p,
	                               const G4int n,
                                  const G4VPhysicalVolume* pRep);
   virtual G4double GetCubicVolume();
   virtual G4GeometryType  GetEntityType() const;
   virtual G4ThreeVector GetPointOnSurface() const;
   virtual std::ostream& StreamInfo(std::ostream& os) const;
   virtual void DescribeYourselfTo (G4VGraphicsScene& scene) const;
   virtual G4VisExtent   GetExtent        () const;
   virtual G4Polyhedron* CreatePolyhedron () const;
   virtual G4NURBS*      CreateNURBS      () const;
   virtual G4Polyhedron* GetPolyhedron () const;
   virtual const G4VSolid* GetConstituentSolid(G4int no) const;
   virtual       G4VSolid* GetConstituentSolid(G4int no);
   virtual const G4DisplacedSolid* GetDisplacedSolidPtr() const; 
   virtual       G4DisplacedSolid* GetDisplacedSolidPtr(); 

//   ClassDef(TG4RootSolid,0)  // Class defining a G4Solid based on a ROOT shape
};
#endif
