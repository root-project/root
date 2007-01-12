// @(#)root/g4root:$Name:  $:$Id: TG4RootSolid.cxx,v 1.2 2006/11/22 17:29:54 rdm Exp $
// Author: Andrei Gheata   07/08/06

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TG4RootSolid                                                         //
//                                                                      //
// GEANT4 solid implemented by a ROOT shape. Visualization methods      //
// not implemented.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGeoShape.h"
#include "TGeoBBox.h"

#include "G4VoxelLimits.hh"
#include "G4AffineTransform.hh"
#include "G4VPVParameterisation.hh"
#include "G4VGraphicsScene.hh"
#include "G4Polyhedron.hh"
#include "G4NURBS.hh"
#include "G4NURBSbox.hh"
#include "G4VisExtent.hh"

#include "TG4RootSolid.h"
#include "TMath.h"

//ClassImp(TG4RootSolid)
static const Double_t gCm = 1./cm;

//______________________________________________________________________________
TG4RootSolid::TG4RootSolid(TGeoShape *shape)
             :G4VSolid(shape->GetName())
{
// Constructor.
   fShape = shape;
}
   
//______________________________________________________________________________
G4bool TG4RootSolid::CalculateExtent(const EAxis /*pAxis*/,
                                     const G4VoxelLimits& /*pVoxelLimit*/,
				                         const G4AffineTransform& /*pTransform*/,
				                         G4double& /*pMin*/, G4double& /*pMax*/) const
{
// Calculate the minimum and maximum extent of the solid, when under the
// specified transform, and within the specified limits. If the solid
// is not intersected by the region, return false, else return true.
   G4cout << "Warning: TG4RootSolid::CalculateExtent() not implemented" << G4endl;
   return false;
}
   
//______________________________________________________________________________
EInside TG4RootSolid::Inside(const G4ThreeVector& p) const
{
// Returns kOutside if the point at offset p is outside the shapes
// boundaries plus Tolerance/2, kSurface if the point is <= Tolerance/2
// from a surface, otherwise kInside.
   Double_t pt[3];
   pt[0] = p.x()*gCm; pt[1] = p.y()*gCm; pt[2] = p.z()*gCm;
   Bool_t in = fShape->Contains(pt);
   // Temporary computation of safety due to the fact that TGeoShape does
   // not return kSurface
   G4double safety = fShape->Safety(pt, in)*cm;
   if (TMath::Abs(safety) < 0.5*kCarTolerance) return kSurface;
   if (in) return kInside;
   return kOutside;
}   

//______________________________________________________________________________
G4ThreeVector TG4RootSolid::SurfaceNormal(const G4ThreeVector& p) const
// Returns the outwards pointing unit normal of the shape for the
// surface closest to the point at offset p.
{
   Double_t pt[3], dir[3], norm[3];
   pt[0] = p.x()*gCm; pt[1] = p.y()*gCm; pt[2] = p.z()*gCm;
   dir[0] = 0.0;  dir[1] =0.0;   dir[2] = 1.0;
   fShape->ComputeNormal(pt,dir,norm);
   pt[0] += norm[0] * 2.* kCarTolerance; 
   pt[1] += norm[1] * 2.* kCarTolerance; 
   pt[2] += norm[2] * 2.* kCarTolerance;
   // Do a trick that should work if the point p is on the surface...
   G4ThreeVector n(norm[0], norm[1], norm[2]);
   Bool_t in = fShape->Contains(pt);
   if (!in) return n;
   n.set(-norm[0], -norm[1], -norm[2]);
   return n;
}   

//______________________________________________________________________________
G4double TG4RootSolid::DistanceToIn(const G4ThreeVector& p, const G4ThreeVector& v) const
{
// Return the distance along the normalised vector v to the shape,
// from the point at offset p. If there is no intersection, return
// kInfinity. The first intersection resulting from `leaving' a
// surface/volume is discarded. Hence, it is tolerant of points on
// the surface of the shape.
   Double_t pt[3], dir[3];
   pt[0] = p.x()*gCm; pt[1] = p.y()*gCm; pt[2] = p.z()*gCm;
   dir[0] = v.x(); dir[1] =v.y(); dir[2] = v.z();
   G4double dist = fShape->DistFromOutside(pt,dir,3)*cm;
   if (dist < TGeoShape::Big()) return dist;
   return kInfinity;
}      

//______________________________________________________________________________
G4double TG4RootSolid::DistanceToIn(const G4ThreeVector& p) const
{
// Calculate the distance to the nearest surface of a shape from an
// outside point. The distance can be an underestimate.
   Double_t pt[3];
   pt[0] = p.x()*gCm; pt[1] = p.y()*gCm; pt[2] = p.z()*gCm;
   G4double safety = fShape->Safety(pt, kFALSE)*cm;
   return safety;
}      

//______________________________________________________________________________
G4double TG4RootSolid::DistanceToOut(const G4ThreeVector& p,
				   const G4ThreeVector& v,
				   const G4bool calcNorm,
				   G4bool *validNorm,
				   G4ThreeVector *n) const
{
// Return the distance along the normalised vector v to the shape,
// from a point at an offset p inside or on the surface of the shape.
// Intersections with surfaces, when the point is < Tolerance/2 from a
// surface must be ignored.
// If calcNorm==true:
//    validNorm set true if the solid lies entirely behind or on the
//              exiting surface.
//    n set to exiting outwards normal vector (undefined Magnitude).
//    validNorm set to false if the solid does not lie entirely behind
//              or on the exiting surface
// If calcNorm==false:
//    validNorm and n are unused.
//
// Must be called as solid.DistanceToOut(p,v) or by specifying all
// the parameters.
   Double_t pt[3], dir[3], norm[3];
   pt[0] = p.x()*gCm; pt[1] = p.y()*gCm; pt[2] = p.z()*gCm;
   dir[0] = v.x(); dir[1] =v.y(); dir[2] = v.z();
   G4double dist = fShape->DistFromInside(pt,dir,3)*cm;
   if (calcNorm) *validNorm = true;
   if (dist < 0.5*kCarTolerance) dist = 0.;
   else {
      pt[0] += dist*dir[0];
      pt[1] += dist*dir[1];
      pt[2] += dist*dir[2];
   }   
   if (calcNorm) {
      fShape->ComputeNormal(pt,dir,norm);
      *n = G4ThreeVector(norm[0],norm[1],norm[2]);
   }
   return dist;
}         

//______________________________________________________________________________
G4double TG4RootSolid::DistanceToOut(const G4ThreeVector& p) const
{
// Calculate the distance to the nearest surface of a shape from an
// inside point. The distance can be an underestimate.
   Double_t pt[3];
   pt[0] = p.x()*gCm; pt[1] = p.y()*gCm; pt[2] = p.z()*gCm;
   G4double safety = fShape->Safety(pt, kTRUE)*cm;
   return safety;
}

//______________________________________________________________________________
void TG4RootSolid::ComputeDimensions(G4VPVParameterisation* /*p*/, const G4int /*n*/,
                       const G4VPhysicalVolume* /*pRep*/)
{
// Throw exception if ComputeDimensions called frrom an illegal
// derived class. It should not be called with this interface.
   G4cout << "Warning: TG4RootSolid::ComputeDimensions() not implemented" << G4endl;
}

//______________________________________________________________________________
G4double TG4RootSolid::GetCubicVolume()
{
// Returns an estimation of the solid volume in internal units.
// This method may be overloaded by derived classes to compute the
// exact geometrical quantity for solids where this is possible,
// or anyway to cache the computed value.
// Note: the computed value is NOT cached.
   G4double capacity = fShape->Capacity() * cm3;
   return capacity;
}

//______________________________________________________________________________
G4GeometryType TG4RootSolid::GetEntityType() const
{
// Provide identification of the class of an object.
// (required for persistency and STEP interface)
   return G4String(fShape->ClassName());
}   

//______________________________________________________________________________
G4ThreeVector TG4RootSolid::GetPointOnSurface() const
{
// Returns a random point located on the surface of the solid.
   G4cout << "Warning: TG4RootSolid::GetPointOnSurface() not implemented" << G4endl;
   return G4ThreeVector(0.,0.,0.);
}

//______________________________________________________________________________
std::ostream& TG4RootSolid::StreamInfo(std::ostream& os) const
{
// Dumps contents of the solid to a stream.
  os << "-----------------------------------------------------------\n"
     << "    *** Dump for solid - " << GetName() << " ***\n"
     << "    ===================================================\n"
     << " Solid type: ROOT solid / "<< fShape->ClassName() << "\n"
     << " Bounding box: \n"
     << "    half length X: " << ((TGeoBBox*)fShape)->GetDX()*cm/mm << " mm \n"
     << "    half length Y: " << ((TGeoBBox*)fShape)->GetDY()*cm/mm << " mm \n"
     << "    half length Z: " << ((TGeoBBox*)fShape)->GetDZ()*cm/mm << " mm \n"
     << "-----------------------------------------------------------\n";

  return os;
}

// Visualization functions

//______________________________________________________________________________
void TG4RootSolid::DescribeYourselfTo(G4VGraphicsScene& /*scene*/) const
{
// A "double dispatch" function which identifies the solid
// to the graphics scene.
}

//______________________________________________________________________________
G4VisExtent TG4RootSolid::GetExtent() const
{
// Provide extent (bounding box) as possible hint to the graphics view.
   G4double dx = ((TGeoBBox*)fShape)->GetDX()*cm;
   G4double dy = ((TGeoBBox*)fShape)->GetDY()*cm;
   G4double dz = ((TGeoBBox*)fShape)->GetDZ()*cm;
   const Double_t *origin = ((TGeoBBox*)fShape)->GetOrigin();
   G4double ox = origin[0]*cm;
   G4double oy = origin[1]*cm;
   G4double oz = origin[2]*cm;
   return G4VisExtent (-dx+ox, dx+ox, -dy+oy, dy+oy, -dz+oz, dz+oz);
}

//______________________________________________________________________________
G4Polyhedron* TG4RootSolid::CreatePolyhedron () const
{
   return NULL;
}
   
//______________________________________________________________________________
G4NURBS* TG4RootSolid::CreateNURBS() const
{
// Create a G4Polyhedron/G4NURBS/...  (It is the caller's responsibility
// to delete it).  A null pointer means "not created".
   return NULL;
}
   
//______________________________________________________________________________
G4Polyhedron* TG4RootSolid::GetPolyhedron () const
{
// Smart access function - creates on request and stores for future
// access.  A null pointer means "not available".
   return NULL;
}

//______________________________________________________________________________
const G4VSolid* TG4RootSolid::GetConstituentSolid(G4int /*no*/) const
{
   return NULL;
}

//______________________________________________________________________________
G4VSolid* TG4RootSolid::GetConstituentSolid(G4int /*no*/)
{
// If the solid is made up from a Boolean operation of two solids,
// return the "no" solid. If the solid is not a "Boolean", return 0.
   return NULL;
}

//______________________________________________________________________________
const G4DisplacedSolid* TG4RootSolid::GetDisplacedSolidPtr() const
{
   return NULL;
}

//______________________________________________________________________________
G4DisplacedSolid* TG4RootSolid::GetDisplacedSolidPtr()
{
   return NULL;
}
