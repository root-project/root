// @(#):$Id$
// Author: Mihaela Gheata   03/08/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_____________________________________________________________________________
// TGeoHalfSpace - A half-space defined by:
//            p[3] - an arbitrary point on the plane
//            n[3] - normal at the plane in point P
//    A half-space is not really a shape, because it is infinite. The normal
//    points "outside" the half-space
//_____________________________________________________________________________

#include "Riostream.h"
#include "TGeoHalfSpace.h"
#include "TMath.h"

ClassImp(TGeoHalfSpace)

//_____________________________________________________________________________
TGeoHalfSpace::TGeoHalfSpace()
{
// Dummy constructor
   SetShapeBit(TGeoShape::kGeoHalfSpace);
   SetShapeBit(TGeoShape::kGeoInvalidShape);
   memset(fP, 0, 3*sizeof(Double_t));
   memset(fN, 0, 3*sizeof(Double_t));
}

//_____________________________________________________________________________
TGeoHalfSpace::TGeoHalfSpace(const char *name, Double_t *p, Double_t *n)
              :TGeoBBox(name, 0,0,0)
{
// Constructor with name, point on the plane and normal
   SetShapeBit(TGeoShape::kGeoHalfSpace);
   SetShapeBit(TGeoShape::kGeoInvalidShape);
   Double_t param[6];
   memcpy(param, p, 3*sizeof(Double_t));
   memcpy(&param[3], n, 3*sizeof(Double_t));
   SetDimensions(param);
}

//_____________________________________________________________________________
TGeoHalfSpace::TGeoHalfSpace(Double_t *param)
              :TGeoBBox(0,0,0)
{
// Default constructor specifying minimum and maximum radius
   SetShapeBit(TGeoShape::kGeoHalfSpace);
   SetShapeBit(TGeoShape::kGeoInvalidShape);
   SetDimensions(param);
}

//_____________________________________________________________________________
TGeoHalfSpace::~TGeoHalfSpace()
{
// destructor
}

//_____________________________________________________________________________
void TGeoHalfSpace::ComputeNormal(const Double_t * /*point*/, const Double_t *dir, Double_t *norm)
{
// Compute normal to closest surface from POINT.
   memcpy(norm, fN, 3*sizeof(Double_t));
   if (norm[0]*dir[0]+norm[1]*dir[1]+norm[2]*dir[2]<0) {
      norm[0] = -norm[0];
      norm[1] = -norm[1];
      norm[2] = -norm[2];
   }
}

//_____________________________________________________________________________
Bool_t TGeoHalfSpace::Contains(const Double_t *point) const
{
// test if point is inside the half-space
   Double_t r[3];
   r[0] = fP[0]-point[0];
   r[1] = fP[1]-point[1];
   r[2] = fP[2]-point[2];
   Double_t rdotn = r[0]*fN[0]+r[1]*fN[1]+r[2]*fN[2];
   if (rdotn < 0) return kFALSE;
   return kTRUE;
}

//_____________________________________________________________________________
Int_t TGeoHalfSpace::DistancetoPrimitive(Int_t /*px*/, Int_t /*py*/)
{
// A half-space does not have a mesh primitive
   return 999;
}

//_____________________________________________________________________________
Double_t TGeoHalfSpace::DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from inside point to the plane
   Double_t r[3];
   r[0] = fP[0]-point[0];
   r[1] = fP[1]-point[1];
   r[2] = fP[2]-point[2];
   Double_t rdotn = r[0]*fN[0]+r[1]*fN[1]+r[2]*fN[2];
   if (iact<3 && safe) {
      *safe = rdotn;
      if (iact==0) return TGeoShape::Big();
      if ((iact==1) && (*safe>step)) return TGeoShape::Big();
   }
   // compute distance to plane
   Double_t snxt = TGeoShape::Big();
   Double_t ddotn = dir[0]*fN[0]+dir[1]*fN[1]+dir[2]*fN[2];
   if (TMath::Abs(ddotn)<TGeoShape::Tolerance()) return snxt;
   snxt = rdotn/ddotn;
   if (snxt<0) return TGeoShape::Big();
   return snxt;
}

//_____________________________________________________________________________
Double_t TGeoHalfSpace::DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from inside point to the plane
   Double_t r[3];
   r[0] = fP[0]-point[0];
   r[1] = fP[1]-point[1];
   r[2] = fP[2]-point[2];
   Double_t rdotn = r[0]*fN[0]+r[1]*fN[1]+r[2]*fN[2];
   if (iact<3 && safe) {
      *safe = -rdotn;
      if (iact==0) return TGeoShape::Big();
      if ((iact==1) && (step<*safe)) return TGeoShape::Big();
   }
   // compute distance to plane
   Double_t snxt = TGeoShape::Big();
   Double_t ddotn = dir[0]*fN[0]+dir[1]*fN[1]+dir[2]*fN[2];
   if (TMath::Abs(ddotn)<TGeoShape::Tolerance()) return snxt;
   snxt = rdotn/ddotn;
   if (snxt<0) return TGeoShape::Big();
   return snxt;
}

//_____________________________________________________________________________
TGeoVolume *TGeoHalfSpace::Divide(TGeoVolume * /*voldiv*/, const char * /*divname*/, Int_t /*iaxis*/, Int_t /*ndiv*/,
                             Double_t /*start*/, Double_t /*step*/)
{
// Divide the shape along one axis.
   Error("Divide", "Half-spaces cannot be divided");
   return 0;
}

//_____________________________________________________________________________
void TGeoHalfSpace::GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const
{
// Returns numbers of vertices, segments and polygons composing the shape mesh.
   nvert = 0;
   nsegs = 0;
   npols = 0;
}

//_____________________________________________________________________________
void TGeoHalfSpace::InspectShape() const
{
// print shape parameters
   printf("*** Shape %s: TGeoHalfSpace ***\n", GetName());
   printf("    Point    : %11.5f, %11.5f, %11.5f\n", fP[0], fP[1], fP[2]);
   printf("    Normal   : %11.5f, %11.5f, %11.5f\n", fN[0], fN[1], fN[2]);
}

//_____________________________________________________________________________
Double_t TGeoHalfSpace::Safety(const Double_t *point, Bool_t /*in*/) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   Double_t r[3];
   r[0] = fP[0]-point[0];
   r[1] = fP[1]-point[1];
   r[2] = fP[2]-point[2];
   Double_t rdotn = r[0]*fN[0]+r[1]*fN[1]+r[2]*fN[2];
   return TMath::Abs(rdotn);
}

//_____________________________________________________________________________
void TGeoHalfSpace::SavePrimitive(std::ostream &out, Option_t * /*option*/ /*= ""*/)
{
// Save a primitive as a C++ statement(s) on output stream "out".
   if (TObject::TestBit(kGeoSavePrimitive)) return;
   out << "   // Shape: " << GetName() << " type: " << ClassName() << std::endl;
   out << "   point[0] = " << fP[0] << ";" << std::endl;
   out << "   point[1] = " << fP[1] << ";" << std::endl;
   out << "   point[2] = " << fP[2] << ";" << std::endl;
   out << "   norm[0]  = " << fN[0] << ";" << std::endl;
   out << "   norm[1]  = " << fN[1] << ";" << std::endl;
   out << "   norm[2]  = " << fN[2] << ";" << std::endl;
   out << "   TGeoShape *" << GetPointerName() << " = new TGeoHalfSpace(\"" << GetName() << "\", point,norm);" << std::endl;
   TObject::SetBit(TGeoShape::kGeoSavePrimitive);
}

//_____________________________________________________________________________
void TGeoHalfSpace::SetDimensions(Double_t *param)
{
// Set half-space parameters as stored in an array.
   memcpy(fP, param, 3*sizeof(Double_t));
   memcpy(fN, &param[3], 3*sizeof(Double_t));
   Double_t nsq = TMath::Sqrt(fN[0]*fN[0]+fN[1]*fN[1]+fN[2]*fN[2]);
   fN[0] /= nsq;
   fN[1] /= nsq;
   fN[2] /= nsq;
}

//_____________________________________________________________________________
void TGeoHalfSpace::Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const
{
// Check the inside status for each of the points in the array.
// Input: Array of point coordinates + vector size
// Output: Array of Booleans for the inside of each point
   for (Int_t i=0; i<vecsize; i++) inside[i] = Contains(&points[3*i]);
}

//_____________________________________________________________________________
void TGeoHalfSpace::ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize)
{
// Compute the normal for an array o points so that norm.dot.dir is positive
// Input: Arrays of point coordinates and directions + vector size
// Output: Array of normal directions
   for (Int_t i=0; i<vecsize; i++) ComputeNormal(&points[3*i], &dirs[3*i], &norms[3*i]);
}

//_____________________________________________________________________________
void TGeoHalfSpace::DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t* step) const
{
// Compute distance from array of input points having directions specisied by dirs. Store output in dists
   for (Int_t i=0; i<vecsize; i++) dists[i] = DistFromInside(&points[3*i], &dirs[3*i], 3, step[i]);
}

//_____________________________________________________________________________
void TGeoHalfSpace::DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t* step) const
{
// Compute distance from array of input points having directions specisied by dirs. Store output in dists
   for (Int_t i=0; i<vecsize; i++) dists[i] = DistFromOutside(&points[3*i], &dirs[3*i], 3, step[i]);
}

//_____________________________________________________________________________
void TGeoHalfSpace::Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const
{
// Compute safe distance from each of the points in the input array.
// Input: Array of point coordinates, array of statuses for these points, size of the arrays
// Output: Safety values
   for (Int_t i=0; i<vecsize; i++) safe[i] = Safety(&points[3*i], inside[i]);
}
