// @(#)root/geom:$Id$// Author: Ben Salisbury   21/11/24

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGeoTriangle
\ingroup Geometry_classes

TGeoTriangle (replacement for TGeoFacet) to represent a single triangle face
of a triangle mesh for use with TGeoTessellated. Contains functionality to
detect ray triangle intersections and closest point to point computation.
*/

#include "Tessellated/TGeoTriangle.h"

#include <cstdlib>  // for abs
#include <iostream> // for operator<<, cout, ostream, endl, basic_ostream
#include <limits>   // for numeric_limits

#include "TGeoShape.h" // for TGeoShape

namespace Tessellated {

///////////////////////////////////////////////////////////////////////////////
// ClassImp(TGeoTriangle);

namespace XYZVectorHelper {
// Helper functions to make the transition from TVector3 to XYZVector a little easier.

std::array<Double_t, 3> ToArray(const ROOT::Math::XYZVector &vec)
{
   std::array<Double_t, 3> a;
   vec.GetCoordinates(a.begin(), a.end());
   return a;
}

ROOT::Math::XYZVector Orthogonal(const ROOT::Math::XYZVector &vec)
{
   const Double_t fX = vec.X();
   const Double_t fY = vec.Y();
   const Double_t fZ = vec.Z();

   Double_t xx = fX < 0.0 ? -fX : fX;
   Double_t yy = fY < 0.0 ? -fY : fY;
   Double_t zz = fZ < 0.0 ? -fZ : fZ;
   if (xx < yy) {
      return xx < zz ? ROOT::Math::XYZVector(0, fZ, -fY) : ROOT::Math::XYZVector(fY, -fX, 0);
   } else {
      return yy < zz ? ROOT::Math::XYZVector(-fZ, 0, fX) : ROOT::Math::XYZVector(fY, -fX, 0);
   }
}

void Print(const ROOT::Math::XYZVector &vec)
{
   Printf("ROOT::Math::XYZVector (x,y,z)=(%f,%f,%f) (rho,theta,phi)=(%f,%f,%f)", vec.X(), vec.Y(), vec.Z(), TMath::Sqrt(vec.Mag2()),
          vec.Theta() * TMath::RadToDeg(), vec.Phi() * TMath::RadToDeg());
}
}; // namespace XYZVectorHelper

namespace TGeoTriangleInternal {

////////////////////////////////////////////////////////////////////////////////
/// Test if two doubles are identical

Bool_t EqualTo(Double_t a, Double_t b, Double_t accuracy)
{
   return std::abs(a - b) < accuracy;
}

////////////////////////////////////////////////////////////////////////////////
/// Test if double a is smaller than double b
Bool_t SmallerThan(Double_t a, Double_t b, Double_t accuracy)
{
   return (a - accuracy) < b;
}

////////////////////////////////////////////////////////////////////////////////
/// Test if double a is larger than double b
Bool_t LargerThan(Double_t a, Double_t b, Double_t accuracy)
{
   return (a + accuracy) > b;
}

}; // namespace TGeoTriangleInternal

////////////////////////////////////////////////////////////////////////////////
TGeoTriangle::TGeoTriangle(const std::array<UInt_t, 3> &indices) : TObject(), fIndices(indices) {}

////////////////////////////////////////////////////////////////////////////////
TGeoTriangle::TGeoTriangle(const std::vector<ROOT::Math::XYZVector> *points, const std::array<UInt_t, 3> &indices)
   : TObject(), fIndices(indices)
{
   SetPoints(points);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the center, normal and the largest distance from center to the
/// corner points as "radius", allowing for quick point in triangle exclusions

void TGeoTriangle::Setup()
{
   fCenter = CalculateCenter();
   fNormal = CalculateNormal();
}

////////////////////////////////////////////////////////////////////////////////
/// Flip the sequence of the vertices, essentially flipping the triangle
/// orientation

void TGeoTriangle::Flip()
{
   Int_t iv0 = fIndices[0];
   fIndices[0] = fIndices[2];
   fIndices[2] = iv0;
   Setup();
}
////////////////////////////////////////////////////////////////////////////////
/// Compute the center of the triangle
/// \return ROOT::Math::XYZVector describing the triangle center

ROOT::Math::XYZVector TGeoTriangle::CalculateCenter() const
{
   ROOT::Math::XYZVector center{0.0, 0.0, 0.0};

   for (size_t index = 0; index < (size_t)sNumberOfVertices; ++index) {
      center += Point(index);
   }

   center *= (1.0 / 3.0);
   return center;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the normal of the triangle. The normals of the triangles in the mesh
/// have to point outwards.
/// \return ROOT::Math::XYZVector describing the triangle normal

ROOT::Math::XYZVector TGeoTriangle::CalculateNormal() const
{
   return ((Point(1) - Point(0)).Cross(Point(2) - Point(1))).Unit();
}

////////////////////////////////////////////////////////////////////////////////
/// Check if a connected neighbour facet has compatible normal

Bool_t TGeoTriangle::IsNeighbour(const TGeoTriangle &other, Bool_t &requireFlip) const
{

   // Find a connecting segment
   Bool_t neighbour = kFALSE;
   Int_t line1[2], line2[2];
   Int_t npoints = 0;
   for (Int_t i = 0; i < sNumberOfVertices; ++i) {
      UInt_t ivert = fIndices[i];
      // Check if the other facet has the same vertex

      for (Int_t j = 0; j < sNumberOfVertices; ++j) {
         if (ivert == other.Indices()[j]) {
            line1[npoints] = i;
            line2[npoints] = j;
            if (++npoints == 2) {
               neighbour = kTRUE;
               Bool_t order1 = line1[1] == line1[0] + 1;
               Bool_t order2 = line2[1] == (line2[0] + 1) % sNumberOfVertices;
               requireFlip = (order1 == order2);
               return neighbour;
            }
         }
      }
   }
   return neighbour;
}

////////////////////////////////////////////////////////////////////////////////
/// Moeller-Trumbore ray triangle-intersection
/// Compute distance from origin along line until triangle is hit. Returns
/// std::numeric_limits<double>::infinity() if not hit. Implementation taken from Sandro Wenzel's
/// ROOT PR https://github.com/root-project/root/pull/21045
///
/// \param[in] origin
/// \param[in] direction
/// \return Distance from origin to triangle along direction

Double_t TGeoTriangle::DistanceFrom(const ROOT::Math::XYZVector &origin, const ROOT::Math::XYZVector &direction) const
{
   constexpr double EPS = sAccuracy;
   constexpr double rayEPS = sAccuracy;
   const ROOT::Math::XYZVector &v0 = Point(0);
   const ROOT::Math::XYZVector &v1 = Point(1);
   const ROOT::Math::XYZVector &v2 = Point(2);

   ROOT::Math::XYZVector e1{v1.X() - v0.X(), v1.Y() - v0.Y(), v1.Z() - v0.Z()};
   ROOT::Math::XYZVector e2{v2.X() - v0.X(), v2.Y() - v0.Y(), v2.Z() - v0.Z()};
   auto p = direction.Cross(e2);
   auto det = e1.Dot(p);
   if (std::abs(det) <= EPS) {
      return sINF;
   }

   ROOT::Math::XYZVector tvec{origin.X() - v0.X(), origin.Y() - v0.Y(), origin.Z() - v0.Z()};
   auto invDet = 1.0 / det;
   auto u = tvec.Dot(p) * invDet;
   if (u < 0.0 || u > 1.0) {
      return sINF;
   }
   auto q = tvec.Cross(e1);
   auto v = direction.Dot(q) * invDet;
   if (v < 0.0 || u + v > 1.0) {
      return sINF;
   }
   auto t = e2.Dot(q) * invDet;
   // If t is larger than rayEPS we have a intersection in the direction of the ray, otherwise it is a intersection in
   // the opposite direction return (t > rayEPS) ? t : sINF; Actually, we want to also return the distances in the
   // opposite direction. It allows to cross check certain things. It is also helpful with origin almost on triangle
   // situations
   return t;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate the closest point on the triangle to point
///
/// \param[in] point
/// \return closest point on triangle to point

ROOT::Math::XYZVector TGeoTriangle::ClosestPointToPoint(const ROOT::Math::XYZVector &point) const
{
   Double_t t = DistanceFrom(point, fNormal);
   // Point is not projectable onto triangle
   if (t == sINF) {
      return ClosestPointOfEdgesToPoint(point);
   }
   // Otherwise, closest point on triangle is projection:
   return point + t * fNormal;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculates Closest point of triangle edges to given point point
///
/// \param[in] point
/// return closestPoint

ROOT::Math::XYZVector TGeoTriangle::ClosestPointOfEdgesToPoint(const ROOT::Math::XYZVector &point) const
{
   double_t smallestdistance = sINF;
   ROOT::Math::XYZVector edgedir{0.0, 0.0, 0.0};
   ROOT::Math::XYZVector closestpoint{0.0, 0.0, 0.0};
   ROOT::Math::XYZVector current{0.0, 0.0, 0.0};
   for (UInt_t index = 0; index < sNumberOfVertices; ++index) {
      const ROOT::Math::XYZVector &startedge = Point((index + 1) % sNumberOfVertices);
      const ROOT::Math::XYZVector &endedge = Point(index);
      edgedir = endedge - startedge;

      current = ClosestPointOfEdgeToPoint(point, startedge, edgedir);
      double_t distance = TMath::Sqrt((closestpoint - point).Mag2());
      if (smallestdistance < distance) {
         smallestdistance = distance;
         closestpoint = current;
      }
   }
   return closestpoint;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate closest point of edge to point
///
/// \param[in] point
/// \param[in] edge
/// \param[in] edgedirection
/// return closest point

ROOT::Math::XYZVector TGeoTriangle::ClosestPointOfEdgeToPoint(const ROOT::Math::XYZVector &point,
                                                              const ROOT::Math::XYZVector &edge,
                                                              const ROOT::Math::XYZVector &edgedirection) const
{
   ROOT::Math::XYZVector edgetopoint = point - edge;
   Double_t isleft = edgetopoint.Dot(edgedirection);
   if (TGeoTriangleInternal::SmallerThan(isleft, 0, sAccuracy)) {
      return edge;
   }
   Double_t isright = edgedirection.Dot(edgedirection);
   if (TGeoTriangleInternal::LargerThan(isleft, isright, sAccuracy)) {
      return (edge + edgedirection);
   }
   Double_t scale = isleft / isright;
   return edge + (edgedirection * scale);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the area of the triangle
///
/// \return Double_t giving the area of the triangle

Double_t TGeoTriangle::Area() const
{
   ROOT::Math::XYZVector result = ROOT::Math::XYZVector{0.0, 0.0, 0.0};

   for (UInt_t ptn = 0; ptn < sNumberOfVertices; ++ptn) {
      result += Point(ptn).Cross(Point((ptn + 1) % sNumberOfVertices));
   }

   return std::abs(0.5 * fNormal.Dot(result));
}

////////////////////////////////////////////////////////////////////////////////
/// Check validity of triangle

Bool_t TGeoTriangle::IsValid() const
{
   constexpr double kTolerance = 1.e-10;
   Bool_t is_valid = kTRUE;
   const ROOT::Math::XYZVector e1 = Point(1) - Point(0);
   if (e1.Mag2() < kTolerance) {
      std::cout << "Triangle edge e1 is degenerated\n";
      is_valid = kFALSE;
   }
   const ROOT::Math::XYZVector e2 = Point(2) - Point(1);
   if (e2.Mag2() < kTolerance) {
      std::cout << "Triangle edge e2 is degenerated\n";
      is_valid = kFALSE;
   }
   const ROOT::Math::XYZVector e3 = Point(0) - Point(2);
   if (e3.Mag2() < kTolerance) {
      std::cout << "Triangle edge e3 is degenerated\n";
      is_valid = kFALSE;
   }
   const ROOT::Math::XYZVector normal = e1.Cross(e2);
   if (normal.Mag2() < kTolerance) {
      std::cout << "Triangle normal is degenerated\n";
      is_valid = kFALSE;
   }

   // Compute surface area
   Double_t surfaceArea = Area();
   if (surfaceArea < kTolerance) {
      std::cout << "TGeoTriangle has zero surface area\n";
      is_valid = kFALSE;
   }

   return is_valid;
}

////////////////////////////////////////////////////////////////////////////////
/// Print the triangle information to stdout

void TGeoTriangle::Print(Option_t * /*option*/) const
{
   std::cout << "TGeoTriangle:\n";
   Tessellated::XYZVectorHelper::Print(Point(0));
   Tessellated::XYZVectorHelper::Print(Point(1));
   Tessellated::XYZVectorHelper::Print(Point(2));
   std::cout << "Normal \n";
   Tessellated::XYZVectorHelper::Print(fNormal);
   std::cout << "Center \n";
   Tessellated::XYZVectorHelper::Print(fCenter);
   std::cout << "Area: " << Area();

   std::cout << std::endl;
   return;
}

}; // namespace Tessellated