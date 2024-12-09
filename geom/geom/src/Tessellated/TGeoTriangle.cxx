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
TGeoTriangle::TGeoTriangle(const std::vector<TVector3> *points, const std::array<UInt_t, 3> &indices)
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
   Double_t radiussqr = (Point(0) - fCenter).Mag2();
   fRadiusSqr = radiussqr;
   radiussqr = (Point(1) - fCenter).Mag2();
   if (radiussqr > fRadiusSqr) {
      fRadiusSqr = radiussqr;
   }
   radiussqr = (Point(2) - fCenter).Mag2();
   if (radiussqr > fRadiusSqr) {
      fRadiusSqr = radiussqr;
   }
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
/// \return TVector3 describing the triangle center

TVector3 TGeoTriangle::CalculateCenter() const
{
   TVector3 center{0.0, 0.0, 0.0};

   for (size_t index = 0; index < (size_t)sNumberOfVertices; ++index) {
      center += Point(index);
   }

   center *= (1.0 / 3.0);
   return center;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the normal of the triangle. The normals of the triangles in the mesh
/// have to point outwards.
/// \return TVector3 describing the triangle normal

TVector3 TGeoTriangle::CalculateNormal() const
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
/// Test if this triangle is hit by the ray given by origin and direction
///
/// \param[in] origin
/// \param[in] direction
/// \return TriangleIntersection_t

TGeoTriangle::TriangleIntersection_t TGeoTriangle::IsIntersected(const TVector3 &origin, const TVector3 &direction) const
{
   TriangleIntersection_t result;
   IntersectsPlane(origin, direction, TGeoShape::Tolerance(), result);
   if (result.fIntersectionType != TGeoTriangle::IntersectionType::kNone) {
      if (IsPointContained(result.fIntersectionPoint)) {
         if (result.fIntersectionType == TGeoTriangle::IntersectionType::kLiesOnPlane) {
            result.fDistance = (result.fIntersectionPoint - origin).Mag();
         }
         return result;
      }
   }
   result.fIntersectionType = TGeoTriangle::IntersectionType::kNone;
   result.fDistance = std::numeric_limits<double>::max();

   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Test if the ray given by linepoint and linedirection intersect the plane
/// in which the triangle lies
///
/// \param[in] linepoint
/// \param[in] linedirection
/// \param[in] accuracy Double_t giving the accuracy
/// \param[out] result TriangleIntersection_t object containing
///                    the intersected triangle

void TGeoTriangle::IntersectsPlane(const TVector3 &linepoint, const TVector3 &linedirection, Double_t accuracy,
                                TriangleIntersection_t &result) const
{
   result.fIntersectionType = TGeoTriangle::IntersectionType::kNone;
   result.fDirDotNormal = linedirection.Dot(fNormal);

   Double_t nominator = (fCenter - linepoint).Dot(fNormal);

   if (std::abs(nominator) < TGeoShape::Tolerance()) { // linepoint lies on plane
      result.fIntersectionPoint = linepoint;
      result.fIntersectionType = TGeoTriangle::IntersectionType::kLiesOnPlane;
   }
   if (TGeoTriangleInternal::EqualTo(result.fDirDotNormal, 0, accuracy)) { // line and plane are parallel
      result.fIntersectionType = TGeoTriangle::IntersectionType::kNone;
      return;
   }
   Double_t scale = nominator / result.fDirDotNormal;

   if (scale < 0) {
      result.fIntersectionType = TGeoTriangle::IntersectionType::kInOppositeDirection;
      result.fIntersectionPoint = linepoint + (linedirection * scale);
      result.fDistance = (result.fIntersectionPoint - linepoint).Mag();

   } else {
      result.fIntersectionType = TGeoTriangle::IntersectionType::kInDirection;
      result.fIntersectionPoint = linepoint + (linedirection * scale);
      result.fDistance = (result.fIntersectionPoint - linepoint).Mag();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Test if the point contained by the triangle.
/// Barycentric point in triangle test
///
/// @param[in] point
/// @return Bool_t

Bool_t TGeoTriangle::IsPointContained(const TVector3 &point) const
{
   if ((point - fCenter).Mag2() > fRadiusSqr) {
      return kFALSE;
   }
   const TVector3 AToC = Point(2) - Point(0);
   const TVector3 AToB = Point(1) - Point(0);
   const Double_t DotCC = AToC.Dot(AToC);
   const Double_t DotCB = AToC.Dot(AToB);
   const Double_t DotBB = AToB.Dot(AToB);
   const Double_t InvDenom = 1.0 / (DotCC * DotBB - DotCB * DotCB);
   if (InvDenom < 1e5 && InvDenom > 0.1) {
      // Input dependent part
      const TVector3 AToPoint = point - Point(0);
      const Double_t dotCPoint = AToC.Dot(AToPoint);
      const Double_t dotBPoint = AToB.Dot(AToPoint);

      const Double_t u = (DotBB * dotCPoint - DotCB * dotBPoint) * InvDenom;
      const Double_t v = (DotCC * dotBPoint - DotCB * dotCPoint) * InvDenom;

      if ((u >= -TGeoShape::Tolerance()) && (v >= -TGeoShape::Tolerance()) && u + v >= 0 && u + v <= 1.01) {
         return IsPointContainedCenterCast(point);
      } else {
         return (u >= 0) && (v >= 0) && u + v <= 1;
      }
   } else {
      return IsPointContainedCenterCast(point);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Public helper function for IsPoiontContained
/// Alternative point in triangle test by finding closest point of the triangle
/// edges to the point, and comparing if the ray from center to that edge point
/// is parallel or antiparallel to the direction from the test point to the edge
/// point
///
/// \param[in] point
/// \return Bool_t indicating, that point is inside this triangle

Bool_t TGeoTriangle::IsPointContainedCenterCast(const TVector3 &point) const
{
   auto closestPoint = ClosestPoint_t{};
   ClosestPointOfEdgesToPoint(point, closestPoint);
   TVector3 dir = closestPoint.fClosestPoint - point;

   TVector3 centerdir = (closestPoint.fClosestPoint - fCenter).Unit();

   return dir.Mag() < TGeoShape::Tolerance() || dir.Dot(centerdir) > 0 ||
          (dir.Mag() < TGeoShape::Tolerance() && dir.Dot(centerdir) > -TGeoShape::Tolerance());
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate the closest point on the triangle to point
///
/// \param[in] point
/// \return ClosestPoint_t

TGeoTriangle::ClosestPoint_t TGeoTriangle::ClosestPointToPoint(const TVector3 &point) const
{
   TriangleIntersection_t intersection = IsIntersected(point, fNormal);
   auto closestPoint = ClosestPoint_t{intersection.fIntersectionPoint, intersection.fDistance};
   if (intersection.fIntersectionType == TGeoTriangle::IntersectionType::kNone ||
       !IsPointContained(intersection.fIntersectionPoint)) {
      ClosestPointOfEdgesToPoint(point, closestPoint);
   }

   return closestPoint;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculates Closest point of triangle edges to given point point
///
/// \param[in] point
/// \param[out] closestPoint

void TGeoTriangle::ClosestPointOfEdgesToPoint(const TVector3 &point, ClosestPoint_t &closestPoint) const
{
   closestPoint.fDistance = 1e30;
   TVector3 edgedir = TVector3{0.0, 0.0, 0.0};
   auto current = ClosestPoint_t{{0.0, 0.0, 0.0}, 1e30};
   for (UInt_t index = 0; index < sNumberOfVertices; ++index) {
      const TVector3 &startedge = Point((index + 1) % sNumberOfVertices);
      const TVector3 &endedge = Point(index);
      edgedir = endedge - startedge;

      ClosestPointOfEdgeToPoint(point, startedge, edgedir, current);
      if (current.fDistance < closestPoint.fDistance) {
         closestPoint = current;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate closest point of edge to point
///
/// \param[in] point
/// \param[in] edge
/// \param[in] edgedirection
/// \param[out] closestPoint

void TGeoTriangle::ClosestPointOfEdgeToPoint(const TVector3 &point, const TVector3 &edge, const TVector3 &edgedirection,
                                          ClosestPoint_t &closestPoint) const
{
   TVector3 edgetopoint = point - edge;
   Double_t isleft = edgetopoint.Dot(edgedirection);
   if (TGeoTriangleInternal::SmallerThan(isleft, 0, sAccuracy)) {
      closestPoint.fDistance = edgetopoint.Mag();
      closestPoint.fClosestPoint = edge;
      return;
   }
   Double_t isright = edgedirection.Dot(edgedirection);
   if (TGeoTriangleInternal::LargerThan(isleft, isright, sAccuracy)) {
      closestPoint.fClosestPoint = (edge + edgedirection);
      closestPoint.fDistance = (point - closestPoint.fClosestPoint).Mag();
      return;
   }
   Double_t scale = isleft / isright;
   closestPoint.fClosestPoint = edge + (edgedirection * scale);
   closestPoint.fDistance = (point - closestPoint.fClosestPoint).Mag();
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the area of the triangle
///
/// \return Double_t giving the area of the triangle

Double_t TGeoTriangle::Area() const
{
   TVector3 result = TVector3{0.0, 0.0, 0.0};

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
   const TVector3 e1 = Point(1) - Point(0);
   if (e1.Mag2() < kTolerance) {
      std::cout << "Triangle edge e1 is degenerated\n";
      is_valid = kFALSE;
   }
   const TVector3 e2 = Point(2) - Point(1);
   if (e2.Mag2() < kTolerance) {
      std::cout << "Triangle edge e2 is degenerated\n";
      is_valid = kFALSE;
   }
   const TVector3 e3 = Point(0) - Point(2);
   if (e3.Mag2() < kTolerance) {
      std::cout << "Triangle edge e3 is degenerated\n";
      is_valid = kFALSE;
   }
   const TVector3 normal = e1.Cross(e2);
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
   Point(0).Print();
   Point(1).Print();
   Point(2).Print();
   std::cout << "Normal \n";
   fNormal.Print();
   std::cout << "Center \n";
   fCenter.Print();
   std::cout << "Area: " << Area();

   std::cout << std::endl;
   return;
}

}; // namespace Tessellated