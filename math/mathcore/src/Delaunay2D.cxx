// @(#)root/mathcore:$Id: Delaunay2D.h,v 1.00
// Authors: Daniel Funke, Lorenzo Moneta, Olivier Couet

/*************************************************************************
 * Copyright (C) 2015 ROOT Math Team                                     *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Implementation file for class Delaunay2D

#include "Math/Delaunay2D.h"
#include "Rtypes.h"
#include "TError.h"

//#include <thread>

// use the CDT library if we do not use CGAL
#ifndef HAS_CGAL
#include "CDT/CDT.h"
#endif

#include <algorithm>
#include <cstdlib>

#include <iostream>
#include <limits>


namespace ROOT {

   namespace Math {


/// class constructor from array of data points
Delaunay2D::Delaunay2D(int n, const double * x, const double * y, const double * z,
                       double xmin, double xmax, double ymin, double ymax)
{
   // Delaunay2D normal constructor

   fX            = x;
   fY            = y;
   fZ            = z;
   fZout         = 0.;
   fNpoints      = n;
   fOffsetX      = 0;
   fOffsetY      = 0;
   fScaleFactorX = 0;
   fScaleFactorY = 0;
   fNdt          = 0;
   fXNmax        = 0;
   fXNmin        = 0;
   fYNmin        = 0;
   fYNmax        = 0;

   SetInputPoints(n,x,y,z,xmin,xmax,ymin,ymax);

}

/// set the input points
void Delaunay2D::SetInputPoints(int n, const double * x, const double * y, const double * z,
                           double xmin, double xmax, double ymin, double ymax) {


   fInit         = kFALSE;

   if (n == 0 || !x || !y || !z ) return;

   if (xmin >= xmax) {
      xmin = *std::min_element(x, x+n);
      xmax = *std::max_element(x, x+n);

      ymin = *std::min_element(y, y+n);
      ymax = *std::max_element(y, y+n);
   }

   fOffsetX      = -(xmax+xmin)/2.;
   fOffsetY      = -(ymax+ymin)/2.;

   if ( (xmax-xmin) != 0 ) {
      fScaleFactorX = 1./(xmax-xmin);
      fXNmax        = Linear_transform(xmax, fOffsetX, fScaleFactorX); //xTransformer(xmax);
      fXNmin        = Linear_transform(xmin, fOffsetX, fScaleFactorX); //xTransformer(xmin);
   } else {
      // we can't find triangle in this case
      fInit = true;
   }

   if (ymax-ymin != 0) {
      fScaleFactorY = 1./(ymax-ymin);
      fYNmax        = Linear_transform(ymax, fOffsetY, fScaleFactorY); //yTransformer(ymax);
      fYNmin        = Linear_transform(ymin, fOffsetY, fScaleFactorY); //yTransformer(ymin);
   } else {
      fInit = true;
   }

#ifndef HAS_CGAL
   fXCellStep    = 0.;
   fYCellStep    = 0.;
#endif
}

//______________________________________________________________________________
double Delaunay2D::Interpolate(double x, double y)
{
   // Return the interpolated z value corresponding to the given (x,y) point

   // Initialise the Delaunay algorithm if needed.
   // CreateTrianglesDataStructure computes fXoffset, fYoffset,
   // fXScaleFactor and fYScaleFactor;
   // needed in this function.
   FindAllTriangles();

   // handle case no triangles are found  return default value (i.e. 0)
   // to do: if points are aligned in one direction we could return in
   // some case the 1d interpolated value
   if (fNdt == 0) {
      return fZout;
   }

   // Find the z value corresponding to the point (x,y).
   double xx, yy;
   xx = Linear_transform(x, fOffsetX, fScaleFactorX); //xx = xTransformer(x);
   yy = Linear_transform(y, fOffsetY, fScaleFactorY); //yy = yTransformer(y);
   double zz = DoInterpolateNormalized(xx, yy);

   // the case of points on a regular grid (i.e. points on triangle edges) it is now handles in
   // DoInterpolateNormalized

   return zz;
}

//______________________________________________________________________________
void Delaunay2D::FindAllTriangles()
{

   if (fInit)
      return;
   else
      fInit = kTRUE;

   // Function used internally only. It creates the data structures needed to
   // compute the Delaunay triangles.

   // Offset fX and fY so they average zero, and scale so the average
   // of the X and Y ranges is one. The normalized version of fX and fY used
   // in Interpolate.

   DoNormalizePoints(); // call backend specific point normalization

   DoFindTriangles(); // call backend specific triangle finding

   fNdt = fTriangles.size();
}

// backend specific implementations

#ifndef HAS_CGAL

// Triangle implementation (default case)

/// Triangle implementation for points normalization
void Delaunay2D::DoNormalizePoints() {
   for (Int_t n = 0; n < fNpoints; n++) {
      fXN.push_back(Linear_transform(fX[n], fOffsetX, fScaleFactorX));
      fYN.push_back(Linear_transform(fY[n], fOffsetY, fScaleFactorY));
   }

   //also initialize fXCellStep and FYCellStep
   fXCellStep = fNCells / (fXNmax - fXNmin);
   fYCellStep = fNCells / (fYNmax - fYNmin);
}

/// Triangle implementation for finding all the triangles
void Delaunay2D::DoFindTriangles() {

   int i;
   std::vector<CDT::V2d<double>> points(fNpoints);
   for (i = 0; i < fNpoints; ++i) points[i] = CDT::V2d<double>::make(fXN[i], fYN[i]);
   CDT::RemoveDuplicates(points);
   if (fNpoints-points.size() > 0)
      Warning("DoFindTriangles",
              "This TGraph2D has duplicated vertices. To remove them call RemoveDuplicates before drawing");

   CDT::Triangulation<double> cdt;
   cdt.insertVertices(points);
   cdt.eraseSuperTriangle();

   auto AllTriangles      = cdt.triangles;
   auto AllVertices       = cdt.vertices;
   int  NumberOfTriangles = cdt.triangles.size();

   fTriangles.resize(NumberOfTriangles);

   for(i = 0; i < NumberOfTriangles; i++){
      Triangle tri;
      const auto& t = AllTriangles[i];

      const auto& v0  = AllVertices[t.vertices[0]];
      tri.x[0]   = v0.x;
      tri.y[0]   = v0.y;
      tri.idx[0] = t.vertices[0];

      const auto& v1  = AllVertices[t.vertices[1]];
      tri.x[1]   = v1.x;
      tri.y[1]   = v1.y;
      tri.idx[1] = t.vertices[1];

      const auto& v2  = AllVertices[t.vertices[2]];
      tri.x[2]   = v2.x;
      tri.y[2]   = v2.y;
      tri.idx[2] = t.vertices[2];

      // see comment in header for CGAL fallback section
      tri.invDenom = 1 / ( (tri.y[1] - tri.y[2])*(tri.x[0] - tri.x[2]) + (tri.x[2] - tri.x[1])*(tri.y[0] - tri.y[2]) );

      fTriangles[i] = tri;

      auto bx = std::minmax({tri.x[0], tri.x[1], tri.x[2]});
      auto by = std::minmax({tri.y[0], tri.y[1], tri.y[2]});

      unsigned int cellXmin = CellX(bx.first);
      unsigned int cellXmax = CellX(bx.second);

      unsigned int cellYmin = CellY(by.first);
      unsigned int cellYmax = CellY(by.second);

      for(unsigned int j = cellXmin; j <= cellXmax; j++) {
         for(unsigned int k = cellYmin; k <= cellYmax; k++) {
            fCells[Cell(j,k)].insert(i);
         }
      }
   }
}

/// Triangle implementation for interpolation
/// Finds the Delaunay triangle that the point (xi,yi) sits in (if any) and
/// calculate a z-value for it by linearly interpolating the z-values that
/// make up that triangle.
/// Relay that all the triangles have been found before
/// see comment in class description (in Delaunay2D.h) for implementation details:
/// finding barycentric coordinates and computing the interpolation
double Delaunay2D::DoInterpolateNormalized(double xx, double yy)
{

   // compute barycentric coordinates of a point P(xx,yy,zz)
   auto bayCoords = [&](const unsigned int t) -> std::tuple<double, double, double> {
      double la = ((fTriangles[t].y[1] - fTriangles[t].y[2]) * (xx - fTriangles[t].x[2]) +
                   (fTriangles[t].x[2] - fTriangles[t].x[1]) * (yy - fTriangles[t].y[2])) *
                  fTriangles[t].invDenom;
      double lb = ((fTriangles[t].y[2] - fTriangles[t].y[0]) * (xx - fTriangles[t].x[2]) +
                   (fTriangles[t].x[0] - fTriangles[t].x[2]) * (yy - fTriangles[t].y[2])) *
                  fTriangles[t].invDenom;

      return std::make_tuple(la, lb, (1 - la - lb));
   };

   // function to test if a point with barycentric coordinates (a,b,c) is inside the triangle
   // If the point is outside one or more of the coordinate are negative.
   // If the point is on a triangle edge, one of the coordinate (the one not part of the edge) is zero.
   // Due to numerical error, it can happen that if the point is at the edge the result is a small negative value.
   // Use then a tolerance (of - eps) to still consider the point within the triangle
   auto inTriangle = [](const std::tuple<double, double, double> &coords) -> bool {
      constexpr double eps = -4 * std::numeric_limits<double>::epsilon();
      return std::get<0>(coords) > eps && std::get<1>(coords) > eps && std::get<2>(coords) > eps;
   };

   int cX = CellX(xx);
   int cY = CellY(yy);

   if (cX < 0 || cX > fNCells || cY < 0 || cY > fNCells)
      return fZout; // TODO some more fancy interpolation here

   for (unsigned int t : fCells[Cell(cX, cY)]) {

      auto coords = bayCoords(t);

      // std::cout << "result of bayCoords " << std::get<0>(coords) <<
      //     "  " << std::get<1>(coords)  << "   " << std::get<2>(coords) << std::endl;

      if (inTriangle(coords)) {

         // we found the triangle -> interpolate using the barycentric interpolation

         return std::get<0>(coords) * fZ[fTriangles[t].idx[0]] + std::get<1>(coords) * fZ[fTriangles[t].idx[1]] +
                std::get<2>(coords) * fZ[fTriangles[t].idx[2]];
      }
   }

   // no triangle found return standard value
   return fZout;
}

#else //HAS_CGAL: case of using GCAL

/// CGAL implementation of normalize points
void Delaunay2D::DonormalizePoints() {
   for (Int_t n = 0; n < fNpoints; n++) {
      //Point p(xTransformer(fX[n]), yTransformer(fY[n]));
      Point p(linear_transform(fX[n], fOffsetX, fScaleFactorX),
              linear_transform(fY[n], fOffsetY, fScaleFactorY));

      fNormalizedPoints.insert(std::make_pair(p, n));
   }
}

/// CGAL implementation for finding triangles
void Delaunay2D::DoFindTriangles() {
   fCGALdelaunay.insert(fNormalizedPoints.begin(), fNormalizedPoints.end());

   std::transform(fCGALdelaunay.finite_faces_begin(),
                  fCGALdelaunay.finite_faces_end(), std::back_inserter(fTriangles),
                  [] (const Delaunay::Face face) -> Triangle {

                     Triangle tri;

                     auto transform = [&] (const unsigned int i) {
                        tri.x[i] = face.vertex(i)->point().x();
                        tri.y[i] = face.vertex(i)->point().y();
                        tri.idx[i] = face.vertex(i)->info();
                     };

                     transform(0);
                     transform(1);
                     transform(2);

                     return tri;

                  });
}

/// CGAL implementation for interpolation
double Delaunay2D::DoInterpolateNormalized(double xx, double yy)
{
   // Finds the Delaunay triangle that the point (xi,yi) sits in (if any) and
   // calculate a z-value for it by linearly interpolating the z-values that
   // make up that triangle.

   // initialise the Delaunay algorithm if needed
    FindAllTriangles();

   //coordinate computation
   Point p(xx, yy);

   std::vector<std::pair<Point, Coord_type> > coords;
   auto nn = CGAL::natural_neighbor_coordinates_2(fCGALdelaunay, p,
                                                  std::back_inserter(coords));

   //std::cout << std::this_thread::get_id() << ": Found " << coords.size() << " points" << std::endl;

   if(!nn.third) // neighbour finding was NOT successful, return standard value
      return fZout;

   Coord_type res = CGAL::linear_interpolation(coords.begin(), coords.end(),
                                               nn.second, Value_access(fNormalizedPoints, fZ));

   //std::cout << std::this_thread::get_id() << ": Result " << res << std::endl;

   return res;
}
#endif // HAS_GCAL

} // namespace Math
} // namespace ROOT
