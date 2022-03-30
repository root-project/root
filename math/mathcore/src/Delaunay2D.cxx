// @(#)root/mathcore:$Id: Delaunay2D.h,v 1.00
// Author: Daniel Funke, Lorenzo Moneta

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

//#include <thread>

// in case we do not use CGAL
#ifndef HAS_CGAL
// use the triangle library
#include "triangle.h"
#endif

#include <algorithm>
#include <stdlib.h>

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


#ifdef THREAD_SAFE
   fInit         = Initialization::UNINITIALIZED;
#else
   fInit         = kFALSE;
#endif

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
   //printf("Normalized space extends from (%f,%f) to (%f,%f)\n", fXNmin, fYNmin, fXNmax, fYNmax);


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

   // Wrong zeros may appear when points sit on a regular grid.
   // The following line try to avoid this problem.
   if (zz==0) zz = DoInterpolateNormalized(xx+0.0001, yy);

   return zz;
}

//______________________________________________________________________________
void Delaunay2D::FindAllTriangles()
{

#ifdef THREAD_SAFE
   //treat the common case first
   if(fInit.load(std::memory_order::memory_order_relaxed) == Initialization::INITIALIZED)
      return;

   Initialization cState = Initialization::UNINITIALIZED;
   if(fInit.compare_exchange_strong(cState, Initialization::INITIALIZING,
                                    std::memory_order::memory_order_release, std::memory_order::memory_order_relaxed))
   {
      // the value of fInit was indeed UNINIT, we replaced it atomically with initializing
      // performing the initialzing now
#else
      if (fInit) return; else fInit = kTRUE;
#endif

      // Function used internally only. It creates the data structures needed to
      // compute the Delaunay triangles.

      // Offset fX and fY so they average zero, and scale so the average
      // of the X and Y ranges is one. The normalized version of fX and fY used
      // in Interpolate.

      DoNormalizePoints(); // call backend specific point normalization

      DoFindTriangles(); // call backend specific triangle finding

      fNdt = fTriangles.size();

#ifdef THREAD_SAFE
      fInit = Initialization::INITIALIZED;
   } else while(cState != Initialization::INITIALIZED) {
         //the value of fInit was NOT UNINIT, so we have to spin until we reach INITIALEZED
         cState = fInit.load(std::memory_order::memory_order_relaxed);
      }
#endif

}


// backend specific implementations

#ifdef HAS_CGAL
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

   if(!nn.third) //neighbor finding was NOT successfull, return standard value
      return fZout;

   //printf("found neighbors %u\n", coords.size());

   Coord_type res = CGAL::linear_interpolation(coords.begin(), coords.end(),
                                               nn.second, Value_access(fNormalizedPoints, fZ));

   //std::cout << std::this_thread::get_id() << ": Result " << res << std::endl;

   return res;
}

#else // HAS_CGAL

/// Triangle implementation for normalizing the points
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

   auto initStruct = [] (triangulateio & s) {
      s.pointlist = nullptr;              /* In / out */
      s.pointattributelist = nullptr;     /* In / out */
      s.pointmarkerlist = nullptr;        /* In / out */
      s.numberofpoints = 0;               /* In / out */
      s.numberofpointattributes = 0;      /* In / out */

      s.trianglelist = nullptr;           /* In / out */
      s.triangleattributelist = nullptr;  /* In / out */
      s.trianglearealist = nullptr;       /* In only */
      s.neighborlist = nullptr;           /* Out only */
      s.numberoftriangles = 0;            /* In / out */
      s.numberofcorners = 0;              /* In / out */
      s.numberoftriangleattributes = 0;   /* In / out */

      s.segmentlist = nullptr;            /* In / out */
      s.segmentmarkerlist = nullptr;      /* In / out */
      s.numberofsegments = 0;             /* In / out */

      s.holelist = nullptr;               /* In / pointer to array copied out */
      s.numberofholes = 0;                /* In / copied out */

      s.regionlist = nullptr;             /* In / pointer to array copied out */
      s.numberofregions = 0;              /* In / copied out */

      s.edgelist = nullptr;               /* Out only */
      s.edgemarkerlist = nullptr;         /* Not used with Voronoi diagram; out only */
      s.normlist = nullptr;               /* Used only with Voronoi diagram; out only */
      s.numberofedges = 0;                /* Out only */
   };

   auto freeStruct = [] (triangulateio & s) {
      if(s.pointlist != nullptr) free(s.pointlist);                         /* In / out */
      if(s.pointattributelist != nullptr) free(s.pointattributelist);       /* In / out */
      if(s.pointmarkerlist != nullptr) free(s.pointmarkerlist);             /* In / out */

      if(s.trianglelist != nullptr) free(s.trianglelist);                   /* In / out */
      if(s.triangleattributelist != nullptr) free(s.triangleattributelist); /* In / out */
      if(s.trianglearealist != nullptr) free(s.trianglearealist);           /* In only */
      if(s.neighborlist != nullptr) free(s.neighborlist);                   /* Out only */

      if(s.segmentlist != nullptr) free(s.segmentlist);                     /* In / out */
      if(s.segmentmarkerlist != nullptr) free(s.segmentmarkerlist);         /* In / out */

      if(s.holelist != nullptr) free(s.holelist);             /* In / pointer to array copied out */

      if(s.regionlist != nullptr) free(s.regionlist);         /* In / pointer to array copied out */

      if(s.edgelist != nullptr) free(s.edgelist);             /* Out only */
      if(s.edgemarkerlist != nullptr) free(s.edgemarkerlist); /* Not used with Voronoi diagram; out only */
      if(s.normlist != nullptr) free(s.normlist);             /* Used only with Voronoi diagram; out only */
   };

   struct triangulateio in, out;
   initStruct(in); initStruct(out);

   /* Define input points. */

   in.numberofpoints = fNpoints;
   in.pointlist = (REAL *) malloc(in.numberofpoints * 2 * sizeof(REAL));

   for (Int_t i = 0; i < fNpoints; ++i) {
      in.pointlist[2 * i] = fXN[i];
      in.pointlist[2 * i + 1] = fYN[i];
   }

   triangulate((char *) "zQN", &in, &out, nullptr);

   fTriangles.resize(out.numberoftriangles);
   for(int t = 0; t < out.numberoftriangles; ++t){
      Triangle tri;

      auto transform = [&] (const unsigned int v) {
         //each triangle as numberofcorners vertices ( = 3)
         tri.idx[v] = out.trianglelist[t*out.numberofcorners + v];

         //printf("triangle %u vertex %u: point %u/%i\n", t, v, tri.idx[v], out.numberofpoints);

         //pointlist is [x0 y0 x1 y1 ...]
         tri.x[v] = in.pointlist[tri.idx[v] * 2 + 0];

         //printf("\t x: %f\n", tri.x[v]);

         tri.y[v] = in.pointlist[tri.idx[v] * 2 + 1];

         //printf("\t y: %f\n", tri.y[v]);
      };

      transform(0);
      transform(1);
      transform(2);

      //see comment in header for CGAL fallback section
      tri.invDenom = 1 / ( (tri.y[1] - tri.y[2])*(tri.x[0] - tri.x[2]) + (tri.x[2] - tri.x[1])*(tri.y[0] - tri.y[2]) );

      fTriangles[t] = tri;

      auto bx = std::minmax({tri.x[0], tri.x[1], tri.x[2]});
      auto by = std::minmax({tri.y[0], tri.y[1], tri.y[2]});

      unsigned int cellXmin = CellX(bx.first);
      unsigned int cellXmax = CellX(bx.second);

      unsigned int cellYmin = CellY(by.first);
      unsigned int cellYmax = CellY(by.second);

      for(unsigned int i = cellXmin; i <= cellXmax; ++i) {
         for(unsigned int j = cellYmin; j <= cellYmax; ++j) {
            //printf("(%u,%u) = %u\n", i, j, Cell(i,j));
            fCells[Cell(i,j)].insert(t);
         }
      }
   }

   freeStruct(in); freeStruct(out);
}

/// Triangle implementation for interpolation
/// Finds the Delaunay triangle that the point (xi,yi) sits in (if any) and
/// calculate a z-value for it by linearly interpolating the z-values that
/// make up that triangle.
double Delaunay2D::DoInterpolateNormalized(double xx, double yy)
{

   // relay that ll the triangles have been found
   ///  FindAllTriangles();

    //see comment in header for CGAL fallback section
    auto bayCoords = [&] (const unsigned int t) -> std::tuple<double, double, double> {
       double la = ( (fTriangles[t].y[1] - fTriangles[t].y[2])*(xx - fTriangles[t].x[2])
                    + (fTriangles[t].x[2] - fTriangles[t].x[1])*(yy - fTriangles[t].y[2]) ) * fTriangles[t].invDenom;
       double lb = ( (fTriangles[t].y[2] - fTriangles[t].y[0])*(xx - fTriangles[t].x[2])
                    + (fTriangles[t].x[0] - fTriangles[t].x[2])*(yy - fTriangles[t].y[2]) ) * fTriangles[t].invDenom;

       return std::make_tuple(la, lb, (1 - la - lb));
    };

    auto inTriangle = [] (const std::tuple<double, double, double> & coords) -> bool {
       return std::get<0>(coords) >= 0 && std::get<1>(coords) >= 0 && std::get<2>(coords) >= 0;
    };

   int cX = CellX(xx);
   int cY = CellY(yy);

   if(cX < 0 || cX > fNCells || cY < 0 || cY > fNCells)
      return fZout; //TODO some more fancy interpolation here

    for(unsigned int t : fCells[Cell(cX, cY)]){
       auto coords = bayCoords(t);

       if(inTriangle(coords)){
          //we found the triangle -> interpolate using the barycentric interpolation
          return std::get<0>(coords) * fZ[fTriangles[t].idx[0]]
                 + std::get<1>(coords) * fZ[fTriangles[t].idx[1]]
               + std::get<2>(coords) * fZ[fTriangles[t].idx[2]];

       }
    }

    //debugging

    /*
    for(unsigned int t = 0; t < fNdt; ++t) {
       auto coords = bayCoords(t);

       if(inTriangle(coords)){

          //brute force found a triangle -> grid not
          printf("Found triangle %u for (%f,%f) -> (%u,%u)\n", t, xx,yy, cX, cY);
          printf("Triangles in grid cell: ");
          for(unsigned int x : fCells[Cell(cX, cY)])
             printf("%u ", x);
          printf("\n");

          printf("Triangle %u is in cells: ", t);
          for(unsigned int i = 0; i <= fNCells; ++i)
             for(unsigned int j = 0; j <= fNCells; ++j)
                if(fCells[Cell(i,j)].count(t))
                   printf("(%u,%u) ", i, j);
          printf("\n");
          for(unsigned int i = 0; i < 3; ++i)
             printf("\tpoint %u (%u): (%f,%f) -> (%u,%u)\n", i, fTriangles[t].idx[i], fTriangles[t].x[i], fTriangles[t].y[i], CellX(fTriangles[t].x[i]), CellY(fTriangles[t].y[i]));

          //we found the triangle -> interpolate using the barycentric interpolation
          return std::get<0>(coords) * fZ[fTriangles[t].idx[0]]
                 + std::get<1>(coords) * fZ[fTriangles[t].idx[1]]
                 + std::get<2>(coords) * fZ[fTriangles[t].idx[2]];

       }
    }

    printf("Could not find a triangle for point (%f,%f)\n", xx, yy);
    */

    //no triangle found return standard value
   return fZout;
}
#endif //HAS_CGAL

} // namespace Math
} // namespace ROOT

