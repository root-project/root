// @(#)root/mathcore:$Id: Delaunay2D.h,v 1.00
// Authors: Daniel Funke, Lorenzo Moneta, Olivier Couet

/*************************************************************************
 * Copyright (C) 2015 ROOT Math Team                                     *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Header file for class Delaunay2D

#ifndef ROOT_Math_Delaunay2D
#define ROOT_Math_Delaunay2D

#include <map>
#include <vector>
#include <set>
#include <functional>


namespace ROOT {



   namespace Math {

/**

   Class to generate a Delaunay triangulation of a 2D set of points.
   Algorithm based on [CDT](https://github.com/artem-ogre/CDT), a C++ library for
   generating constraint or conforming Delaunay triangulations.

   After having found the triangles using the above library,  barycentric coordinates are used
   to test whether a point is inside a triangle (inTriangle test) and for interpolation.
   All this below is implemented in the DoInterpolateNormalized function.

   Given triangle ABC and point P, P can be expressed by

     P.x = la * A.x + lb * B.x + lc * C.x
     P.y = la * A.y + lb * B.y + lc * C.y

   with lc = 1 - la - lb

     P.x = la * A.x + lb * B.x + (1-la-lb) * C.x
     P.y = la * A.y + lb * B.y + (1-la-lb) * C.y

   Rearranging yields

     la * (A.x - C.x) + lb * (B.x - C.x) = P.x - C.x
     la * (A.y - C.y) + lb * (B.y - C.y) = P.y - C.y

   Thus

     la = ( (B.y - C.y)*(P.x - C.x) + (C.x - B.x)*(P.y - C.y) ) / ( (B.y - C.y)*(A.x - C.x) + (C.x - B.x)*(A.y - C.y) )
     lb = ( (C.y - A.y)*(P.x - C.x) + (A.x - C.x)*(P.y - C.y) ) / ( (B.y - C.y)*(A.x - C.x) + (C.x - B.x)*(A.y - C.y) )
     lc = 1 - la - lb

   We save the inverse denominator to speedup computation

     invDenom = 1 / ( (B.y - C.y)*(A.x - C.x) + (C.x - B.x)*(A.y - C.y) )

     P is in triangle (including edges if

     0 <= [la, lb, lc] <= 1

     The interpolation of P.z is

     P.z = la * A.z + lb * B.z + lc * C.z

   To speed up localisation of points (to see to which triangle belong) a grid is laid over the internal coordinate space.
   A reference to triangle ABC is added to _all_ grid cells that include ABC's bounding box.
   The size of the grid is defined to be 25x25

   \ingroup MathCore
 */


class Delaunay2D  {

public:

   struct Triangle {
      double x[3];           // x of triangle vertices
      double y[3];           // y of triangle vertices
      unsigned int idx[3];   // point corresponding to vertices
      double invDenom;       // cached inv denominator for computing barycentric coordinates (see above)
   };

   typedef std::vector<Triangle> Triangles;

public:


   Delaunay2D(int n, const double *x, const double * y, const double * z, double xmin=0, double xmax=0, double ymin=0, double ymax=0);

   /// set the input points for building the graph
   void SetInputPoints(int n, const double *x, const double * y, const double * z, double xmin=0, double xmax=0, double ymin=0, double ymax=0);

   /// Return the Interpolated z value corresponding to the given (x,y) point.
   /// Note that in case no Delaunay triangles are found, for example when the
   /// points are aligned, then a default value of zero is always return.
   /// See the class documentation for  how the interpolation is computed.
   double  Interpolate(double x, double y);

   /// Find all triangles
   void      FindAllTriangles();

   /// return the number of triangles
   int    NumberOfTriangles() const {return fNdt;}

   double  XMin() const {return fXNmin;}
   double  XMax() const {return fXNmax;}
   double  YMin() const {return fYNmin;}
   double  YMax() const {return fYNmax;}

   /// set z value to be returned for points  outside the region
   void      SetZOuterValue(double z=0.) { fZout = z; }

   /// return the user defined Z-outer value
   double ZOuterValue() const { return fZout; }

   // iterators on the found triangles
   Triangles::const_iterator begin() const { return fTriangles.begin(); }
   Triangles::const_iterator end()  const { return fTriangles.end(); }


private:

   // internal methods


   inline double Linear_transform(double x, double offset, double factor){
      return (x+offset)*factor;
   }

   /// internal function to normalize the points.
   /// See the class documentation for the details on how it is computed.
   void DoNormalizePoints();

   /// internal function to find the triangle
   void DoFindTriangles();

   /// internal method to compute the interpolation
   double  DoInterpolateNormalized(double x, double y);



private:
   // class is not copyable
   Delaunay2D(const Delaunay2D&); // Not implemented
   Delaunay2D& operator=(const Delaunay2D&); // Not implemented

protected:

   int         fNdt;           ///<! Number of Delaunay triangles found
   int         fNpoints;       ///<! Number of data points

   const double   *fX;         ///<! Pointer to X array (managed externally)
   const double   *fY;         ///<! Pointer to Y array
   const double   *fZ;         ///<! Pointer to Z array

   double    fXNmin;           ///<! Minimum value of fXN
   double    fXNmax;           ///<! Maximum value of fXN
   double    fYNmin;           ///<! Minimum value of fYN
   double    fYNmax;           ///<! Maximum value of fYN

   double    fOffsetX;         ///<! Normalization offset X
   double    fOffsetY;         ///<! Normalization offset Y

   double    fScaleFactorX;    ///<! Normalization factor X
   double    fScaleFactorY;    ///<! Normalization factor Y

   double    fZout;            ///<! Height for points lying outside the convex hull

   bool      fInit;            ///<! True if FindAllTriangles() has been performed


   Triangles   fTriangles;     ///<! Triangles of Triangulation

   std::vector<double> fXN; ///<! normalized X
   std::vector<double> fYN; ///<! normalized Y

   static const int fNCells = 25; ///<! number of cells to divide the normalized space
   double fXCellStep; ///<! inverse denominator to calculate X cell = fNCells / (fXNmax - fXNmin)
   double fYCellStep; ///<! inverse denominator to calculate X cell = fNCells / (fYNmax - fYNmin)
   std::set<unsigned int> fCells[(fNCells+1)*(fNCells+1)]; ///<! grid cells with containing triangles

   inline unsigned int Cell(unsigned int x, unsigned int y) const {
      return x*(fNCells+1) + y;
   }

   inline int CellX(double x) const {
      return (x - fXNmin) * fXCellStep;
   }

   inline int CellY(double y) const {
      return (y - fYNmin) * fYCellStep;
   }

};


} // namespace Math
} // namespace ROOT


#endif
