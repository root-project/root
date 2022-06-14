// @(#)root/hist:$Id: TGraphDelaunay2D.cxx,v 1.00
// Author: Olivier Couet, Luke Jones (Royal Holloway, University of London)

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGraph2D.h"
#include "TGraphDelaunay2D.h"

ClassImp(TGraphDelaunay2D);


/** \class TGraphDelaunay2D
    \ingroup Graphs
TGraphDelaunay2D generates a Delaunay triangulation of a TGraph2D.
The algorithm used for finding the triangles is based on on
**Triangle**, a two-dimensional quality mesh generator and
Delaunay triangulator from Jonathan Richard Shewchuk.
See [http://www.cs.cmu.edu/~quake/triangle.html]
The ROOT::Math::Delaunay2D class provides a wrapper for using
the **Triangle** library.

This implementation provides large improvements in terms of computational performances
compared to the legacy one available in TGraphDelaunay, and it is by default
used in TGraph2D. The old, legacy implementation can be still used when calling
`TGraph2D::GetHistogram` and `TGraph2D::Draw` with the `old` option.

Definition of Delaunay triangulation (After B. Delaunay):
For a set S of points in the Euclidean plane, the unique triangulation DT(S)
of S such that no point in S is inside the circumcircle of any triangle in
DT(S). DT(S) is the dual of the Voronoi diagram of S. If n is the number of
points in S, the Voronoi diagram of S is the partitioning of the plane
containing S points into n convex polygons such that each polygon contains
exactly one point and every point in a given polygon is closer to its
central point than to any other. A Voronoi diagram is sometimes also known
as a Dirichlet tessellation.

\image html tgraph2d_delaunay.png

[This applet](http://www.cs.cornell.edu/Info/People/chew/Delaunay.html)
gives a nice practical view of Delaunay triangulation and Voronoi diagram.
*/

/// TGraphDelaunay2D normal constructor
TGraphDelaunay2D::TGraphDelaunay2D(TGraph2D *g ) :
   TNamed("TGraphDelaunay2D","TGraphDelaunay2D"),
   fGraph2D(g),
   fDelaunay((g) ? g->GetN() : 0, (g) ? g->GetX() : nullptr, (g) ? g->GetY() : nullptr, (g) ? g->GetZ() : nullptr ,
             (g) ? g->GetXmin() : 0, (g) ? g->GetXmax() : 0,
             (g) ? g->GetYmin() : 0, (g) ? g->GetYmax() : 0 )

{}

