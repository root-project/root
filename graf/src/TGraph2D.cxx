// @(#)root/graf:$Name:  $:$Id: TGraph2D.cxx,v 1.00
// Author: Olivier Couet   23/10/03
// Author: Luke Jones (Royal Holloway, University of London) April 2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGraph2D.h"
#include "TMath.h"
#include "TPolyLine.h"
#include "TPolyMarker.h"
#include "TVirtualPad.h"
#include "TView.h"

ClassImp(TGraph2D)

//______________________________________________________________________________
//
// A Graph2D is a graphics object made of three arrays X, Y and Z
// with npoints each.
//
// This class uses the Delaunay triangles technique to interpolate and 
// render the data set. 
// This class linearly interpolate a Z value for any (X,Y) point given some 
// existing (X,Y,Z) points. The existing (X,Y,Z) points can be randomly 
// scattered. The algorithm work by joining the existing points to make 
// (Delaunay) triangles in (X,Y). These are then used to define flat planes 
// in (X,Y,Z) over which to interpolate. The interpolated surface thus takes 
// the form of tessellating triangles at various angles. Output can take the 
// form of a 2D histogram or a vector. The triangles found can be drawn in 3D.
//
// This software cannot be guaranteed to work under all circumstances. They 
// were originally written to work with a few hundred points in an XY space 
// with similar X and Y ranges.
//


const Int_t maxstored    = 2500;
const Int_t maxntris2try = 100000;


//______________________________________________________________________________
TGraph2D::TGraph2D(): TNamed(), TAttLine(), TAttFill(1,1001), TAttMarker()
{
   // Graph default constructor

   fHistogram = 0;
}


//______________________________________________________________________________
TGraph2D::TGraph2D(Int_t n, Double_t *x, Double_t *y, Double_t *z, Option_t *)
         : TNamed("Graph2D","Graph2D"), TAttLine(), TAttFill(1,1001), TAttMarker()
{
   // Produce a 2D histogram of z values linearly interpolated from the
   // vectors rx, ry, rz using Delaunay triangulation.

   fNp     = n;
   fMargin = 0.;
   fNpx    = 40;
   fNpy    = 40;
   fZout   = 0.;

   fNdt       = 0;
   fNxt       = 0;
   fNhull     = 0;
   fHistogram = 0;

   fX = new Double_t[fNp+1];
   fY = new Double_t[fNp+1];
   fZ = new Double_t[fNp+1];

   fTried = new Int_t[maxstored];

   fHullPoints = new Int_t[fNp];
   fOrder      = new Int_t[fNp];
   fDist       = new Double_t[fNp];

   // Copy the input vectors into local arrays
   // and compute the minimum and maximum of the x and y arrays.
   fXmin = x[0];
   fXmax = x[0];
   fYmin = y[0];
   fYmax = y[0];

   Int_t N;
   for (N=0; N<fNp; N++) {
      fX[N+1] = x[N];
      fY[N+1] = y[N];
      fZ[N+1] = z[N];
      if (fXmin > x[N]) fXmin = x[N];
      if (fXmax < x[N]) fXmax = x[N];
      if (fYmin > y[N]) fYmin = y[N];
      if (fYmax < y[N]) fYmax = y[N];
   }

   Double_t xrange, yrange, avrange;
   fXoffset     = -(fXmax+fXmin)/2.;
   fYoffset     = -(fYmax+fYmin)/2.;
   xrange       = fXmax-fXmin;
   yrange       = fYmax-fYmin;
   avrange      = (xrange+yrange)/2.;
   fScaleFactor = 1./avrange;

   // Offset fX and fY so they average zero,
   // and scale so the average of the X and Y ranges is one.
   fXmax = (fXmax+fXoffset)*fScaleFactor;
   fXmin = (fXmin+fXoffset)*fScaleFactor;
   fYmax = (fYmax+fYoffset)*fScaleFactor;
   fYmin = (fYmin+fYoffset)*fScaleFactor;

   for (N=1; N<=fNp; N++) {
      fX[N] = (fX[N]+fXoffset)*fScaleFactor;
      fY[N] = (fY[N]+fYoffset)*fScaleFactor;
   }

   FindHull();
}


//______________________________________________________________________________
TGraph2D::~TGraph2D()
{
   // TGraph2D destructor.

   if (fX)          delete [] fX;
   if (fY)          delete [] fY;
   if (fZ)          delete [] fZ;
   if (fTried)      delete [] fTried;
   if (fHullPoints) delete [] fHullPoints;
   if (fOrder)      delete [] fOrder;   
   if (fDist)       delete [] fDist;
   if (fHistogram) {
      delete fHistogram;
      fHistogram = 0;
   }
}


//______________________________________________________________________________
Double_t TGraph2D::ComputeZ(Double_t xx, Double_t yy)
{
   // Find the Delaunay triangle that the point (xx,yy) sits in (if any) and 
   // calculate a z-value for it by linearly interpolating the z-values that 
   // make up that triangle.

   Double_t thevalue;

   Int_t IT, ntris_tried, tri, P, N, M;
   Int_t I,J,K,L,Z,F,D,O1,O2,A,B,T1,T2,T3;
   Int_t ndegen=0,degen=0,fdegen=0,o1degen=0,o2degen=0;
   Int_t thistri;
   Double_t vxN,vyN;
   Double_t d1,d2,d3,c1,c2,dko1,dko2,dfo1;
   Double_t dfo2,sin_sum,cfo1k,co2o1k,co2o1f;

   Bool_t shouldbein;
   Double_t dx1,dx2,dx3,dy1,dy2,dy3,U,V,dxz[3],dyz[3];

   // the input point will be point zero.
   fX[0] = xx;
   fY[0] = yy;

   // set the output value to the default value for now
   thevalue = fZout;

   // some counting
   ntris_tried = 0;

   // no point in proceeding if xx or yy are silly
   if ((xx>fXmax) || (xx<fXmin) || (yy>fYmax) || (yy<fYmin)) return thevalue;

   // check existing Delaunay triangles for a good one
   for (IT=1; IT<=fNdt; IT++) {
      if (fTried[IT-1] > 0) {
         tri = fTried[IT-1];
         P   = tri%1000;
         N   = ((tri%1000000)-P)/1000;
         M   = (tri-N*1000-P)/1000000;
         // P, N and M form a previously found Delaunay triangle, does it 
         // enclose the point?
         if (Enclose(P,N,M,0)) {
            // yes, we have the triangle
            thevalue = Interpolate(P,N,M,0);
            return thevalue;
         }
      } else {
         Error("ComputeZ", "Negative/zero Delaunay triangle ? %d %d %g %g %d",IT,fNdt,xx,yy,fTried[IT-1]);
      }
   }

   // is this point inside the convex hull?
   shouldbein = InHull(0,-1);
   if (!shouldbein) return thevalue;

   // it must be in a Delaunay triangle - find it...

   // order mass points by distance in mass plane from desired point
   for (N=1; N<=fNp; N++) {
      vxN = fX[N];
      vyN = fY[N];
      fDist[N-1] = TMath::Sqrt((xx-vxN)*(xx-vxN)+(yy-vyN)*(yy-vyN));
   }

   // sort array 'dist' to find closest points
   TMath::Sort(fNp, fDist, fOrder, kFALSE);
   for (N=1; N<=fNp; N++) fOrder[N-1]++;

   // loop over triplets of close points to try to find a triangle that 
   // encloses the point.
   for (K=3; K<=fNp; K++) {
      M = fOrder[K-1];
      for (J=2; J<=K-1; J++) {
         N = fOrder[J-1];
         for (I=1; I<=J-1; I++) {
            P = fOrder[I-1];
            if (ntris_tried > maxntris2try) {
               // perhaps this point isn't in the hull after all
///            Warning("ComputeZ", 
///                    "Abandoning the effort to find a Delaunay triangle (and thus interpolated Z-value) for point %g %g"
///                    ,xx,yy);
               return thevalue;
            }
            ntris_tried++;
            // check the points aren't colinear
            d1 = TMath::Sqrt((fX[P]-fX[N])*(fX[P]-fX[N])+(fY[P]-fY[N])*(fY[P]-fY[N]));
            d2 = TMath::Sqrt((fX[P]-fX[M])*(fX[P]-fX[M])+(fY[P]-fY[M])*(fY[P]-fY[M]));
            d3 = TMath::Sqrt((fX[N]-fX[M])*(fX[N]-fX[M])+(fY[N]-fY[M])*(fY[N]-fY[M]));
            if ((d1+d2<=d3) || (d1+d3<=d2) || (d2+d3<=d1)) goto L90;

            // does the triangle enclose the point?
            if (!Enclose(P,N,M,0)) goto L90;

            // is it a Delaunay triangle? (ie. are there any other points 
            // inside the circle that is defined by its vertices?)

            // has this triangle already been tested for Delaunay'ness?
            tri = TriEncode(P,N,M);
            for (IT=maxstored-fNxt+1; IT<=maxstored; IT++) {
               if (tri == TMath::Abs(fTried[IT-1])) {
                  if (fTried[IT-1] < 0) {
                     // yes, and it is not a Delaunay triangle, forget about it
                     goto L90;
                  } else {
                     Error("ComputeZ", "Positive non-Delaunay triangle ? %g %g %d %d %d",
                           xx,yy,IT,fNxt,fNdt);
                     thevalue  = Interpolate(P,N,M,0);
                     return thevalue;
                  }
               }
            }

            // test the triangle for Delaunay'ness

            thistri = tri;

            // loop over all other points testing each to see if it's 
            // inside the triangle's circle
            ndegen = 0;
            for ( Z=1; Z<=fNp; Z++) {
               if ((Z==P) || (Z==N) || (Z==M)) goto L50;
               // An easy first check is to see if point Z is inside the triangle 
               // (if it's in the triangle it's also in the circle)

               // point Z cannot be inside the triangle if it's further from (xx,yy) 
               // than the furthest pointing making up the triangle - test this
               for (L=1; L<=fNp; L++) {
                  if (fOrder[L-1] == Z) {
                     if ((L<I) || (L<J) || (L<K)) {
                        // point Z is nearer to (xx,yy) than M, N or P - it could be in the 
                        // triangle so call enclose to find out
                        if (Enclose(P,N,M,Z)) {
                           // it is inside the triangle and so this can't be a Del' triangle
                           FileIt(-thistri);
                           goto L90;
                        }
                     } else {
                        // there's no way it could be in the triangle so there's no point 
                        // calling enclose
                        goto L1;
                     }
                  }
               }
               // is point Z colinear with any pair of the triangle points?
L1:
               if (((fX[P]-fX[Z])*(fY[P]-fY[N])) == ((fY[P]-fY[Z])*(fX[P]-fX[N]))) {
                  // Z is colinear with P and N
                  A = P;
                  B = N;
               } else if (((fX[P]-fX[Z])*(fY[P]-fY[M])) == ((fY[P]-fY[Z])*(fX[P]-fX[M]))) {
                  // Z is colinear with P and M
                  A = P;
                  B = M;
               } else if (((fX[N]-fX[Z])*(fY[N]-fY[M])) == ((fY[N]-fY[Z])*(fX[N]-fX[M]))) {
                  // Z is colinear with N and M
                  A = N;
                  B = M;
               } else {
                  A = 0;
                  B = 0;
               }
               if (A != 0) {
                  // point Z is colinear with 2 of the triangle points, if it lies 
                  // between them it's in the circle otherwise it's outside
                  if (fX[A] != fX[B]) {
                     if (((fX[Z]-fX[A])*(fX[Z]-fX[B])) < 0) {
                        FileIt(-thistri);
                        goto L90;
                     } else if (((fX[Z]-fX[A])*(fX[Z]-fX[B])) == 0) {
                        // At least two points are sitting on top of each other, we will
                        // treat these as one and not consider this a 'multiple points lying
                        // on a common circle' situation. It is a sign something could be
                        // wrong though, especially if the two coincident points have
                        // different fZ's. If they don't then this is harmless.
                        Warning("ComputeZ", "Two of these three points are coincident %d %d %d",A,B,Z);
                     }
                  } else {
                     if (((fY[Z]-fY[A])*(fY[Z]-fY[B])) < 0) {
                        FileIt(-thistri);
                        goto L90;
                     } else if (((fY[Z]-fY[A])*(fY[Z]-fY[B])) == 0) {
                        // At least two points are sitting on top of each other - see above.
                        Warning("ComputeZ", "Two of these three points are coincident %d %d %d",A,B,Z);
                     }
                  }
                  // point is outside the circle, move to next point
                  goto L50;
               }

               // if point Z were to look at the triangle, which point would it see 
               // lying between the other two? (we're going to form a quadrilateral 
               // from the points, and then demand certain properties of that
               // quadrilateral)
               dxz[0] = fX[P]-fX[Z];
               dyz[0] = fY[P]-fY[Z];
               dxz[1] = fX[N]-fX[Z];
               dyz[1] = fY[N]-fY[Z];
               dxz[2] = fX[M]-fX[Z];
               dyz[2] = fY[M]-fY[Z];
               for(L=1; L<=3; L++) {
                  dx1 = dxz[L-1];
                  dx2 = dxz[L%3];
                  dx3 = dxz[(L+1)%3];
                  dy1 = dyz[L-1];
                  dy2 = dyz[L%3];
                  dy3 = dyz[(L+1)%3];

                  U = (dy3*dx2-dx3*dy2)/(dy1*dx2-dx1*dy2);
                  V = (dy3*dx1-dx3*dy1)/(dy2*dx1-dx2*dy1);

                  if ((U>=0) && (V>=0)) {
                     // vector (dx3,dy3) is expressible as a sum of the other two vectors 
                     // with positive coefficents -> i.e. it lies between the other two vectors
                     if (L == 1) {
                        F  = M;
                        O1 = P;
                        O2 = N;
                     } else if (L == 2) {
                        F  = P;
                        O1 = N;
                        O2 = M;
                     } else {
                        F  = N;
                        O1 = M;
                        O2 = P;
                     }
                     goto L2;
                  }
               }
               Error("ComputeZ", "Should not get to here");
               // may as well soldier on
               F  = M;
               O1 = P;
               O2 = N;
L2:
               // this is not a valid quadrilateral if the diagonals don't cross, 
               // check that points F and Z lie on opposite side of the line O1-O2,
               // this is true if the angle F-O1-Z is greater than O2-O1-Z and O2-O1-F
               cfo1k  = ((fX[F]-fX[O1])*(fX[Z]-fX[O1])+(fY[F]-fY[O1])*(fY[Z]-fY[O1]))/
                        TMath::Sqrt(((fX[F]-fX[O1])*(fX[F]-fX[O1])+(fY[F]-fY[O1])*(fY[F]-fY[O1]))*
                        ((fX[Z]-fX[O1])*(fX[Z]-fX[O1])+(fY[Z]-fY[O1])*(fY[Z]-fY[O1])));
               co2o1k = ((fX[O2]-fX[O1])*(fX[Z]-fX[O1])+(fY[O2]-fY[O1])*(fY[Z]-fY[O1]))/
                        TMath::Sqrt(((fX[O2]-fX[O1])*(fX[O2]-fX[O1])+(fY[O2]-fY[O1])*(fY[O2]-fY[O1]))*
                        ((fX[Z]-fX[O1])*(fX[Z]-fX[O1])  + (fY[Z]-fY[O1])*(fY[Z]-fY[O1])));
               co2o1f = ((fX[O2]-fX[O1])*(fX[F]-fX[O1])+(fY[O2]-fY[O1])*(fY[F]-fY[O1]))/
                        TMath::Sqrt(((fX[O2]-fX[O1])*(fX[O2]-fX[O1])+(fY[O2]-fY[O1])*(fY[O2]-fY[O1]))*
                        ((fX[F]-fX[O1])*(fX[F]-fX[O1]) + (fY[F]-fY[O1])*(fY[F]-fY[O1]) ));
               if ((cfo1k>co2o1k) || (cfo1k>co2o1f)) {
                  // not a valid quadrilateral - point Z is definitely outside the circle
                  goto L50;
               }
               // calculate the 2 internal angles of the quadrangle formed by joining
               // points Z and F to points O1 and O2, at Z and F. If they sum to less
               // than 180 degrees then Z lies outside the circle
               dko1    = TMath::Sqrt((fX[Z]-fX[O1])*(fX[Z]-fX[O1])+(fY[Z]-fY[O1])*(fY[Z]-fY[O1]));
               dko2    = TMath::Sqrt((fX[Z]-fX[O2])*(fX[Z]-fX[O2])+(fY[Z]-fY[O2])*(fY[Z]-fY[O2]));
               dfo1    = TMath::Sqrt((fX[F]-fX[O1])*(fX[F]-fX[O1])+(fY[F]-fY[O1])*(fY[F]-fY[O1]));
               dfo2    = TMath::Sqrt((fX[F]-fX[O2])*(fX[F]-fX[O2])+(fY[F]-fY[O2])*(fY[F]-fY[O2]));
               c1      = ((fX[Z]-fX[O1])*(fX[Z]-fX[O2])+(fY[Z]-fY[O1])*(fY[Z]-fY[O2]))/dko1/dko2;
               c2      = ((fX[F]-fX[O1])*(fX[F]-fX[O2])+(fY[F]-fY[O1])*(fY[F]-fY[O2]))/dfo1/dfo2;
               sin_sum = c1*TMath::Sqrt(1-c2*c2)+c2*TMath::Sqrt(1-c1*c1);

               // When being called from paw, sin_sum doesn't always come out as zero 
               // when it should do.
               if (sin_sum < -1.E-6) {
                  // Z is inside the circle, this is not a Delaunay triangle
                  FileIt(-thistri);
                  goto L90;
               } else if (TMath::Abs(sin_sum) <= 1.E-6) {
                  // point Z lies on the circumference of the circle (within rounding errors) 
                  // defined by the triangle, so there is potential for degeneracy in the 
                  // triangle set (Delaunay triangulation does not give a unique way to split
                  // a polygon whose points lie on a circle into constituent triangles). Make
                  // a note of the additional point number.
                  ndegen++;
                  degen   = Z;
                  fdegen  = F;
                  o1degen = O1;
                  o2degen = O2;
               }
L50:
            continue;
            }
            // This is a good triangle
            if (ndegen > 0) {
               // but is degenerate with at least one other,
               // haven't figured out what to do if more than 4 points are involved
               if (ndegen > 1) {
                  Error("ComputeZ", 
		         "More than 4 points lying on a circle. No decision making process formulated for triangulating this region in a non-arbitrary way %d %d %d %d",
                         P,N,M,degen);
                  return thevalue;
               }

               // we have a quadrilateral which can be split down either diagonal
               // (D<->F or O1<->O2) to form valid Delaunay triangles. Choose diagonal
               // with highest average z-value. Whichever we choose we will have
               // verified two triangles as good and two as bad, only note the good ones
               D  = degen;
               F  = fdegen;
               O1 = o1degen;
               O2 = o2degen;
               if ((fZ[O1]+fZ[O2]) > (fZ[D]+fZ[F])) {
                  // best diagonalisation of quadrilateral is current one, we have 
                  // the triangle
                  T1 = P;
                  T2 = N;
                  T3 = M;
                  // file the good triangles as good
                  FileIt(thistri);
                  tri = TriEncode(D,O1,O2);
                  FileIt(tri);
               } else {
                  // use other diagonal to split quadrilateral, use triangle formed by 
                  // point F, the degnerate point D and whichever of O1 and O2 create 
                  // an enclosing triangle
                  T1 = F;
                  T2 = D;
                  if (Enclose(F,D,O1,0)) {
                     T3 = O1;
                  } else {
                     T3 = O2;
                  }
                  // file the good triangles as good and the original one as bad
                  FileIt(-thistri);
                  tri = TriEncode(F,D,O1);
                  FileIt(tri);
                  tri = TriEncode(F,D,O2);
                  FileIt(tri);
               }
            } else {
               // this is a Delaunay triangle, file it as such
               FileIt(thistri);
               T1 = P;
               T2 = N;
               T3 = M;
            }
            // do the interpolation
            thevalue = Interpolate(T1,T2,T3,0);
            return thevalue;
L90:
            continue;
         }
      }
   }
   if (shouldbein) {
      Error("ComputeZ", 
            "Point outside hull when expected inside: this point could be dodgy %g %g %d",
             xx, yy, ntris_tried);
   }
   return thevalue;
}


//______________________________________________________________________________
void TGraph2D::CreateHistogram()
{
   // Book the 2D histogram fHistogram with a margin around the hull.

   Double_t hxmax = fXmax/fScaleFactor-fXoffset;
   Double_t hymax = fYmax/fScaleFactor-fYoffset;
   Double_t hxmin = fXmin/fScaleFactor-fXoffset;
   Double_t hymin = fYmin/fScaleFactor-fYoffset;
   Double_t xwid  = hxmax - hxmin;
   Double_t ywid  = hymax - hymin;
   hxmax = hxmax + fMargin * xwid;
   hymax = hymax + fMargin * ywid;
   hxmin = hxmin - fMargin * xwid;
   hymin = hymin - fMargin * ywid;

   fHistogram = new TH2D(GetName(),GetTitle(),fNpx ,hxmin, hxmax,
                                              fNpy, hymin, hymax);
}


//______________________________________________________________________________
Int_t TGraph2D::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Compute distance from point px,py to a graph

   Int_t distance = 9999;
   if (fHistogram) distance = fHistogram->DistancetoPrimitive(px,py);
   return distance;
}

//______________________________________________________________________________
Bool_t TGraph2D::Enclose(Int_t T1, Int_t T2, Int_t T3, Int_t Ex) const
{
   // Is point E inside the triangle T1-T2-T3 ?

   Int_t E=0,A=0,B=0;
   Double_t dx1,dx2,dx3,dy1,dy2,dy3,U,V;

   Bool_t enclose = kFALSE;

   E = TMath::Abs(Ex);
      
   // First ask if point E is colinear with any pair of the triangle points
   A = 0;
   if (((fX[T1]-fX[E])*(fY[T1]-fY[T2])) == ((fY[T1]-fY[E])*(fX[T1]-fX[T2]))) {
   //     E is colinear with T1 and T2
      A = T1;
      B = T2;
   } else if (((fX[T1]-fX[E])*(fY[T1]-fY[T3])) == ((fY[T1]-fY[E])*(fX[T1]-fX[T3]))) {
   //     E is colinear with T1 and T3
      A = T1;
      B = T3;
   } else if (((fX[T2]-fX[E])*(fY[T2]-fY[T3])) == ((fY[T2]-fY[E])*(fX[T2]-fX[T3]))) {
   //     E is colinear with T2 and T3
      A = T2;
      B = T3;
   }
   if (A != 0) {
   //     point E is colinear with 2 of the triangle points, if it lies 
   //     between them it's in the circle otherwise it's outside
      if (fX[A] != fX[B]) {
         if (((fX[E]-fX[A])*(fX[E]-fX[B])) <= 0) {
            enclose = kTRUE;
            return enclose;
         }
      } else {
         if (((fY[E]-fY[A])*(fY[E]-fY[B])) <= 0) {
            enclose = kTRUE;
            return enclose;
         }
      }
   //     point is outside the triangle
      return enclose;
   }

   // E is not colinear with any pair of triangle points, if it is inside
   // the triangle then the vector from E to one of the corners must be 
   // expressible as a sum with positive coefficients of the vectors from 
   // the two other corners to E. Say vector3=U*vector1+V*vector2

   // vector1==T1->E
   dx1 = fX[E]-fX[T1];
   dy1 = fY[E]-fY[T1];
   // vector2==T2->E
   dx2 = fX[E]-fX[T2];
   dy2 = fY[E]-fY[T2];
   // vector3==E->T3
   dx3 = fX[T3]-fX[E];
   dy3 = fY[T3]-fY[E];

   U = (dx2*dy3-dx3*dy2)/(dx2*dy1-dx1*dy2);
   V = (dx1*dy3-dx3*dy1)/(dx1*dy2-dx2*dy1);

   if ((U>=0) && (V>=0)) enclose = kTRUE;

   return enclose;
}


//______________________________________________________________________________
void TGraph2D::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{   
   // Execute action corresponding to one event

   if (fHistogram) fHistogram->ExecuteEvent(event, px, py);
}


//______________________________________________________________________________
void TGraph2D::FileIt(Int_t tri)
{
   // File the triangle 'tri' in the fTried array. Delaunay triangles 
   // (tri>0) are stored sequentially from fTried[0] onwards. Non-Delaunay 
   // triangles are stored sequentially from fTried[maxstored-1] backwards. 
   // If the array cannot hold all the triangles, Delaunay triangles get 
   // priority - non-Delaunay triangles are overwritten and subsequent 
   // non-Delaunay triangles overwrite previous non-Delaunay triangles.

   if (tri > 0) {
      // store a new Delaunay triangle
      fNdt++;
      if (fNdt> maxstored) {
         Error("FileIt", 
               "No space left in fTried, the maxstored parameter should be increased %d",
               tri);
      } else {
         fTried[fNdt-1] = tri;
      }
      // we may have overwritten a non-Delaunay triangle - update fNxt
      fNxt = TMath::Min(maxstored-fNdt,fNxt);
   } else if ((maxstored-fNxt) > fNdt) {
      // store a new non-Delaunay triangle - we have space to do this without 
      // overwriting anything
      fTried[maxstored-fNxt-1] = tri;
      fNxt++;
   } else if (maxstored > fNdt) {
      // store a new non-Delaunay triangle - there is still space to do this 
      // but we will have to overwrite old non-Delaunay triangles
      fTried[maxstored-1] = tri;
      fNxt = 1;
   }
}


//______________________________________________________________________________
void TGraph2D::FillHistogram()
{
   // Call ComputeZ at each bin centre to build up interpolated 2D histogram

   Double_t x, y, z, sx, sy;

   Double_t hxmin = fHistogram->GetXaxis()->GetXmin();
   Double_t hxmax = fHistogram->GetXaxis()->GetXmax();
   Double_t hymin = fHistogram->GetYaxis()->GetXmin();
   Double_t hymax = fHistogram->GetYaxis()->GetXmax();

   Double_t dx = (hxmax-hxmin)/fNpx;
   Double_t dy = (hymax-hymin)/fNpy;

   for (Int_t ix=1; ix<=fNpx; ix++) {
      x  = hxmin+(ix-0.5)*dx;
      sx = (x+fXoffset)*fScaleFactor;
      for (Int_t iy=1; iy<=fNpy; iy++) {
         y  = hymin+(iy-0.5)*dy;
         sy = (y+fYoffset)*fScaleFactor;
         z  = ComputeZ(sx, sy);
         fHistogram->Fill(x, y, z);
      }
   }
}


//______________________________________________________________________________
void TGraph2D::FindHull()
{
   //
   // Author: Luke Jones (Royal Holloway, University of London), April 2002
   //
   // Find those points which make up the convex hull of the set. If the xy
   // plane were a sheet of wood, and the points were nails hammered into it
   // at the respective coordinates, then if an elastic band were stretched
   // over all the nails it would form the shape of the convex hull. Those
   // nails in contact with it are the points that make up the hull.
   //

   Int_t N,nhull_tmp;
   Bool_t in;

   fNhull = 0;
   nhull_tmp = 0;
   for(N=1; N<=fNp; N++) {
      // if the point is not inside the hull of the set of all points 
      // bar it, then it is part of the hull of the set of all points 
      // including it
      in = InHull(N,N);
      if (!in) {
         // cannot increment fNhull directly - InHull needs to know that 
         // the hull has not yet been completely found
         nhull_tmp++;
         fHullPoints[nhull_tmp-1] = N;
      }
   }
   fNhull = nhull_tmp;
}


//______________________________________________________________________________
Bool_t TGraph2D::InHull(Int_t E, Int_t X) const
{
   //
   // Author: Luke Jones (Royal Holloway, University of London), April 2002
   //
   // Is point E inside the hull defined by all points apart from X ?
   //

   Int_t n1,n2,N,M,Ntry;
   Double_t lastdphi,dd1,dd2,dx1,dx2,dx3,dy1,dy2,dy3;
   Double_t U,V,vNv1,vNv2,phi1,phi2,dphi,xx,yy;

   Bool_t DTinhull = kFALSE;

   xx = fX[E];
   yy = fY[E];

   if (fNhull > 0) {
      //  The hull has been found - no need to use any points other than 
      //  those that make up the hull
      Ntry = fNhull;
   } else {
      //  The hull has not yet been found, will have to try every point
      Ntry = fNp;
   }

   //  N1 and N2 will represent the two points most separated by angle
   //  from point E. Initially the angle between them will be <180 degs.
   //  But subsequent points will increase the N1-E-N2 angle. If it 
   //  increases above 180 degrees then point E must be surrounded by 
   //  points - it is not part of the hull.
   n1 = 1;
   n2 = 2;
   if (n1 == X) {
      n1 = n2;
      n2++;
   } else if (n2 == X) {
      n2++;
   }

   //  Get the angle N1-E-N2 and set it to lastdphi
   dx1  = xx-fX[n1];
   dy1  = yy-fY[n1];
   dx2  = xx-fX[n2];
   dy2  = yy-fY[n2];
   phi1 = TMath::ATan2(dy1,dx1);
   phi2 = TMath::ATan2(dy2,dx2);
   dphi = (phi1-phi2)-((Int_t)((phi1-phi2)/TMath::TwoPi())*TMath::TwoPi());
   if (dphi < 0) dphi = dphi+TMath::TwoPi();
   lastdphi = dphi;
   for (N=1; N<=Ntry; N++) {
      if (fNhull > 0) {
         // Try hull point N
         M = fHullPoints[N-1];
      } else {
         M = N;
      }
      if ((M!=n1) && (M!=n2) && (M!=X)) {
         // Can the vector E->M be represented as a sum with positive 
         // coefficients of vectors E->N1 and E->N2?
         dx1 = xx-fX[n1];
         dy1 = yy-fY[n1];
         dx2 = xx-fX[n2];
         dy2 = yy-fY[n2];
         dx3 = xx-fX[M];
         dy3 = yy-fY[M];

         dd1 = (dx2*dy1-dx1*dy2);
         dd2 = (dx1*dy2-dx2*dy1);

         if (dd1*dd2!=0) {
            U = (dx2*dy3-dx3*dy2)/dd1;
            V = (dx1*dy3-dx3*dy1)/dd2;
            if ((U<0) || (V<0)) {
               // No, it cannot - point M does not lie inbetween N1 and N2 as 
               // viewed from E. Replace either N1 or N2 to increase the 
               // N1-E-N2 angle. The one to replace is the one which makes the
               // smallest angle with E->M
               vNv1 = (dx1*dx3+dy1*dy3)/TMath::Sqrt(dx1*dx1+dy1*dy1);
               vNv2 = (dx2*dx3+dy2*dy3)/TMath::Sqrt(dx2*dx2+dy2*dy2);
               if (vNv1 > vNv2) {
                  n1   = M;
                  phi1 = TMath::ATan2(dy3,dx3);
                  phi2 = TMath::ATan2(dy2,dx2);
               } else {
                  n2   = M;
                  phi1 = TMath::ATan2(dy1,dx1);
                  phi2 = TMath::ATan2(dy3,dx3);
               }
               dphi = (phi1-phi2)-((Int_t)((phi1-phi2)/TMath::TwoPi())*TMath::TwoPi());
               if (dphi < 0) dphi = dphi+TMath::TwoPi();
               if (((dphi-TMath::Pi())*(lastdphi-TMath::Pi())) < 0) {
                  // The addition of point M means the angle N1-E-N2 has risen 
                  // above 180 degs, the point is in the hull.
                  goto L10;
               }
               lastdphi = dphi;
            }
         }
      }
   }
   // Point E is not surrounded by points - it is not in the hull.
   goto L999;
L10:
   DTinhull = kTRUE;
L999:
   return DTinhull;
}


//______________________________________________________________________________
Double_t TGraph2D::Interpolate(Int_t TI1, Int_t TI2, Int_t TI3, Int_t E) const
{
   // Find the z-value at point E given that it lies 
   // on the plane defined by T1,T2,T3

   Int_t tmp;
   Bool_t swap;
   Double_t x1,x2,x3,y1,y2,y3,f1,f2,f3,U,V,W;

   Int_t T1 = TI1;
   Int_t T2 = TI2;
   Int_t T3 = TI3;

L1:
   swap = kFALSE;
   if (T2 > T1) { tmp = T1; T1 = T2; T2 = tmp; swap = kTRUE;}
   if (T3 > T2) { tmp = T2; T2 = T3; T3 = tmp; swap = kTRUE;}
   if (swap) goto L1;

   x1 = fX[T1];
   x2 = fX[T2];
   x3 = fX[T3];
   y1 = fY[T1];
   y2 = fY[T2];
   y3 = fY[T3];
   f1 = fZ[T1];
   f2 = fZ[T2];
   f3 = fZ[T3];
   U  = (f1*(y2-y3)+f2*(y3-y1)+f3*(y1-y2))/(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2));
   V  = (f1*(x2-x3)+f2*(x3-x1)+f3*(x1-x2))/(y1*(x2-x3)+y2*(x3-x1)+y3*(x1-x2));
   W  = f1-U*x1-V*y1;

   return U*fX[E]+V*fY[E]+W;
}


//______________________________________________________________________________
void TGraph2D::Paint(Option_t *option)
{
   // Paint this graph with its current attributes

   if (strstr(option,"TRI") || strstr(option,"tri")) {
      if (!fHistogram) {
         CreateHistogram();
         FillHistogram();
      }
      PaintTriangles();
      PaintMarkers();
   } else{
      if (!fHistogram) {
         CreateHistogram();
         FillHistogram();
      }
      fHistogram->Paint(option);
   }
}


//______________________________________________________________________________
void TGraph2D::PaintMarkers()
{
   Double_t temp1[3],temp2[3];

   Double_t *x = new Double_t[fNp]; 
   Double_t *y = new Double_t[fNp];

   TView *view = gPad->GetView();

   if (!view) {
      gPad->Range(-1,-1,1,1);
      view = new TView(1);
      view->SetRange(fHistogram->GetXaxis()->GetXmin(),
                     fHistogram->GetYaxis()->GetXmin(),
                     fHistogram->GetMinimum(),
                     fHistogram->GetXaxis()->GetXmax(),
                     fHistogram->GetYaxis()->GetXmax(),
                     fHistogram->GetMaximum());
   }

   for (Int_t N=0; N<fNp; N++) {
      temp1[0] = fX[N+1]/fScaleFactor-fXoffset;
      temp1[1] = fY[N+1]/fScaleFactor-fYoffset;
      temp1[2] = fZ[N+1];
      view->WCtoNDC(temp1, &temp2[0]);
      x[N] = temp2[0];
      y[N] = temp2[1];
   }
   SetMarkerStyle(20);
   SetMarkerSize(0.4);
   SetMarkerColor(0);
   TAttMarker::Modify();
   gPad->PaintPolyMarker(fNp,x,y);
   SetMarkerStyle(24);
   SetMarkerColor(1);
   TAttMarker::Modify();
   gPad->PaintPolyMarker(fNp,x,y);

   delete [] x;
   delete [] y;
}


//______________________________________________________________________________
void TGraph2D::PaintTriangles()
{
   Double_t x[4];
   Double_t y[4];
   Double_t temp1[3],temp2[3];
   Int_t T0,T[3];

   TView *view = gPad->GetView();

   if (!view) {
      gPad->Range(-1,-1,1,1);
      view = new TView(1);
      view->SetRange(fHistogram->GetXaxis()->GetXmin(),
                     fHistogram->GetYaxis()->GetXmin(),
                     fHistogram->GetMinimum(),
                     fHistogram->GetXaxis()->GetXmax(),
                     fHistogram->GetYaxis()->GetXmax(),
                     fHistogram->GetMaximum());
   }

   SetFillColor(5);
   SetFillStyle(1001);
   TAttFill::Modify();
   SetLineColor(1);
   TAttLine::Modify();

   for (Int_t N=0; N<fNdt; N++) {
      T0   = fTried[N];
      T[0] = T0/1000000;
      T[1] = (T0%1000000)/1000;
      T[2] = T0%1000;
      for (Int_t t=0; t<3; t++) {
         temp1[0] = fX[T[t]]/fScaleFactor-fXoffset;
         temp1[1] = fY[T[t]]/fScaleFactor-fYoffset;
         temp1[2] = fZ[T[t]];
         view->WCtoNDC(temp1, &temp2[0]);
         x[t] = temp2[0];
         y[t] = temp2[1];
      }
      x[3] = x[0];
      y[3] = y[0];
      gPad->PaintFillArea(3,x,y);
      gPad->PaintPolyLine(4,x,y);
   }
}


//______________________________________________________________________________
TH1 *TGraph2D::Project(Option_t *option) const
{
   // Project a 3-d graph into 1 or 2-d histograms depending on the
   // option parameter
   // option may contain a combination of the characters x,y,z
   // option = "x" return the x projection into a TH1D histogram
   // option = "y" return the y projection into a TH1D histogram
   // option = "xy" return the x versus y projection into a TH2D histogram
   // option = "yx" return the y versus x projection into a TH2D histogram

   TString opt = option; opt.ToLower();

   Int_t pcase = 0;
   if (opt.Contains("x"))  pcase = 1;
   if (opt.Contains("y"))  pcase = 2;
   if (opt.Contains("xy")) pcase = 3;
   if (opt.Contains("yx")) pcase = 4;
    
   // Create the projection histogram
   TH1D *h1 = 0;
   TH2D *h2 = 0; 
   Int_t nch = strlen(GetName()) +opt.Length() +2;
   char *name = new char[nch];
   sprintf(name,"%s_%s",GetName(),option);
   nch = strlen(GetTitle()) +opt.Length() +2;
   char *title = new char[nch];
   sprintf(title,"%s_%s",GetTitle(),option);

   Double_t hxmin = fXmin/fScaleFactor-fXoffset;
   Double_t hxmax = fXmax/fScaleFactor-fXoffset;
   Double_t hymin = fYmin/fScaleFactor-fYoffset;
   Double_t hymax = fYmax/fScaleFactor-fYoffset;

   switch (pcase) {
      case 1:
         // "x"
         h1 = new TH1D(name,title,fNpx,hxmin,hxmax);
         break;
      case 2:
         // "y"
         h1 = new TH1D(name,title,fNpy,hymin,hymax);
         break;
      case 3:
         // "xy"
         h2 = new TH2D(name,title,fNpx,hxmin,hxmax,fNpy,hymin,hymax);
         break;
      case 4:
         // "yx"
         h2 = new TH2D(name,title,fNpy,hymin,hymax,fNpx,hxmin,hxmax);
         break;
   }

   delete [] name;
   delete [] title;
   TH1 *h = h1;
   if (h2) h = h2;
   if (h == 0) return 0;

   // Fill the projected histogram
   Double_t entries = 0;
   for (Int_t N=1; N<=fNp; N++) {
      switch (pcase) {
         case 1:
            // "x"
            h1->Fill(fX[N]/fScaleFactor-fXoffset, fZ[N]);
            break;
         case 2:
            // "y"
            h1->Fill(fY[N]/fScaleFactor-fYoffset, fZ[N]);
            break;
         case 3:
            // "xy"
            h2->Fill(fX[N]/fScaleFactor-fXoffset, 
                     fY[N]/fScaleFactor-fYoffset,
                     fZ[N]);
            break;
         case 4:
            // "yx"
            h2->Fill(fY[N]/fScaleFactor-fYoffset, 
                     fX[N]/fScaleFactor-fXoffset,
                     fZ[N]);
            break;
      }
      entries += fZ[N];
   }
   h->SetEntries(entries);
   return h;
}


//______________________________________________________________________________
void TGraph2D::SetMargin(Double_t m)
{
   // Set the extra space (in %) around interpolated area for the 2D histogram

   if (m<0 || m>1) {
      Warning("SetMargin","The margin must be >= 0 && <= 1, fMargin set to 0.1");
      fMargin = 0.1;
   } else {
      fMargin = m;
   }
   Update();
}


//______________________________________________________________________________
void TGraph2D::SetNpx(Int_t npx)
{
   // Set the number of bins along X used to draw the function

   if (npx < 4) {
      Warning("SetNpx","Number of points must be >4 && < 500, fNpx set to 4");
      fNpx = 4;
   } else if(npx > 500) {
      Warning("SetNpx","Number of points must be >4 && < 500, fNpx set to 500");
      fNpx = 100000;
   } else {
      fNpx = npx;
   }
   Update();
}


//______________________________________________________________________________
void TGraph2D::SetNpy(Int_t npy)
{
   // Set the number of bins along Y used to draw the function

   if (npy < 4) {
      Warning("SetNpy","Number of points must be >4 && < 500, fNpy set to 4");
      fNpy = 4;
   } else if(npy > 500) {
      Warning("SetNpy","Number of points must be >4 && < 500, fNpy set to 500");
      fNpy = 100000;
   } else {
      fNpy = npy;
   }
   Update();
}


//______________________________________________________________________________
void TGraph2D::SetMarginBinsContent(Double_t z)
{
   // Set the histogram bin height for points lying outside the convex hull ie:
   // the bins in the margin.

   fZout = z;
   Update();
}


//______________________________________________________________________________
void TGraph2D::SetTitle(const char* title)
{
   fTitle = title;
   if (fHistogram) fHistogram->SetTitle(title);
}


//______________________________________________________________________________
Int_t TGraph2D::TriEncode(Int_t T1, Int_t T2, Int_t T3) const
{
   // Form the point numbers into a single number to represent the triangle

   Int_t triencode = 0;
   Int_t MinT = T1;
   Int_t MaxT = T1;
   if (T2 > MaxT) MaxT = T2;
   if (T3 > MaxT) MaxT = T3;
   if (T2 < MinT) MinT = T2;
   if (T3 < MinT) MinT = T3;
   
   triencode = 1000000*MaxT+MinT;
   if ((T1!=MaxT) && (T1!=MinT)) {
      triencode = triencode+1000*T1;
   } else if ((T2!=MaxT) && (T2!=MinT)) {
      triencode = triencode+1000*T2;
   } else if ((T3!=MaxT) && (T3!=MinT)) {
      triencode = triencode+1000*T3;
   } else {
      Error("TriEncode", "Should not get to here");
   }
   return triencode;
}


//_______________________________________________________________________
void TGraph2D::Update()
{
   // Called each time fHistogram should be recreated.

   delete fHistogram;
   fHistogram = 0;
}
