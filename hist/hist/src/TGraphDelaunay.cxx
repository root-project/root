// @(#)root/hist:$Id: TGraphDelaunay.cxx,v 1.00
// Author: Olivier Couet, Luke Jones (Royal Holloway, University of London)

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMath.h"
#include "TGraph2D.h"
#include "TGraphDelaunay.h"

ClassImp(TGraphDelaunay)


//______________________________________________________________________________
//
// TGraphDelaunay generates a Delaunay triangulation of a TGraph2D. This
// triangulation code derives from an implementation done by Luke Jones
// (Royal Holloway, University of London) in April 2002 in the PAW context.
//
// This software cannot be guaranteed to work under all circumstances. They
// were originally written to work with a few hundred points in an XY space
// with similar X and Y ranges.
//
// Definition of Delaunay triangulation (After B. Delaunay):
// For a set S of points in the Euclidean plane, the unique triangulation DT(S)
// of S such that no point in S is inside the circumcircle of any triangle in
// DT(S). DT(S) is the dual of the Voronoi diagram of S. If n is the number of
// points in S, the Voronoi diagram of S is the partitioning of the plane
// containing S points into n convex polygons such that each polygon contains
// exactly one point and every point in a given polygon is closer to its
// central point than to any other. A Voronoi diagram is sometimes also known
// as a Dirichlet tessellation.
//Begin_Html
/*
<img src="gif/dtvd.gif">
<br>
<a href="http://www.cs.cornell.edu/Info/People/chew/Delaunay.html">This applet</a>
gives a nice practical view of Delaunay triangulation and Voronoi diagram.
*/
//End_Html


//______________________________________________________________________________
TGraphDelaunay::TGraphDelaunay()
            : TNamed("TGraphDelaunay","TGraphDelaunay")
{
   // TGraphDelaunay default constructor

   fGraph2D      = 0;
   fX            = 0;
   fY            = 0;
   fZ            = 0;
   fNpoints      = 0;
   fTriedSize    = 0;
   fZout         = 0.;
   fNdt          = 0;
   fNhull        = 0;
   fHullPoints   = 0;
   fXN           = 0;
   fYN           = 0;
   fOrder        = 0;
   fDist         = 0;
   fPTried       = 0;
   fNTried       = 0;
   fMTried       = 0;
   fInit         = kFALSE;
   fXNmin        = 0.;
   fXNmax        = 0.;
   fYNmin        = 0.;
   fYNmax        = 0.;
   fXoffset      = 0.;
   fYoffset      = 0.;
   fXScaleFactor = 0.;
   fYScaleFactor = 0.;

   SetMaxIter();
}


//______________________________________________________________________________
TGraphDelaunay::TGraphDelaunay(TGraph2D *g)
            : TNamed("TGraphDelaunay","TGraphDelaunay")
{
   // TGraphDelaunay normal constructor

   fGraph2D      = g;
   fX            = fGraph2D->GetX();
   fY            = fGraph2D->GetY();
   fZ            = fGraph2D->GetZ();
   fNpoints      = fGraph2D->GetN();
   fTriedSize    = 0;
   fZout         = 0.;
   fNdt          = 0;
   fNhull        = 0;
   fHullPoints   = 0;
   fXN           = 0;
   fYN           = 0;
   fOrder        = 0;
   fDist         = 0;
   fPTried       = 0;
   fNTried       = 0;
   fMTried       = 0;
   fInit         = kFALSE;
   fXNmin        = 0.;
   fXNmax        = 0.;
   fYNmin        = 0.;
   fYNmax        = 0.;
   fXoffset      = 0.;
   fYoffset      = 0.;
   fXScaleFactor = 0.;
   fYScaleFactor = 0.;

   SetMaxIter();
}


//______________________________________________________________________________
TGraphDelaunay::~TGraphDelaunay()
{
   // TGraphDelaunay destructor.

   if (fPTried)     delete [] fPTried;
   if (fNTried)     delete [] fNTried;
   if (fMTried)     delete [] fMTried;
   if (fHullPoints) delete [] fHullPoints;
   if (fOrder)      delete [] fOrder;
   if (fDist)       delete [] fDist;
   if (fXN)         delete [] fXN;
   if (fYN)         delete [] fYN;

   fPTried     = 0;
   fNTried     = 0;
   fMTried     = 0;
   fHullPoints = 0;
   fOrder      = 0;
   fDist       = 0;
   fXN         = 0;
   fYN         = 0;
}


//______________________________________________________________________________
Double_t TGraphDelaunay::ComputeZ(Double_t x, Double_t y)
{
   // Return the z value corresponding to the (x,y) point in fGraph2D

   // Initialise the Delaunay algorithm if needed.
   // CreateTrianglesDataStructure computes fXoffset, fYoffset,
   // fXScaleFactor and fYScaleFactor;
   // needed in this function.
   if (!fInit) {
      CreateTrianglesDataStructure();
      FindHull();
      fInit = kTRUE;
   }

   // Find the z value corresponding to the point (x,y).
   Double_t xx, yy;
   xx = (x+fXoffset)*fXScaleFactor;
   yy = (y+fYoffset)*fYScaleFactor;
   Double_t zz = Interpolate(xx, yy);

   // Wrong zeros may appear when points sit on a regular grid.
   // The following line try to avoid this problem.
   if (zz==0) zz = Interpolate(xx+0.0001, yy);

   return zz;
}


//______________________________________________________________________________
void TGraphDelaunay::CreateTrianglesDataStructure()
{
   // Function used internally only. It creates the data structures needed to
   // compute the Delaunay triangles.

   // Offset fX and fY so they average zero, and scale so the average
   // of the X and Y ranges is one. The normalized version of fX and fY used
   // in Interpolate.
   Double_t xmax = fGraph2D->GetXmax();
   Double_t ymax = fGraph2D->GetYmax();
   Double_t xmin = fGraph2D->GetXmin();
   Double_t ymin = fGraph2D->GetYmin();
   fXoffset      = -(xmax+xmin)/2.;
   fYoffset      = -(ymax+ymin)/2.;
   fXScaleFactor  = 1./(xmax-xmin);
   fYScaleFactor  = 1./(ymax-ymin);
   fXNmax        = (xmax+fXoffset)*fXScaleFactor;
   fXNmin        = (xmin+fXoffset)*fXScaleFactor;
   fYNmax        = (ymax+fYoffset)*fYScaleFactor;
   fYNmin        = (ymin+fYoffset)*fYScaleFactor;
   fXN           = new Double_t[fNpoints+1];
   fYN           = new Double_t[fNpoints+1];
   for (Int_t n=0; n<fNpoints; n++) {
      fXN[n+1] = (fX[n]+fXoffset)*fXScaleFactor;
      fYN[n+1] = (fY[n]+fYoffset)*fYScaleFactor;
   }

   // If needed, creates the arrays to hold the Delaunay triangles.
   // A maximum number of 2*fNpoints is guessed. If more triangles will be
   // find, FillIt will automatically enlarge these arrays.
   fTriedSize = 2*fNpoints;
   fPTried    = new Int_t[fTriedSize];
   fNTried    = new Int_t[fTriedSize];
   fMTried    = new Int_t[fTriedSize];
}


//______________________________________________________________________________
Bool_t TGraphDelaunay::Enclose(Int_t t1, Int_t t2, Int_t t3, Int_t e) const
{
   // Is point e inside the triangle t1-t2-t3 ?

   Double_t x[4],y[4],xp, yp;
   x[0] = fXN[t1];
   x[1] = fXN[t2];
   x[2] = fXN[t3];
   x[3] = x[0];
   y[0] = fYN[t1];
   y[1] = fYN[t2];
   y[2] = fYN[t3];
   y[3] = y[0];
   xp   = fXN[e];
   yp   = fYN[e];
   return TMath::IsInside(xp, yp, 4, x, y);
}


//______________________________________________________________________________
void TGraphDelaunay::FileIt(Int_t p, Int_t n, Int_t m)
{
   // Files the triangle defined by the 3 vertices p, n and m into the
   // fxTried arrays. If these arrays are to small they are automatically
   // expanded.

   Bool_t swap;
   Int_t tmp, ps = p, ns = n, ms = m;

   // order the vertices before storing them
L1:
   swap = kFALSE;
   if (ns > ps) { tmp = ps; ps = ns; ns = tmp; swap = kTRUE;}
   if (ms > ns) { tmp = ns; ns = ms; ms = tmp; swap = kTRUE;}
   if (swap) goto L1;

   // expand the triangles storage if needed
   if (fNdt>=fTriedSize) {
      Int_t newN   = 2*fTriedSize;
      Int_t *savep = new Int_t [newN];
      Int_t *saven = new Int_t [newN];
      Int_t *savem = new Int_t [newN];
      memcpy(savep,fPTried,fTriedSize*sizeof(Int_t));
      memset(&savep[fTriedSize],0,(newN-fTriedSize)*sizeof(Int_t));
      delete [] fPTried;
      memcpy(saven,fNTried,fTriedSize*sizeof(Int_t));
      memset(&saven[fTriedSize],0,(newN-fTriedSize)*sizeof(Int_t));
      delete [] fNTried;
      memcpy(savem,fMTried,fTriedSize*sizeof(Int_t));
      memset(&savem[fTriedSize],0,(newN-fTriedSize)*sizeof(Int_t));
      delete [] fMTried;
      fPTried    = savep;
      fNTried    = saven;
      fMTried    = savem;
      fTriedSize = newN;
   }

   // store a new Delaunay triangle
   fNdt++;
   fPTried[fNdt-1] = ps;
   fNTried[fNdt-1] = ns;
   fMTried[fNdt-1] = ms;
}


//______________________________________________________________________________
void TGraphDelaunay::FindAllTriangles()
{
   // Attempt to find all the Delaunay triangles of the point set. It is not
   // guaranteed that it will fully succeed, and no check is made that it has
   // fully succeeded (such a check would be possible by referencing the points
   // that make up the convex hull). The method is to check if each triangle
   // shares all three of its sides with other triangles. If not, a point is
   // generated just outside the triangle on the side(s) not shared, and a new
   // triangle is found for that point. If this method is not working properly
   // (many triangles are not being found) it's probably because the new points
   // are too far beyond or too close to the non-shared sides. Fiddling with
   // the size of the `alittlebit' parameter may help.

   if (fAllTri) return; else fAllTri = kTRUE;

   Double_t xcntr,ycntr,xm,ym,xx,yy;
   Double_t sx,sy,nx,ny,mx,my,mdotn,nn,a;
   Int_t t1,t2,pa,na,ma,pb,nb,mb,p1=0,p2=0,m,n,p3=0;
   Bool_t s[3];
   Double_t alittlebit = 0.0001;

   // start with a point that is guaranteed to be inside the hull (the
   // centre of the hull). The starting point is shifted "a little bit"
   // otherwise, in case of triangles aligned on a regular grid, we may
   // found none of them.
   xcntr = 0;
   ycntr = 0;
   for (n=1; n<=fNhull; n++) {
      xcntr = xcntr+fXN[fHullPoints[n-1]];
      ycntr = ycntr+fYN[fHullPoints[n-1]];
   }
   xcntr = xcntr/fNhull+alittlebit;
   ycntr = ycntr/fNhull+alittlebit;
   // and calculate it's triangle
   Interpolate(xcntr,ycntr);

   // loop over all Delaunay triangles (including those constantly being
   // produced within the loop) and check to see if their 3 sides also
   // correspond to the sides of other Delaunay triangles, i.e. that they
   // have all their neighbours.
   t1 = 1;
   while (t1 <= fNdt) {
      // get the three points that make up this triangle
      pa = fPTried[t1-1];
      na = fNTried[t1-1];
      ma = fMTried[t1-1];

      // produce three integers which will represent the three sides
      s[0]  = kFALSE;
      s[1]  = kFALSE;
      s[2]  = kFALSE;
      // loop over all other Delaunay triangles
      for (t2=1; t2<=fNdt; t2++) {
         if (t2 != t1) {
            // get the points that make up this triangle
            pb = fPTried[t2-1];
            nb = fNTried[t2-1];
            mb = fMTried[t2-1];
            // do triangles t1 and t2 share a side?
            if ((pa==pb && na==nb) || (pa==pb && na==mb) || (pa==nb && na==mb)) {
               // they share side 1
               s[0] = kTRUE;
            } else if ((pa==pb && ma==nb) || (pa==pb && ma==mb) || (pa==nb && ma==mb)) {
               // they share side 2
               s[1] = kTRUE;
            } else if ((na==pb && ma==nb) || (na==pb && ma==mb) || (na==nb && ma==mb)) {
               // they share side 3
               s[2] = kTRUE;
            }
         }
         // if t1 shares all its sides with other Delaunay triangles then
         // forget about it
         if (s[0] && s[1] && s[2]) continue;
      }
      // Looks like t1 is missing a neighbour on at least one side.
      // For each side, take a point a little bit beyond it and calculate
      // the Delaunay triangle for that point, this should be the triangle
      // which shares the side.
      for (m=1; m<=3; m++) {
         if (!s[m-1]) {
            // get the two points that make up this side
            if (m == 1) {
               p1 = pa;
               p2 = na;
               p3 = ma;
            } else if (m == 2) {
               p1 = pa;
               p2 = ma;
               p3 = na;
            } else if (m == 3) {
               p1 = na;
               p2 = ma;
               p3 = pa;
            }
            // get the coordinates of the centre of this side
            xm = (fXN[p1]+fXN[p2])/2.;
            ym = (fYN[p1]+fYN[p2])/2.;
            // we want to add a little to these coordinates to get a point just
            // outside the triangle; (sx,sy) will be the vector that represents
            // the side
            sx = fXN[p1]-fXN[p2];
            sy = fYN[p1]-fYN[p2];
            // (nx,ny) will be the normal to the side, but don't know if it's
            // pointing in or out yet
            nx    = sy;
            ny    = -sx;
            nn    = TMath::Sqrt(nx*nx+ny*ny);
            nx    = nx/nn;
            ny    = ny/nn;
            mx    = fXN[p3]-xm;
            my    = fYN[p3]-ym;
            mdotn = mx*nx+my*ny;
            if (mdotn > 0) {
               // (nx,ny) is pointing in, we want it pointing out
               nx = -nx;
               ny = -ny;
            }
            // increase/decrease xm and ym a little to produce a point
            // just outside the triangle (ensuring that the amount added will
            // be large enough such that it won't be lost in rounding errors)
            a  = TMath::Abs(TMath::Max(alittlebit*xm,alittlebit*ym));
            xx = xm+nx*a;
            yy = ym+ny*a;
            // try and find a new Delaunay triangle for this point
            Interpolate(xx,yy);

            // this side of t1 should now, hopefully, if it's not part of the
            // hull, be shared with a new Delaunay triangle just calculated by Interpolate
         }
      }
      t1++;
   }
}


//______________________________________________________________________________
void TGraphDelaunay::FindHull()
{
   // Finds those points which make up the convex hull of the set. If the xy
   // plane were a sheet of wood, and the points were nails hammered into it
   // at the respective coordinates, then if an elastic band were stretched
   // over all the nails it would form the shape of the convex hull. Those
   // nails in contact with it are the points that make up the hull.

   Int_t n,nhull_tmp;
   Bool_t in;

   if (!fHullPoints) fHullPoints = new Int_t[fNpoints];

   nhull_tmp = 0;
   for(n=1; n<=fNpoints; n++) {
      // if the point is not inside the hull of the set of all points
      // bar it, then it is part of the hull of the set of all points
      // including it
      in = InHull(n,n);
      if (!in) {
         // cannot increment fNhull directly - InHull needs to know that
         // the hull has not yet been completely found
         nhull_tmp++;
         fHullPoints[nhull_tmp-1] = n;
      }
   }
   fNhull = nhull_tmp;
}


//______________________________________________________________________________
Bool_t TGraphDelaunay::InHull(Int_t e, Int_t x) const
{
   // Is point e inside the hull defined by all points apart from x ?

   Int_t n1,n2,n,m,ntry;
   Double_t lastdphi,dd1,dd2,dx1,dx2,dx3,dy1,dy2,dy3;
   Double_t u,v,vNv1,vNv2,phi1,phi2,dphi,xx,yy;

   Bool_t deTinhull = kFALSE;

   xx = fXN[e];
   yy = fYN[e];

   if (fNhull > 0) {
      //  The hull has been found - no need to use any points other than
      //  those that make up the hull
      ntry = fNhull;
   } else {
      //  The hull has not yet been found, will have to try every point
      ntry = fNpoints;
   }

   //  n1 and n2 will represent the two points most separated by angle
   //  from point e. Initially the angle between them will be <180 degs.
   //  But subsequent points will increase the n1-e-n2 angle. If it
   //  increases above 180 degrees then point e must be surrounded by
   //  points - it is not part of the hull.
   n1 = 1;
   n2 = 2;
   if (n1 == x) {
      n1 = n2;
      n2++;
   } else if (n2 == x) {
      n2++;
   }

   //  Get the angle n1-e-n2 and set it to lastdphi
   dx1  = xx-fXN[n1];
   dy1  = yy-fYN[n1];
   dx2  = xx-fXN[n2];
   dy2  = yy-fYN[n2];
   phi1 = TMath::ATan2(dy1,dx1);
   phi2 = TMath::ATan2(dy2,dx2);
   dphi = (phi1-phi2)-((Int_t)((phi1-phi2)/TMath::TwoPi())*TMath::TwoPi());
   if (dphi < 0) dphi = dphi+TMath::TwoPi();
   lastdphi = dphi;
   for (n=1; n<=ntry; n++) {
      if (fNhull > 0) {
         // Try hull point n
         m = fHullPoints[n-1];
      } else {
         m = n;
      }
      if ((m!=n1) && (m!=n2) && (m!=x)) {
         // Can the vector e->m be represented as a sum with positive
         // coefficients of vectors e->n1 and e->n2?
         dx1 = xx-fXN[n1];
         dy1 = yy-fYN[n1];
         dx2 = xx-fXN[n2];
         dy2 = yy-fYN[n2];
         dx3 = xx-fXN[m];
         dy3 = yy-fYN[m];

         dd1 = (dx2*dy1-dx1*dy2);
         dd2 = (dx1*dy2-dx2*dy1);

         if (dd1*dd2!=0) {
            u = (dx2*dy3-dx3*dy2)/dd1;
            v = (dx1*dy3-dx3*dy1)/dd2;
            if ((u<0) || (v<0)) {
               // No, it cannot - point m does not lie inbetween n1 and n2 as
               // viewed from e. Replace either n1 or n2 to increase the
               // n1-e-n2 angle. The one to replace is the one which makes the
               // smallest angle with e->m
               vNv1 = (dx1*dx3+dy1*dy3)/TMath::Sqrt(dx1*dx1+dy1*dy1);
               vNv2 = (dx2*dx3+dy2*dy3)/TMath::Sqrt(dx2*dx2+dy2*dy2);
               if (vNv1 > vNv2) {
                  n1   = m;
                  phi1 = TMath::ATan2(dy3,dx3);
                  phi2 = TMath::ATan2(dy2,dx2);
               } else {
                  n2   = m;
                  phi1 = TMath::ATan2(dy1,dx1);
                  phi2 = TMath::ATan2(dy3,dx3);
               }
               dphi = (phi1-phi2)-((Int_t)((phi1-phi2)/TMath::TwoPi())*TMath::TwoPi());
               if (dphi < 0) dphi = dphi+TMath::TwoPi();
               if (((dphi-TMath::Pi())*(lastdphi-TMath::Pi())) < 0) {
                  // The addition of point m means the angle n1-e-n2 has risen
                  // above 180 degs, the point is in the hull.
                  goto L10;
               }
               lastdphi = dphi;
            }
         }
      }
   }
   // Point e is not surrounded by points - it is not in the hull.
   goto L999;
L10:
   deTinhull = kTRUE;
L999:
   return deTinhull;
}


//______________________________________________________________________________
Double_t TGraphDelaunay::InterpolateOnPlane(Int_t TI1, Int_t TI2, Int_t TI3, Int_t e) const
{
   // Finds the z-value at point e given that it lies
   // on the plane defined by t1,t2,t3

   Int_t tmp;
   Bool_t swap;
   Double_t x1,x2,x3,y1,y2,y3,f1,f2,f3,u,v,w;

   Int_t t1 = TI1;
   Int_t t2 = TI2;
   Int_t t3 = TI3;

   // order the vertices
L1:
   swap = kFALSE;
   if (t2 > t1) { tmp = t1; t1 = t2; t2 = tmp; swap = kTRUE;}
   if (t3 > t2) { tmp = t2; t2 = t3; t3 = tmp; swap = kTRUE;}
   if (swap) goto L1;

   x1 = fXN[t1];
   x2 = fXN[t2];
   x3 = fXN[t3];
   y1 = fYN[t1];
   y2 = fYN[t2];
   y3 = fYN[t3];
   f1 = fZ[t1-1];
   f2 = fZ[t2-1];
   f3 = fZ[t3-1];
   u  = (f1*(y2-y3)+f2*(y3-y1)+f3*(y1-y2))/(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2));
   v  = (f1*(x2-x3)+f2*(x3-x1)+f3*(x1-x2))/(y1*(x2-x3)+y2*(x3-x1)+y3*(x1-x2));
   w  = f1-u*x1-v*y1;

   return u*fXN[e]+v*fYN[e]+w;
}


//______________________________________________________________________________
Double_t TGraphDelaunay::Interpolate(Double_t xx, Double_t yy)
{
   // Finds the Delaunay triangle that the point (xi,yi) sits in (if any) and
   // calculate a z-value for it by linearly interpolating the z-values that
   // make up that triangle.

   Double_t thevalue;

   Int_t it, ntris_tried, p, n, m;
   Int_t i,j,k,l,z,f,d,o1,o2,a,b,t1,t2,t3;
   Int_t ndegen=0,degen=0,fdegen=0,o1degen=0,o2degen=0;
   Double_t vxN,vyN;
   Double_t d1,d2,d3,c1,c2,dko1,dko2,dfo1;
   Double_t dfo2,sin_sum,cfo1k,co2o1k,co2o1f;

   Bool_t shouldbein;
   Double_t dx1,dx2,dx3,dy1,dy2,dy3,u,v,dxz[3],dyz[3];

   // initialise the Delaunay algorithm if needed
   if (!fInit) {
      CreateTrianglesDataStructure();
      FindHull();
      fInit = kTRUE;
   }

   // create vectors needed for sorting
   if (!fOrder) {
      fOrder = new Int_t[fNpoints];
      fDist  = new Double_t[fNpoints];
   }

   // the input point will be point zero.
   fXN[0] = xx;
   fYN[0] = yy;

   // set the output value to the default value for now
   thevalue = fZout;

   // some counting
   ntris_tried = 0;

   // no point in proceeding if xx or yy are silly
   if ((xx>fXNmax) || (xx<fXNmin) || (yy>fYNmax) || (yy<fYNmin)) return thevalue;

   // check existing Delaunay triangles for a good one
   for (it=1; it<=fNdt; it++) {
      p = fPTried[it-1];
      n = fNTried[it-1];
      m = fMTried[it-1];
      // p, n and m form a previously found Delaunay triangle, does it
      // enclose the point?
      if (Enclose(p,n,m,0)) {
         // yes, we have the triangle
         thevalue = InterpolateOnPlane(p,n,m,0);
         return thevalue;
      }
   }

   // is this point inside the convex hull?
   shouldbein = InHull(0,-1);
   if (!shouldbein) return thevalue;

   // it must be in a Delaunay triangle - find it...

   // order mass points by distance in mass plane from desired point
   for (it=1; it<=fNpoints; it++) {
      vxN = fXN[it];
      vyN = fYN[it];
      fDist[it-1] = TMath::Sqrt((xx-vxN)*(xx-vxN)+(yy-vyN)*(yy-vyN));
   }

   // sort array 'fDist' to find closest points
   TMath::Sort(fNpoints, fDist, fOrder, kFALSE);
   for (it=0; it<fNpoints; it++) fOrder[it]++;

   // loop over triplets of close points to try to find a triangle that
   // encloses the point.
   for (k=3; k<=fNpoints; k++) {
      m = fOrder[k-1];
      for (j=2; j<=k-1; j++) {
         n = fOrder[j-1];
         for (i=1; i<=j-1; i++) {
            p = fOrder[i-1];
            if (ntris_tried > fMaxIter) {
               // perhaps this point isn't in the hull after all
///            Warning("Interpolate",
///                    "Abandoning the effort to find a Delaunay triangle (and thus interpolated z-value) for point %g %g"
///                    ,xx,yy);
               return thevalue;
            }
            ntris_tried++;
            // check the points aren't colinear
            d1 = TMath::Sqrt((fXN[p]-fXN[n])*(fXN[p]-fXN[n])+(fYN[p]-fYN[n])*(fYN[p]-fYN[n]));
            d2 = TMath::Sqrt((fXN[p]-fXN[m])*(fXN[p]-fXN[m])+(fYN[p]-fYN[m])*(fYN[p]-fYN[m]));
            d3 = TMath::Sqrt((fXN[n]-fXN[m])*(fXN[n]-fXN[m])+(fYN[n]-fYN[m])*(fYN[n]-fYN[m]));
            if ((d1+d2<=d3) || (d1+d3<=d2) || (d2+d3<=d1)) goto L90;

            // does the triangle enclose the point?
            if (!Enclose(p,n,m,0)) goto L90;

            // is it a Delaunay triangle? (ie. are there any other points
            // inside the circle that is defined by its vertices?)

            // test the triangle for Delaunay'ness

            // loop over all other points testing each to see if it's
            // inside the triangle's circle
            ndegen = 0;
            for ( z=1; z<=fNpoints; z++) {
               if ((z==p) || (z==n) || (z==m)) goto L50;
               // An easy first check is to see if point z is inside the triangle
               // (if it's in the triangle it's also in the circle)

               // point z cannot be inside the triangle if it's further from (xx,yy)
               // than the furthest pointing making up the triangle - test this
               for (l=1; l<=fNpoints; l++) {
                  if (fOrder[l-1] == z) {
                     if ((l<i) || (l<j) || (l<k)) {
                        // point z is nearer to (xx,yy) than m, n or p - it could be in the
                        // triangle so call enclose to find out

                        // if it is inside the triangle this can't be a Delaunay triangle
                        if (Enclose(p,n,m,z)) goto L90;
                     } else {
                        // there's no way it could be in the triangle so there's no point
                        // calling enclose
                        goto L1;
                     }
                  }
               }
               // is point z colinear with any pair of the triangle points?
L1:
               if (((fXN[p]-fXN[z])*(fYN[p]-fYN[n])) == ((fYN[p]-fYN[z])*(fXN[p]-fXN[n]))) {
                  // z is colinear with p and n
                  a = p;
                  b = n;
               } else if (((fXN[p]-fXN[z])*(fYN[p]-fYN[m])) == ((fYN[p]-fYN[z])*(fXN[p]-fXN[m]))) {
                  // z is colinear with p and m
                  a = p;
                  b = m;
               } else if (((fXN[n]-fXN[z])*(fYN[n]-fYN[m])) == ((fYN[n]-fYN[z])*(fXN[n]-fXN[m]))) {
                  // z is colinear with n and m
                  a = n;
                  b = m;
               } else {
                  a = 0;
                  b = 0;
               }
               if (a != 0) {
                  // point z is colinear with 2 of the triangle points, if it lies
                  // between them it's in the circle otherwise it's outside
                  if (fXN[a] != fXN[b]) {
                     if (((fXN[z]-fXN[a])*(fXN[z]-fXN[b])) < 0) {
                        goto L90;
                     } else if (((fXN[z]-fXN[a])*(fXN[z]-fXN[b])) == 0) {
                        // At least two points are sitting on top of each other, we will
                        // treat these as one and not consider this a 'multiple points lying
                        // on a common circle' situation. It is a sign something could be
                        // wrong though, especially if the two coincident points have
                        // different fZ's. If they don't then this is harmless.
                        Warning("Interpolate", "Two of these three points are coincident %d %d %d",a,b,z);
                     }
                  } else {
                     if (((fYN[z]-fYN[a])*(fYN[z]-fYN[b])) < 0) {
                        goto L90;
                     } else if (((fYN[z]-fYN[a])*(fYN[z]-fYN[b])) == 0) {
                        // At least two points are sitting on top of each other - see above.
                        Warning("Interpolate", "Two of these three points are coincident %d %d %d",a,b,z);
                     }
                  }
                  // point is outside the circle, move to next point
                  goto L50;
               }

               // if point z were to look at the triangle, which point would it see
               // lying between the other two? (we're going to form a quadrilateral
               // from the points, and then demand certain properties of that
               // quadrilateral)
               dxz[0] = fXN[p]-fXN[z];
               dyz[0] = fYN[p]-fYN[z];
               dxz[1] = fXN[n]-fXN[z];
               dyz[1] = fYN[n]-fYN[z];
               dxz[2] = fXN[m]-fXN[z];
               dyz[2] = fYN[m]-fYN[z];
               for(l=1; l<=3; l++) {
                  dx1 = dxz[l-1];
                  dx2 = dxz[l%3];
                  dx3 = dxz[(l+1)%3];
                  dy1 = dyz[l-1];
                  dy2 = dyz[l%3];
                  dy3 = dyz[(l+1)%3];

                  // u et v are used only to know their sign. The previous
                  // code computed them with a division which was long and
                  // might be a division by 0. It is now replaced by a
                  // multiplication.
                  u = (dy3*dx2-dx3*dy2)*(dy1*dx2-dx1*dy2);
                  v = (dy3*dx1-dx3*dy1)*(dy2*dx1-dx2*dy1);

                  if ((u>=0) && (v>=0)) {
                     // vector (dx3,dy3) is expressible as a sum of the other two vectors
                     // with positive coefficents -> i.e. it lies between the other two vectors
                     if (l == 1) {
                        f  = m;
                        o1 = p;
                        o2 = n;
                     } else if (l == 2) {
                        f  = p;
                        o1 = n;
                        o2 = m;
                     } else {
                        f  = n;
                        o1 = m;
                        o2 = p;
                     }
                     goto L2;
                  }
               }
///            Error("Interpolate", "Should not get to here");
               // may as well soldier on
               f  = m;
               o1 = p;
               o2 = n;
L2:
               // this is not a valid quadrilateral if the diagonals don't cross,
               // check that points f and z lie on opposite side of the line o1-o2,
               // this is true if the angle f-o1-z is greater than o2-o1-z and o2-o1-f
               cfo1k  = ((fXN[f]-fXN[o1])*(fXN[z]-fXN[o1])+(fYN[f]-fYN[o1])*(fYN[z]-fYN[o1]))/
                        TMath::Sqrt(((fXN[f]-fXN[o1])*(fXN[f]-fXN[o1])+(fYN[f]-fYN[o1])*(fYN[f]-fYN[o1]))*
                        ((fXN[z]-fXN[o1])*(fXN[z]-fXN[o1])+(fYN[z]-fYN[o1])*(fYN[z]-fYN[o1])));
               co2o1k = ((fXN[o2]-fXN[o1])*(fXN[z]-fXN[o1])+(fYN[o2]-fYN[o1])*(fYN[z]-fYN[o1]))/
                        TMath::Sqrt(((fXN[o2]-fXN[o1])*(fXN[o2]-fXN[o1])+(fYN[o2]-fYN[o1])*(fYN[o2]-fYN[o1]))*
                        ((fXN[z]-fXN[o1])*(fXN[z]-fXN[o1])  + (fYN[z]-fYN[o1])*(fYN[z]-fYN[o1])));
               co2o1f = ((fXN[o2]-fXN[o1])*(fXN[f]-fXN[o1])+(fYN[o2]-fYN[o1])*(fYN[f]-fYN[o1]))/
                        TMath::Sqrt(((fXN[o2]-fXN[o1])*(fXN[o2]-fXN[o1])+(fYN[o2]-fYN[o1])*(fYN[o2]-fYN[o1]))*
                        ((fXN[f]-fXN[o1])*(fXN[f]-fXN[o1]) + (fYN[f]-fYN[o1])*(fYN[f]-fYN[o1]) ));
               if ((cfo1k>co2o1k) || (cfo1k>co2o1f)) {
                  // not a valid quadrilateral - point z is definitely outside the circle
                  goto L50;
               }
               // calculate the 2 internal angles of the quadrangle formed by joining
               // points z and f to points o1 and o2, at z and f. If they sum to less
               // than 180 degrees then z lies outside the circle
               dko1    = TMath::Sqrt((fXN[z]-fXN[o1])*(fXN[z]-fXN[o1])+(fYN[z]-fYN[o1])*(fYN[z]-fYN[o1]));
               dko2    = TMath::Sqrt((fXN[z]-fXN[o2])*(fXN[z]-fXN[o2])+(fYN[z]-fYN[o2])*(fYN[z]-fYN[o2]));
               dfo1    = TMath::Sqrt((fXN[f]-fXN[o1])*(fXN[f]-fXN[o1])+(fYN[f]-fYN[o1])*(fYN[f]-fYN[o1]));
               dfo2    = TMath::Sqrt((fXN[f]-fXN[o2])*(fXN[f]-fXN[o2])+(fYN[f]-fYN[o2])*(fYN[f]-fYN[o2]));
               c1      = ((fXN[z]-fXN[o1])*(fXN[z]-fXN[o2])+(fYN[z]-fYN[o1])*(fYN[z]-fYN[o2]))/dko1/dko2;
               c2      = ((fXN[f]-fXN[o1])*(fXN[f]-fXN[o2])+(fYN[f]-fYN[o1])*(fYN[f]-fYN[o2]))/dfo1/dfo2;
               sin_sum = c1*TMath::Sqrt(1-c2*c2)+c2*TMath::Sqrt(1-c1*c1);

               // sin_sum doesn't always come out as zero when it should do.
               if (sin_sum < -1.E-6) {
                  // z is inside the circle, this is not a Delaunay triangle
                  goto L90;
               } else if (TMath::Abs(sin_sum) <= 1.E-6) {
                  // point z lies on the circumference of the circle (within rounding errors)
                  // defined by the triangle, so there is potential for degeneracy in the
                  // triangle set (Delaunay triangulation does not give a unique way to split
                  // a polygon whose points lie on a circle into constituent triangles). Make
                  // a note of the additional point number.
                  ndegen++;
                  degen   = z;
                  fdegen  = f;
                  o1degen = o1;
                  o2degen = o2;
               }
L50:
            continue;
            }
            // This is a good triangle
            if (ndegen > 0) {
               // but is degenerate with at least one other,
               // haven't figured out what to do if more than 4 points are involved
///            if (ndegen > 1) {
///               Error("Interpolate",
///                     "More than 4 points lying on a circle. No decision making process formulated for triangulating this region in a non-arbitrary way %d %d %d %d",
///                     p,n,m,degen);
///               return thevalue;
///            }

               // we have a quadrilateral which can be split down either diagonal
               // (d<->f or o1<->o2) to form valid Delaunay triangles. Choose diagonal
               // with highest average z-value. Whichever we choose we will have
               // verified two triangles as good and two as bad, only note the good ones
               d  = degen;
               f  = fdegen;
               o1 = o1degen;
               o2 = o2degen;
               if ((fZ[o1-1]+fZ[o2-1]) > (fZ[d-1]+fZ[f-1])) {
                  // best diagonalisation of quadrilateral is current one, we have
                  // the triangle
                  t1 = p;
                  t2 = n;
                  t3 = m;
                  // file the good triangles
                  FileIt(p, n, m);
                  FileIt(d, o1, o2);
               } else {
                  // use other diagonal to split quadrilateral, use triangle formed by
                  // point f, the degnerate point d and whichever of o1 and o2 create
                  // an enclosing triangle
                  t1 = f;
                  t2 = d;
                  if (Enclose(f,d,o1,0)) {
                     t3 = o1;
                  } else {
                     t3 = o2;
                  }
                  // file the good triangles
                  FileIt(f, d, o1);
                  FileIt(f, d, o2);
               }
            } else {
               // this is a Delaunay triangle, file it
               FileIt(p, n, m);
               t1 = p;
               t2 = n;
               t3 = m;
            }
            // do the interpolation
            thevalue = InterpolateOnPlane(t1,t2,t3,0);
            return thevalue;
L90:
            continue;
         }
      }
   }
   if (shouldbein) {
      Error("Interpolate",
            "Point outside hull when expected inside: this point could be dodgy %g %g %d",
             xx, yy, ntris_tried);
   }
   return thevalue;
}


//______________________________________________________________________________
void TGraphDelaunay::SetMaxIter(Int_t n)
{
   // Defines the number of triangles tested for a Delaunay triangle
   // (number of iterations) before abandoning the search

   fAllTri  = kFALSE;
   fMaxIter = n;
}


//______________________________________________________________________________
void TGraphDelaunay::SetMarginBinsContent(Double_t z)
{
   // Sets the histogram bin height for points lying outside the convex hull ie:
   // the bins in the margin.

   fZout = z;
}
