// @(#)root/hist:$Name:  $:$Id: TGraphDelaunay.cxx,v 1.00
// Author: Olivier Couet, Luke Jones (Royal Holloway, University of London)

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TMath.h"
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

   fGraph2D    = 0;
   fX          = 0;
   fY          = 0;
   fZ          = 0;
   fNpoints    = 0;
   fTriedSize  = 0;
   fZout       = 0.;
   fNdt        = 0;
   fNhull      = 0;
   fHullPoints = 0;
   fXN         = 0;
   fYN         = 0;
   fOrder      = 0;
   fDist       = 0;
   fPTried     = 0;
   fNTried     = 0;
   fMTried     = 0;
   fInit       = kFALSE;

   SetMaxIter();
}


//______________________________________________________________________________
TGraphDelaunay::TGraphDelaunay(TGraph2D *g)
            : TNamed("TGraphDelaunay","TGraphDelaunay")
{
   // TGraphDelaunay default constructor

   fGraph2D    = g;
   fX          = fGraph2D->GetX();
   fY          = fGraph2D->GetY();
   fZ          = fGraph2D->GetZ();
   fNpoints    = fGraph2D->GetN();
   fTriedSize  = 0;
   fZout       = 0.;
   fNdt        = 0;
   fNhull      = 0;
   fHullPoints = 0;
   fXN         = 0;
   fYN         = 0;
   fOrder      = 0;
   fDist       = 0;
   fPTried     = 0;
   fNTried     = 0;
   fMTried     = 0;
   fInit       = kFALSE;

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

   Double_t xx, yy;
   xx = (x+fXoffset)*fScaleFactor;
   yy = (y+fYoffset)*fScaleFactor;
   return Interpolate(xx, yy);
}


//______________________________________________________________________________
void TGraphDelaunay::CreateTrianglesDataStructure()
{
   // Fonction used internally only. It creates the data structures needed to
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
   fScaleFactor  = 2./((xmax-xmin)+(ymax-ymin));
   fXNmax        = (xmax+fXoffset)*fScaleFactor;
   fXNmin        = (xmin+fXoffset)*fScaleFactor;
   fYNmax        = (ymax+fYoffset)*fScaleFactor;
   fYNmin        = (ymin+fYoffset)*fScaleFactor;
   fXN           = new Double_t[fNpoints+1];
   fYN           = new Double_t[fNpoints+1];
   for (Int_t N=0; N<fNpoints; N++) {
      fXN[N+1] = (fX[N]+fXoffset)*fScaleFactor;
      fYN[N+1] = (fY[N]+fYoffset)*fScaleFactor;
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
Bool_t TGraphDelaunay::Enclose(Int_t T1, Int_t T2, Int_t T3, Int_t E) const
{
   // Is point E inside the triangle T1-T2-T3 ?

   Int_t A = 0, B = 0;
   Double_t dx1,dx2,dx3,dy1,dy2,dy3,U,V;

   // First ask if point E is colinear with any pair of the triangle points
   if (((fXN[T1]-fXN[E])*(fYN[T1]-fYN[T2])) == ((fYN[T1]-fYN[E])*(fXN[T1]-fXN[T2]))) {
      // E is colinear with T1 and T2
      A = T1;
      B = T2;
   } else if (((fXN[T1]-fXN[E])*(fYN[T1]-fYN[T3])) == ((fYN[T1]-fYN[E])*(fXN[T1]-fXN[T3]))) {
      // E is colinear with T1 and T3
      A = T1;
      B = T3;
   } else if (((fXN[T2]-fXN[E])*(fYN[T2]-fYN[T3])) == ((fYN[T2]-fYN[E])*(fXN[T2]-fXN[T3]))) {
      // E is colinear with T2 and T3
      A = T2;
      B = T3;
   }
   if (A != 0) {
      // point E is colinear with 2 of the triangle points, if it lies 
      // between them it's in the circle otherwise it's outside
      if (fXN[A] != fXN[B]) {
         if (((fXN[E]-fXN[A])*(fXN[E]-fXN[B])) <= 0) return kTRUE;
      } else {
         if (((fYN[E]-fYN[A])*(fYN[E]-fYN[B])) <= 0) return kTRUE;
      }
      // point is outside the triangle
      return kFALSE;
   }

   // E is not colinear with any pair of triangle points, if it is inside
   // the triangle then the vector from E to one of the corners must be 
   // expressible as a sum with positive coefficients of the vectors from 
   // the two other corners to E. Say vector3=U*vector1+V*vector2

   // vector1==T1->E
   dx1 = fXN[E]-fXN[T1];
   dy1 = fYN[E]-fYN[T1];
   // vector2==T2->E
   dx2 = fXN[E]-fXN[T2];
   dy2 = fYN[E]-fYN[T2];
   // vector3==E->T3
   dx3 = fXN[T3]-fXN[E];
   dy3 = fYN[T3]-fYN[E];

   U = (dx2*dy3-dx3*dy2)/(dx2*dy1-dx1*dy2);
   V = (dx1*dy3-dx3*dy1)/(dx1*dy2-dx2*dy1);

   if ((U>=0) && (V>=0)) return kTRUE;

   return kFALSE;
}


//______________________________________________________________________________
void TGraphDelaunay::FileIt(Int_t P, Int_t N, Int_t M)
{
   // Files the triangle defined by the 3 vertices P, N and M into the 
   // fxTried arrays. If these arrays are to small they are automatically
   // expanded.

   Bool_t swap;
   Int_t tmp, Ps = P, Ns = N, Ms = M;

   // order the vertices before storing them
L1:
   swap = kFALSE;
   if (Ns > Ps) { tmp = Ps; Ps = Ns; Ns = tmp; swap = kTRUE;}
   if (Ms > Ns) { tmp = Ns; Ns = Ms; Ms = tmp; swap = kTRUE;}
   if (swap) goto L1;

   // expand the triangles storage if needed
   if (fNdt> fTriedSize) {
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
   fPTried[fNdt-1] = Ps;
   fNTried[fNdt-1] = Ns;
   fMTried[fNdt-1] = Ms;
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
   Double_t sx,sy,nx,ny,mx,my,mdotn,nn,A;
   Int_t T1,T2,Pa,Na,Ma,Pb,Nb,Mb,P1=0,P2=0,M,N,P3=0;
   Bool_t s[3];
   Double_t alittlebit = 0.0001;

   // start with a point that is guaranteed to be inside the hull (the 
   // centre of the hull). The starting point is shifted "a little bit" 
   // otherwise, in case of triangles aligned on a regular grid, we may
   // found none of them.
   xcntr = 0;
   ycntr = 0;
   for (N=1; N<=fNhull; N++) {
      xcntr = xcntr+fXN[fHullPoints[N-1]];
      ycntr = ycntr+fYN[fHullPoints[N-1]];
   }
   xcntr = xcntr/fNhull+alittlebit;
   ycntr = ycntr/fNhull+alittlebit;
   // and calculate it's triangle
   Interpolate(xcntr,ycntr);

   // loop over all Delaunay triangles (including those constantly being 
   // produced within the loop) and check to see if their 3 sides also 
   // correspond to the sides of other Delaunay triangles, i.e. that they 
   // have all their neighbours.
   T1 = 1;
   while (T1 <= fNdt) {
      // get the three points that make up this triangle
      Pa = fPTried[T1-1];
      Na = fNTried[T1-1];
      Ma = fMTried[T1-1];

      // produce three integers which will represent the three sides
      s[0]  = kFALSE;
      s[1]  = kFALSE;
      s[2]  = kFALSE;
      // loop over all other Delaunay triangles
      for (T2=1; T2<=fNdt; T2++) {
         if (T2 != T1) {
            // get the points that make up this triangle
            Pb = fPTried[T2-1];
            Nb = fNTried[T2-1];
            Mb = fMTried[T2-1];
            // do triangles T1 and T2 share a side?
            if ((Pa==Pb && Na==Nb) || (Pa==Pb && Na==Mb) || (Pa==Nb && Na==Mb)) {
               // they share side 1
               s[0] = kTRUE;
            } else if ((Pa==Pb && Ma==Nb) || (Pa==Pb && Ma==Mb) || (Pa==Nb && Ma==Mb)) {
               // they share side 2
               s[1] = kTRUE;
            } else if ((Na==Pb && Ma==Nb) || (Na==Pb && Ma==Mb) || (Na==Nb && Ma==Mb)) {
               // they share side 3
               s[2] = kTRUE;
            }
         }
         // if T1 shares all its sides with other Delaunay triangles then 
         // forget about it
         if (s[0] && s[1] && s[2]) continue;
      }
      // Looks like T1 is missing a neighbour on at least one side.
      // For each side, take a point a little bit beyond it and calculate 
      // the Delaunay triangle for that point, this should be the triangle 
      // which shares the side.
      for (M=1; M<=3; M++) {
         if (!s[M-1]) {
            // get the two points that make up this side
            if (M == 1) {
               P1 = Pa;
               P2 = Na;
               P3 = Ma;
            } else if (M == 2) {
               P1 = Pa;
               P2 = Ma;
               P3 = Na;
            } else if (M == 3) {
               P1 = Na;
               P2 = Ma;
               P3 = Pa;
            }
            // get the coordinates of the centre of this side
            xm = (fXN[P1]+fXN[P2])/2.;
            ym = (fYN[P1]+fYN[P2])/2.;
            // we want to add a little to these coordinates to get a point just
            // outside the triangle; (sx,sy) will be the vector that represents 
            // the side
            sx = fXN[P1]-fXN[P2];
            sy = fYN[P1]-fYN[P2];
            // (nx,ny) will be the normal to the side, but don't know if it's 
            // pointing in or out yet
            nx    = sy;
            ny    = -sx;
            nn    = TMath::Sqrt(nx*nx+ny*ny);
            nx    = nx/nn;
            ny    = ny/nn;
            mx    = fXN[P3]-xm;
            my    = fYN[P3]-ym;
            mdotn = mx*nx+my*ny;
            if (mdotn > 0) {
               // (nx,ny) is pointing in, we want it pointing out
               nx = -nx;
               ny = -ny;
            }
            // increase/decrease xm and ym a little to produce a point 
            // just outside the triangle (ensuring that the amount added will 
            // be large enough such that it won't be lost in rounding errors)
            A  = TMath::Abs(TMath::Max(alittlebit*xm,alittlebit*ym));
            xx = xm+nx*A;
            yy = ym+ny*A;
            // try and find a new Delaunay triangle for this point
            Interpolate(xx,yy);

            // this side of T1 should now, hopefully, if it's not part of the 
            // hull, be shared with a new Delaunay triangle just calculated by Interpolate
         }
      }
      T1++;
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

   Int_t N,nhull_tmp;
   Bool_t in;

   if (!fHullPoints) fHullPoints = new Int_t[fNpoints];

   nhull_tmp = 0;
   for(N=1; N<=fNpoints; N++) {
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
Bool_t TGraphDelaunay::InHull(Int_t E, Int_t X) const
{
   // Is point E inside the hull defined by all points apart from X ?

   Int_t n1,n2,N,M,Ntry;
   Double_t lastdphi,dd1,dd2,dx1,dx2,dx3,dy1,dy2,dy3;
   Double_t U,V,vNv1,vNv2,phi1,phi2,dphi,xx,yy;

   Bool_t DTinhull = kFALSE;

   xx = fXN[E];
   yy = fYN[E];

   if (fNhull > 0) {
      //  The hull has been found - no need to use any points other than 
      //  those that make up the hull
      Ntry = fNhull;
   } else {
      //  The hull has not yet been found, will have to try every point
      Ntry = fNpoints;
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
   dx1  = xx-fXN[n1];
   dy1  = yy-fYN[n1];
   dx2  = xx-fXN[n2];
   dy2  = yy-fYN[n2];
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
         dx1 = xx-fXN[n1];
         dy1 = yy-fYN[n1];
         dx2 = xx-fXN[n2];
         dy2 = yy-fYN[n2];
         dx3 = xx-fXN[M];
         dy3 = yy-fYN[M];

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
Double_t TGraphDelaunay::InterpolateOnPlane(Int_t TI1, Int_t TI2, Int_t TI3, Int_t E) const
{
   // Finds the z-value at point E given that it lies 
   // on the plane defined by T1,T2,T3

   Int_t tmp;
   Bool_t swap;
   Double_t x1,x2,x3,y1,y2,y3,f1,f2,f3,U,V,W;

   Int_t T1 = TI1;
   Int_t T2 = TI2;
   Int_t T3 = TI3;

   // order the vertices
L1:
   swap = kFALSE;
   if (T2 > T1) { tmp = T1; T1 = T2; T2 = tmp; swap = kTRUE;}
   if (T3 > T2) { tmp = T2; T2 = T3; T3 = tmp; swap = kTRUE;}
   if (swap) goto L1;

   x1 = fXN[T1];
   x2 = fXN[T2];
   x3 = fXN[T3];
   y1 = fYN[T1];
   y2 = fYN[T2];
   y3 = fYN[T3];
   f1 = fZ[T1-1];
   f2 = fZ[T2-1];
   f3 = fZ[T3-1];
   U  = (f1*(y2-y3)+f2*(y3-y1)+f3*(y1-y2))/(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2));
   V  = (f1*(x2-x3)+f2*(x3-x1)+f3*(x1-x2))/(y1*(x2-x3)+y2*(x3-x1)+y3*(x1-x2));
   W  = f1-U*x1-V*y1;

   return U*fXN[E]+V*fYN[E]+W;
}


//______________________________________________________________________________
Double_t TGraphDelaunay::Interpolate(Double_t xx, Double_t yy)
{
   // Finds the Delaunay triangle that the point (xi,yi) sits in (if any) and 
   // calculate a z-value for it by linearly interpolating the z-values that 
   // make up that triangle.

   Double_t thevalue;

   Int_t IT, ntris_tried, P, N, M;
   Int_t I,J,K,L,Z,F,D,O1,O2,A,B,T1,T2,T3;
   Int_t ndegen=0,degen=0,fdegen=0,o1degen=0,o2degen=0;
   Double_t vxN,vyN;
   Double_t d1,d2,d3,c1,c2,dko1,dko2,dfo1;
   Double_t dfo2,sin_sum,cfo1k,co2o1k,co2o1f;

   Bool_t shouldbein;
   Double_t dx1,dx2,dx3,dy1,dy2,dy3,U,V,dxz[3],dyz[3];

   // initialise the Delaunay algorithm
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
   for (IT=1; IT<=fNdt; IT++) {
      P = fPTried[IT-1];
      N = fNTried[IT-1];
      M = fMTried[IT-1];
      // P, N and M form a previously found Delaunay triangle, does it 
      // enclose the point?
      if (Enclose(P,N,M,0)) {
         // yes, we have the triangle
         thevalue = InterpolateOnPlane(P,N,M,0);
         return thevalue;
      }
   }

   // is this point inside the convex hull?
   shouldbein = InHull(0,-1);
   if (!shouldbein) return thevalue;

   // it must be in a Delaunay triangle - find it...

   // order mass points by distance in mass plane from desired point
   for (IT=1; IT<=fNpoints; IT++) {
      vxN = fXN[IT];
      vyN = fYN[IT];
      fDist[IT-1] = TMath::Sqrt((xx-vxN)*(xx-vxN)+(yy-vyN)*(yy-vyN));
   }

   // sort array 'fDist' to find closest points
   TMath::Sort(fNpoints, fDist, fOrder, kFALSE);
   for (IT=0; IT<fNpoints; IT++) fOrder[IT]++;

   // loop over triplets of close points to try to find a triangle that 
   // encloses the point.
   for (K=3; K<=fNpoints; K++) {
      M = fOrder[K-1];
      for (J=2; J<=K-1; J++) {
         N = fOrder[J-1];
         for (I=1; I<=J-1; I++) {
            P = fOrder[I-1];
            if (ntris_tried > fMaxIter) {
               // perhaps this point isn't in the hull after all
///            Warning("Interpolate", 
///                    "Abandoning the effort to find a Delaunay triangle (and thus interpolated Z-value) for point %g %g"
///                    ,xx,yy);
               return thevalue;
            }
            ntris_tried++;
            // check the points aren't colinear
            d1 = TMath::Sqrt((fXN[P]-fXN[N])*(fXN[P]-fXN[N])+(fYN[P]-fYN[N])*(fYN[P]-fYN[N]));
            d2 = TMath::Sqrt((fXN[P]-fXN[M])*(fXN[P]-fXN[M])+(fYN[P]-fYN[M])*(fYN[P]-fYN[M]));
            d3 = TMath::Sqrt((fXN[N]-fXN[M])*(fXN[N]-fXN[M])+(fYN[N]-fYN[M])*(fYN[N]-fYN[M]));
            if ((d1+d2<=d3) || (d1+d3<=d2) || (d2+d3<=d1)) goto L90;

            // does the triangle enclose the point?
            if (!Enclose(P,N,M,0)) goto L90;

            // is it a Delaunay triangle? (ie. are there any other points 
            // inside the circle that is defined by its vertices?)

            // test the triangle for Delaunay'ness

            // loop over all other points testing each to see if it's 
            // inside the triangle's circle
            ndegen = 0;
            for ( Z=1; Z<=fNpoints; Z++) {
               if ((Z==P) || (Z==N) || (Z==M)) goto L50;
               // An easy first check is to see if point Z is inside the triangle 
               // (if it's in the triangle it's also in the circle)

               // point Z cannot be inside the triangle if it's further from (xx,yy) 
               // than the furthest pointing making up the triangle - test this
               for (L=1; L<=fNpoints; L++) {
                  if (fOrder[L-1] == Z) {
                     if ((L<I) || (L<J) || (L<K)) {
                        // point Z is nearer to (xx,yy) than M, N or P - it could be in the 
                        // triangle so call enclose to find out

                        // if it is inside the triangle this can't be a Delaunay triangle
                        if (Enclose(P,N,M,Z)) goto L90;
                     } else {
                        // there's no way it could be in the triangle so there's no point 
                        // calling enclose
                        goto L1;
                     }
                  }
               }
               // is point Z colinear with any pair of the triangle points?
L1:
               if (((fXN[P]-fXN[Z])*(fYN[P]-fYN[N])) == ((fYN[P]-fYN[Z])*(fXN[P]-fXN[N]))) {
                  // Z is colinear with P and N
                  A = P;
                  B = N;
               } else if (((fXN[P]-fXN[Z])*(fYN[P]-fYN[M])) == ((fYN[P]-fYN[Z])*(fXN[P]-fXN[M]))) {
                  // Z is colinear with P and M
                  A = P;
                  B = M;
               } else if (((fXN[N]-fXN[Z])*(fYN[N]-fYN[M])) == ((fYN[N]-fYN[Z])*(fXN[N]-fXN[M]))) {
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
                  if (fXN[A] != fXN[B]) {
                     if (((fXN[Z]-fXN[A])*(fXN[Z]-fXN[B])) < 0) {
                        goto L90;
                     } else if (((fXN[Z]-fXN[A])*(fXN[Z]-fXN[B])) == 0) {
                        // At least two points are sitting on top of each other, we will
                        // treat these as one and not consider this a 'multiple points lying
                        // on a common circle' situation. It is a sign something could be
                        // wrong though, especially if the two coincident points have
                        // different fZ's. If they don't then this is harmless.
                        Warning("Interpolate", "Two of these three points are coincident %d %d %d",A,B,Z);
                     }
                  } else {
                     if (((fYN[Z]-fYN[A])*(fYN[Z]-fYN[B])) < 0) {
                        goto L90;
                     } else if (((fYN[Z]-fYN[A])*(fYN[Z]-fYN[B])) == 0) {
                        // At least two points are sitting on top of each other - see above.
                        Warning("Interpolate", "Two of these three points are coincident %d %d %d",A,B,Z);
                     }
                  }
                  // point is outside the circle, move to next point
                  goto L50;
               }

               // if point Z were to look at the triangle, which point would it see 
               // lying between the other two? (we're going to form a quadrilateral 
               // from the points, and then demand certain properties of that
               // quadrilateral)
               dxz[0] = fXN[P]-fXN[Z];
               dyz[0] = fYN[P]-fYN[Z];
               dxz[1] = fXN[N]-fXN[Z];
               dyz[1] = fYN[N]-fYN[Z];
               dxz[2] = fXN[M]-fXN[Z];
               dyz[2] = fYN[M]-fYN[Z];
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
///            Error("Interpolate", "Should not get to here");
               // may as well soldier on
               F  = M;
               O1 = P;
               O2 = N;
L2:
               // this is not a valid quadrilateral if the diagonals don't cross, 
               // check that points F and Z lie on opposite side of the line O1-O2,
               // this is true if the angle F-O1-Z is greater than O2-O1-Z and O2-O1-F
               cfo1k  = ((fXN[F]-fXN[O1])*(fXN[Z]-fXN[O1])+(fYN[F]-fYN[O1])*(fYN[Z]-fYN[O1]))/
                        TMath::Sqrt(((fXN[F]-fXN[O1])*(fXN[F]-fXN[O1])+(fYN[F]-fYN[O1])*(fYN[F]-fYN[O1]))*
                        ((fXN[Z]-fXN[O1])*(fXN[Z]-fXN[O1])+(fYN[Z]-fYN[O1])*(fYN[Z]-fYN[O1])));
               co2o1k = ((fXN[O2]-fXN[O1])*(fXN[Z]-fXN[O1])+(fYN[O2]-fYN[O1])*(fYN[Z]-fYN[O1]))/
                        TMath::Sqrt(((fXN[O2]-fXN[O1])*(fXN[O2]-fXN[O1])+(fYN[O2]-fYN[O1])*(fYN[O2]-fYN[O1]))*
                        ((fXN[Z]-fXN[O1])*(fXN[Z]-fXN[O1])  + (fYN[Z]-fYN[O1])*(fYN[Z]-fYN[O1])));
               co2o1f = ((fXN[O2]-fXN[O1])*(fXN[F]-fXN[O1])+(fYN[O2]-fYN[O1])*(fYN[F]-fYN[O1]))/
                        TMath::Sqrt(((fXN[O2]-fXN[O1])*(fXN[O2]-fXN[O1])+(fYN[O2]-fYN[O1])*(fYN[O2]-fYN[O1]))*
                        ((fXN[F]-fXN[O1])*(fXN[F]-fXN[O1]) + (fYN[F]-fYN[O1])*(fYN[F]-fYN[O1]) ));
               if ((cfo1k>co2o1k) || (cfo1k>co2o1f)) {
                  // not a valid quadrilateral - point Z is definitely outside the circle
                  goto L50;
               }
               // calculate the 2 internal angles of the quadrangle formed by joining
               // points Z and F to points O1 and O2, at Z and F. If they sum to less
               // than 180 degrees then Z lies outside the circle
               dko1    = TMath::Sqrt((fXN[Z]-fXN[O1])*(fXN[Z]-fXN[O1])+(fYN[Z]-fYN[O1])*(fYN[Z]-fYN[O1]));
               dko2    = TMath::Sqrt((fXN[Z]-fXN[O2])*(fXN[Z]-fXN[O2])+(fYN[Z]-fYN[O2])*(fYN[Z]-fYN[O2]));
               dfo1    = TMath::Sqrt((fXN[F]-fXN[O1])*(fXN[F]-fXN[O1])+(fYN[F]-fYN[O1])*(fYN[F]-fYN[O1]));
               dfo2    = TMath::Sqrt((fXN[F]-fXN[O2])*(fXN[F]-fXN[O2])+(fYN[F]-fYN[O2])*(fYN[F]-fYN[O2]));
               c1      = ((fXN[Z]-fXN[O1])*(fXN[Z]-fXN[O2])+(fYN[Z]-fYN[O1])*(fYN[Z]-fYN[O2]))/dko1/dko2;
               c2      = ((fXN[F]-fXN[O1])*(fXN[F]-fXN[O2])+(fYN[F]-fYN[O1])*(fYN[F]-fYN[O2]))/dfo1/dfo2;
               sin_sum = c1*TMath::Sqrt(1-c2*c2)+c2*TMath::Sqrt(1-c1*c1);

               // sin_sum doesn't always come out as zero when it should do.
               if (sin_sum < -1.E-6) {
                  // Z is inside the circle, this is not a Delaunay triangle
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
///            if (ndegen > 1) {
///               Error("Interpolate", 
///                     "More than 4 points lying on a circle. No decision making process formulated for triangulating this region in a non-arbitrary way %d %d %d %d",
///                     P,N,M,degen);
///               return thevalue;
///            }

               // we have a quadrilateral which can be split down either diagonal
               // (D<->F or O1<->O2) to form valid Delaunay triangles. Choose diagonal
               // with highest average z-value. Whichever we choose we will have
               // verified two triangles as good and two as bad, only note the good ones
               D  = degen;
               F  = fdegen;
               O1 = o1degen;
               O2 = o2degen;
               if ((fZ[O1-1]+fZ[O2-1]) > (fZ[D-1]+fZ[F-1])) {
                  // best diagonalisation of quadrilateral is current one, we have 
                  // the triangle
                  T1 = P;
                  T2 = N;
                  T3 = M;
                  // file the good triangles
                  FileIt(P, N, M);
                  FileIt(D, O1, O2);
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
                  // file the good triangles
                  FileIt(F, D, O1);
                  FileIt(F, D, O2);
               }
            } else {
               // this is a Delaunay triangle, file it
               FileIt(P, N, M);
               T1 = P;
               T2 = N;
               T3 = M;
            }
            // do the interpolation
            thevalue = InterpolateOnPlane(T1,T2,T3,0);
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
