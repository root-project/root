// @(#)root/geom:$Id$
// Author: Mihaela Gheata   5/01/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//____________________________________________________________________________
// TGeoPolygon - Arbitrary polygon class
//____________________________________________________________________________
//
// A polygon is a 2D shape defined by vertices in the XY plane. It is used by
// TGeoXtru class for computing Contains() and Safety(). Only the pointers to
// the actual lists of XY values are used - these are not owned by the class.
//
// To check if a point in XY plane is contained by a polygon, this is split
// into an outscribed convex polygon and the remaining polygons of its subtracton
// from the outscribed one. A point is INSIDE if it is
// contained by the outscribed polygon but NOT by the remaining ones. Since these
// can also be arbitrary polygons at their turn, a tree structure is formed:
//
//  P = Pconvex - (Pconvex-P)           where (-) means 'subtraction'
//  Pconvex-P = P1 + P2 + ...           where (+) means 'union'
//
//  *Note that P1, P2, ... do not intersect each other and they are defined
//   by subsets of the list of vertices of P. They can be split in the same
//   way as P*
//
// Therefore, if C(P) represents the Boolean : 'does P contains a given point?',
// then:
//
// C(P) = C(Pconvex) .and. not(C(P1) | C(P2) | ...)
//
// For creating a polygon without TGeoXtru class, one has to call the constructor
// TGeoPolygon(nvert) and then SetXY(Double_t *x, Double_t *y) providing the
// arrays of X and Y vertex positions (defined clockwise) that have to 'live' longer
// than the polygon they will describe. This complication is due to efficiency reasons.
// At the end one has to call the FinishPolygon() method.

#include "TObjArray.h"
#include "TGeoPolygon.h"
#include "TMath.h"
#include "TGeoShape.h"
#include "TGeoManager.h"
#include "TVirtualGeoPainter.h"

ClassImp(TGeoPolygon)

//_____________________________________________________________________________
TGeoPolygon::TGeoPolygon()
{
// Dummy constructor.
   fNvert   = 0;
   fNconvex = 0;
   fInd     = 0;
   fIndc    = 0;
   fX       = 0;
   fY       = 0;
   fDaughters = 0;
   SetConvex(kFALSE);
   TObject::SetBit(kGeoFinishPolygon, kFALSE);
}

//_____________________________________________________________________________
TGeoPolygon::TGeoPolygon(Int_t nvert)
{
// Default constructor.
   if (nvert<3) {
      Fatal("Ctor", "Invalid number of vertices %i", nvert);
      return;
   }
   fNvert   = nvert;
   fNconvex = 0;
   fInd     = new Int_t[nvert];
   fIndc    = 0;
   fX       = 0;
   fY       = 0;
   fDaughters = 0;
   SetConvex(kFALSE);
   TObject::SetBit(kGeoFinishPolygon, kFALSE);
   SetNextIndex();
}

//_____________________________________________________________________________
TGeoPolygon::~TGeoPolygon()
{
// Destructor
   if (fInd)  delete [] fInd;
   if (fIndc) delete [] fIndc;
   if (fDaughters) {
      fDaughters->Delete();
      delete fDaughters;
   }
}
//_____________________________________________________________________________
Double_t TGeoPolygon::Area() const
{
// Computes area of the polygon in [length^2].
   Int_t ic,i,j;
   Double_t area = 0;
   // Compute area of the convex part
   for (ic=0; ic<fNvert; ic++) {
      i = fInd[ic];
      j = fInd[(ic+1)%fNvert];
      area += 0.5*(fX[i]*fY[j]-fX[j]*fY[i]);
   }
   return TMath::Abs(area);
}

//_____________________________________________________________________________
Bool_t TGeoPolygon::Contains(const Double_t *point) const
{
// Check if a point given by X = point[0], Y = point[1] is inside the polygon.
   Int_t i;
   TGeoPolygon *poly;
   for (i=0; i<fNconvex; i++)
      if (!IsRightSided(point, fIndc[i], fIndc[(i+1)%fNconvex])) return kFALSE;
   if (!fDaughters) return kTRUE;
   Int_t nd = fDaughters->GetEntriesFast();
   for (i=0; i<nd; i++) {
      poly = (TGeoPolygon*)fDaughters->UncheckedAt(i);
      if (poly->Contains(point)) return kFALSE;
   }
   return kTRUE;
}

//_____________________________________________________________________________
void TGeoPolygon::ConvexCheck()
{
// Check polygon convexity.
   if (fNvert==3) {
      SetConvex();
      return;
   }
   Int_t j,k;
   Double_t point[2];
   for (Int_t i=0; i<fNvert; i++) {
      j = (i+1)%fNvert;
      k = (i+2)%fNvert;
      point[0] = fX[fInd[k]];
      point[1] = fY[fInd[k]];
      if (!IsRightSided(point, fInd[i], fInd[j])) return;
   }
   SetConvex();
}

//_____________________________________________________________________________
void TGeoPolygon::Draw(Option_t *)
{
// Draw the polygon.
   if (!gGeoManager) return;
   gGeoManager->GetGeomPainter()->DrawPolygon(this);
}

//_____________________________________________________________________________
void TGeoPolygon::FinishPolygon()
{
// Decompose polygon in a convex outscribed part and a list of daughter
// polygons that have to be substracted to get the actual one.
   TObject::SetBit(kGeoFinishPolygon);
   // check convexity
   ConvexCheck();
   // find outscribed convex polygon indices
   OutscribedConvex();
   if (IsConvex()) {
//      printf(" -> polygon convex -> same indices\n");
      memcpy(fIndc, fInd, fNvert*sizeof(Int_t));
      return;
   }
//   printf(" -> polygon NOT convex\n");
   // make daughters if necessary
   if (IsConvex()) return;
   // ... algorithm here
   if (!fDaughters) fDaughters = new TObjArray();
   TGeoPolygon *poly = 0;
   Int_t indconv = 0;
   Int_t indnext, indback;
   Int_t nskip;
   while (indconv < fNconvex) {
      indnext = (indconv+1)%fNconvex;
      nskip = fIndc[indnext]-fIndc[indconv];
      if (nskip<0) nskip+=fNvert;
      if (nskip==1) {
         indconv++;
         continue;
      }
      // gap -> make polygon
      poly = new TGeoPolygon(nskip+1);
      poly->SetXY(fX,fY);
      poly->SetNextIndex(fInd[fIndc[indconv]]);
      poly->SetNextIndex(fInd[fIndc[indnext]]);
      indback = fIndc[indnext]-1;
      if (indback < 0) indback+=fNvert;
      while (indback != fIndc[indconv]) {
         poly->SetNextIndex(fInd[indback]);
         indback--;
         if (indback < 0) indback+=fNvert;
      }
      poly->FinishPolygon();
      fDaughters->Add(poly);
      indconv++;
   }
   for (indconv=0; indconv<fNconvex; indconv++) fIndc[indconv] = fInd[fIndc[indconv]];
}

//_____________________________________________________________________________
void TGeoPolygon::GetVertices(Double_t *x, Double_t *y) const
{
// Fill list of vertices into provided arrays.
   memcpy(x, fX, fNvert*sizeof(Double_t));
   memcpy(y, fY, fNvert*sizeof(Double_t));
}

//_____________________________________________________________________________
void TGeoPolygon::GetConvexVertices(Double_t *x, Double_t *y) const
{
// Fill list of vertices of the convex outscribed polygon into provided arrays.
   for (Int_t ic=0; ic<fNconvex; ic++) {
      x[ic] = fX[fIndc[ic]];
      y[ic] = fY[fIndc[ic]];
   }
}

//_____________________________________________________________________________
Bool_t TGeoPolygon::IsRightSided(const Double_t *point, Int_t ind1, Int_t ind2) const
{
// Check if POINT is right-sided with respect to the segment defined by IND1 and IND2.
   Double_t dot = (point[0]-fX[ind1])*(fY[ind2]-fY[ind1]) -
                  (point[1]-fY[ind1])*(fX[ind2]-fX[ind1]);
   if (!IsClockwise()) dot = -dot;
   if (dot<-1.E-10) return kFALSE;
   return kTRUE;
}

//_____________________________________________________________________________
Bool_t TGeoPolygon::IsSegConvex(Int_t i1, Int_t i2) const
{
// Check if a segment [0..fNvert-1] belongs to the outscribed convex pgon.
   if (i2<0) i2=(i1+1)%fNvert;
   Double_t point[2];
   for (Int_t i=0; i<fNvert; i++) {
      if (i==i1 || i==i2) continue;
      point[0] = fX[fInd[i]];
      point[1] = fY[fInd[i]];
      if (!IsRightSided(point, fInd[i1], fInd[i2])) return kFALSE;
   }
   return kTRUE;
}

//_____________________________________________________________________________
Bool_t TGeoPolygon::IsIllegalCheck() const
{
// Check for illegal crossings between non-consecutive segments
   if (fNvert<4) return kFALSE;
   Bool_t is_illegal = kFALSE;
   Double_t x1,y1,x2,y2,x3,y3,x4,y4;
   for (Int_t i=0; i<fNvert-2; i++) {
      // Check segment i
      for (Int_t j=i+2; j<fNvert; j++) {
         // Versus segment j
         if (i==0 && j==(fNvert-1)) continue;
         x1 = fX[i];
         y1 = fY[i];
         x2 = fX[i+1];
         y2 = fY[i+1];
         x3 = fX[j];
         y3 = fY[j];
         x4 = fX[(j+1)%fNvert];
         y4 = fY[(j+1)%fNvert];
         if (TGeoShape::IsSegCrossing(x1,y1,x2,y2,x3,y3,x4,y4)) {
            Error("IsIllegalCheck", "Illegal crossing of segment %d vs. segment %d", i,j);
            is_illegal = kTRUE;
         }
      }
   }
   return is_illegal;
}

//_____________________________________________________________________________
void TGeoPolygon::OutscribedConvex()
{
// Compute indices for the outscribed convex polygon.
   fNconvex = 0;
   Int_t iseg = 0;
   Int_t ivnew;
   Bool_t conv;
   Int_t *indconv = new Int_t[fNvert];
   memset(indconv, 0, fNvert*sizeof(Int_t));
   while (iseg<fNvert) {
      if (!IsSegConvex(iseg)) {
         if (iseg+2 > fNvert) break;
         ivnew = (iseg+2)%fNvert;
         conv = kFALSE;
         // check iseg with next vertices
         while (ivnew) {
            if (IsSegConvex(iseg, ivnew)) {
               conv = kTRUE;
               break;
            }
            ivnew = (ivnew+1)%fNvert;
         }
         if (!conv) {
//            Error("OutscribedConvex","NO convex line connection to vertex %d\n", iseg);
            iseg++;
            continue;
         }
      } else {
         ivnew = (iseg+1)%fNvert;
      }
      // segment belonging to convex outscribed poligon
      if (!fNconvex) indconv[fNconvex++] = iseg;
      else if (indconv[fNconvex-1] != iseg) indconv[fNconvex++] = iseg;
      if (iseg<fNvert-1) indconv[fNconvex++] = ivnew;
      if (ivnew<iseg) break;
      iseg = ivnew;
   }
   if (!fNconvex) {
      delete [] indconv;
      Fatal("OutscribedConvex","cannot build outscribed convex");
      return;
   }
   fIndc = new Int_t[fNvert];
   memcpy(fIndc, indconv, fNconvex*sizeof(Int_t)); // does not contain real indices yet
   delete [] indconv;
}

//_____________________________________________________________________________
Double_t TGeoPolygon::Safety(const Double_t *point, Int_t &isegment) const
{
// Compute minimum distance from POINT to any segment. Returns segment index.
   Int_t i1, i2;
   Double_t p1[2], p2[3];
   Double_t lsq, ssq, dx, dy, dpx, dpy, u;
   Double_t safe=1E30;
   Int_t isegmin=0;
   for (i1=0; i1<fNvert; i1++) {
      if (TGeoShape::IsSameWithinTolerance(safe,0)) {
         isegment = isegmin;
         return 0.;
      }
      i2 = (i1+1)%fNvert;
      p1[0] = fX[i1];
      p1[1] = fY[i1];
      p2[0] = fX[i2];
      p2[1] = fY[i2];

      dx = p2[0] - p1[0];
      dy = p2[1] - p1[1];
      dpx = point[0] - p1[0];
      dpy = point[1] - p1[1];

      lsq = dx*dx + dy*dy;
      if (TGeoShape::IsSameWithinTolerance(lsq,0)) {
         ssq = dpx*dpx + dpy*dpy;
         if (ssq < safe) {
            safe = ssq;
            isegmin = i1;
         }
         continue;
      }
      u = (dpx*dx + dpy*dy)/lsq;
      if (u>1) {
         dpx = point[0]-p2[0];
         dpy = point[1]-p2[1];
      } else {
         if (u>=0) {
            dpx -= u*dx;
            dpy -= u*dy;
         }
      }
      ssq = dpx*dpx + dpy*dpy;
      if (ssq < safe) {
         safe = ssq;
         isegmin = i1;
      }
   }
   isegment = isegmin;
   safe = TMath::Sqrt(safe);
//   printf("== segment %d: (%f, %f) - (%f, %f) safe=%f\n", isegment, fX[isegment],fY[isegment],fX[(isegment+1)%fNvert],fY[(isegment+1)%fNvert],safe);
   return safe;
}

//_____________________________________________________________________________
void TGeoPolygon::SetNextIndex(Int_t index)
{
// Sets the next polygone index. If index<0 sets all indices consecutive
// in increasing order.
   Int_t i;
   if (index <0) {
      for (i=0; i<fNvert; i++) fInd[i] = i;
      return;
   }
   if (fNconvex >= fNvert) {
      Error("SetNextIndex", "all indices already set");
      return;
   }
   fInd[fNconvex++] = index;
   if (fNconvex == fNvert) {
      if (!fX || !fY) return;
      Double_t area = 0.0;
      for (i=0; i<fNvert; i++) area += fX[fInd[i]]*fY[fInd[(i+1)%fNvert]]-fX[fInd[(i+1)%fNvert]]*fY[fInd[i]];
      if (area<0) TObject::SetBit(kGeoACW, kFALSE);
      else        TObject::SetBit(kGeoACW, kTRUE);
   }
}

//_____________________________________________________________________________
void TGeoPolygon::SetXY(Double_t *x, Double_t *y)
{
// Set X/Y array pointer for the polygon and daughters.
   Int_t i;
   fX = x;
   fY = y;
   Double_t area = 0.0;
   for (i=0; i<fNvert; i++) area += fX[fInd[i]]*fY[fInd[(i+1)%fNvert]]-fX[fInd[(i+1)%fNvert]]*fY[fInd[i]];
   if (area<0) TObject::SetBit(kGeoACW, kFALSE);
   else        TObject::SetBit(kGeoACW, kTRUE);

   if (!fDaughters) return;
   TGeoPolygon *poly;
   Int_t nd = fDaughters->GetEntriesFast();
   for (i=0; i<nd; i++) {
      poly = (TGeoPolygon*)fDaughters->At(i);
      if (poly) poly->SetXY(x,y);
   }
}
