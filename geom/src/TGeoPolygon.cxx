// @(#)root/geom:$Name:  $:$Id: TGeoPolygon.cxx,v 1.1 2004/01/20 15:43:30 brun Exp $
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
// To check if a point in XY plane is contained by a polygon, this is splitted
// into an outscribed convex polygon and the remaining polygons of its subtracton
// from the outscribed one. A point is INSIDE if it is 
// contained by the outscribed polygon but NOT by the remaining ones. Since these
// can also be arbitrary polygons at their turn, a tree structure is formed:
//
//  P = Pconvex - (Pconvex-P)           where (-) means 'subtraction'
//  Pconvex-P = P1 + P2 + ...           where (+) means 'union'
//
//  *Note that P1, P2, ... do not intersect each other and they are defined
//   by subsets of the list of vertices of P. They can be splitted in the same
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
   printf("=== Polygon with %i vertices\n", fNvert);
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
Bool_t TGeoPolygon::Contains(Double_t *point) const
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
void TGeoPolygon::FinishPolygon()
{
   TObject::SetBit(kGeoFinishPolygon);
   // check convexity
   ConvexCheck();
   // find outscribed convex polygon indices
   OutscribedConvex();
   if (IsConvex()) {
      printf(" -> polygon convex -> same indices\n");
      memcpy(fIndc, fInd, fNvert*sizeof(Int_t));
      return;
   }   
   printf(" -> polygon NOT convex\n");
   printf("Convex indices:\n");
   for (Int_t i=0; i<fNconvex; i++) printf(" %i ",fInd[fIndc[i]]);
   printf("\n");
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
      printf(" making daughter with %i vertices\n", nskip+1);
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
Bool_t TGeoPolygon::IsRightSided(Double_t *point, Int_t ind1, Int_t ind2) const
{
// Check if POINT is right-sided with respect to the segment defined by IND1 and IND2.
   Double_t dot = (point[0]-fX[ind1])*(fY[ind2]-fY[ind1]) -
                  (point[1]-fY[ind1])*(fX[ind2]-fX[ind1]);
   if (dot<0) return kFALSE;
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
      Fatal("OutscribedConvex","cannot build outscribed convex");
      return;
   }
   fIndc = new Int_t[fNconvex];
   memcpy(fIndc, indconv, fNconvex*sizeof(Int_t)); // does not contain real indices yet
   delete [] indconv;
}

//_____________________________________________________________________________
Double_t TGeoPolygon::Safety(Double_t * /*point*/, Int_t & /*isegment*/) const
{
// Compute minimum distance from POINT to any segment. Returns segment index.
   Warning("Safety", "not yet implemented");
   return 0.;
}

//_____________________________________________________________________________
void TGeoPolygon::SetNextIndex(Int_t index)
{
// Sets the next polygone index. If index<0 sets all indices consecutive
// in increasing order.
   if (index <0) {
      for (Int_t i=0; i<fNvert; i++) fInd[i] = i;
      return;
   }
   if (fNconvex >= fNvert) {
      Error("SetNextIndex", "all indices already set");
      return;
   }
   fInd[fNconvex++] = index;  
   printf(" %i ", index);
   if (fNconvex == fNvert) printf ("\n");   
}




   
