// @(#)root/geom:$Name:  $:$Id:$
// Author: Andrei Gheata   28/07/03

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_____________________________________________________________________________
// TGeoTorus - Torus segment class. A torus has 5 parameters :
//            R    - axial radius
//            Rmin - inner radius
//            Rmax - outer radius 
//            Phi1 - starting phi
//            Dphi - phi extent
//
//_____________________________________________________________________________

//Begin_Html
/*
<img src="gif/t_ctorus.gif">
*/
//End_Html

#include "TROOT.h"

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoTube.h"
#include "TVirtualGeoPainter.h"
#include "TGeoTorus.h"

ClassImp(TGeoTorus)

//_____________________________________________________________________________
TGeoTorus::TGeoTorus()
{
// Default constructor
   SetBit(TGeoShape::kGeoTorus);
   fR    = 0.0;
   fRmin = 0.0;
   fRmax = 0.0;
   fPhi1 = 0.0;
   fDphi = 0.0;
}   

//_____________________________________________________________________________
TGeoTorus::TGeoTorus(Double_t r, Double_t rmin, Double_t rmax, Double_t phi1, Double_t dphi)
          :TGeoBBox(0, 0, 0)
{
// Constructor without name.
   SetBit(TGeoShape::kGeoTorus);
   SetTorusDimensions(r, rmin, rmax, phi1, dphi);
   if ((fRmin<0) || (fRmax<0)) 
      SetBit(kGeoRunTimeShape);
   ComputeBBox();
}

//_____________________________________________________________________________
TGeoTorus::TGeoTorus(const char *name, Double_t r, Double_t rmin, Double_t rmax, Double_t phi1, Double_t dphi)
          :TGeoBBox(name, 0, 0, 0)
{
// Constructor with name.
   SetBit(TGeoShape::kGeoTorus);
   SetTorusDimensions(r, rmin, rmax, phi1, dphi);
   if ((fRmin<0) || (fRmax<0)) 
      SetBit(kGeoRunTimeShape);
   ComputeBBox();
}

//_____________________________________________________________________________
TGeoTorus::TGeoTorus(Double_t *param)
          :TGeoBBox(0, 0, 0)
{
// Constructor based on an array of parameters.
// param[0] = R
// param[1] = Rmin
// param[2] = Rmax
// param[3] = Phi1
// param[4] = Dphi
   SetBit(TGeoShape::kGeoTorus);
   SetDimensions(param);
   if (fRmin<0 || fRmax<0) SetBit(kGeoRunTimeShape);
   ComputeBBox();
}

//_____________________________________________________________________________
void TGeoTorus::ComputeBBox()
{
// Compute bounding box of the torus.
   fDZ = fRmax;
   if (fDphi == 360.) {
      fDX = fDY = fR+fRmax;
      return;
   }
   Double_t xc[4];
   Double_t yc[4];
   xc[0] = (fR+fRmax)*TMath::Cos(fPhi1*kDegRad);
   yc[0] = (fR+fRmax)*TMath::Sin(fPhi1*kDegRad);
   xc[1] = (fR+fRmax)*TMath::Cos((fPhi1+fDphi)*kDegRad);
   yc[1] = (fR+fRmax)*TMath::Sin((fPhi1+fDphi)*kDegRad);
   xc[2] = (fR-fRmax)*TMath::Cos(fPhi1*kDegRad);
   yc[2] = (fR-fRmax)*TMath::Sin(fPhi1*kDegRad);
   xc[3] = (fR-fRmax)*TMath::Cos((fPhi1+fDphi)*kDegRad);
   yc[3] = (fR-fRmax)*TMath::Sin((fPhi1+fDphi)*kDegRad);
      
   Double_t xmin = xc[TMath::LocMin(4, &xc[0])];
   Double_t xmax = xc[TMath::LocMax(4, &xc[0])]; 
   Double_t ymin = yc[TMath::LocMin(4, &yc[0])]; 
   Double_t ymax = yc[TMath::LocMax(4, &yc[0])];
   Double_t ddp = -fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp<=fDphi) xmax = fR+fRmax;
   ddp = 90-fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=fDphi) ymax = fR+fRmax;
   ddp = 180-fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=fDphi) xmin = -(fR+fRmax);
   ddp = 270-fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=fDphi) ymin = -(fR+fRmax);
   fOrigin[0] = (xmax+xmin)/2;
   fOrigin[1] = (ymax+ymin)/2;
   fOrigin[2] = 0;
   fDX = (xmax-xmin)/2;
   fDY = (ymax-ymin)/2;
}

//-----------------------------------------------------------------------------   
void TGeoTorus::ComputeNormal(Double_t * /*point*/, Double_t * /*dir*/, Double_t * /*norm*/)
{
// Compute normal to closest surface from POINT. 
}

//_____________________________________________________________________________
Bool_t TGeoTorus::Contains(Double_t *point) const
{
// Test if point is inside the torus.
   // check phi range
   if (fDphi!=360) {
      Double_t phi = TMath::ATan2(point[1], point[0]) * kRadDeg;
      if (phi < 0) phi+=360.0;
      Double_t ddp = phi-fPhi1;
      if (ddp<0) ddp+=360.;
      if (ddp>fDphi) return kFALSE;
   }
   //check radius
   Double_t rxy = TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t radsq = (rxy-fR)*(rxy-fR) + point[2]*point[2];
   if (radsq<fRmin*fRmin) return kFALSE;
   if (radsq>fRmax*fRmax) return kFALSE;
   return kTRUE;
}   

//_____________________________________________________________________________
Int_t TGeoTorus::DistancetoPrimitive(Int_t px, Int_t py)
{
// Compute closest distance from point px,py to each vertex.
   Int_t n = gGeoManager->GetNsegments()+1;
   Int_t numPoints = n*(n-1);
   if (fRmin>0) numPoints *= 2;
   else if (fDphi<360) numPoints += 2;
   return ShapeDistancetoPrimitive(numPoints, px, py);
}

//_____________________________________________________________________________
Double_t TGeoTorus::Daxis(Double_t *pt, Double_t *dir, Double_t t) const
{
// Computes distance to axis of the torus from point pt + t*dir;
   Double_t p[3];
   for (Int_t i=0; i<3; i++) p[i] = pt[i]+t*dir[i];
   Double_t rxy = TMath::Sqrt(p[0]*p[0]+p[1]*p[1]);
   return TMath::Sqrt((rxy-fR)*(rxy-fR)+p[2]*p[2]);
}   

//_____________________________________________________________________________
Double_t TGeoTorus::DDaxis(Double_t *pt, Double_t *dir, Double_t t) const
{
// Computes derivative w.r.t. t of the distance to axis of the torus from point pt + t*dir;
   Double_t p[3];
   for (Int_t i=0; i<3; i++) p[i] = pt[i]+t*dir[i];
   Double_t rxy = TMath::Sqrt(p[0]*p[0]+p[1]*p[1]);
   if (rxy<1E-4) return ((p[2]*dir[2]-fR*TMath::Sqrt(dir[0]*dir[0]+dir[1]*dir[1]))/TMath::Sqrt(fR*fR+p[2]*p[2]));
   Double_t d = TMath::Sqrt((rxy-fR)*(rxy-fR)+p[2]*p[2]);
   if (d==0) return 0.;
   Double_t dd = (p[0]*dir[0]+p[1]*dir[1]+p[2]*dir[2] - (p[0]*dir[0]+p[1]*dir[1])*fR/rxy)/d;
   return dd;
}   

//_____________________________________________________________________________
Double_t TGeoTorus::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// Compute distance from inside point to surface of the torus.
   if (iact<3 && *safe) {
      *safe = Safety(point, kTRUE);
      if (iact==0) return kBig;
      if ((iact==1) && (step<=*safe)) return kBig;
   }
   Double_t daxis, rxy2, dd;
   Bool_t hasphi = (fDphi<360)?kTRUE:kFALSE;
   Bool_t hasrmin = (fRmin>0)?kTRUE:kFALSE;
   Double_t c1,s1,c2,s2,cm,sm,cdfi;
   Double_t pt[3];
   Double_t invdir[3];
   Double_t din = kBig;
   Double_t dout = kBig;
   Int_t i;
   for (i=0; i<3; i++) invdir[i] = -dir[i];
   if (hasphi) {
      // Torus segment case.
      Double_t phi1=fPhi1*kDegRad;
      Double_t phi2=(fPhi1+fDphi)*kDegRad;
      c1=TMath::Cos(phi1);
      s1=TMath::Sin(phi1);
      c2=TMath::Cos(phi2);
      s2=TMath::Sin(phi2);
      Double_t fio=0.5*(phi1+phi2);
      cm=TMath::Cos(fio);
      sm=TMath::Sin(fio);
      cdfi=TMath::Cos(0.5*(phi2-phi1));
      while (hasrmin) {
         // We are in between the two torus surfaces, within phi range.
         // Check if we cross the inner bounding ring
         rxy2 = point[0]*point[0]+point[1]*point[1];
         if (TMath::Abs(point[2])>fRmin || rxy2<(fR-fRmin)*(fR-fRmin) || rxy2>(fR+fRmin)*(fR+fRmin)) {
         // we are outside the inner bounding ring
            din = TGeoTubeSeg::DistToInS(point,dir,fR-fRmin,fR+fRmin, fRmin, c1,s1,c2,s2,cm,sm,cdfi);
            if (din>1E10) break;
            for (i=0; i<3; i++) pt[i] = point[i]+(din-1E-6)*dir[i];
            dd = ToBoundary(pt, dir, fRmin);
            if (dd<1E10) {
               din += dd - 1E-6;
               break;
            }
            // propagate inside the bounding ring
            din += 1E-6;
            for (i=0; i<3; i++) pt[i] = point[i]+din*dir[i];
         } else {
            din = 0;
            dd = ToBoundary(point, dir, fRmin);
            if (dd<1E10) {
               din += dd;
               break;
            }   
            memcpy(pt, point, 3*sizeof(Double_t));
         }
         // propagate to exit of bounding ring
         dd = TGeoTubeSeg::DistToOutS(point,dir,fR-fRmin,fR+fRmin, fRmin, c1,s1,c2,s2,cm,sm);
         dd += 1E-6;
         for (i=0; i<3; i++) pt[i] += dd*dir[i];
         // we have exited the ring again
         din += dd;
         dd = TGeoTubeSeg::DistToInS(point,dir,fR-fRmin,fR+fRmin, fRmin, c1,s1,c2,s2,cm,sm,cdfi);
         if (dd>1E10) {
            din = kBig;
            break;
         }   
         dd -= 1E-6;
         din += dd;
         for (i=0; i<3; i++) pt[i] += dd*dir[i];
         dd = ToBoundary(pt, dir, fRmin);
         if (dd<1E10) {
            din += dd;
            break;
         }   
         din = kBig;
         break;
      }   
      // Compute the distance to exiting outer torus.
      dout = TGeoTubeSeg::DistToOutS(point,dir,fR-fRmax,fR+fRmax, fRmax, c1,s1,c2,s2,cm,sm); 
      for (i=0; i<3; i++) pt[i] = point[i]+dout*dir[i];
      daxis = Daxis(pt,dir,0);
      if (daxis-fRmax>0) {
         // We have crossed the outer torus on the way
         // -> compute distance from new point back
         dout -= ToBoundary(pt, invdir, fRmax);
      } else {
         // We have just crossed an endcap -> do nothing
      }         
      return TMath::Min(din,dout);
   }            
   // Case with full phi range
   if (hasrmin) {
      // We are in between the two torus surfaces.
      // Check if we cross the inner bounding ring
      rxy2 = point[0]*point[0]+point[1]*point[1];
      if (TMath::Abs(point[2])>fRmin || rxy2<(fR-fRmin)*(fR-fRmin) || rxy2>(fR+fRmin)*(fR+fRmin)) {
         din = TGeoTube::DistToInS(point,dir,fR-fRmin,fR+fRmin, fRmin);
         daxis = Daxis(pt,dir,0);
         if (din<1E10 && daxis>=fRmin) {
            for (i=0; i<3; i++) pt[i] = point[i]+din*dir[i];
            din += ToBoundary(pt,dir,fRmin);
         }
      } else {
         din = ToBoundary(point,dir,fRmin);
      }
   }
   dout = TGeoTube::DistToOutS(point,dir,fR-fRmax,fR+fRmax, fRmax); 
   for (i=0; i<3; i++) pt[i] = point[i]+dout*dir[i];
   dout -= ToBoundary(pt, invdir, fRmax);
   return TMath::Min(din,dout);
}

//_____________________________________________________________________________
Double_t TGeoTorus::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// Compute distance from outside point to surface of the torus.
   if (iact<3 && *safe) {
      *safe = Safety(point, kFALSE);
      if (iact==0) return kBig;
      if ((iact==1) && (step<=*safe)) return kBig;
   }
   Double_t daxis;
   Bool_t hasphi = (fDphi<360)?kTRUE:kFALSE;
   Bool_t hasrmin = (fRmin>0)?kTRUE:kFALSE;
   Double_t c1,s1,c2,s2,cm,sm,cdfi;
   Double_t rxy2,dd;
   Double_t snext;
   Double_t pt[3];
   Double_t invdir[3];
   Int_t i;
   for (i=0; i<3; i++) invdir[i] = -dir[i];
   if (hasphi) {
      // Torus segment case.
      Bool_t inphi = kFALSE;
      Double_t phi=TMath::ATan2(point[1], point[0])*kRadDeg;;
      if (phi<0) phi+=360;
      Double_t ddp = phi-fPhi1;
      if (ddp<0) ddp+=360;;
      if (ddp<=fDphi) inphi=kTRUE;
      Double_t phi1=fPhi1*kDegRad;
      Double_t phi2=(fPhi1+fDphi)*kDegRad;
      c1=TMath::Cos(phi1);
      s1=TMath::Sin(phi1);
      c2=TMath::Cos(phi2);
      s2=TMath::Sin(phi2);
      Double_t fio=0.5*(phi1+phi2);
      cm=TMath::Cos(fio);
      sm=TMath::Sin(fio);
      cdfi=TMath::Cos(0.5*(phi2-phi1));
      
      // check if we are inside the hole of the torus
      if (hasrmin) {
         daxis = Daxis(point, dir, 0);
         if (inphi) {
            if (daxis<fRmax) {
               // We are inside the outer torus, in the phi range.
               if (daxis>=fRmin) {
                  Warning("DistToIn", "point is actually inside - returning 0");
                  return 0.;
               }   
               // We are inside the hole -> we will maybe cross the inner surface 
               // of the torus. Propagate until exiting the bounding ring of inner torus.
               snext = TGeoTubeSeg::DistToOutS(point,dir,fR-fRmin,fR+fRmin, fRmin, c1,s1,c2,s2,cm,sm);
               for (i=0; i<3; i++) pt[i] = point[i]+snext*dir[i];
               daxis = Daxis(pt,dir,0);
               if (daxis-fRmin>0) {
                  // We have crossed the inner torus on the way
                  // -> compute distance from new point back
                  snext -= ToBoundary(pt, invdir, fRmin);
                  return snext;
               } else {
                  // We have just crossed an endcap -> check if we cross the torus from
                  // the new point.
                  for (i=0; i<3; i++) pt[i] += 1E-6*dir[i];
                  snext += DistToIn(pt, dir, 3) + 1E-6;
                  return snext;
               }
            } else {
               // We are outside Rmax within phi range and we can only cross the 
               // outer surface;
               rxy2 = point[0]*point[0]+point[1]*point[1];
               if (TMath::Abs(point[2])>fRmax || rxy2<(fR-fRmax)*(fR-fRmax) || rxy2>(fR+fRmax)*(fR+fRmax)) {
                  // we are outside the bounding ring
                  snext = TGeoTubeSeg::DistToInS(point,dir,fR-fRmax,fR+fRmax, fRmax, c1,s1,c2,s2,cm,sm,cdfi);
                  if (snext>1E10) return kBig;
                  for (i=0; i<3; i++) pt[i] = point[i]+(snext-1E-6)*dir[i];
                  dd = ToBoundary(pt, dir, fRmax);
                  if (dd<1E10) {
                     snext += dd - 1E-6;
                     return snext;
                  }
                  // propagate inside the bounding ring
                  snext += 1E-6;
                  for (i=0; i<3; i++) pt[i] = point[i]+snext*dir[i];
               } else {
                  snext = 0;
                  dd = ToBoundary(point, dir, fRmax);
                  if (dd<1E10) {
                     snext += dd;
                     return snext;
                  }   
                  memcpy(pt, point, 3*sizeof(Double_t));
               }
               // propagate to exit of bounding ring
               dd = TGeoTubeSeg::DistToOutS(point,dir,fR-fRmax,fR+fRmax, fRmax, c1,s1,c2,s2,cm,sm);
               dd += 1E-6;
               for (i=0; i<3; i++) pt[i] += dd*dir[i];
               // we have exited the ring again
               snext += dd;
               dd = TGeoTubeSeg::DistToInS(point,dir,fR-fRmax,fR+fRmax, fRmax, c1,s1,c2,s2,cm,sm,cdfi);
               if (dd>1E10) return kBig;
               dd -= 1E-6;
               snext += dd;
               for (i=0; i<3; i++) pt[i] += dd*dir[i];
               dd = ToBoundary(pt, dir, fRmax);
               if (dd<1E10) {
                  snext += dd;
                  return snext;
               }   
               return kBig;
            }
         } else {
            // We are in the empty space between the 2 caps. We might cross the caps
            // the outer torus or enter trough the hole and cross the inner one. 
            // -> check crossing with bounding ring of outer torus
            snext = TGeoTubeSeg::DistToInS(point,dir,fR-fRmax,fR+fRmax, fRmax,c1,s1,c2,s2,cm,sm,cdfi);
            if (snext>1E10) return kBig;
            snext -= 1E-6;
            for (i=0; i<3; i++) pt[i] = point[i]+snext*dir[i];
            daxis = Daxis(pt,dir,0);
            if (daxis<=fRmax) {
               // we have entered an endcap
               if (daxis>=fRmin) return (snext+1E-6);
               // we are entering a hole
               snext += 2E-6;
               for (i=0; i<3; i++) pt[i] += 2E-6*dir[i];
               // we are now inside the inner part
               dd = TGeoTubeSeg::DistToOutS(pt,dir,fR-fRmin,fR+fRmin, fRmin, c1,s1,c2,s2,cm,sm);
               if (dd>1E10) {
                  Error("DistToIn", "we entered the hole and cannot exit");
                  return kBig;
               }   
               for (i=0; i<3; i++) pt[i] += dd*dir[i];
               snext += dd;
               daxis = Daxis(pt,dir,0);
               if (daxis-fRmin>0) {
                  // We have crossed the inner torus on the way
                  // -> compute distance from new point back
                  snext -= ToBoundary(pt, invdir, fRmin);
                  return snext;
               } 
               // we have crossed the hole not touching anything
               return kBig;
            } else {
               dd = ToBoundary(pt, dir, fRmax);
               if (dd<1E10) return (snext+dd);
               snext += 2E-6;
               // make sure pt is inside bounding ring
               for (i=0; i<3; i++) pt[i] = point[i]+snext*dir[i];
               dd = TGeoTubeSeg::DistToOutS(pt,dir,fR-fRmax,fR+fRmax, fRmax,c1,s1,c2,s2,cm,sm) + 1E-6;
               snext += dd;
               for (i=0; i<3; i++) pt[i] += dd*dir[i];
               // --> propagate again 
               dd = TGeoTubeSeg::DistToInS(pt,dir,fR-fRmax,fR+fRmax, fRmax,c1,s1,c2,s2,cm,sm,cdfi);
               if (dd>1E10) return kBig;
               dd -= 1E-6;
               snext += dd;
               for (i=0; i<3; i++) pt[i] += dd*dir[i];
               dd = ToBoundary(pt, dir, fRmax);
               if (dd>1E10) return kBig;
               snext += dd;
               return snext;
            }   
         }
      } else {
         // No inner radius -> check the bounding ring
         rxy2 = point[0]*point[0]+point[1]*point[1];
         if (TMath::Abs(point[2])>fRmax || rxy2<(fR-fRmax)*(fR-fRmax) || rxy2>(fR+fRmax)*(fR+fRmax)) {
            // we are outside the bounding ring
            snext = TGeoTubeSeg::DistToInS(point,dir,fR-fRmax,fR+fRmax, fRmax, c1,s1,c2,s2,cm,sm,cdfi);
            if (snext>1E10) return kBig;
            for (i=0; i<3; i++) pt[i] = point[i]+(snext-1E-6)*dir[i];
            snext += ToBoundary(pt, dir, fRmax) - 1E-6;
            return snext;
         } else {
            snext = ToBoundary(point, dir, fRmax);
            return snext;
         }      
      }
   }     
   // Full torus case.
   if (hasrmin) {
      // Check if we are in the hole.            
      daxis = Daxis(point, dir, 0);
      if (daxis<fRmax) {
         if (daxis>=fRmin) {
            Warning("DistToIn", "point is actually inside - returning 0");
            return 0.;
         }   
         // Point is in the hole -> we will cross for sure the inner torus.
         snext = TGeoTube::DistToOutS(point,dir,fR-fRmin,fR+fRmin, fRmin);
         for (i=0; i<3; i++) pt[i] = point[i]+snext*dir[i];
         // We have crossed the inner torus on the way
         // -> compute distance from new point back
         snext -= ToBoundary(pt, invdir, fRmin);
         return snext;         
      }
   }
   // We are fully outside a complete torus -> propagate to bounding ring.
   rxy2 = point[0]*point[0]+point[1]*point[1];
   if (TMath::Abs(point[2])>fRmax || rxy2<(fR-fRmax)*(fR-fRmax) || rxy2>(fR+fRmax)*(fR+fRmax)) {
      // we are outside the bounding ring
      snext = TGeoTube::DistToInS(point,dir,fR-fRmax,fR+fRmax, fRmax);
      if (snext>1E10) return kBig;
      snext -= 1E-6;
      for (i=0; i<3; i++) pt[i] = point[i]+snext*dir[i];
      dd = ToBoundary(pt, dir, fRmax);
      if (dd<1E10) return (snext+dd);
      snext += 2E-6;
      // make sure pt is inside bounding ring
      for (i=0; i<3; i++) pt[i] = point[i]+snext*dir[i];
   } else {
      snext = ToBoundary(pt,dir,fRmax);
      if (snext<1E10) return snext;
      snext = 0;
      memcpy(pt, point, 3*sizeof(Double_t));
   }
   // We are now in the bounding ring, but not in the torus.
   // --> propagate to the exit of the bounding ring
   dd = TGeoTube::DistToOutS(pt,dir,fR-fRmax,fR+fRmax, fRmax) + 1E-6;
   snext += dd;
   for (i=0; i<3; i++) pt[i] += dd*dir[i];
   // --> propagate again 
   dd = TGeoTube::DistToInS(pt,dir,fR-fRmax,fR+fRmax, fRmax);
   snext += dd;
   if (dd>1E10) return kBig;
   for (i=0; i<3; i++) pt[i] += (dd-1E-6)*dir[i];
   dd = ToBoundary(pt, dir, fRmax)-1E-6;
   if (dd>1E10) return kBig;
   snext += dd;
   return snext;
}      

//_____________________________________________________________________________
TGeoVolume *TGeoTorus::Divide(TGeoVolume * /*voldiv*/, const char * /*divname*/, Int_t /*iaxis*/, Int_t /*ndiv*/, 
                              Double_t /*start*/, Double_t /*step*/) 
{
//--- Divide this torus shape belonging to volume "voldiv" into ndiv volumes
// called divname, from start position with the given step. 
   return 0;
}

//_____________________________________________________________________________
const char *TGeoTorus::GetAxisName(Int_t iaxis) const
{
// Returns name of axis IAXIS.
   switch (iaxis) {
      case 1:
         return "R";
      case 2:
         return "PHI";
      case 3:
         return "Z";
      default:
         return "UNDEFINED";
   }
}   
   
//_____________________________________________________________________________
Double_t TGeoTorus::GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const
{
// Get range of shape for a given axis.
   xlo = 0;
   xhi = 0;
   Double_t dx = 0;
   switch (iaxis) {
      case 1:
         xlo = fRmin;
         xhi = fRmax;
         dx = xhi-xlo;
         return dx;
      case 2:
         xlo = fPhi1;
         xhi = fPhi1+fDphi;
         dx = fDphi;
         return dx;
      case 3:
         dx = 0;
         return dx;
   }
   return dx;
}         
   
//_____________________________________________________________________________
void TGeoTorus::GetBoundingCylinder(Double_t *param) const
{
//--- Fill vector param[4] with the bounding cylinder parameters. The order
// is the following : Rmin, Rmax, Phi1, Phi2, dZ
   param[0] = (fR-fRmax); // Rmin
   param[1] = (fR+fRmax); // Rmax
   param[2] = fPhi1;    // Phi1
   param[3] = fPhi1+fDphi;  // Phi2
}   
 
//_____________________________________________________________________________
TGeoShape *TGeoTorus::GetMakeRuntimeShape(TGeoShape * /*mother*/, TGeoMatrix * /*mat*/) const
{
   if (!TestBit(kGeoRunTimeShape)) return 0;
   Error("GetMakeRuntimeShape", "parametrized toruses not supported");
   return 0;
}
      
//_____________________________________________________________________________
void TGeoTorus::InspectShape() const
{
// print shape parameters
   printf("*** TGeoTorus parameters ***\n");
   printf("    R    = %11.5f\n", fR);
   printf("    Rmin = %11.5f\n", fRmin);
   printf("    Rmax = %11.5f\n", fRmax);
   printf("    Phi1 = %11.5f\n", fPhi1);
   printf("    Dphi = %11.5f\n", fDphi);
   TGeoBBox::InspectShape();
}

//_____________________________________________________________________________
void *TGeoTorus::Make3DBuffer(const TGeoVolume *vol) const
{
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return 0;
   return painter->MakeTorus3DBuffer(vol);
}

//_____________________________________________________________________________
void TGeoTorus::Paint(Option_t *option)
{
// paint this shape according to option
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   TGeoVolume *vol = gGeoManager->GetCurrentVolume();
   if (vol->GetShape() != (TGeoShape*)this) return;
   painter->PaintTorus(this, option);
}

//_____________________________________________________________________________
void TGeoTorus::PaintNext(TGeoHMatrix *glmat, Option_t *option)
{
// paint this shape according to option
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   painter->PaintTorus(this, option, glmat);
}

//_____________________________________________________________________________
Double_t TGeoTorus::Safety(Double_t *point, Bool_t in) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   Double_t saf[3];
   Int_t i;
   Double_t rxy = TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t rad = TMath::Sqrt((rxy-fR)*(rxy-fR) + point[2]*point[2]);
   saf[0] = rad-fRmin;
   saf[1] = fRmax-rad;
   if (fDphi==360) {
      if (in) return TMath::Min(saf[0],saf[1]);
      for (i=0; i<2; i++) saf[i]=-saf[i];
      return TMath::Max(saf[0], saf[1]);
   }   

   Double_t phi1 = fPhi1*kDegRad;
   Double_t phi2 = (fPhi1+fDphi)*kDegRad;
   Double_t c1 = TMath::Cos(phi1);
   Double_t s1 = TMath::Sin(phi1);
   Double_t c2 = TMath::Cos(phi2);
   Double_t s2 = TMath::Sin(phi2);

   saf[2] = TGeoShape::SafetyPhi(point,in,c1,s1,c2,s2);
   if (in) return saf[TMath::LocMin(3,saf)];

   for (i=0; i<3; i++) saf[i]=-saf[i];
   return saf[TMath::LocMax(3,saf)];
}

//_____________________________________________________________________________
void TGeoTorus::SetTorusDimensions(Double_t r, Double_t rmin, Double_t rmax,
                          Double_t phi1, Double_t dphi)
{
   fR = r;
   fRmin = rmin;
   fRmax = rmax;
   fPhi1 = phi1;
   if (fPhi1<0) fPhi1+=360.;
   fDphi = dphi;
}

//_____________________________________________________________________________
void TGeoTorus::SetDimensions(Double_t *param)
{
   SetTorusDimensions(param[0], param[1], param[2], param[3], param[4]);
}

//_____________________________________________________________________________
void TGeoTorus::SetPoints(Double_t *buff) const
{
// Create torus mesh points
   if (!buff) return;
   Int_t n = gGeoManager->GetNsegments()+1;
   Double_t phin, phout;
   Double_t dpin = 360./(n-1);
   Double_t dpout = fDphi/(n-1);
   Double_t co,so,ci,si;
   Bool_t havermin = (fRmin==0)?kFALSE:kTRUE;
   Int_t i,j;
   Int_t indx = 0;
   // loop outer mesh -> n*n points [0, 3*n*n-1]
   for (i=0; i<n; i++) {
      phout = (fPhi1+i*dpout)*kDegRad;
      co = TMath::Cos(phout);
      so = TMath::Sin(phout);
      for (j=0; j<n-1; j++) {
         phin = j*dpin*kDegRad;
         ci = TMath::Cos(phin);
         si = TMath::Sin(phin);
         buff[indx++] = (fR+fRmax*ci)*co;
         buff[indx++] = (fR+fRmax*ci)*so;
         buff[indx++] = fRmax*si;
      }
   }     
    
   if (havermin) {
    // loop inner mesh -> n*n points [3*n*n, 6*n*n-1]
      for (i=0; i<n; i++) {
         phout = (fPhi1+i*dpout)*kDegRad;
         co = TMath::Cos(phout);
         so = TMath::Sin(phout);
         for (j=0; j<n-1; j++) {
            phin = j*dpin*kDegRad;
            ci = TMath::Cos(phin);
            si = TMath::Sin(phin);
            buff[indx++] = (fR+fRmin*ci)*co;
            buff[indx++] = (fR+fRmin*ci)*so;
            buff[indx++] = fRmin*si;
         }
      }  
   } else {
      if (fDphi!=360.) {
      // just add extra 2 points on the centers of the 2 phi cuts [3*n*n, 3*n*n+1]
         buff[indx++] = fR*TMath::Cos(fPhi1*kDegRad);
         buff[indx++] = fR*TMath::Sin(fPhi1*kDegRad);
         buff[indx++] = 0;
         buff[indx++] = fR*TMath::Cos((fPhi1+fDphi)*kDegRad);
         buff[indx++] = fR*TMath::Sin((fPhi1+fDphi)*kDegRad);
         buff[indx++] = 0;
      }
   }      
}        

//_____________________________________________________________________________
void TGeoTorus::SetPoints(Float_t *buff) const
{
// Create torus mesh points
   if (!buff) return;
   Int_t n = gGeoManager->GetNsegments()+1;
   Double_t phin, phout;
   Double_t dpin = 360./(n-1);
   Double_t dpout = fDphi/(n-1);
   Double_t co,so,ci,si;
   Bool_t havermin = (fRmin==0)?kFALSE:kTRUE;
   Int_t i,j;
   Int_t indx = 0;
   // loop outer mesh -> n*n points [0, n*n-1]
   // plane i = 0, n-1  point j = 0, n-1  ipoint = n*i + j
   for (i=0; i<n; i++) {
      phout = (fPhi1+i*dpout)*kDegRad;
      co = TMath::Cos(phout);
      so = TMath::Sin(phout);
      for (j=0; j<n-1; j++) {
         phin = j*dpin*kDegRad;
         ci = TMath::Cos(phin);
         si = TMath::Sin(phin);
         buff[indx++] = (fR+fRmax*ci)*co;
         buff[indx++] = (fR+fRmax*ci)*so;
         buff[indx++] = fRmax*si;
      }
   }     
    
   if (havermin) {
      // loop inner mesh -> n*n points [n*n, 2*n*n-1]
      // plane i = 0, n-1  point j = 0, n-1  ipoint = n*n + n*i + j
      for (i=0; i<n; i++) {
         phout = (fPhi1+i*dpout)*kDegRad;
         co = TMath::Cos(phout);
         so = TMath::Sin(phout);
         for (j=0; j<n-1; j++) {
            phin = j*dpin*kDegRad;
            ci = TMath::Cos(phin);
            si = TMath::Sin(phin);
            buff[indx++] = (fR+fRmin*ci)*co;
            buff[indx++] = (fR+fRmin*ci)*so;
            buff[indx++] = fRmin*si;
         }
      }  
   } else {
      if (fDphi!=360.) {
      // just add extra 2 points on the centers of the 2 phi cuts [n*n, n*n+1]
      // ip1 = n*(n-1) + 0;
      // ip2 = n*(n-1) + 1
         buff[indx++] = fR*TMath::Cos(fPhi1*kDegRad);
         buff[indx++] = fR*TMath::Sin(fPhi1*kDegRad);
         buff[indx++] = 0;
         buff[indx++] = fR*TMath::Cos((fPhi1+fDphi)*kDegRad);
         buff[indx++] = fR*TMath::Sin((fPhi1+fDphi)*kDegRad);
         buff[indx++] = 0;
      }
   }      
}        
//_____________________________________________________________________________
void TGeoTorus::Sizeof3D() const
{
// fill size of this 3-D object
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   Int_t n = gGeoManager->GetNsegments()+1;
   Int_t numPoints = n*(n-1);
   Int_t numSegs   = (2*n-1)*(n-1);
   Int_t numPolys  = (n-1)*(n-1);

   Bool_t hasrmin = (fRmin>0)?kTRUE:kFALSE;
   Bool_t hasphi  = (fDphi<360)?kTRUE:kFALSE;
   if (hasrmin) numPoints *= 2;
   else if (hasphi) numPoints += 2;
   if (hasrmin) {
      numSegs   += (2*n-1)*(n-1);
      numPolys  += (n-1)*(n-1);
   }   
   if (hasphi) {
      numSegs   += 2*(n-1);
      numPolys  += 2*(n-1);
   }   
    
   painter->AddSize3D(numPoints, numSegs, numPolys);
}

//_____________________________________________________________________________
Double_t TGeoTorus::ToBoundary(Double_t *pt, Double_t *dir, Double_t r) const
{
// Returns distance to the surface or the torus (fR,r) from a point, along
// a direction. Point is close enough to the boundary so that the distance 
// to the torus is decreasing while moving along the given direction.
   Double_t step, daxis, ddaxis, snext, epsil;
   
   ddaxis = DDaxis(pt, dir, 0); 
   if (ddaxis>=0) {
//      Error("ToBoundary", "derivative from point (%f, %f, %f) positive", pt[0],pt[1],pt[2]);
      return kBig;
   }   
   daxis = Daxis(pt, dir, 0);
   epsil = daxis - r;
   if (epsil<0) return kBig;
   snext = 0;
   Int_t istep = 0;
   while (epsil > 1E-12) {
      istep++;
      step = -epsil/ddaxis;
      snext += step;
      ddaxis = DDaxis(pt, dir, snext);
      if (ddaxis>=0) return kBig;
      daxis = Daxis(pt, dir, snext);
      epsil = daxis - r;
   }
   return snext;   
}      

      
   

