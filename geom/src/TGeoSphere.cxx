// @(#)root/geom:$Name:  $:$Id: TGeoSphere.cxx,v 1.2 2002/07/10 19:24:16 brun Exp $
// Author: Andrei Gheata   31/01/02
// TGeoSphere::Contains() DistToOut() implemented by Mihaela Gheata

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TVirtualGeoPainter.h"

#include "TGeoSphere.h"

/*************************************************************************
 * TGeoSphere - spherical shell class. It takes 6 parameters : 
 *           - inner and outer radius Rmin, Rmax
 *           - the theta limits Tmin, Tmax
 *           - the phi limits Pmin, Pmax (the sector in phi is considered
 *             starting from Pmin to Pmax counter-clockwise
 *
 *************************************************************************/
//Begin_Html
/*
<img src="gif/TGeoSphere.gif">
*/
//End_Html

ClassImp(TGeoSphere)
   
//-----------------------------------------------------------------------------
TGeoSphere::TGeoSphere()
{
// Default constructor
   SetBit(TGeoShape::kGeoSph);
   fNz = 0;
   fNseg = 0;
   fRmin = 0.0;
   fRmax = 0.0;
   fTheta1 = 0.0;
   fTheta2 = 180.0;
   fPhi1 = 0.0;
   fPhi2 = 360.0;
}   
//-----------------------------------------------------------------------------
TGeoSphere::TGeoSphere(Double_t rmin, Double_t rmax, Double_t theta1,
                       Double_t theta2, Double_t phi1, Double_t phi2)
           :TGeoBBox(0, 0, 0)
{
// Default constructor specifying minimum and maximum radius
   SetBit(TGeoShape::kGeoSph);
   SetSphDimensions(rmin, rmax, theta1, theta2, phi1, phi2);
   ComputeBBox();
   SetNumberOfDivisions(20);
}
//-----------------------------------------------------------------------------
TGeoSphere::TGeoSphere(Double_t *param, Int_t nparam)
{
// Default constructor specifying minimum and maximum radius
// param[0] = Rmin
// param[1] = Rmax
   SetBit(TGeoShape::kGeoSph);
   SetDimensions(param);
   ComputeBBox();
   SetNumberOfDivisions(20);
}
//-----------------------------------------------------------------------------
TGeoSphere::~TGeoSphere()
{
// destructor
}
//-----------------------------------------------------------------------------   
void TGeoSphere::ComputeBBox()
{
// compute bounding box of the sphere
//   Double_t xmin, xmax, ymin, ymax, zmin, zmax;
   if (TMath::Abs(fTheta2-fTheta1) == 180) {
      if (TMath::Abs(fPhi2-fPhi1) == 360) {
         TGeoBBox::SetBoxDimensions(fRmax, fRmax, fRmax);
         memset(fOrigin, 0, 3*sizeof(Double_t));
         return;
      }
   }   
   Double_t st1 = TMath::Sin(fTheta1*kDegRad);
   Double_t st2 = TMath::Sin(fTheta2*kDegRad);
   Double_t r1min, r1max, r2min, r2max, rmin, rmax;
   r1min = TMath::Min(fRmax*st1, fRmax*st2);
   r1max = TMath::Max(fRmax*st1, fRmax*st2);
   r2min = TMath::Min(fRmin*st1, fRmin*st2);
   r2max = TMath::Max(fRmin*st1, fRmin*st2);
   if (((fTheta1<=90) && (fTheta2>=90)) || ((fTheta2<=90) && (fTheta1>=90))) {
      r1max = fRmax;
      r2max = fRmin;
   }
   rmin = TMath::Min(r1min, r2min);
   rmax = TMath::Max(r1max, r2max);

   Double_t xc[4];
   Double_t yc[4];
   xc[0] = rmax*TMath::Cos(fPhi1*kDegRad);
   yc[0] = rmax*TMath::Sin(fPhi1*kDegRad);
   xc[1] = rmax*TMath::Cos(fPhi2*kDegRad);
   yc[1] = rmax*TMath::Sin(fPhi2*kDegRad);
   xc[2] = rmin*TMath::Cos(fPhi1*kDegRad);
   yc[2] = rmin*TMath::Sin(fPhi1*kDegRad);
   xc[3] = rmin*TMath::Cos(fPhi2*kDegRad);
   yc[3] = rmin*TMath::Sin(fPhi2*kDegRad);

   Double_t xmin = xc[TMath::LocMin(4, &xc[0])];
   Double_t xmax = xc[TMath::LocMax(4, &xc[0])]; 
   Double_t ymin = yc[TMath::LocMin(4, &yc[0])]; 
   Double_t ymax = yc[TMath::LocMax(4, &yc[0])];
   Double_t dp = fPhi2-fPhi1;
   if (dp<0) dp+=360;
   Double_t ddp = -fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=dp) xmax = rmax;
   ddp = 90-fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=dp) ymax = rmax;
   ddp = 180-fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=dp) xmin = -rmax;
   ddp = 270-fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=dp) ymin = -rmax;
   xc[0] = fRmax*TMath::Cos(fTheta1*kDegRad);  
   xc[1] = fRmax*TMath::Cos(fTheta2*kDegRad);  
   xc[2] = fRmin*TMath::Cos(fTheta1*kDegRad);  
   xc[3] = fRmin*TMath::Cos(fTheta2*kDegRad);  
   Double_t zmin = xc[TMath::LocMin(4, &xc[0])];
   Double_t zmax = xc[TMath::LocMax(4, &xc[0])]; 


   fOrigin[0] = (xmax+xmin)/2;
   fOrigin[1] = (ymax+ymin)/2;
   fOrigin[2] = (zmax+zmin)/2;;
   fDX = (xmax-xmin)/2;
   fDY = (ymax-ymin)/2;
   fDZ = (zmax-zmin)/2;
}   
//-----------------------------------------------------------------------------
Bool_t TGeoSphere::Contains(Double_t *point) const
{
// test if point is inside this sphere
   // check Rmin<=R<=Rmax
   Double_t r2=point[0]*point[0]+point[1]*point[1]+point[2]*point[2];
   if ((r2<fRmin*fRmin) || (r2>fRmax*fRmax)) return kFALSE;
   // check phi range
   Double_t phi = TMath::ATan2(point[1], point[0]) * kRadDeg;
   if (phi < 0 ) phi+=360.;
   Double_t dphi = fPhi2 -fPhi1;
   if (dphi < 0) dphi+=360.;
   Double_t ddp = phi - fPhi1;
   if (ddp < 0) ddp += 360.;
   if (ddp > 360.) ddp -= 360;
   if (ddp > dphi) return kFALSE;    
   r2 = point[0]*point[0]+point[1]*point[1];
   r2=TMath::Sqrt(r2);
   // check theta range
   Double_t theta = TMath::ATan2(r2, point[2])*kRadDeg;
   if (theta < 0) theta+=180.;
   if ((theta<fTheta1) || (theta>fTheta2)) return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
Double_t TGeoSphere::DistToSurf(Double_t *point, Double_t *dir) const
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   return kBig;;
}
//-----------------------------------------------------------------------------
Int_t TGeoSphere::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute closest distance from point px,py to each corner
   Int_t n = fNseg+1;
   Int_t nz = fNz+1;
   const Int_t numPoints = 2*n*nz;
   return ShapeDistancetoPrimitive(numPoints, px, py);
}
//-----------------------------------------------------------------------------
Double_t TGeoSphere::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from outside point to surface of the sphere
//   Warning("DistToIn", "BBOX");
   return TGeoBBox::DistToIn(point, dir, iact, step, safe);
}   
//-----------------------------------------------------------------------------
Double_t TGeoSphere::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from outside point to surface of the sphere
   Double_t saf[6];
   Double_t rxy2 = point[0]*point[0]+point[1]*point[1];
   Double_t rxy = TMath::Sqrt(rxy2);
   Double_t r2 = rxy2+point[2]*point[2];
   Double_t r=TMath::Sqrt(r2);
   Bool_t rzero=kFALSE;
   if (r==0) rzero=kTRUE;
   //localize theta
   Bool_t has_theta = ((fTheta2-fTheta1)>=180)?kFALSE:kTRUE;
   Double_t th=0.;
   if (has_theta && (!rzero)) {
      th = TMath::ACos(point[2]/r)*kRadDeg;
   }
   //localize phi
   Double_t phi1=fPhi1; 
   Double_t phi2=fPhi2;
   Double_t phi=0.;
   if (phi2<=phi1) phi2+=360.;
   Bool_t has_phi = ((phi2-phi1)==360.)?kFALSE:kTRUE;
   if (has_phi) {
      phi=TMath::ATan2(point[1], point[0])*kRadDeg;
      if (phi<phi1) phi+=360.;
   }   
   if (iact<3 && safe) {
      saf[0]=(fRmin==0)?kBig:r-fRmin;
      saf[1]=fRmax-r;
      if (!has_theta) {
         saf[2]=saf[3]=kBig;
      } else {
         saf[2] = r*TMath::Sin((th-fTheta1)*kDegRad);
         if (saf[2]<0) saf[2]=kBig;
         saf[3] = r*TMath::Sin((fTheta2-th)*kDegRad);
         if (saf[3]<0) saf[3]=kBig;
      }
      if (!has_phi) {
         saf[4]=saf[5]=kBig;
      } else {
         Double_t dph1=phi-phi1;
         if (dph1>90.) saf[4]=kBig;            
         else saf[4]=rxy*TMath::Sin(dph1*kDegRad);
         Double_t dph2=phi2-phi;
         if (dph2>90.) saf[5]=kBig;            
         else saf[5]=rxy*TMath::Sin(dph2*kDegRad);
      }
      *safe = saf[TMath::LocMin(6, &saf[0])];
      if (iact==0) return kBig;
      if (iact==1) {
         if (step < *safe) return step;
      }   
   }
   // compute distance to shape
   Double_t *norm=gGeoManager->GetNormalChecked();
   if (rzero) {
      memcpy(norm, dir, 3*sizeof(Double_t));
      return fRmax;
   }
   // first do rmin
   Double_t s=kBig;
   Double_t b=point[0]*dir[0]+point[1]*dir[1]+point[2]*dir[2];
   Double_t c = 0;
   Double_t d=0;
   Double_t xi, yi, zi;
   
   if (fRmin>0) {
      c = r2-fRmin*fRmin;
      d=b*b-c;
      if (d>=0) {
         s=-b-TMath::Sqrt(d);
         if (s>=0) {
            xi=point[0]+s*dir[0];
            yi=point[1]+s*dir[1];
            zi=point[2]+s*dir[2];
            // check theta
            Double_t thi = TMath::ACos(zi/fRmin)*kRadDeg;
            if ((thi>=fTheta1) && (thi<=fTheta2)) {
               // check phi
               Double_t phii=TMath::ATan2(yi,xi)*kRadDeg;
               if (phii<phi1) phii+=360.;
               if ((phii-phi1)<=(phi2-phi1)) {
                  norm[0]=-xi/fRmin;
                  norm[1]=-yi/fRmin;
                  norm[2]=-zi/fRmin;
                  return s;
               }
            }
         }
      }            
   }
   // now do rmax
   c = r2-fRmax*fRmax;
   d=b*b-c;
   if (d>=0) {
      s=-b+TMath::Sqrt(d);
      if (s>=0) {
         xi=point[0]+s*dir[0];
         yi=point[1]+s*dir[1];
         zi=point[2]+s*dir[2];
         // check theta
         Double_t thi = TMath::ACos(zi/fRmax)*kRadDeg;
         if ((thi>=fTheta1) && (thi<=fTheta2)) {
            // check phi
            Double_t phii=TMath::ATan2(yi,xi)*kRadDeg;
            if (phii<phi1) phii+=360.;
            if ((phii-phi1)<=(phi2-phi1)) {
               norm[0]=xi/fRmax;
               norm[1]=yi/fRmax;
               norm[2]=zi/fRmax;
               return s;
            }
         }
      }
   }      
   // check lower theta conical surface
   return kBig;            
}   
//-----------------------------------------------------------------------------
TGeoVolume *TGeoSphere::Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Double_t step) 
{
// Divide all range of iaxis in range/step cells 
   Error("Divide", "Division in all range not implemented");
   return voldiv;
}      
//-----------------------------------------------------------------------------
void TGeoSphere::InspectShape() const
{
// print shape parameters
   printf("*** TGeoSphere parameters ***\n");
   printf("    Rmin = %11.5f\n", fRmin);
   printf("    Rmax = %11.5f\n", fRmax);
   printf("    Th1  = %11.5f\n", fTheta1);
   printf("    Th2  = %11.5f\n", fTheta2);
   printf("    Ph1  = %11.5f\n", fPhi1);
   printf("    Ph2  = %11.5f\n", fPhi2);
   TGeoBBox::InspectShape();
}
//-----------------------------------------------------------------------------
void TGeoSphere::Paint(Option_t *option)
{
// paint this shape according to option
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   TGeoVolume *vol = gGeoManager->GetCurrentVolume();
   if (vol->GetShape() != (TGeoShape*)this) return;
   painter->PaintSphere(vol, option);
}
//-----------------------------------------------------------------------------
void TGeoSphere::NextCrossing(TGeoParamCurve *c, Double_t *point) const
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoSphere::Safety(Double_t *point, Double_t *spoint, Option_t *option) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return kBig;
}
//-----------------------------------------------------------------------------
void TGeoSphere::SetSphDimensions(Double_t rmin, Double_t rmax, Double_t theta1,
                               Double_t theta2, Double_t phi1, Double_t phi2)
{
   if (rmin >= rmax) {
      Error("SetDimensions", "invalid parameters rmin/rmax");
      return;
   }
   fRmin = rmin;
   fRmax = rmax;
   if (theta1 >= theta2) {
      Error("SetDimensions", "invalid parameters theta1/theta2");
      return;
   }
   fTheta1 = theta1;
   fTheta2 = theta2;
   fPhi1 = phi1;
   fPhi2 = phi2;
}   
//-----------------------------------------------------------------------------
void TGeoSphere::SetDimensions(Double_t *param)
{
   Double_t rmin = param[0];
   Double_t rmax = param[1];
   Double_t theta1 = 0;
   Double_t theta2 = 180.;
   Double_t phi1 = 0;
   Double_t phi2 = 360.;
//   if (nparam > 2) theta1 = param[2];
//   if (nparam > 3) theta2 = param[3];
//   if (nparam > 4) phi1   = param[4];
//   if (nparam > 5) phi2   = param[5];
   SetSphDimensions(rmin, rmax, theta1, theta2, phi1, phi2);
}   
//-----------------------------------------------------------------------------
void TGeoSphere::SetNumberOfDivisions(Int_t p)
{
   fNseg = p;
   Double_t dphi = fPhi2 - fPhi1;
   if (dphi<0) dphi+=360;
   Double_t dtheta = TMath::Abs(fTheta2-fTheta1);
   fNz = Int_t(fNseg*dtheta/dphi) +1;
}
//-----------------------------------------------------------------------------
void TGeoSphere::SetPoints(Double_t *buff) const
{
// create sphere mesh points
   Int_t i,j ;
    if (buff) {
        Double_t dphi = fPhi2-fPhi1;
        if (dphi<0) dphi+=360;
        Double_t dtheta = fTheta2-fTheta1;

        Int_t n            = fNseg + 1;
        dphi = dphi/fNseg;
        dtheta = dtheta/fNz;
        Double_t z, theta, phi, cphi, sphi;
        Int_t indx=0;
        for (i = 0; i < fNz+1; i++)
        {
            theta = (fTheta1+i*dtheta)*kDegRad;
            z = fRmin * TMath::Cos(theta); // fSinPhiTab[i];
            Float_t zi = fRmin*TMath::Sin(theta);
//            printf("plane %i nseg=%i z=%f:\n", i,n,z);
            for (j = 0; j < n; j++)
            {
                phi = (fPhi1+j*dphi)*kDegRad;
                cphi = TMath::Cos(phi);
                sphi = TMath::Sin(phi);
                buff[indx++] = zi * cphi;
                buff[indx++] = zi * sphi;
                buff[indx++] = z;
//                printf("%i %f %f %f\n", j, buff[3*j], buff[3*j+1], buff[3*j+2]);
            }
            z = fRmax * TMath::Cos(theta);
            zi = fRmax* TMath::Sin(theta);
//            printf("outer points for plane %i:\n", i);
            for (j = 0; j < n; j++)
            {
                phi = (fPhi1+j*dphi)*kDegRad;
                cphi = TMath::Cos(phi);
                sphi = TMath::Sin(phi);
                buff[indx++] = zi * cphi;
                buff[indx++] = zi * sphi;
                buff[indx++] = z;
//                printf("%i %f %f %f\n", j, buff[n+3*j], buff[n+3*j+1], buff[n+3*j+2]);
            }
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoSphere::SetPoints(Float_t *buff) const
{
// create sphere mesh points
   Int_t i,j ;
    if (buff) {
        Double_t dphi = fPhi2-fPhi1;
        if (dphi<0) dphi+=360;
        Double_t dtheta = fTheta2-fTheta1;

        Int_t n            = fNseg + 1;
        dphi = dphi/fNseg;
        dtheta = dtheta/fNz;
        Double_t z, theta, phi, cphi, sphi;
        Int_t indx=0;
        for (i = 0; i < fNz+1; i++)
        {
            theta = (fTheta1+i*dtheta)*kDegRad;
            z = fRmin * TMath::Cos(theta); // fSinPhiTab[i];
            Float_t zi = fRmin*TMath::Sin(theta);
//            printf("plane %i nseg=%i z=%f:\n", i,n,z);
            for (j = 0; j < n; j++)
            {
                phi = (fPhi1+j*dphi)*kDegRad;
                cphi = TMath::Cos(phi);
                sphi = TMath::Sin(phi);
                buff[indx++] = zi * cphi;
                buff[indx++] = zi * sphi;
                buff[indx++] = z;
//                printf("%i %f %f %f\n", j, buff[3*j], buff[3*j+1], buff[3*j+2]);
            }
            z = fRmax * TMath::Cos(theta);
            zi = fRmax* TMath::Sin(theta);
//            printf("outer points for plane %i:\n", i);
            for (j = 0; j < n; j++)
            {
                phi = (fPhi1+j*dphi)*kDegRad;
                cphi = TMath::Cos(phi);
                sphi = TMath::Sin(phi);
                buff[indx++] = zi * cphi;
                buff[indx++] = zi * sphi;
                buff[indx++] = z;
//                printf("%i %f %f %f\n", j, buff[n+3*j], buff[n+3*j+1], buff[n+3*j+2]);
            }
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoSphere::Sizeof3D() const
{
// fill size of this 3-D object
    TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
    if (!painter) return;
    
    Int_t n;
    n = fNseg+1;
    Int_t nz = fNz+1;
    Bool_t specialCase = kFALSE;

    if (TMath::Abs(TMath::Sin(2*(fPhi2 - fPhi1))) <= 0.01)  //mark this as a very special case, when
          specialCase = kTRUE;                                  //we have to draw this PCON like a TUBE

    Int_t numPoints = 2*n*nz;
    Int_t numSegs   = 4*(nz*n-1+(specialCase == kTRUE));
    Int_t numPolys  = 2*(nz*n-1+(specialCase == kTRUE));
    painter->AddSize3D(numPoints, numSegs, numPolys);
}
