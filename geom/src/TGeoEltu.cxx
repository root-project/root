// @(#)root/geom:$Name:  $:$Id: TGeoEltu.cxx,v 1.2 2002/07/10 19:24:16 brun Exp $
// Author: Mihaela Gheata   05/06/02

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
#include "TGeoEltu.h"

/*************************************************************************
 * TGeoEltu - elliptical tube class. It takes 3 parameters : 
 * semi-axis of the ellipse along x, semi-asix of the ellipse along y
 * and half-length dz. 
 *
 *************************************************************************/
//Begin_Html
/*
<img src="gif/TGeoEltu.gif">
*/
//End_Html


ClassImp(TGeoEltu)
   
//-----------------------------------------------------------------------------
TGeoEltu::TGeoEltu()
{
// Dummy constructor
   SetBit(TGeoShape::kGeoEltu);
}   
//-----------------------------------------------------------------------------
TGeoEltu::TGeoEltu(Double_t a, Double_t b, Double_t dz)
           :TGeoTube(a, b, dz)
{
// Default constructor specifying X and Y semiaxis length
   SetBit(TGeoShape::kGeoEltu);
   SetEltuDimensions(a, b, dz);
   ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoEltu::TGeoEltu(Double_t *param)
{
// Default constructor specifying minimum and maximum radius
// param[0] =  A
// param[1] =  B
// param[2] = dz
   SetBit(TGeoShape::kGeoEltu);
   SetDimensions(param);
   ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoEltu::~TGeoEltu()
{
// destructor
}
//-----------------------------------------------------------------------------   
void TGeoEltu::ComputeBBox()
{
// compute bounding box of the tube
   fDX = fRmin;
   fDY = fRmax;
   fDZ = fDz;
}   
//-----------------------------------------------------------------------------
Bool_t TGeoEltu::Contains(Double_t *point) const
{
// test if point is inside the elliptical tube
   if (TMath::Abs(point[2]) > fDz) return kFALSE;
   Double_t r2 = (point[0]*point[0])/(fRmin*fRmin)+(point[1]*point[1])/(fRmin*fRmin);
   if (r2>1.)  return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
Int_t TGeoEltu::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute closest distance from point px,py to each vertex
   Int_t n = gGeoManager->GetNsegments();
   const Int_t numPoints=4*n;
   return ShapeDistancetoPrimitive(numPoints, px, py);
}   
//-----------------------------------------------------------------------------
Double_t TGeoEltu::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from inside point to surface of the tube
   Double_t a2=fRmin*fRmin;
   Double_t b2=fRmax*fRmax;
   Double_t safz1=fDz-point[2];
   Double_t safz2=fDz+point[2];
   
   if (iact<3 && safe) {
      Double_t x0=TMath::Abs(point[0]);
      Double_t y0=TMath::Abs(point[1]);
      Double_t x1=x0;
      Double_t y1=TMath::Sqrt((fRmin-x0)*(fRmin+x0))*fRmax/fRmin;
      Double_t y2=y0;
      Double_t x2=TMath::Sqrt((fRmax-y0)*(fRmax+y0))*fRmin/fRmax;
      Double_t d1=(x1-x0)*(x1-x0)+(y1-y0)*(y1-y0);
      Double_t d2=(x2-x0)*(x2-x0)+(y2-y0)*(y2-y0);
      Double_t x3,y3;
   
      Double_t safr=kBig;
      Double_t safz = TMath::Min(safz1,safz2);
      for (Int_t i=0; i<8; i++) {
         if (fRmax<fRmin) {
            x3=0.5*(x1+x2);
            y3=TMath::Sqrt((fRmin-x3)*(fRmin+x3))*fRmax/fRmin;;
         } else {
            y3=0.5*(y1+y2);   
            x3=TMath::Sqrt((fRmax-y3)*(fRmax+y3))*fRmin/fRmax;
         }
         if (d1<d2) {
            x2=x3;
            y2=y3;
            d2=(x2-x0)*(x2-x0)+(y2-y0)*(y2-y0);
         } else {
            x1=x3;
            y1=y3;
            d1=(x1-x0)*(x1-x0)+(y1-y0)*(y1-y0);
         }
      }
      safr=TMath::Sqrt(d1)-1.0E-3;   
      *safe = TMath::Min(safz, safr);
      if (iact==0) return kBig;
      if ((iact==1) && (*safe>step)) return step;
   }
   // compute distance to surface 
   // Do Z
   Double_t snxt = kBig;
   if (dir[2]>0) {
      snxt=safz1/dir[2];
   } else {
      if (dir[2]<0) snxt=-safz2/dir[2];
   } 
   // do eliptical surface
   Double_t sz = snxt;
   Double_t xz=point[0]+dir[0]*sz;
   Double_t yz=point[1]+dir[1]*sz;
   if ((xz*xz/a2+yz*yz/b2)<=1) return snxt;
   Double_t u=dir[0]*dir[0]*b2+dir[1]*dir[1]*a2;
   Double_t v=point[0]*dir[0]*b2+point[1]*dir[1]*a2;
   Double_t w=point[0]*point[0]*b2+point[1]*point[1]*a2-a2*b2;
   Double_t d=v*v-u*w;
   if (d<0) return snxt;
   if (u==0) return snxt;
   Double_t sd=TMath::Sqrt(d);
   Double_t tau1=(-v+sd)/u;
   Double_t tau2=(-v+sd)/u;
   
   if (tau1<0) {
      if (tau2<0) return snxt;
   } else {
      snxt=tau1;
      if ((tau2>0) && (tau2<tau1)) return tau2;
   }
   return snxt;      
}
//-----------------------------------------------------------------------------
Double_t TGeoEltu::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from outside point to surface of the tube and safe distance
   Double_t snxt=kBig;
   Double_t safz=TMath::Abs(point[2]-fDz);
   Double_t a2=fRmin*fRmin;
   Double_t b2=fRmax*fRmax;
   if (iact<3 && safe) {
      Double_t x0=TMath::Abs(point[0]);
      Double_t y0=TMath::Abs(point[1]);
      *safe=0.;
      if ((x0*x0/a2+y0*y0/b2)>=1) {
         Double_t phi1=0;
         Double_t phi2=0.5*TMath::Pi();
         Double_t phi3;
         Double_t x3,y3,d;
         for (Int_t i=0; i<10; i++) {
            phi3=(phi1+phi2)*0.5;
            x3=fRmin*TMath::Cos(phi3);
            y3=fRmax*TMath::Sin(phi3);
            d=y3*a2*(x0-x3)-x3*b2*(y0-y3);
            if (d<0) phi1=phi3;
            else phi2=phi3;
         }
         *safe=TMath::Sqrt((x0-x3)*(x0-x3)+(y0-y3)*(y0-y3));
      }
      if (safz>0) {
         *safe=TMath::Sqrt((*safe)*(*safe)+safz*safz);
      } 
      if (iact==0) return kBig;
      if ((iact==1) && (step<*safe)) return step;
   }
   // compute vector distance
   if ((safz>0) && (point[2]*dir[2]>=0)) return kBig;
   Double_t zi;
   if (dir[2]!=0) {
      Double_t u=dir[0]*dir[0]*b2+dir[1]*dir[1]*a2;
      Double_t v=point[0]*dir[0]*b2+point[1]*dir[1]*a2;
      Double_t w=point[0]*point[0]*b2+point[1]*point[1]*a2-a2*b2;
      Double_t d=v*v-u*w;
      if (d<0) return kBig;
      Double_t dsq=TMath::Sqrt(d);
      Double_t tau[2];
      tau[0]=(-v+dsq)/u;
      tau[1]=(-v-dsq)/u;
      for (Int_t j=0; j<2; j++) {
         if (tau[j]>=0) {
            zi=point[2]+tau[j]*dir[2];
            if ((TMath::Abs(zi)-fDz)<0)
               snxt=TMath::Min(snxt,tau[j]);
         } 
      }
   }   
   // do z
   zi=kBig;
   if (safz>0) {
      if (point[2]>0) zi=fDz;
      if (point[2]<0) zi=-fDz;     
      Double_t tauz=(zi-point[2])/dir[2];
      Double_t xz=point[0]+dir[0]*tauz;
      Double_t yz=point[1]+dir[1]*tauz;
      if ((xz*xz/a2+yz*yz/b2)<=1) snxt=tauz;
   }
   return snxt;   
}
//-----------------------------------------------------------------------------
Double_t TGeoEltu::DistToSurf(Double_t *point, Double_t *dir) const
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   return 0.0;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoEltu::Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                             Double_t start, Double_t step) 
{
   Error("Divide", "Elliptical tubes divisions not implemenetd");
   return voldiv;
}   
//-----------------------------------------------------------------------------
TGeoVolume *TGeoEltu::Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Double_t step) 
{
// Divide all range of iaxis in range/step cells 
   Error("Divide", "Division in all range not implemented");
   return voldiv;
}      
//-----------------------------------------------------------------------------
TGeoShape *TGeoEltu::GetMakeRuntimeShape(TGeoShape *mother) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestBit(kGeoRunTimeShape)) return 0;
   if (mother->IsRunTimeShape() || !mother->TestBit(kGeoEltu)) {
      Error("GetMakeRuntimeShape", "invalid mother");
      return 0;
   }
   Double_t a, b, dz;
   a = fRmin;
   b = fRmax;
   dz = fDz;
   if (fDz<0) dz=((TGeoEltu*)mother)->GetDz();
   if (fRmin<0)
      a = ((TGeoEltu*)mother)->GetA();
   if (fRmax<0) 
      a = ((TGeoEltu*)mother)->GetB();

   return (new TGeoEltu(a, b, dz));
}
//-----------------------------------------------------------------------------
void TGeoEltu::InspectShape() const
{
// print shape parameters
   printf("*** TGeoEltu parameters ***\n");
   printf("    A    = %11.5f\n", fRmin);
   printf("    B    = %11.5f\n", fRmax);
   printf("    dz   = %11.5f\n", fDz);
   TGeoBBox::InspectShape();
}
//-----------------------------------------------------------------------------
void TGeoEltu::NextCrossing(TGeoParamCurve *c, Double_t *point) const
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoEltu::Safety(Double_t *point, Double_t *spoint, Option_t *option) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return kBig;
}
//-----------------------------------------------------------------------------
void TGeoEltu::SetEltuDimensions(Double_t a, Double_t b, Double_t dz)
{
   if ((a<=0) || (b<0) || (dz<0)) {
      SetBit(kGeoRunTimeShape);
   }
   fRmin=a;
   fRmax=b;
   fDz=dz;
}   
//-----------------------------------------------------------------------------
void TGeoEltu::SetDimensions(Double_t *param)
{
   Double_t a    = param[0];
   Double_t b    = param[1];
   Double_t dz   = param[2];
   SetEltuDimensions(a, b, dz);
}   
//-----------------------------------------------------------------------------
void TGeoEltu::SetPoints(Double_t *buff) const
{
// create tube mesh points
    Double_t dz;
    Int_t j, n;

    n = gGeoManager->GetNsegments();
    Double_t dphi = 360./n;
    Double_t phi = 0;
    Double_t cph,sph;
    dz = fDz;

    Int_t indx = 0;
    Double_t r2,r;
    Double_t a2=fRmin*fRmin;
    Double_t b2=fRmax*fRmax;

    if (buff) {

        for (j = 0; j < n; j++) {
            buff[indx+6*n] = buff[indx] = 0;
            indx++;
            buff[indx+6*n] = buff[indx] = 0;
            indx++;
            buff[indx+6*n] = dz;
            buff[indx]     =-dz;
            indx++;
        }
        for (j = 0; j < n; j++) {
            phi = j*dphi*kDegRad;
            sph=TMath::Sin(phi);
            cph=TMath::Sin(phi);
            r2=(a2*b2)/(b2+(a2-b2)*sph*sph);
            r=TMath::Sqrt(r2);
            buff[indx+6*n] = buff[indx] = r*cph;
            indx++;
            buff[indx+6*n] = buff[indx] = r*sph;
            indx++;
            buff[indx+6*n]= dz;
            buff[indx]    =-dz;
            indx++;
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoEltu::SetPoints(Float_t *buff) const
{
// create tube mesh points
    Double_t dz;
    Int_t j, n;

    n = gGeoManager->GetNsegments();
    Double_t dphi = 360./n;
    Double_t phi = 0;
    Double_t cph,sph;
    dz = fDz;

    Int_t indx = 0;
    Double_t r2,r;
    Double_t a2=fRmin*fRmin;
    Double_t b2=fRmax*fRmax;

    if (buff) {

        for (j = 0; j < n; j++) {
            buff[indx+6*n] = buff[indx] = 0;
            indx++;
            buff[indx+6*n] = buff[indx] = 0;
            indx++;
            buff[indx+6*n] = dz;
            buff[indx]     =-dz;
            indx++;
        }
        for (j = 0; j < n; j++) {
            phi = j*dphi*kDegRad;
            sph=TMath::Sin(phi);
            cph=TMath::Cos(phi);
            r2=(a2*b2)/(b2+(a2-b2)*sph*sph);
            r=TMath::Sqrt(r2);
            buff[indx+6*n] = buff[indx] = r*cph;
            indx++;
            buff[indx+6*n] = buff[indx] = r*sph;
            indx++;
            buff[indx+6*n]= dz;
            buff[indx]    =-dz;
            indx++;
        }
    }
}
