/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author :  Andrei Gheata  - date Thu 31 Jan 2002 01:47:40 PM CET
// TGeoTube::Contains() and DistToOut/In() implemented by Mihaela Gheata

#include "TROOT.h"

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoPainter.h"
#include "TGeoTube.h"

/*************************************************************************
 * TGeoTube - cylindrical tube class. It takes 3 parameters : 
 * inner radius, outer radius and half-length dz. 
 *
 *************************************************************************/
//Begin_Html
/*
<img src="gif/TGeoTube.gif">
*/
//End_Html

/*************************************************************************
 * TGeoTubeSeg - a phi segment of a tube. Has 5 parameters :
 *            - the same 3 as a tube;
 *            - first phi limit (in degrees)
 *            - second phi limit 
 *
 *************************************************************************/
//Begin_Html
/*
<img src="gif/TGeoTubs.gif">
*/
//End_Html

/*************************************************************************
 * TGeoCtub - a tube segment cut with 2 planes. Has 11 parameters :
 *            - the same 5 as a tube segment;
 *            - x, y, z components of the normal to the -dZ cut plane in
 *              point (0, 0, -dZ);
 *            - x, y, z components of the normal to the +dZ cut plane in
 *              point (0, 0, dZ);
 *
 *************************************************************************/
//Begin_Html
/*
<img src="gif/TGeoCtub.gif">
*/
//End_Html


ClassImp(TGeoTube)
   
//-----------------------------------------------------------------------------
TGeoTube::TGeoTube()
{
// Default constructor
   SetBit(TGeoShape::kGeoTube);
   fRmin = 0.0;
   fRmax = 0.0;
   fDz   = 0.0;
}   
//-----------------------------------------------------------------------------
TGeoTube::TGeoTube(Double_t rmin, Double_t rmax, Double_t dz)
           :TGeoBBox(0, 0, 0)
{
// Default constructor specifying minimum and maximum radius
   SetBit(TGeoShape::kGeoTube);
   SetTubeDimensions(rmin, rmax, dz);
   if ((fDz<0) || (fRmin<0) || (fRmax<0)) {
      SetBit(kGeoRunTimeShape);
//      if (fRmax<=fRmin) SetBit(kGeoInvalidShape);
//      printf("tube : dz=%f rmin=%f rmax=%f\n", dz, rmin, rmax);
   }
   ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoTube::TGeoTube(Double_t *param)
{
// Default constructor specifying minimum and maximum radius
// param[0] = Rmin
// param[1] = Rmax
// param[2] = dz
   SetBit(TGeoShape::kGeoTube);
   SetDimensions(param);
   if ((fDz<0) || (fRmin<0) || (fRmax<0)) SetBit(kGeoRunTimeShape);
   ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoTube::~TGeoTube()
{
// destructor
}
//-----------------------------------------------------------------------------   
void TGeoTube::ComputeBBox()
{
// compute bounding box of the tube
   fDX = fDY = fRmax;
   fDZ = fDz;
}   
//-----------------------------------------------------------------------------
Bool_t TGeoTube::Contains(Double_t *point)
{
// test if point is inside this tube
   if (TMath::Abs(point[2]) > fDz) return kFALSE;
   Double_t r2 = point[0]*point[0]+point[1]*point[1];
   if ((r2<fRmin*fRmin) || (r2>fRmax*fRmax)) return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
Int_t TGeoTube::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute closest distance from point px,py to each corner
   Int_t n = TGeoManager::kGeoDefaultNsegments;
   const Int_t numPoints = 4*n;
   return ShapeDistancetoPrimitive(numPoints, px, py);
}
//-----------------------------------------------------------------------------
Double_t TGeoTube::DistToOutS(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe,
                              Double_t rmin, Double_t rmax, Double_t dz)
{
// compute distance from inside point to surface of the tube (static)
   Double_t saf[3];
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   if (iact<3 && safe) {
      if (rmin>1E-10) saf[0] = r-rmin;
      else saf[0] = kBig;
      saf[1] = rmax-r;
      saf[2] = dz-TMath::Abs(point[2]);
      *safe = saf[TMath::LocMin(3, &saf[0])];
      if (iact==0) return kBig;
      if ((iact==1) && (*safe>step)) return step;
   }
   // compute distance to surface 
   // Do Z
   Double_t sz = kBig;
   if (dir[2]>1E-20) 
      sz = (dz-point[2])/dir[2];
   else
      if (dir[2]<-1E-20) sz = -(dz+point[2])/dir[2];
   // Do R
   Double_t t1=dir[0]*dir[0]+dir[1]*dir[1];  
   Double_t t2=point[0]*dir[0]+point[1]*dir[1];  
   Double_t t3=point[0]*point[0]+point[1]*point[1]; 
   if (t1<0) return sz;
   Double_t b=t2/t1;
   Double_t sr, c=0, d=0;
   // inner cylinder
   if (rmin>1E-10) {
      c=(t3-rmin*rmin)/t1;
      d=b*b-c;
      if (d>=0) {
         sr=-b-TMath::Sqrt(d);
         if (sr>0) return TMath::Min(sz,sr);
      }
   }
   // outer cylinder
   c=(t3-rmax*rmax)/t1;
   d=TMath::Max(b*b-c, 0.);
   sr=-b+TMath::Sqrt(d);
   if (sr>0) return TMath::Min(sz,sr);
   return kBig;      
}
//-----------------------------------------------------------------------------
Double_t TGeoTube::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from inside point to surface of the tube
   Double_t saf[3];
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   if (iact<3 && safe) {
      if (fRmin>1E-10) saf[0] = r-fRmin;
      else saf[0] = kBig;
      saf[1] = fRmax-r;
      saf[2] = fDz-TMath::Abs(point[2]);
      *safe = TMath::Min(saf[0], TMath::Min(saf[1],saf[2]));
      if (iact==0) return kBig;
      if ((iact==1) && (*safe>step)) return step;
   }
   // compute distance to surface 
   // Do Z
   Double_t sz = kBig;
   if (dir[2]>1E-20) 
      sz = (fDz-point[2])/dir[2];
   else
      if (dir[2]<-1E-20) sz = -(fDz+point[2])/dir[2];
   // Do R
   Double_t t1=dir[0]*dir[0]+dir[1]*dir[1];  
   Double_t t2=point[0]*dir[0]+point[1]*dir[1];  
   Double_t t3=point[0]*point[0]+point[1]*point[1]; 
   if (t1<0) return sz;
   Double_t b=t2/t1;
   Double_t sr, c=0, d=0;
   // inner cylinder
   if (fRmin>1E-10) {
      c=(t3-fRmin*fRmin)/t1;
      d=b*b-c;
      if (d>=0) {
         sr=-b-TMath::Sqrt(d);
         if (sr>0) return TMath::Min(sz,sr);
      }
   }
   // outer cylinder
   c=(t3-fRmax*fRmax)/t1;
   d=TMath::Max(b*b-c, 0.);
   sr=-b+TMath::Sqrt(d);
   if (sr>0) return TMath::Min(sz,sr);
   return kBig;      
}
//-----------------------------------------------------------------------------
Double_t TGeoTube::DistToInS(Double_t *point, Double_t *dir, Double_t rmin, Double_t rmax, Double_t dz)
{
// static method to compute distance from outside point to a tube with given parameters
   Double_t *norm = gGeoManager->GetNormalChecked();
   Double_t rsq = point[0]*point[0]+point[1]*point[1];

   // check Z planes
   Double_t xi, yi, zi;
   Double_t s = kBig;
   if (TMath::Abs(point[2])>dz) {
      if ((point[2]*dir[2])<0) {
         s = (TMath::Abs(point[2])-dz)/TMath::Abs(dir[2]);
         xi = point[0]+s*dir[0];
         yi = point[1]+s*dir[1];
         Double_t r2=xi*xi+yi*yi;
         if (((rmin*rmin)<=r2) && (r2<=(rmax*rmax))) {
            norm[0]=norm[1]=0;
            norm[2]=(point[2]>0)?1:-1;
            return s;
         }
      }
   }      
   
   // check outer cyl. surface
   Double_t t1=dir[0]*dir[0]+dir[1]*dir[1];
   if (TMath::Abs(t1)<1E-32) return kBig;
   Double_t t2=point[0]*dir[0]+point[1]*dir[1];
   Double_t t3=rsq;
   Double_t b=t2/t1;
   Double_t c,d;
   // only r>rmax has to be considered
   if (rsq>rmax*rmax) {
      c=(t3-rmax*rmax)/t1;
      d=b*b-c;
      if (d>=0) {
         s=-b-TMath::Sqrt(d);
         if (s>=0) {
            zi=point[2]+s*dir[2];
            if (TMath::Abs(zi)<=dz) {
               xi=point[0]+s*dir[0];
               yi=point[1]+s*dir[1];
               norm[0] = xi/rmax;
               norm[1] = yi/rmax;
               norm[2] = 0;
               return s;
            }
         }
      }
   }         
   // check inner cylinder
   if (rmin>0) {
      c=(t3-rmin*rmin)/t1;
      d=b*b-c;
      if (d>=0) {
         s=-b+TMath::Sqrt(d);
         if (s>=0) {
            zi=point[2]+s*dir[2];
            if (TMath::Abs(zi)<=dz) {
               xi=point[0]+s*dir[0];
               yi=point[1]+s*dir[1];
               norm[0] = -xi/rmin;
               norm[1] = -yi/rmin;
               norm[2] = 0;
               return s;
            }
         }
      }
   }         
   return kBig;
}   
//-----------------------------------------------------------------------------
Double_t TGeoTube::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from outside point to surface of the tube and safe distance
   // fist localize point w.r.t tube
   Double_t saf[4];
   Double_t rsq = point[0]*point[0]+point[1]*point[1];
   Double_t r = TMath::Sqrt(rsq);
   if (iact<3 && *safe) {
      saf[0] = -fDz-point[2];
      saf[1] = point[2]-fDz;
      saf[2] = fRmin-r;
      saf[3] = r-fRmax;
      *safe = saf[TMath::LocMax(4,&saf[0])];
      if (iact==0) return kBig;
      if ((iact==1) && (step<=*safe)) return step;
   }
   // find distance to shape
   return DistToInS(point, dir, fRmin, fRmax, fDz);
}
//-----------------------------------------------------------------------------
Double_t TGeoTube::DistToSurf(Double_t *point, Double_t *dir)
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   return 0.0;
}
//-----------------------------------------------------------------------------
void TGeoTube::Draw(Option_t *option)
{
// draw this shape according to option
}
//-----------------------------------------------------------------------------
TGeoShape *TGeoTube::GetMakeRuntimeShape(TGeoShape *mother) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestBit(kGeoRunTimeShape)) return 0;
   if (mother->IsRunTimeShape() || !mother->TestBit(kGeoTube)) {
      Error("GetMakeRuntimeShape", "invalid mother");
      return 0;
   }
   Double_t rmin, rmax, dz;
   rmin = fRmin;
   rmax = fRmax;
   dz = fDz;
   if (fDz<0) dz=((TGeoTube*)mother)->GetDz();
   if (fRmin<0)
      rmin = ((TGeoTube*)mother)->GetRmin();
   if (fRmax<0) 
      rmax = ((TGeoTube*)mother)->GetRmax();

   return (new TGeoTube(rmin, rmax, dz));
}
//-----------------------------------------------------------------------------
void TGeoTube::InspectShape()
{
// print shape parameters
   printf("*** TGeoTube parameters ***\n");
   printf("    Rmin = %11.5f\n", fRmin);
   printf("    Rmax = %11.5f\n", fRmax);
   printf("    dz   = %11.5f\n", fDz);
   TGeoBBox::InspectShape();
}
//-----------------------------------------------------------------------------
void TGeoTube::Paint(Option_t *option)
{
// paint this shape according to option
   TGeoPainter *painter = (TGeoPainter*)gGeoManager->GetMakeDefPainter();
   if (!painter) return;
   TGeoVolume *vol = gGeoManager->GetCurrentVolume();
   if (vol->GetShape() != (TGeoShape*)this) return;
   painter->PaintTube(vol, option);
}
//-----------------------------------------------------------------------------
void TGeoTube::NextCrossing(TGeoParamCurve *c, Double_t *point)
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoTube::Safety(Double_t *point, Double_t *spoint, Option_t *option)
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return 0.0;
}
//-----------------------------------------------------------------------------
void TGeoTube::SetTubeDimensions(Double_t rmin, Double_t rmax, Double_t dz)
{
   if (rmin>=0) {
      if (rmax>0) {
         if (rmin<rmax) {
         // normal rmin/rmax
            fRmin = rmin;
            fRmax = rmax;
         } else {
            fRmin = rmax;
            fRmax = rmin;
            Warning("SetTubeDimensions", "rmin>rmax Switch rmin<->rmax");
         }
      } else {
         // run-time
         fRmin = rmin;
         fRmax = rmax;
      }
   } else {
      // run-time
      fRmin = rmin;
      fRmax = rmax;
   }               
   fDz   = dz;
}   
//-----------------------------------------------------------------------------
void TGeoTube::SetDimensions(Double_t *param)
{
   Double_t rmin = param[0];
   Double_t rmax = param[1];
   Double_t dz   = param[2];
   SetTubeDimensions(rmin, rmax, dz);
}   
//-----------------------------------------------------------------------------
void TGeoTube::SetPoints(Double_t *buff) const
{
// create tube mesh points
    Double_t dz;
    Int_t j, n;

    n = TGeoManager::kGeoDefaultNsegments;
    Double_t dphi = 360./n;
    Double_t phi = 0;
    dz = fDz;

    Int_t indx = 0;


    if (buff) {

        for (j = 0; j < n; j++) {
            phi = j*dphi*kDegRad;
            buff[indx+6*n] = buff[indx] = fRmin * TMath::Cos(phi);
            indx++;
            buff[indx+6*n] = buff[indx] = fRmin * TMath::Sin(phi);
            indx++;
            buff[indx+6*n] = dz;
            buff[indx]     =-dz;
            indx++;
        }
        for (j = 0; j < n; j++) {
            phi = j*dphi*kDegRad;
            buff[indx+6*n] = buff[indx] = fRmax * TMath::Cos(phi);
            indx++;
            buff[indx+6*n] = buff[indx] = fRmax * TMath::Sin(phi);
            indx++;
            buff[indx+6*n]= dz;
            buff[indx]    =-dz;
            indx++;
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoTube::SetPoints(Float_t *buff) const
{
// create tube mesh points
    Double_t dz;
    Int_t j, n;

    n = TGeoManager::kGeoDefaultNsegments;
    Double_t dphi = 360./n;
    Double_t phi = 0;
    dz = fDz;

    Int_t indx = 0;


    if (buff) {

        for (j = 0; j < n; j++) {
            phi = j*dphi*kDegRad;
            buff[indx+6*n] = buff[indx] = fRmin * TMath::Cos(phi);
            indx++;
            buff[indx+6*n] = buff[indx] = fRmin * TMath::Sin(phi);
            indx++;
            buff[indx+6*n] = dz;
            buff[indx]     =-dz;
            indx++;
        }
        for (j = 0; j < n; j++) {
            phi = j*dphi*kDegRad;
            buff[indx+6*n] = buff[indx] = fRmax * TMath::Cos(phi);
            indx++;
            buff[indx+6*n] = buff[indx] = fRmax * TMath::Sin(phi);
            indx++;
            buff[indx+6*n]= dz;
            buff[indx]    =-dz;
            indx++;
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoTube::Sizeof3D() const
{
// fill size of this 3-D object
    Int_t n = TGeoManager::kGeoDefaultNsegments;
    gSize3D.numPoints += n*4;
    gSize3D.numSegs   += n*8;
    gSize3D.numPolys  += n*4;
}

ClassImp(TGeoTubeSeg)
   
//-----------------------------------------------------------------------------
TGeoTubeSeg::TGeoTubeSeg()
{
// Default constructor
   SetBit(TGeoShape::kGeoTubeSeg);
   fPhi1 = fPhi2 = 0.0;
}   
//-----------------------------------------------------------------------------
TGeoTubeSeg::TGeoTubeSeg(Double_t rmin, Double_t rmax, Double_t dz,
                          Double_t phi1, Double_t phi2)
            :TGeoTube(rmin, rmax, dz)
{
// Default constructor specifying minimum and maximum radius
   SetBit(TGeoShape::kGeoTubeSeg);
   SetTubsDimensions(rmin, rmax, dz, phi1, phi2);
   ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoTubeSeg::TGeoTubeSeg(Double_t *param)
{
// Default constructor specifying minimum and maximum radius
// param[0] = Rmin
// param[1] = Rmax
// param[2] = dz
// param[3] = phi1
// param[4] = phi2
   SetBit(TGeoShape::kGeoTubeSeg);
   SetDimensions(param);
   ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoTubeSeg::~TGeoTubeSeg()
{
// destructor
}
//-----------------------------------------------------------------------------   
void TGeoTubeSeg::ComputeBBox()
{
// compute bounding box of the tube segment
   Double_t xc[4];
   Double_t yc[4];
   xc[0] = fRmax*TMath::Cos(fPhi1*kDegRad);
   yc[0] = fRmax*TMath::Sin(fPhi1*kDegRad);
   xc[1] = fRmax*TMath::Cos(fPhi2*kDegRad);
   yc[1] = fRmax*TMath::Sin(fPhi2*kDegRad);
   xc[2] = fRmin*TMath::Cos(fPhi1*kDegRad);
   yc[2] = fRmin*TMath::Sin(fPhi1*kDegRad);
   xc[3] = fRmin*TMath::Cos(fPhi2*kDegRad);
   yc[3] = fRmin*TMath::Sin(fPhi2*kDegRad);

   Double_t xmin = xc[TMath::LocMin(4, &xc[0])];
   Double_t xmax = xc[TMath::LocMax(4, &xc[0])]; 
   Double_t ymin = yc[TMath::LocMin(4, &yc[0])]; 
   Double_t ymax = yc[TMath::LocMax(4, &yc[0])];

   Double_t dp = fPhi2-fPhi1;
   if (dp<0) dp+=360;
   Double_t ddp = -fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=dp) xmax = fRmax;
   ddp = 90-fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=dp) ymax = fRmax;
   ddp = 180-fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=dp) xmin = -fRmax;
   ddp = 270-fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=dp) ymin = -fRmax;
   fOrigin[0] = (xmax+xmin)/2;
   fOrigin[1] = (ymax+ymin)/2;
   fOrigin[2] = 0;
   fDX = (xmax-xmin)/2;
   fDY = (ymax-ymin)/2;
   fDZ = fDz;
}   
//-----------------------------------------------------------------------------
Bool_t TGeoTubeSeg::Contains(Double_t *point)
{
// test if point is inside this tube segment
   // first check if point is inside the tube
   if (!TGeoTube::Contains(point)) return kFALSE;
   Double_t phi = TMath::ATan2(point[1], point[0]) * kRadDeg;
   if (phi < 0 ) phi+=360.;
   Double_t dphi = fPhi2 -fPhi1;
   if (dphi < 0) dphi+=360.;
   Double_t ddp = phi-fPhi1;
   if (ddp<0) ddp += 360.;
//   if (ddp>360) ddp-=360;
   if (ddp > dphi) return kFALSE;
   return kTRUE;    
}
//-----------------------------------------------------------------------------
Int_t TGeoTubeSeg::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute closest distance from point px,py to each corner
   Int_t n = TGeoManager::kGeoDefaultNsegments+1;
   const Int_t numPoints = 4*n;
   return ShapeDistancetoPrimitive(numPoints, px, py);
}
//-----------------------------------------------------------------------------
Double_t TGeoTubeSeg::DistToPhiMin(Double_t *point, Double_t *dir, Double_t s1, Double_t c1,
                                   Double_t s2, Double_t c2, Double_t sm, Double_t cm)
{
// compute distance from poin to both phi planes. Return minimum.
   Double_t sfi1=kBig;
   Double_t sfi2=kBig;
   Double_t s=0;
   Double_t un = dir[0]*s1-dir[1]*c1;
   if (un!=0) {
      s=(point[1]*c1-point[0]*s1)/un;
      if (s>=0) {
         if (((point[1]+s*dir[1])*cm-(point[0]+s*dir[0])*sm)<=0) sfi1=s;
      }   
   }
   un = dir[0]*s2-dir[1]*c2;    
   if (un!=0) {
      s=(point[1]*c2-point[0]*s2)/un;
      if (s>=0) {
         if (((point[1]+s*dir[1])*cm-(point[0]+s*dir[0])*sm)>=0) sfi2=s;
      }   
   }
   return TMath::Min(sfi1, sfi2);
}
//-----------------------------------------------------------------------------
Double_t TGeoTubeSeg::DistToOutS(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe,
                                 Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2)
{
// compute distance from inside point to surface of the tube segment (static)
   Double_t saf[4];
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t ph1 = phi1*kDegRad;
   Double_t ph2 = phi2*kDegRad;
   if (ph2<ph1) ph2+=2.*TMath::Pi();
   Double_t phim = 0.5*(ph1+ph2);
   Double_t c1 = TMath::Cos(ph1);
   Double_t c2 = TMath::Cos(ph2);
   Double_t s1 = TMath::Sin(ph1);
   Double_t s2 = TMath::Sin(ph2);
   Double_t cm = TMath::Cos(phim);
   Double_t sm = TMath::Sin(phim);
   
   if (iact<3 && safe) {
      if (rmin>1E-10) saf[0] = r-rmin;
      else saf[0] = kBig;
      saf[1] = rmax-r;
      saf[2] = dz-TMath::Abs(point[2]);
      if ((point[1]*cm-point[1]*sm)<=0)
         saf[3] = TMath::Abs(point[0]*s1-point[1]*c1);
      else
         saf[3] = TMath::Abs(point[0]*s2-point[1]*c2);
      *safe = saf[TMath::LocMin(4, &saf[0])];
      if (iact==0) return kBig;
      if ((iact==1) && (*safe>step)) return step;
   }
   // compute distance to surface 
   // Do Z
   Double_t sz = kBig;
   if (dir[2]>1E-20) 
      sz = (dz-point[2])/dir[2];
   else
      if (dir[2]<-1E-20) sz = -(dz+point[2])/dir[2];
   // Do R
   Double_t t1=dir[0]*dir[0]+dir[1]*dir[1];  
   Double_t t2=point[0]*dir[0]+point[1]*dir[1];  
   Double_t t3=point[0]*point[0]+point[1]*point[1]; 
   // track parralel to Z
   if (t1==0) return sz;
   Double_t b=t2/t1;
   Double_t sr=kBig, c=0, d=0;
   Bool_t skip_outer = kFALSE;
   // inner cylinder
   if (rmin>1E-10) {
      c=(t3-rmin*rmin)/t1;
      d=b*b-c;
      if (d>=0) {
         sr=-b-TMath::Sqrt(d);
         if (sr>0)
            skip_outer = kTRUE;
      }
   }
   // outer cylinder
   if (!skip_outer) {
      c=(t3-rmax*rmax)/t1;
      d=TMath::Max(b*b-c, 0.);
      sr=-b+TMath::Sqrt(d);
      if (sr<0) sr=kBig;
   }
   // phi planes
   Double_t sfmin=TGeoTubeSeg::DistToPhiMin(point, dir, s1, c1, s2, c2, sm, cm);;
   return TMath::Min(TMath::Min(sz,sr), sfmin);      
}
//-----------------------------------------------------------------------------
Double_t TGeoTubeSeg::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from inside point to surface of the tube segment
   Double_t saf[4];
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t phi1 = fPhi1*kDegRad;
   Double_t phi2 = fPhi2*kDegRad;
   if (phi2<phi1) phi2+=2.*TMath::Pi();
   Double_t phim = 0.5*(phi1+phi2);
   Double_t c1 = TMath::Cos(phi1);
   Double_t c2 = TMath::Cos(phi2);
   Double_t s1 = TMath::Sin(phi1);
   Double_t s2 = TMath::Sin(phi2);
   Double_t cm = TMath::Cos(phim);
   Double_t sm = TMath::Sin(phim);
   
   if (iact<3 && safe) {
      if (fRmin>1E-10) saf[0] = r-fRmin;
      else saf[0] = kBig;
      saf[1] = fRmax-r;
      saf[2] = fDz-TMath::Abs(point[2]);
      if ((point[1]*cm-point[1]*sm)<=0)
         saf[3] = TMath::Abs(point[0]*s1-point[1]*c1);
      else
         saf[3] = TMath::Abs(point[0]*s2-point[1]*c2);
      *safe = saf[TMath::LocMin(4, &saf[0])];
      if (iact==0) return kBig;
      if ((iact==1) && (*safe>step)) return step;
   }
   // compute distance to surface 
   // Do Z
   Double_t sz = kBig;
   if (dir[2]>1E-20) 
      sz = (fDz-point[2])/dir[2];
   else
      if (dir[2]<-1E-20) sz = -(fDz+point[2])/dir[2];
   // Do R
   Double_t t1=dir[0]*dir[0]+dir[1]*dir[1];  
   Double_t t2=point[0]*dir[0]+point[1]*dir[1];  
   Double_t t3=point[0]*point[0]+point[1]*point[1]; 
   // track parralel to Z
   if (t1==0) return sz;
   Double_t b=t2/t1;
   Double_t sr=kBig, c=0, d=0;
   Bool_t skip_outer = kFALSE;
   // inner cylinder
   if (fRmin>1E-10) {
      c=(t3-fRmin*fRmin)/t1;
      d=b*b-c;
      if (d>=0) {
         sr=-b-TMath::Sqrt(d);
         if (sr>0)
            skip_outer = kTRUE;
      }
   }
   // outer cylinder
   if (!skip_outer) {
      c=(t3-fRmax*fRmax)/t1;
      d=TMath::Max(b*b-c, 0.);
      sr=-b+TMath::Sqrt(d);
      if (sr<0) sr=kBig;
   }
   // phi planes
   Double_t sfmin=DistToPhiMin(point, dir, s1, c1, s2, c2, sm, cm);;
   return TMath::Min(TMath::Min(sz,sr), sfmin);      
}
//-----------------------------------------------------------------------------
Double_t TGeoTubeSeg::DistToInS(Double_t *point, Double_t *dir, Double_t rmin, Double_t rmax, 
                                Double_t dz, Double_t c1, Double_t s1, Double_t c2, Double_t s2,
                                Double_t cfio, Double_t sfio, Double_t cdfi)
{
// static method to compute distance to arbitrary tube segment from outside point
   Double_t *norm = gGeoManager->GetNormalChecked();
   Double_t r2, cpsi;
   Double_t rsq = point[0]*point[0]+point[1]*point[1];
   // check Z planes
   Double_t xi, yi, zi;
   Double_t s = kBig;
   if (TMath::Abs(point[2])>dz) {
      if ((point[2]*dir[2])<0) {
         s = (TMath::Abs(point[2])-dz)/TMath::Abs(dir[2]);
         xi = point[0]+s*dir[0];
         yi = point[1]+s*dir[1];
         r2=xi*xi+yi*yi;
         if (((rmin*rmin)<=r2) && (r2<=(rmax*rmax))) {
            norm[0]=norm[1]=0;
            norm[2]=(point[2]>0)?1:-1;
            cpsi=(xi*cfio+yi*sfio)/TMath::Sqrt(r2);
            if (cpsi>=cdfi) return s;
         }
      }
   }      
   
   // check outer cyl. surface
   Double_t t1=dir[0]*dir[0]+dir[1]*dir[1];
   if (TMath::Abs(t1)<1E-32) return kBig;
   Double_t t2=point[0]*dir[0]+point[1]*dir[1];
   Double_t t3=rsq;
   Double_t b=t2/t1;
   Double_t c,d;
   // only r>rmax has to be considered
   if (rsq>rmax*rmax) {
      c=(t3-rmax*rmax)/t1;
      d=b*b-c;
      if (d>=0) {
         s=-b-TMath::Sqrt(d);
         if (s>=0) {
            zi=point[2]+s*dir[2];
            if (TMath::Abs(zi)<=dz) {
               xi=point[0]+s*dir[0];
               yi=point[1]+s*dir[1];
               norm[0] = xi/rmax;
               norm[1] = yi/rmax;
               norm[2] = 0;
               cpsi=(xi*cfio+yi*sfio)/rmax;
               if (cpsi>=cdfi) return s;
            }
         }
      }
   }         
   // check inner cylinder
   Double_t snxt=kBig;
   if (rmin>0) {
      c=(t3-rmin*rmin)/t1;
      d=b*b-c;
      if (d>=0) {
         s=-b+TMath::Sqrt(d);
         if (s>=0) {
            zi=point[2]+s*dir[2];
            if (TMath::Abs(zi)<=dz) {
               xi=point[0]+s*dir[0];
               yi=point[1]+s*dir[1];
               norm[0] = -xi/rmin;
               norm[1] = -yi/rmin;
               norm[2] = 0;
               cpsi=(xi*cfio+yi*sfio)/rmin;
               if (cpsi>=cdfi) snxt=s;
            }
         }
      }
   }         
   // check phi planes
   Double_t un=dir[0]*s1-dir[1]*c1;
   if (un != 0) {
      s=(point[1]*c1-point[0]*s1)/un;
      if (s>=0) {
         zi=point[2]+s*dir[2];
         if (TMath::Abs(zi)<=dz) {
            xi=point[0]+s*dir[0];
            yi=point[1]+s*dir[1];
            r2=xi*xi+yi*yi;
            if ((rmin*rmin<=r2) && (r2<=rmax*rmax)) {
               if ((yi*cfio-xi*sfio)<=0) {
                  if (s<snxt) {
                     snxt=s;
                     norm[0] = s1;
                     norm[1] = -c1;
                     norm[2] = 0;
                  }
               }
            }
         }            
      }
   }
   un=dir[0]*s2-dir[1]*c2;
   if (un != 0) {
      s=(point[1]*c2-point[0]*s2)/un;
      if (s>=0) {
         zi=point[2]+s*dir[2];
         if (TMath::Abs(zi)<=dz) {
            xi=point[0]+s*dir[0];
            yi=point[1]+s*dir[1];
            r2=xi*xi+yi*yi;
            if ((rmin*rmin<=r2) && (r2<=rmax*rmax)) {
               if ((yi*cfio-xi*sfio)>=0) {
                  if (s<snxt) {
                     snxt=s;
                     norm[0] = -s2;
                     norm[1] = c2;
                     norm[2] = 0;
                  }
               }
            }
         }            
      }
   }
   return snxt;
}   
//-----------------------------------------------------------------------------
Double_t TGeoTubeSeg::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from outside point to surface of the tube segment
   // fist localize point w.r.t tube
   Double_t saf[5];
   Double_t rsq = point[0]*point[0]+point[1]*point[1];
   Double_t r = TMath::Sqrt(rsq);
   Double_t phi1 = fPhi1*kDegRad;
   Double_t phi2 = fPhi2*kDegRad;
   if (phi2<phi1) phi2+=2.*TMath::Pi();
   Double_t c1 = TMath::Cos(phi1);
   Double_t s1 = TMath::Sin(phi1);
   Double_t c2 = TMath::Cos(phi2);
   Double_t s2 = TMath::Sin(phi2);
   Double_t fio = 0.5*(phi1+phi2);
   Double_t cfio = TMath::Cos(fio);
   Double_t sfio = TMath::Sin(fio);
   Double_t dfi = 0.5*(phi2-phi1);
   Double_t cdfi = TMath::Cos(dfi);
   Double_t cpsi;
   
   if (iact<3 && *safe) {
      saf[0] = -fDz-point[2];
      saf[1] = point[2]-fDz;
      saf[2] = fRmin-r;
      saf[3] = r-fRmax;
      if (r>0) {
         cpsi = (point[0]*cfio+point[1]*sfio)/r;
         if (cpsi<cdfi) {
            if ((point[1]*cfio-point[0]*sfio)<0)
               saf[4]=TMath::Abs(point[0]*s1-point[1]*c1);
            else
               saf[4]=TMath::Abs(point[0]*s2-point[1]*c2);
         }      
      }
      *safe = saf[TMath::LocMax(4,&saf[0])];
      if (iact==0) return kBig;
      if ((iact==1) && (step<=*safe)) return step;
   }
   // find distance to shape
   return TGeoTubeSeg::DistToInS(point, dir, fRmin, fRmax, fDz, c1, s1, c2, s2, cfio, sfio, cdfi);
}
//-----------------------------------------------------------------------------
Double_t TGeoTubeSeg::DistToSurf(Double_t *point, Double_t *dir)
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   return kBig;
}
//-----------------------------------------------------------------------------
void TGeoTubeSeg::Draw(Option_t *option)
{
// draw this shape according to option
}
//-----------------------------------------------------------------------------
TGeoShape *TGeoTubeSeg::GetMakeRuntimeShape(TGeoShape *mother) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestBit(kGeoRunTimeShape)) return 0;
   if (mother->IsRunTimeShape() || !mother->TestBit(kGeoTubeSeg)) {
      Error("GetMakeRuntimeShape", "invalid mother");
      return 0;
   }
   Double_t rmin, rmax, dz;
   rmin = fRmin;
   rmax = fRmax;
   dz = fDz;
   if (fDz<0) dz=((TGeoTube*)mother)->GetDz();
   if (fRmin<0)
      rmin = ((TGeoTube*)mother)->GetRmin();
   if ((fRmax<0) || (fRmax<=fRmin))
      rmax = ((TGeoTube*)mother)->GetRmax();
   
   return (new TGeoTubeSeg(rmin, rmax, dz, fPhi1, fPhi2));
}
//-----------------------------------------------------------------------------
void TGeoTubeSeg::InspectShape()
{
// print shape parameters
   printf("*** TGeoTubeSeg parameters ***\n");
   printf("    Rmin = %11.5f\n", fRmin);
   printf("    Rmax = %11.5f\n", fRmax);
   printf("    dz   = %11.5f\n", fDz);
   printf("    phi1 = %11.5f\n", fPhi1);
   printf("    phi2 = %11.5f\n", fPhi2);
   TGeoBBox::InspectShape();
}
//-----------------------------------------------------------------------------
void TGeoTubeSeg::Paint(Option_t *option)
{
// paint this shape according to option
   TGeoPainter *painter = (TGeoPainter*)gGeoManager->GetMakeDefPainter();
   if (!painter) return;
   TGeoVolume *vol = gGeoManager->GetCurrentVolume();
   if (vol->GetShape() != (TGeoShape*)this) return;
   painter->PaintTubs(vol, option);
}
//-----------------------------------------------------------------------------
void TGeoTubeSeg::NextCrossing(TGeoParamCurve *c, Double_t *point)
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoTubeSeg::Safety(Double_t *point, Double_t *spoint, Option_t *option)
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return 0.0;
}
//-----------------------------------------------------------------------------
void TGeoTubeSeg::SetTubsDimensions(Double_t rmin, Double_t rmax, Double_t dz,
                          Double_t phi1, Double_t phi2)
{
   fRmin = rmin;
   fRmax = rmax;
   fDz   = dz;
   fPhi1 = phi1;
   if (fPhi1 < 0) fPhi1+=360.;
   fPhi2 = phi2;
   if (fPhi2 < 0) fPhi2+=360.;
}   
//-----------------------------------------------------------------------------
void TGeoTubeSeg::SetDimensions(Double_t *param)
{
   Double_t rmin = param[0];
   Double_t rmax = param[1];
   Double_t dz   = param[2];
   Double_t phi1 = param[3];
   Double_t phi2 = param[4];
   SetTubsDimensions(rmin, rmax, dz, phi1, phi2);
}   
//-----------------------------------------------------------------------------
void TGeoTubeSeg::SetPoints(Double_t *buff) const
{
// create sphere mesh points
    Double_t dz;
    Int_t j, n;
    Double_t phi, phi1, phi2, dphi;
    phi1 = fPhi1;
    phi2 = fPhi2;
    if (phi2<phi1) phi2+=360.;
    n = TGeoManager::kGeoDefaultNsegments+1;

    dphi = (phi2-phi1)/(n-1);
    dz   = fDz;

    if (buff) {
        Int_t indx = 0;

        for (j = 0; j < n; j++) {
            phi = (phi1+j*dphi)*kDegRad;
            buff[indx+6*n] = buff[indx] = fRmin * TMath::Cos(phi);
            indx++;
            buff[indx+6*n] = buff[indx] = fRmin * TMath::Sin(phi);
            indx++;
            buff[indx+6*n] = dz;
            buff[indx]     =-dz;
            indx++;
        }
        for (j = 0; j < n; j++) {
            phi = (phi1+j*dphi)*kDegRad;
            buff[indx+6*n] = buff[indx] = fRmax * TMath::Cos(phi);
            indx++;
            buff[indx+6*n] = buff[indx] = fRmax * TMath::Sin(phi);
            indx++;
            buff[indx+6*n]= dz;
            buff[indx]    =-dz;
            indx++;
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoTubeSeg::SetPoints(Float_t *buff) const
{
// create sphere mesh points
    Double_t dz;
    Int_t j, n;
    Double_t phi, phi1, phi2, dphi;
    phi1 = fPhi1;
    phi2 = fPhi2;
    if (phi2<phi1) phi2+=360.;
    n = TGeoManager::kGeoDefaultNsegments+1;

    dphi = (phi2-phi1)/(n-1);
    dz   = fDz;

    if (buff) {
        Int_t indx = 0;

        for (j = 0; j < n; j++) {
            phi = (phi1+j*dphi)*kDegRad;
            buff[indx+6*n] = buff[indx] = fRmin * TMath::Cos(phi);
            indx++;
            buff[indx+6*n] = buff[indx] = fRmin * TMath::Sin(phi);
            indx++;
            buff[indx+6*n] = dz;
            buff[indx]     =-dz;
            indx++;
        }
        for (j = 0; j < n; j++) {
            phi = (phi1+j*dphi)*kDegRad;
            buff[indx+6*n] = buff[indx] = fRmax * TMath::Cos(phi);
            indx++;
            buff[indx+6*n] = buff[indx] = fRmax * TMath::Sin(phi);
            indx++;
            buff[indx+6*n]= dz;
            buff[indx]    =-dz;
            indx++;
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoTubeSeg::Sizeof3D() const
{
// fill size of this 3-D object
    Int_t n = TGeoManager::kGeoDefaultNsegments+1;

    gSize3D.numPoints += n*4;
    gSize3D.numSegs   += n*8;
    gSize3D.numPolys  += n*4-2;
}


ClassImp(TGeoCtub)

TGeoCtub::TGeoCtub()
{
// default ctor
   fNlow = 0;
   fNhigh = 0;
}
//-----------------------------------------------------------------------------
TGeoCtub::TGeoCtub(Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2,
                   Double_t lx, Double_t ly, Double_t lz, Double_t hx, Double_t hy, Double_t hz)
         :TGeoTubeSeg(rmin, rmax, dz, phi1, phi2)
{         
// ctor
   fNlow = new Double_t[3];
   fNhigh = new Double_t[3];
   fNlow[0] = lx;
   fNlow[1] = ly;
   fNlow[2] = lz;
   fNhigh[0] = hx;
   fNhigh[1] = hy;
   fNhigh[2] = hz;
   SetBit(kGeoCtub);
   ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoCtub::TGeoCtub(Double_t *params)
{
// ctor with parameters
   fNlow = new Double_t[3];
   fNhigh = new Double_t[3];
   SetCtubDimensions(params[0], params[1], params[2], params[3], params[4], params[5],
                     params[6], params[7], params[8], params[9], params[10]);
   SetBit(kGeoCtub);
}
//-----------------------------------------------------------------------------
TGeoCtub::~TGeoCtub()
{
// dtor
   if (fNlow) delete [] fNlow;
   if (fNhigh) delete [] fNhigh;
}   
//-----------------------------------------------------------------------------
void TGeoCtub::ComputeBBox()
{
// compute minimum bounding box of the ctub
   TGeoTubeSeg::ComputeBBox();
   if ((fNlow[2]>-(1E-10)) || (fNhigh[2]<1E-10)) {
      Error("ComputeBBox", "Wrong definition of cut planes");
      return;
   }   
   Double_t xc=0, yc=0;
   Double_t zmin=0, zmax=0;
   Double_t z1;
   Double_t z[8];
   // check if nxy is in the phi range
   Double_t phi_low = TMath::ATan2(fNlow[1], fNlow[0]) *kRadDeg;
   Double_t phi_hi = TMath::ATan2(fNhigh[1], fNhigh[0]) *kRadDeg;
   Bool_t in_range_low = kFALSE;
   Bool_t in_range_hi = kFALSE;
   
   Int_t i;
   for (i=0; i<2; i++) {
      if (phi_low<0) phi_low+=360.;
      Double_t dphi = fPhi2 -fPhi1;
      if (dphi < 0) dphi+=360.;
      Double_t ddp = phi_low-fPhi1;
      if (ddp<0) ddp += 360.;
      if (ddp <= dphi) {
         xc = fRmin*TMath::Cos(phi_low*kDegRad);
         yc = fRmin*TMath::Sin(phi_low*kDegRad);
         z1 = GetZcoord(xc, yc, -fDz);
         xc = fRmax*TMath::Cos(phi_low*kDegRad);
         yc = fRmax*TMath::Sin(phi_low*kDegRad);
         z1 = TMath::Min(z1, GetZcoord(xc, yc, -fDz));
         if (in_range_low)
            zmin = TMath::Min(zmin, z1);
         else 
            zmin = z1;
         in_range_low = kTRUE;
      }     
      phi_low += 180;
      if (phi_low>360) phi_low-=360.;
   }   

   for (i=0; i<2; i++) {
      if (phi_hi<0) phi_hi+=360.;
      Double_t dphi = fPhi2 -fPhi1;
      if (dphi < 0) dphi+=360.;
      Double_t ddp = phi_hi-fPhi1;
      if (ddp<0) ddp += 360.;
      if (ddp <= dphi) {
         xc = fRmin*TMath::Cos(phi_hi*kDegRad);
         yc = fRmin*TMath::Sin(phi_hi*kDegRad);
         z1 = GetZcoord(xc, yc, fDz);
         xc = fRmax*TMath::Cos(phi_hi*kDegRad);
         yc = fRmax*TMath::Sin(phi_hi*kDegRad);
         z1 = TMath::Max(z1, GetZcoord(xc, yc, fDz));
         if (in_range_hi) 
            zmax = TMath::Max(zmax, z1);
         else 
            zmax = z1;   
         in_range_hi = kTRUE;
      }     
      phi_hi += 180;
      if (phi_hi>360) phi_hi-=360.;
   }   


   xc = fRmin*TMath::Cos(fPhi1*kDegRad);
   yc = fRmin*TMath::Sin(fPhi1*kDegRad);
   z[0] = GetZcoord(xc, yc, -fDz);
   z[4] = GetZcoord(xc, yc, fDz);

   xc = fRmin*TMath::Cos(fPhi2*kDegRad);
   yc = fRmin*TMath::Sin(fPhi2*kDegRad);
   z[1] = GetZcoord(xc, yc, -fDz);
   z[5] = GetZcoord(xc, yc, fDz);
   
   xc = fRmax*TMath::Cos(fPhi1*kDegRad);
   yc = fRmax*TMath::Sin(fPhi1*kDegRad);
   z[2] = GetZcoord(xc, yc, -fDz);
   z[6] = GetZcoord(xc, yc, fDz);

   xc = fRmax*TMath::Cos(fPhi2*kDegRad);
   yc = fRmax*TMath::Sin(fPhi2*kDegRad); 
   z[3] = GetZcoord(xc, yc, -fDz);
   z[7] = GetZcoord(xc, yc, fDz);
   
   z1 = z[TMath::LocMin(4, &z[0])];
   if (in_range_low)
      zmin = TMath::Min(zmin, z1);
   else 
      zmin = z1;
         
   z1 = z[TMath::LocMax(4, &z[4])+4];
   if (in_range_hi) 
      zmax = TMath::Max(zmax, z1);
   else 
      zmax = z1;
         
   fDZ = 0.5*(zmax-zmin);
   fOrigin[2] = 0.5*(zmax+zmin);
}
//-----------------------------------------------------------------------------
Bool_t TGeoCtub::Contains(Double_t *point)
{
// check if point is contained in the cut tube
   // check the lower cut plane
   Double_t zin = point[0]*fNlow[0]+point[1]*fNlow[1]+(point[2]+fDz)*fNlow[2];
   if (zin>0) return kFALSE;
   // check the higher cut plane
   zin = point[0]*fNhigh[0]+point[1]*fNhigh[1]+(point[2]-fDz)*fNhigh[2];
   if (zin>0) return kFALSE;
   // check radius
   Double_t r2 = point[0]*point[0]+point[1]*point[1];
   if ((r2<fRmin*fRmin) || (r2>fRmax*fRmax)) return kFALSE;
   // check phi
   Double_t phi = TMath::ATan2(point[1], point[0]) * kRadDeg;
   if (phi < 0 ) phi+=360.;
   Double_t dphi = fPhi2 -fPhi1;
   if (dphi < 0) dphi+=360.;
   Double_t ddp = phi-fPhi1;
   if (ddp<0) ddp += 360.;
//   if (ddp>360) ddp-=360;
   if (ddp > dphi) return kFALSE;
   return kTRUE;    
}
//-----------------------------------------------------------------------------
Double_t TGeoCtub::GetZcoord(Double_t xc, Double_t yc, Double_t zc) const
{
// compute real Z coordinate of a point belonging to either lower or 
// higher caps (z should be either +fDz or -fDz)
   Double_t newz = 0;
   if (zc<0) newz =  -fDz-(xc*fNlow[0]+yc*fNlow[1])/fNlow[2];
   else      newz = fDz-(xc*fNhigh[0]+yc*fNhigh[1])/fNhigh[2];
   return newz;
}   
//-----------------------------------------------------------------------------
Double_t TGeoCtub::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from outside point to surface of the cut tube
   Double_t saf[5];
   Double_t rsq = point[0]*point[0]+point[1]*point[1];
   Double_t r = TMath::Sqrt(rsq);
   Double_t c1=0,s1=0,c2=0,s2=0,phim=0;
   Double_t fio=0, cfio=0, sfio=0, dfi=0, cdfi=0, cpsi=0;
   Double_t phi1 = fPhi1*kDegRad;
   Double_t phi2 = fPhi2*kDegRad;
   Bool_t tub = kFALSE;
   if ((fPhi2-fPhi1)==360) tub = kTRUE;
   if (!tub) {
      if (phi2<phi1) phi2+=2.*TMath::Pi();
      phim = 0.5*(phi1+phi2);
      c1 = TMath::Cos(phi1);
      c2 = TMath::Cos(phi2);
      s1 = TMath::Sin(phi1);
      s2 = TMath::Sin(phi2);
      fio = 0.5*(phi1+phi2);
      cfio = TMath::Cos(fio);
      sfio = TMath::Sin(fio);
      dfi = 0.5*(phi2-phi1);
      cdfi = TMath::Cos(dfi);
   }

   saf[0] = point[0]*fNlow[0] + point[1]*fNlow[1] + (fDz+point[2])*fNlow[2];
   saf[1] = point[0]*fNhigh[0] + point[1]*fNhigh[1] + (point[2]-fDz)*fNhigh[2];
   if (iact<3 && *safe) {
      saf[2] = fRmin-r;
      saf[3] = r-fRmax;
      if (!tub) {
         if (r>0) {
            cpsi = (point[0]*cfio+point[1]*sfio)/r;
            if (cpsi<cdfi) {
               if ((point[1]*cfio-point[0]*sfio)<0)
                  saf[4]=TMath::Abs(point[0]*s1-point[1]*c1);
               else
                  saf[4]=TMath::Abs(point[0]*s2-point[1]*c2);
            }      
         }
      } else {
         saf[4]=-kBig;
      }      
      *safe = saf[TMath::LocMax(5,&saf[0])];
      if (iact==0) return kBig;
      if ((iact==1) && (step<=*safe)) return step;
   }
   // find distance to shape
   Double_t *norm = gGeoManager->GetNormalChecked();
   Double_t r2;
   Double_t calf = dir[0]*fNlow[0]+dir[1]*fNlow[1]+dir[2]*fNlow[2];
   // check Z planes
   Double_t xi, yi, zi;
   Double_t s = kBig;
   if (saf[0]>0) {
      if (calf<0) {
         s = -saf[0]/calf;
         xi = point[0]+s*dir[0];
         yi = point[1]+s*dir[1];
         r2=xi*xi+yi*yi;
         if (((fRmin*fRmin)<=r2) && (r2<=(fRmax*fRmax))) {
            memcpy(norm, &fNlow[0], 3*sizeof(Double_t));
            if (tub) return s;
            cpsi=(xi*cfio+yi*sfio)/TMath::Sqrt(r2);
            if (cpsi>=cdfi) return s;
         }
      }
   }      
   calf = dir[0]*fNhigh[0]+dir[1]*fNhigh[1]+dir[2]*fNhigh[2];   
   if (saf[1]>0) {
      if (calf<0) {
         s = -saf[1]/calf;
         xi = point[0]+s*dir[0];
         yi = point[1]+s*dir[1];
         r2=xi*xi+yi*yi;
         if (((fRmin*fRmin)<=r2) && (r2<=(fRmax*fRmax))) {
            memcpy(norm, &fNhigh[0], 3*sizeof(Double_t));
            if (tub) return s;
            cpsi=(xi*cfio+yi*sfio)/TMath::Sqrt(r2);
            if (cpsi>=cdfi) return s;
         }
      }
   }      
   
   // check outer cyl. surface
   Double_t t1=dir[0]*dir[0]+dir[1]*dir[1];
   if (TMath::Abs(t1)<1E-32) return kBig;
   Double_t t2=point[0]*dir[0]+point[1]*dir[1];
   Double_t t3=rsq;
   Double_t b=t2/t1;
   Double_t c,d;
   // only r>fRmax has to be considered
   if (r>fRmax) {
      c=(t3-fRmax*fRmax)/t1;
      d=b*b-c;
      if (d>=0) {
         s=-b-TMath::Sqrt(d);
         if (s>=0) {
            xi=point[0]+s*dir[0];
            yi=point[1]+s*dir[1];
            zi=point[2]+s*dir[2];
            if ((-xi*fNlow[0]-yi*fNlow[1]-(zi+fDz)*fNlow[2])>0) {
               if ((-xi*fNhigh[0]-yi*fNhigh[1]+(fDz-zi)*fNhigh[2])>0) {
                  norm[0] = xi/fRmax;
                  norm[1] = yi/fRmax;
                  norm[2] = 0;
                  if (tub) return s;
                  cpsi=(xi*cfio+yi*sfio)/fRmax;
                  if (cpsi>=cdfi) return s;
               }   
            }
         }
      }
   }         
   // check inner cylinder
   Double_t snxt=kBig;
   if (fRmin>0) {
      c=(t3-fRmin*fRmin)/t1;
      d=b*b-c;
      if (d>=0) {
         s=-b+TMath::Sqrt(d);
         if (s>=0) {
            xi=point[0]+s*dir[0];
            yi=point[1]+s*dir[1];
            zi=point[2]+s*dir[2];
            if ((-xi*fNlow[0]-yi*fNlow[1]-(zi+fDz)*fNlow[2])>0) {
               if ((-xi*fNhigh[0]-yi*fNhigh[1]+(fDz-zi)*fNhigh[2])>0) {
                  norm[0] = -xi/fRmin;
                  norm[1] = -yi/fRmin;
                  norm[2] = 0;
                  if (tub) return s;                  
                  cpsi=(xi*cfio+yi*sfio)/fRmin;
                  if (cpsi>=cdfi) snxt=s;
               }   
            }
         }
      }
   }         
   // check phi planes
   if (tub) return snxt;
   Double_t un=dir[0]*s1-dir[1]*c1;
   if (un != 0) {
      s=(point[1]*c1-point[0]*s1)/un;
      if (s>=0) {
         xi=point[0]+s*dir[0];
         yi=point[1]+s*dir[1];
         zi=point[2]+s*dir[2];
         if ((-xi*fNlow[0]-yi*fNlow[1]-(zi+fDz)*fNlow[2])>0) {
            if ((-xi*fNhigh[0]-yi*fNhigh[1]+(fDz-zi)*fNhigh[2])>0) {
               r2=xi*xi+yi*yi;
               if ((fRmin*fRmin<=r2) && (r2<=fRmax*fRmax)) {
                  if ((yi*cfio-xi*sfio)<=0) {
                     if (s<snxt) {
                        snxt=s;
                        norm[0] = s1;
                        norm[1] = -c1;
                        norm[2] = 0;
                     }
                  }
               }
            }            
         }
      }
   }   
   un=dir[0]*s2-dir[1]*c2;
   if (un != 0) {
      s=(point[1]*c2-point[0]*s2)/un;
      if (s>=0) {
         xi=point[0]+s*dir[0];
         yi=point[1]+s*dir[1];
         zi=point[2]+s*dir[2];
         if ((-xi*fNlow[0]-yi*fNlow[1]-(zi+fDz)*fNlow[2])>0) {
            if ((-xi*fNhigh[0]-yi*fNhigh[1]+(fDz-zi)*fNhigh[2])>0) {
               r2=xi*xi+yi*yi;
               if ((fRmin*fRmin<=r2) && (r2<=fRmax*fRmax)) {
                  if ((yi*cfio-xi*sfio)>=0) {
                     if (s<snxt) {
                        snxt=s;
                        norm[0] = -s2;
                        norm[1] = c2;
                        norm[2] = 0;
                     }
                  }
               }
            }            
         }
      }   
   }
   return snxt;
}   
//-----------------------------------------------------------------------------
Double_t TGeoCtub::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from inside point to surface of the cut tube
   Double_t saf[5];
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t c1=0,s1=0,c2=0,s2=0,cm=0,sm=0,phim=0;
   Double_t phi1 = fPhi1*kDegRad;
   Double_t phi2 = fPhi2*kDegRad;
   Bool_t tub = kFALSE;
   if ((fPhi2-fPhi1)==360) tub = kTRUE;
   if (!tub) {
      if (phi2<phi1) phi2+=2.*TMath::Pi();
      phim = 0.5*(phi1+phi2);
      c1 = TMath::Cos(phi1);
      c2 = TMath::Cos(phi2);
      s1 = TMath::Sin(phi1);
      s2 = TMath::Sin(phi2);
      cm = TMath::Cos(phim);
      sm = TMath::Sin(phim);
   }
   if (iact<3 && safe) {
      if (fRmin>1E-10) saf[0] = r-fRmin;
      else saf[0] = kBig;
      saf[1] = fRmax-r;
      saf[2] = -point[0]*fNlow[0] - point[1]*fNlow[1] - (fDz+point[2])*fNlow[2];
      saf[3] = -point[0]*fNhigh[0] - point[1]*fNhigh[1] + (fDz-point[2])*fNhigh[2];
      if (!tub) {
         if ((point[1]*cm-point[1]*sm)<=0)
            saf[4] = TMath::Abs(point[0]*s1-point[1]*c1);
         else
            saf[4] = TMath::Abs(point[0]*s2-point[1]*c2);
      } else {
         saf[4] = kBig;
      }         
      *safe = saf[TMath::LocMin(5, &saf[0])];
      if (iact==0) return kBig;
      if ((iact==1) && (*safe>step)) return step;
   }
   // compute distance to surface 
   // Do Z
   Double_t sz = kBig;
   Double_t *norm = gGeoManager->GetNormalChecked();
   Double_t calf = dir[0]*fNlow[0]+dir[1]*fNlow[1]+dir[2]*fNlow[2];
   if (calf>0) {
      sz = saf[2]/calf;
      memcpy(norm, &fNlow[0], 3*sizeof(Double_t));
   }
   
   Double_t sz1=kBig;
   calf = dir[0]*fNhigh[0]+dir[1]*fNhigh[1]+dir[2]*fNhigh[2];   
   if (calf>0) {
      sz1 = saf[3]/calf;
      if (sz1<sz) {
         sz = sz1;
         memcpy(norm, &fNhigh[0], 3*sizeof(Double_t));
      }   
   }
         
   // Do R
   Double_t t1=dir[0]*dir[0]+dir[1]*dir[1];  
   Double_t t2=point[0]*dir[0]+point[1]*dir[1];  
   Double_t t3=point[0]*point[0]+point[1]*point[1]; 
   // track parralel to Z
   if (t1==0) return sz;
   Double_t b=t2/t1;
   Double_t sr=kBig, c=0, d=0;
   Bool_t skip_outer = kFALSE;
   // inner cylinder
   if (fRmin>1E-10) {
      c=(t3-fRmin*fRmin)/t1;
      d=b*b-c;
      if (d>=0) {
         sr=-b-TMath::Sqrt(d);
         if (sr>0)
            skip_outer = kTRUE;
      }
   }
   // outer cylinder
   if (!skip_outer) {
      c=(t3-fRmax*fRmax)/t1;
      d=TMath::Max(b*b-c, 0.);
      sr=-b+TMath::Sqrt(d);
      if (sr<0) sr=kBig;
   }
   // phi planes
   Double_t sfmin = kBig;
   if (!tub) sfmin=DistToPhiMin(point, dir, s1, c1, s2, c2, sm, cm);;
   return TMath::Min(TMath::Min(sz,sr), sfmin);      
}   
//-----------------------------------------------------------------------------
Double_t TGeoCtub::DistToSurf(Double_t *point, Double_t *dir)
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   return kBig;
}
//-----------------------------------------------------------------------------
TGeoShape *TGeoCtub::GetMakeRuntimeShape(TGeoShape *mother) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestBit(kGeoRunTimeShape)) return 0;
   if (mother->IsRunTimeShape() || !mother->TestBit(kGeoTube)) {
      Error("GetMakeRuntimeShape", "invalid mother");
      return 0;
   }
   Double_t rmin, rmax, dz;
   rmin = fRmin;
   rmax = fRmax;
   dz = fDz;
   if (fDz<0) dz=((TGeoTube*)mother)->GetDz();
   if (fRmin<0)
      rmin = ((TGeoTube*)mother)->GetRmin();
   if ((fRmax<0) || (fRmax<=fRmin))
      rmax = ((TGeoTube*)mother)->GetRmax();
   
   return (new TGeoCtub(rmin, rmax, dz, fPhi1, fPhi2, fNlow[0], fNlow[1], fNlow[2], 
                        fNhigh[0], fNhigh[1], fNhigh[2]));
}
//-----------------------------------------------------------------------------
void TGeoCtub::Draw(Option_t *option)
{
// draw this shape according to option
}
//-----------------------------------------------------------------------------
void TGeoCtub::InspectShape()
{
// print shape parameters
   printf("*** TGeoCtub parameters ***\n");
   printf("    lx = %11.5f\n", fNlow[0]);
   printf("    ly = %11.5f\n", fNlow[1]);
   printf("    lz = %11.5f\n", fNlow[2]);
   printf("    hx = %11.5f\n", fNhigh[0]);
   printf("    hy = %11.5f\n", fNhigh[1]);
   printf("    hz = %11.5f\n", fNhigh[2]);
   TGeoTubeSeg::InspectShape();
}
//-----------------------------------------------------------------------------
void TGeoCtub::NextCrossing(TGeoParamCurve *c, Double_t *point)
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoCtub::Safety(Double_t *point, Double_t *spoint, Option_t *option)
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return 0.0;
}
//-----------------------------------------------------------------------------
void TGeoCtub::SetCtubDimensions(Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2,
                   Double_t lx, Double_t ly, Double_t lz, Double_t hx, Double_t hy, Double_t hz)
{
// set dimensions of a cut tube
   SetTubsDimensions(rmin, rmax, dz, phi1, phi2);
   fNlow[0] = lx;
   fNlow[1] = ly;
   fNlow[2] = lz;
   fNhigh[0] = hx;
   fNhigh[1] = hy;
   fNhigh[2] = hz;
   ComputeBBox();
}
//-----------------------------------------------------------------------------
void TGeoCtub::SetDimensions(Double_t *param)
{
   SetCtubDimensions(param[0], param[1], param[2], param[3], param[4], param[5],
                     param[6], param[7], param[8], param[9], param[10]);
   ComputeBBox();
}
//-----------------------------------------------------------------------------
void TGeoCtub::SetPoints(Double_t *buff) const
{
// create sphere mesh points
    Double_t dz;
    Int_t j, n;
    Double_t phi, phi1, phi2, dphi;
    phi1 = fPhi1;
    phi2 = fPhi2;
    if (phi2<phi1) phi2+=360.;
    n = TGeoManager::kGeoDefaultNsegments+1;

    dphi = (phi2-phi1)/(n-1);
    dz   = fDz;

    if (buff) {
        Int_t indx = 0;

        for (j = 0; j < n; j++) {
            phi = (phi1+j*dphi)*kDegRad;
            buff[indx+6*n] = buff[indx] = fRmin * TMath::Cos(phi);
            indx++;
            buff[indx+6*n] = buff[indx] = fRmin * TMath::Sin(phi);
            indx++;
            buff[indx+6*n] = GetZcoord(buff[indx-2], buff[indx-1], dz);
            buff[indx]     = GetZcoord(buff[indx-2], buff[indx-1], -dz);
            indx++;
        }
        for (j = 0; j < n; j++) {
            phi = (phi1+j*dphi)*kDegRad;
            buff[indx+6*n] = buff[indx] = fRmax * TMath::Cos(phi);
            indx++;
            buff[indx+6*n] = buff[indx] = fRmax * TMath::Sin(phi);
            indx++;
            buff[indx+6*n]= GetZcoord(buff[indx-2], buff[indx-1], dz);
            buff[indx]    = GetZcoord(buff[indx-2], buff[indx-1], -dz);
            indx++;
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoCtub::SetPoints(Float_t *buff) const
{
// create sphere mesh points
    Double_t dz;
    Int_t j, n;
    Double_t phi, phi1, phi2, dphi;
    phi1 = fPhi1;
    phi2 = fPhi2;
    if (phi2<phi1) phi2+=360.;
    n = TGeoManager::kGeoDefaultNsegments+1;

    dphi = (phi2-phi1)/(n-1);
    dz   = fDz;

    if (buff) {
        Int_t indx = 0;

        for (j = 0; j < n; j++) {
            phi = (phi1+j*dphi)*kDegRad;
            buff[indx+6*n] = buff[indx] = fRmin * TMath::Cos(phi);
            indx++;
            buff[indx+6*n] = buff[indx] = fRmin * TMath::Sin(phi);
            indx++;
            buff[indx+6*n] = GetZcoord(buff[indx-2], buff[indx-1], dz);
            buff[indx]     = GetZcoord(buff[indx-2], buff[indx-1], -dz);
            indx++;
        }
        for (j = 0; j < n; j++) {
            phi = (phi1+j*dphi)*kDegRad;
            buff[indx+6*n] = buff[indx] = fRmax * TMath::Cos(phi);
            indx++;
            buff[indx+6*n] = buff[indx] = fRmax * TMath::Sin(phi);
            indx++;
            buff[indx+6*n]= GetZcoord(buff[indx-2], buff[indx-1], dz);
            buff[indx]    = GetZcoord(buff[indx-2], buff[indx-1], -dz);
            indx++;
        }
    }
}
