/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author :  Andrei Gheata  - date Thu 31 Jan 2002 01:47:40 PM CET
// TGeoPara::Contains() implemented by Mihaela Gheata

/*************************************************************************
 * TGeoPara - parallelipeped class. It has 6 parameters :
 *         dx, dy, dz - half lengths in X, Y, Z
 *         alpha - angle w.r.t the Y axis from center of low Y edge to
 *                 center of high Y edge [deg]
 *         theta, phi - polar and azimuthal angles of the segment between
 *                 low and high Z surfaces [deg]
 *
 *************************************************************************/
#include "TROOT.h"

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoPainter.h"
#include "TGeoPara.h"

//Begin_Html
/*
<img src="gif/TGeoPara.gif">
*/
//End_Html

ClassImp(TGeoPara)
   
//-----------------------------------------------------------------------------
TGeoPara::TGeoPara()
{
// Default constructor
   SetBit(TGeoShape::kGeoPara);
   fX = fY = fZ = 0;
   fAlpha = 0;
   fTheta = 0;
   fPhi = 0;
   fTxy = 0;
   fTxz = 0;
   fTyz = 0;
}   
//-----------------------------------------------------------------------------
TGeoPara::TGeoPara(Double_t dx, Double_t dy, Double_t dz, Double_t alpha,
                   Double_t theta, Double_t phi)
           :TGeoBBox(0, 0, 0)
{
// Default constructor specifying minimum and maximum radius
   SetBit(TGeoShape::kGeoPara);
   fX = dx;
   fY = dy;
   fZ = dz;
   fAlpha = alpha;
   fTheta = theta;
   fPhi = phi;
   fTxy = TMath::Tan(alpha*kDegRad);
   Double_t tth = TMath::Tan(theta*kDegRad);
   Double_t ph  = phi*kDegRad;
   fTxz = tth*TMath::Cos(ph);
   fTyz = tth*TMath::Sin(ph);
   if ((fX<0) || (fY<0) || (fZ<0)) {
//      printf("para : %f %f %f\n", fX, fY, fZ);
      SetBit(kGeoRunTimeShape);
   }
   else ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoPara::TGeoPara(Double_t *param)
{
// Default constructor
// param[0] = dx
// param[1] = dy
// param[2] = dz
// param[3] = alpha
// param[4] = theta
// param[5] = phi
   SetBit(TGeoShape::kGeoPara);
   SetDimensions(param);
   if ((fX<0) || (fY<0) || (fZ<0)) SetBit(kGeoRunTimeShape);
   else ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoPara::~TGeoPara()
{
// destructor
}
//-----------------------------------------------------------------------------   
void TGeoPara::ComputeBBox()
{
// compute bounding box
   Double_t dx = fX+fY*TMath::Abs(fTxy)+fZ*TMath::Abs(fTxz);
   Double_t dy = fY+fZ*TMath::Abs(fTyz);
   Double_t dz = fZ;
   TGeoBBox::SetBoxDimensions(dx, dy, dz);
   memset(fOrigin, 0, 3*sizeof(Double_t));
}   
//-----------------------------------------------------------------------------
Bool_t TGeoPara::Contains(Double_t *point)
{
// test if point is inside this sphere
   // test Z range
   if (TMath::Abs(point[2]) > fZ) return kFALSE;
   // check X and Y
   Double_t yt=point[1]-fTyz*point[2];
   if (TMath::Abs(yt) > fY) return kFALSE;
   Double_t xt=point[0]-fTxz*point[2]-fTxy*yt;
   if (TMath::Abs(xt) > fX) return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
Double_t TGeoPara::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from inside point to surface of the para
   Double_t saf[6];
   Double_t snxt = kBig;
   // distance from point to higher Z face
   saf[4] = fZ-point[2];
   // distance from point to lower Z face
   saf[5] = -fZ-point[2];

   // distance from point to center axis on Y 
   Double_t yt = point[1]-fTyz*point[2];      
   // distance from point to higher Y face 
   saf[2] = fY-yt; 
   // distance from point to lower Y face 
   saf[3] = -fY-yt; 
   // cos of angle YZ
   Double_t cty = 1.0/TMath::Sqrt(1.0+fTyz*fTyz);

   // distance from point to center axis on X 
   Double_t xt = point[0]-fTxz*point[2]-fTxy*yt;      
   // distance from point to higher X face 
   saf[0] = fX-xt; 
   // distance from point to lower X face 
   saf[1] = -fX-xt;
   // cos of angle XZ
   Double_t ctx = 1.0/TMath::Sqrt(1.0+fTxy*fTxy+fTxz*fTxz);
   if (iact<3 && safe) {
   // compute safety
      *safe = TMath::Min(saf[0]*ctx, -saf[1]*ctx);
      *safe = TMath::Min(*safe, TMath::Min(saf[2]*cty, -saf[3]*cty));
      *safe = TMath::Min(*safe, TMath::Min(saf[4], -saf[5]));
      if (iact==0) return kBig;
      if (iact==1 && step<*safe) return step; 
   }
   Double_t sn1, sn2, sn3;
   sn1 = sn2 = sn3 = kBig;
   if (dir[2]!=0) sn3=saf[4]/dir[2];
   if (sn3<0)     sn3=saf[5]/dir[2];
   
   Double_t dy = dir[1]-fTyz*dir[2];
   if (dy!=0)     sn2=saf[2]/dy;
   if (sn2<0)     sn2=saf[3]/dy;
   
   Double_t dx = dir[0]-fTxz*dir[2]-fTxy*dy;
   if (dx!=0)     sn1=saf[0]/dx;
   if (sn1<0)     sn1=saf[1]/dx;
   
   snxt = TMath::Min(sn1, TMath::Min(sn2, sn3));
   return snxt;
}
//-----------------------------------------------------------------------------
Double_t TGeoPara::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from inside point to surface of the para
//   Warning("DistToIn", "PARA TOIN");
//   Double_t snxt=kBig;
   Double_t dn31=-fZ-point[2];
   Double_t dn32=fZ-point[2];
   Double_t yt=point[1]-fTyz*point[2];
   Double_t dn21=-fY-yt;
   Double_t dn22=fY-yt;
   Double_t cty=1.0/TMath::Sqrt(1.0+fTyz*fTyz);
   
   Double_t xt=point[0]-fTxy*yt-fTxz*point[2];
   Double_t dn11=-fX-xt;
   Double_t dn12=fX-xt;
   Double_t ctx=1.0/TMath::Sqrt(1.0+fTxy*fTxy+fTxz*fTxz);
   
   Double_t sn3=dn31;
   if (sn3<0) sn3=-dn32;
   Double_t sn2=dn21*cty;
   if (sn2<0) sn2=-dn22*cty;
   Double_t sn1=dn11*ctx;
   if (sn1<0) sn1=-dn12*ctx;
   if (iact<3 && safe) {
   // compute safety
      *safe = TMath::Max(sn1, sn2);
      if (sn3>(*safe)) *safe=sn3;
      if (iact==0) return kBig;
      if (iact==1 && step<*safe) return step; 
   }
   // compute distance to PARA
//   Double_t *norm = gGeoManager->GetNormalChecked();
   Double_t swap;
//   Bool_t upx=kFALSE, upy=kFALSE, upz=kFALSE;
   // check if dir is paralel to Z planes
   if (dir[2]==0) {
      if ((dn32*dn31)>0) return kBig;
      dn31 = 0;
      dn32 = kBig;
   } else {
      dn31=dn31/dir[2];
      dn32=dn32/dir[2];
      if (dir[2]<0) {
         swap=dn31;
         dn31=dn32;
         dn32=swap;
      }   
   }
   if (dn32<0) return kBig;
   Double_t dy=dir[1]-fTyz*dir[2];
   if (dy==0) {
      if ((dn21*dn22)>0) return kBig;
      dn21=0;
      dn22=kBig;
   } else {
      dn21=dn21/dy;
      dn22=dn22/dy;
      if (dy<0) {   
         swap=dn21;
         dn21=dn22;
         dn22=swap;
      }   
   }
   if (dn22<0) return kBig;
   Double_t dx=dir[0]-fTxy*dy-fTxz*dir[2];
   if (dx==0) {
      if ((dn11*dn12)>0) return kBig;
      dn11=0;
      dn12=kBig;
   } else {
      dn11=dn11/dx;
      dn12=dn12/dx;
      if (dx<0) {   
         swap=dn11;
         dn11=dn12;
         dn12=swap;
      }   
   }
   if (dn12<0) return kBig;
   Double_t smin=TMath::Max(dn11,dn21);
   if (dn31>smin) smin=dn31;
   Double_t smax=TMath::Min(dn12,dn22);
   if (dn32<smax) smax=dn32;
   if (smax<=smin) return kBig;
   if (smin<=0) return kBig;
   return smin;
}
//-----------------------------------------------------------------------------
Double_t TGeoPara::DistToSurf(Double_t *point, Double_t *dir)
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   return 0.0;
}
//-----------------------------------------------------------------------------
void TGeoPara::Draw(Option_t *option)
{
// draw this shape according to option
}
//-----------------------------------------------------------------------------
TGeoShape *TGeoPara::GetMakeRuntimeShape(TGeoShape *mother) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestBit(kGeoRunTimeShape)) return 0;
   if (mother->IsRunTimeShape() || !mother->TestBit(kGeoPara)) {
      Error("GetMakeRuntimeShape", "invalid mother");
      return 0;
   }
   Double_t dx, dy, dz;
   if (fX<0) dx=((TGeoPara*)mother)->GetX();
   else dx=fX;
   if (fY<0) dy=((TGeoPara*)mother)->GetY();
   else dy=fY;
   if (fZ<0) dz=((TGeoPara*)mother)->GetZ();
   else dz=fZ;
   return (new TGeoPara(dx, dy, dz, fAlpha, fTheta, fPhi));
}
//-----------------------------------------------------------------------------
void TGeoPara::InspectShape()
{
// print shape parameters
   printf("*** TGeoPara parameters ***\n");
   printf("    dX = %11.5f\n", fX);
   printf("    dY = %11.5f\n", fY);
   printf("    dZ = %11.5f\n", fZ);
   printf("    alpha = %11.5f\n", fAlpha);
   printf("    theta = %11.5f\n", fTheta);
   printf("    phi   = %11.5f\n", fPhi);
   TGeoBBox::InspectShape();
}
//-----------------------------------------------------------------------------
void TGeoPara::Paint(Option_t *option)
{
// paint this shape according to option
   TGeoBBox::Paint(option);
}
//-----------------------------------------------------------------------------
void TGeoPara::NextCrossing(TGeoParamCurve *c, Double_t *point)
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoPara::Safety(Double_t *point, Double_t *spoint, Option_t *option)
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return 0.0;
}
//-----------------------------------------------------------------------------
void TGeoPara::SetDimensions(Double_t *param)
{
   fX     = param[0];
   fY     = param[1];
   fZ     = param[2];
   fAlpha = param[3];
   fTheta = param[4];
   fPhi   = param[5];
   fTxy = TMath::Tan(param[3]*kDegRad);
   Double_t tth = TMath::Tan(param[4]*kDegRad);
   Double_t ph  = param[5]*kDegRad;
   fTxz   = tth*TMath::Cos(ph);
   fTyz   = tth*TMath::Sin(ph);
}   
//-----------------------------------------------------------------------------
void TGeoPara::SetPoints(Double_t *buff) const
{
// create sphere mesh points
   if (!buff) return;
   Double_t TXY = fTxy;
   Double_t TXZ = fTxz;
   Double_t TYZ = fTyz;
   *buff++ = -fZ*TXZ-TXY*fY-fX; *buff++ = -fY-fZ*TYZ; *buff++ = -fZ;
   *buff++ = -fZ*TXZ+TXY*fY-fX; *buff++ = +fY-fZ*TYZ; *buff++ = -fZ;
   *buff++ = -fZ*TXZ+TXY*fY+fX; *buff++ = +fY-fZ*TYZ; *buff++ = -fZ;
   *buff++ = -fZ*TXZ-TXY*fY+fX; *buff++ = -fY-fZ*TYZ; *buff++ = -fZ;
   *buff++ = +fZ*TXZ-TXY*fY-fX; *buff++ = -fY+fZ*TYZ; *buff++ = +fZ;
   *buff++ = +fZ*TXZ+TXY*fY-fX; *buff++ = +fY+fZ*TYZ; *buff++ = +fZ;
   *buff++ = +fZ*TXZ+TXY*fY+fX; *buff++ = +fY+fZ*TYZ; *buff++ = +fZ;
   *buff++ = +fZ*TXZ-TXY*fY+fX; *buff++ = -fY+fZ*TYZ; *buff++ = +fZ;
}
//-----------------------------------------------------------------------------
void TGeoPara::SetPoints(Float_t *buff) const
{
// create sphere mesh points
   if (!buff) return;
   Double_t TXY = fTxy;
   Double_t TXZ = fTxz;
   Double_t TYZ = fTyz;
   *buff++ = -fZ*TXZ-TXY*fY-fX; *buff++ = -fY-fZ*TYZ; *buff++ = -fZ;
   *buff++ = -fZ*TXZ+TXY*fY-fX; *buff++ = +fY-fZ*TYZ; *buff++ = -fZ;
   *buff++ = -fZ*TXZ+TXY*fY+fX; *buff++ = +fY-fZ*TYZ; *buff++ = -fZ;
   *buff++ = -fZ*TXZ-TXY*fY+fX; *buff++ = -fY-fZ*TYZ; *buff++ = -fZ;
   *buff++ = +fZ*TXZ-TXY*fY-fX; *buff++ = -fY+fZ*TYZ; *buff++ = +fZ;
   *buff++ = +fZ*TXZ+TXY*fY-fX; *buff++ = +fY+fZ*TYZ; *buff++ = +fZ;
   *buff++ = +fZ*TXZ+TXY*fY+fX; *buff++ = +fY+fZ*TYZ; *buff++ = +fZ;
   *buff++ = +fZ*TXZ-TXY*fY+fX; *buff++ = -fY+fZ*TYZ; *buff++ = +fZ;
}
//-----------------------------------------------------------------------------
void TGeoPara::Sizeof3D() const
{
// fill size of this 3-D object
   TGeoBBox::Sizeof3D();
}

