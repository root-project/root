// @(#)root/geom:$Name:  $:$Id: TGeoPara.cxx,v 1.18 2003/08/21 10:17:16 brun Exp $
// Author: Andrei Gheata   31/01/02
// TGeoPara::Contains() implemented by Mihaela Gheata

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_____________________________________________________________________________
// TGeoPara - parallelipeped class. It has 6 parameters :
//         dx, dy, dz - half lengths in X, Y, Z
//         alpha - angle w.r.t the Y axis from center of low Y edge to
//                 center of high Y edge [deg]
//         theta, phi - polar and azimuthal angles of the segment between
//                 low and high Z surfaces [deg]
//
//_____________________________________________________________________________
//
//Begin_Html
/*
<img src="gif/t_para.gif">
*/
//End_Html
//Begin_Html
/*
<img src="gif/t_paradivX.gif">
*/
//End_Html
//Begin_Html
/*
<img src="gif/t_paradivY.gif">
*/
//End_Html
//Begin_Html
/*
<img src="gif/t_paradivZ.gif">
*/
//End_Html

#include "TROOT.h"

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoPara.h"

ClassImp(TGeoPara)
   
//_____________________________________________________________________________
TGeoPara::TGeoPara()
{
// Default constructor
   SetShapeBit(TGeoShape::kGeoPara);
   fX = fY = fZ = 0;
   fAlpha = 0;
   fTheta = 0;
   fPhi = 0;
   fTxy = 0;
   fTxz = 0;
   fTyz = 0;
}   

//_____________________________________________________________________________
TGeoPara::TGeoPara(Double_t dx, Double_t dy, Double_t dz, Double_t alpha,
                   Double_t theta, Double_t phi)
           :TGeoBBox(0, 0, 0)
{
// Default constructor specifying minimum and maximum radius
   SetShapeBit(TGeoShape::kGeoPara);
   fX = dx;
   fY = dy;
   fZ = dz;
   fAlpha = alpha;
   fTheta = theta;
   fPhi = phi;
   fTxy = TMath::Tan(alpha*TMath::DegToRad());
   Double_t tth = TMath::Tan(theta*TMath::DegToRad());
   Double_t ph  = phi*TMath::DegToRad();
   fTxz = tth*TMath::Cos(ph);
   fTyz = tth*TMath::Sin(ph);
   if ((fX<0) || (fY<0) || (fZ<0)) {
//      printf("para : %f %f %f\n", fX, fY, fZ);
      SetShapeBit(kGeoRunTimeShape);
   }
   else ComputeBBox();
}

//_____________________________________________________________________________
TGeoPara::TGeoPara(const char *name, Double_t dx, Double_t dy, Double_t dz, Double_t alpha,
                   Double_t theta, Double_t phi)
           :TGeoBBox(name, 0, 0, 0)
{
// Default constructor specifying minimum and maximum radius
   SetShapeBit(TGeoShape::kGeoPara);
   fX = dx;
   fY = dy;
   fZ = dz;
   fAlpha = alpha;
   fTheta = theta;
   fPhi = phi;
   fTxy = TMath::Tan(alpha*TMath::DegToRad());
   Double_t tth = TMath::Tan(theta*TMath::DegToRad());
   Double_t ph  = phi*TMath::DegToRad();
   fTxz = tth*TMath::Cos(ph);
   fTyz = tth*TMath::Sin(ph);
   if ((fX<0) || (fY<0) || (fZ<0)) {
//      printf("para : %f %f %f\n", fX, fY, fZ);
      SetShapeBit(kGeoRunTimeShape);
   }
   else ComputeBBox();
}

//_____________________________________________________________________________
TGeoPara::TGeoPara(Double_t *param)
           :TGeoBBox(0, 0, 0)
{
// Default constructor
// param[0] = dx
// param[1] = dy
// param[2] = dz
// param[3] = alpha
// param[4] = theta
// param[5] = phi
   SetShapeBit(TGeoShape::kGeoPara);
   SetDimensions(param);
   if ((fX<0) || (fY<0) || (fZ<0)) SetShapeBit(kGeoRunTimeShape);
   else ComputeBBox();
}

//_____________________________________________________________________________
TGeoPara::~TGeoPara()
{
// destructor
}

//_____________________________________________________________________________   
void TGeoPara::ComputeBBox()
{
// compute bounding box
   Double_t dx = fX+fY*TMath::Abs(fTxy)+fZ*TMath::Abs(fTxz);
   Double_t dy = fY+fZ*TMath::Abs(fTyz);
   Double_t dz = fZ;
   TGeoBBox::SetBoxDimensions(dx, dy, dz);
   memset(fOrigin, 0, 3*sizeof(Double_t));
}   

//_____________________________________________________________________________   
void TGeoPara::ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm)
{
// Compute normal to closest surface from POINT. 
   Double_t saf[3];
   // distance from point to higher Z face
   saf[0] = TMath::Abs(fZ-TMath::Abs(point[2])); // Z

   Double_t yt = point[1]-fTyz*point[2];      
   saf[1] = TMath::Abs(fY-TMath::Abs(yt));       // Y
   // cos of angle YZ
   Double_t cty = 1.0/TMath::Sqrt(1.0+fTyz*fTyz);

   Double_t xt = point[0]-fTxz*point[2]-fTxy*yt;      
   saf[2] = TMath::Abs(fX-TMath::Abs(xt));       // X
   // cos of angle XZ
   Double_t ctx = 1.0/TMath::Sqrt(1.0+fTxy*fTxy+fTxz*fTxz);
   saf[2] *= ctx;
   saf[1] *= cty;
   Int_t i = TMath::LocMin(3,saf);
   switch (i) {
      case 0:
         norm[0] = norm[1] = 0;
         norm[2] = TMath::Sign(1.,dir[2]);
         return;
      case 1:
         norm[0] = 0;   
         norm[1] = cty;
         norm[2] = - fTyz*cty;
         break;
      case 2:
         norm[0] = TMath::Cos(fTheta*TMath::DegToRad())*TMath::Cos(fAlpha*TMath::DegToRad());
         norm[1] = - TMath::Cos(fTheta*TMath::DegToRad())*TMath::Sin(fAlpha*TMath::DegToRad());
         norm[2] = -TMath::Sin(fTheta*TMath::DegToRad());
   }
   if (norm[0]*dir[0]+norm[1]*dir[1]+norm[2]*dir[2]<0) {
      norm[0] = -norm[0];
      norm[1] = -norm[1];
      norm[2] = -norm[2];
   }         
}

//_____________________________________________________________________________
Bool_t TGeoPara::Contains(Double_t *point) const
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

//_____________________________________________________________________________
Double_t TGeoPara::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from inside point to surface of the para
   if (iact<3 && safe) {
   // compute safety
      *safe = Safety(point, kTRUE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big(); 
   }
   Double_t saf[6];
   Double_t snxt = TGeoShape::Big();
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
   // distance from point to center axis on X 
   Double_t xt = point[0]-fTxz*point[2]-fTxy*yt;      
   // distance from point to higher X face 
   saf[0] = fX-xt; 
   // distance from point to lower X face 
   saf[1] = -fX-xt;
   Double_t sn1, sn2, sn3;
   sn1 = sn2 = sn3 = TGeoShape::Big();
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

//_____________________________________________________________________________
Double_t TGeoPara::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from inside point to surface of the para
//   Warning("DistToIn", "PARA TOIN");
//   Double_t snxt=TGeoShape::Big();
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
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big(); 
   }
   // compute distance to PARA
   Double_t swap;
//   Bool_t upx=kFALSE, upy=kFALSE, upz=kFALSE;
   // check if dir is paralel to Z planes
   if (dir[2]==0) {
      if ((dn32*dn31)>0) return TGeoShape::Big();
      dn31 = 0;
      dn32 = TGeoShape::Big();
   } else {
      dn31=dn31/dir[2];
      dn32=dn32/dir[2];
      if (dir[2]<0) {
         swap=dn31;
         dn31=dn32;
         dn32=swap;
      }   
   }
   if (dn32<0) return TGeoShape::Big();
   Double_t dy=dir[1]-fTyz*dir[2];
   if (dy==0) {
      if ((dn21*dn22)>0) return TGeoShape::Big();
      dn21=0;
      dn22=TGeoShape::Big();
   } else {
      dn21=dn21/dy;
      dn22=dn22/dy;
      if (dy<0) {   
         swap=dn21;
         dn21=dn22;
         dn22=swap;
      }   
   }
   if (dn22<0) return TGeoShape::Big();
   Double_t dx=dir[0]-fTxy*dy-fTxz*dir[2];
   if (dx==0) {
      if ((dn11*dn12)>0) return TGeoShape::Big();
      dn11=0;
      dn12=TGeoShape::Big();
   } else {
      dn11=dn11/dx;
      dn12=dn12/dx;
      if (dx<0) {   
         swap=dn11;
         dn11=dn12;
         dn12=swap;
      }   
   }
   if (dn12<0) return TGeoShape::Big();
   Double_t smin=TMath::Max(dn11,dn21);
   if (dn31>smin) smin=dn31;
   Double_t smax=TMath::Min(dn12,dn22);
   if (dn32<smax) smax=dn32;
   if (smax<=smin) return TGeoShape::Big();
   if (smin<=0) return TGeoShape::Big();
   return smin;
}

//_____________________________________________________________________________
TGeoVolume *TGeoPara::Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                             Double_t start, Double_t step) 
{
//--- Divide this paralelipiped shape belonging to volume "voldiv" into ndiv equal volumes
// called divname, from start position with the given step. Returns pointer
// to created division cell volume. In case a wrong division axis is supplied,
// returns pointer to volume to be divided.
   TGeoShape *shape;           //--- shape to be created
   TGeoVolume *vol;            //--- division volume to be created
   TGeoVolumeMulti *vmulti;    //--- generic divided volume
   TGeoPatternFinder *finder;  //--- finder to be attached
   TString opt = "";           //--- option to be attached
   Double_t end=start+ndiv*step;
    switch (iaxis) {
      case 1:                  //--- divide on X
         shape = new TGeoPara(step/2, fY, fZ,fAlpha,fTheta, fPhi);
         finder = new TGeoPatternParaX(voldiv, ndiv, start, end);
         opt = "X";
         break;
      case 2:                  //--- divide on Y
         shape = new TGeoPara(fX, step/2, fZ, fAlpha, fTheta, fPhi);
         finder = new TGeoPatternParaY(voldiv, ndiv, start, end);
         opt = "Y";
         break;
      case 3:                  //--- divide on Z
         shape = new TGeoPara(fX, fY, step/2, fAlpha, fTheta, fPhi);
         finder = new TGeoPatternParaZ(voldiv, ndiv, start, end);
         opt = "Z";
         break;
      default:
         Error("Divide", "Wrong axis type for division");
         return 0;            
   }
   vol = new TGeoVolume(divname, shape, voldiv->GetMedium());
   vmulti = gGeoManager->MakeVolumeMulti(divname, voldiv->GetMedium());
   vmulti->AddVolume(vol);
   voldiv->SetFinder(finder);
   finder->SetDivIndex(voldiv->GetNdaughters());
   for (Int_t ic=0; ic<ndiv; ic++) {
      voldiv->AddNodeOffset(vol, ic, start+step/2.+ic*step, opt.Data());
      ((TGeoNodeOffset*)voldiv->GetNodes()->At(voldiv->GetNdaughters()-1))->SetFinder(finder);    
   }
   return vmulti;
}   

//_____________________________________________________________________________
Double_t TGeoPara::GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const
{
// Get range of shape for a given axis.
   xlo = 0;
   xhi = 0;
   Double_t dx = 0;
   switch (iaxis) {
      case 1:
         xlo = -fX;
         xhi = fX;
         dx = xhi-xlo;
         return dx;
      case 2:
         xlo = -fY;
         xhi = fY;
         dx = xhi-xlo;
         return dx;
      case 3:
         xlo = -fZ;
         xhi = fZ;
         dx = xhi-xlo;
         return dx;
   }
   return dx;
}         
            
//_____________________________________________________________________________
void TGeoPara::GetBoundingCylinder(Double_t *param) const
{
//--- Fill vector param[4] with the bounding cylinder parameters. The order
// is the following : Rmin, Rmax, Phi1, Phi2
   TGeoBBox::GetBoundingCylinder(param);
}   

//_____________________________________________________________________________
Int_t TGeoPara::GetFittingBox(const TGeoBBox *parambox, TGeoMatrix *mat, Double_t &dx, Double_t &dy, Double_t &dz) const
{
// Fills real parameters of a positioned box inside this. Returns 0 if successfull.
   dx=dy=dz=0;
   if (mat->IsRotation()) {
      Error("GetFittingBox", "cannot handle parametrized rotated volumes");
      return 1; // ### rotation not accepted ###
   }
   //--> translate the origin of the parametrized box to the frame of this box.
   Double_t origin[3];
   mat->LocalToMaster(parambox->GetOrigin(), origin);
   if (!Contains(origin)) {
      Error("GetFittingBox", "wrong matrix - parametrized box is outside this");
      return 1; // ### wrong matrix ###
   }
   //--> now we have to get the valid range for all parametrized axis
   Double_t dd[3];
   dd[0] = parambox->GetDX();
   dd[1] = parambox->GetDY();
   dd[2] = parambox->GetDZ();
   //-> check if Z range is fixed
   if (dd[2]<0) {
      dd[2] = TMath::Min(origin[2]+fZ, fZ-origin[2]); 
      if (dd[2]<0) {
         Error("GetFittingBox", "wrong matrix");
         return 1;
      }
   }
   if (dd[0]>=0 && dd[1]>=0) {
      dx = dd[0];
      dy = dd[1];
      dz = dd[2];
      return 0;
   }
   //-> check now range at Z = origin[2] +/- dd[2]
   Double_t upper[8];
   Double_t lower[8];
   Double_t z=origin[2]-dd[2];
   lower[0]=z*fTxz-fTxy*fY-fX; 
   lower[1]=-fY+z*fTyz;
   lower[2]=z*fTxz+fTxy*fY-fX; 
   lower[3]=fY+z*fTyz;
   lower[4]=z*fTxz+fTxy*fY+fX; 
   lower[5]=fY+z*fTyz;
   lower[6]=z*fTxz-fTxy*fY+fX; 
   lower[7]=-fY+z*fTyz;
   z=origin[2]+dd[2];
   upper[0]=z*fTxz-fTxy*fY-fX; 
   upper[1]=-fY+z*fTyz;
   upper[2]=z*fTxz+fTxy*fY-fX; 
   upper[3]=fY+z*fTyz;
   upper[4]=z*fTxz+fTxy*fY+fX; 
   upper[5]=fY+z*fTyz;
   upper[6]=z*fTxz-fTxy*fY+fX; 
   upper[7]=-fY+z*fTyz;
   
   Double_t ddmin=TGeoShape::Big();
   for (Int_t iaxis=0; iaxis<2; iaxis++) {
      if (dd[iaxis]>=0) continue;
      ddmin=TGeoShape::Big();
      for (Int_t ivert=0; ivert<4; ivert++) {
         ddmin = TMath::Min(ddmin, TMath::Abs(origin[iaxis]-lower[2*ivert+iaxis]));
         ddmin = TMath::Min(ddmin, TMath::Abs(origin[iaxis]-upper[2*ivert+iaxis]));
      }
      dd[iaxis] = ddmin;
   }
   dx = dd[0];
   dy = dd[1];
   dz = dd[2];
   return 0;
}   

//_____________________________________________________________________________
TGeoShape *TGeoPara::GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix * /*mat*/) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestShapeBit(kGeoRunTimeShape)) return 0;
   if (!mother->TestShapeBit(kGeoPara)) {
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

//_____________________________________________________________________________
void TGeoPara::InspectShape() const
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

//_____________________________________________________________________________
Double_t TGeoPara::Safety(Double_t *point, Bool_t in) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   Double_t saf[3];
   // distance from point to higher Z face
   saf[0] = fZ-TMath::Abs(point[2]); // Z

   Double_t yt = point[1]-fTyz*point[2];      
   saf[1] = fY-TMath::Abs(yt);       // Y
   // cos of angle YZ
   Double_t cty = 1.0/TMath::Sqrt(1.0+fTyz*fTyz);

   Double_t xt = point[0]-fTxz*point[2]-fTxy*yt;      
   saf[2] = fX-TMath::Abs(xt);       // X
   // cos of angle XZ
   Double_t ctx = 1.0/TMath::Sqrt(1.0+fTxy*fTxy+fTxz*fTxz);
   saf[2] *= ctx;
   saf[1] *= cty;
   if (in) return saf[TMath::LocMin(3,saf)];
   for (Int_t i=0; i<3; i++) saf[i]=-saf[i];
   return saf[TMath::LocMax(3,saf)];
}

//_____________________________________________________________________________
void TGeoPara::SetDimensions(Double_t *param)
{
   fX     = param[0];
   fY     = param[1];
   fZ     = param[2];
   fAlpha = param[3];
   fTheta = param[4];
   fPhi   = param[5];
   fTxy = TMath::Tan(param[3]*TMath::DegToRad());
   Double_t tth = TMath::Tan(param[4]*TMath::DegToRad());
   Double_t ph  = param[5]*TMath::DegToRad();
   fTxz   = tth*TMath::Cos(ph);
   fTyz   = tth*TMath::Sin(ph);
}   

//_____________________________________________________________________________
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

//_____________________________________________________________________________
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

//_____________________________________________________________________________
void TGeoPara::Sizeof3D() const
{
// fill size of this 3-D object
   TGeoBBox::Sizeof3D();
}

