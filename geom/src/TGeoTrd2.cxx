// @(#)root/geom:$Name:  $:$Id: TGeoTrd2.cxx,v 1.12 2003/01/24 08:38:50 brun Exp $
// Author: Andrei Gheata   31/01/02
// TGeoTrd2::Contains() and DistToOut() implemented by Mihaela Gheata

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
#include "TGeoTrd2.h"


 /*************************************************************************
 * TGeoTrd2 - a trapezoid with both x and y lengths varying with z. It 
 *   has 5 parameters, the half lengths in x at -dz and +dz, the half
 *  lengths in y at -dz and +dz, and the half length in z (dz).
 *
 *************************************************************************/
//Begin_Html
/*
<img src="gif/TGeoTrd2.gif">
*/
//End_Html

ClassImp(TGeoTrd2)
   
//-----------------------------------------------------------------------------
TGeoTrd2::TGeoTrd2()
{
   // dummy ctor
   SetBit(kGeoTrd2);
   fDz = fDx1 = fDx2 = fDy1 = fDy2 = 0;
}
//-----------------------------------------------------------------------------
TGeoTrd2::TGeoTrd2(Double_t dx1, Double_t dx2, Double_t dy1, Double_t dy2, Double_t dz)
         :TGeoBBox(0,0,0)
{
// constructor. 
   SetBit(kGeoTrd2);
   fDx1 = dx1;
   fDx2 = dx2;
   fDy1 = dy1;
   fDy2 = dy2;
   fDz = dz;
   if ((fDx1<0) || (fDx2<0) || (fDy1<0) || (fDy2<0) || (fDz<0)) {
      SetBit(kGeoRunTimeShape);
      printf("trd2 : dx1=%f, dx2=%f, dy1=%f, dy2=%f, dz=%f\n",
              dx1,dx2,dy1,dy2,dz);
   }
   else ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoTrd2::TGeoTrd2(const char * name, Double_t dx1, Double_t dx2, Double_t dy1, Double_t dy2, Double_t dz)
         :TGeoBBox(name, 0,0,0)
{
// constructor. 
   SetBit(kGeoTrd2);
   fDx1 = dx1;
   fDx2 = dx2;
   fDy1 = dy1;
   fDy2 = dy2;
   fDz = dz;
   if ((fDx1<0) || (fDx2<0) || (fDy1<0) || (fDy2<0) || (fDz<0)) {
      SetBit(kGeoRunTimeShape);
      printf("trd2 : dx1=%f, dx2=%f, dy1=%f, dy2=%f, dz=%f\n",
              dx1,dx2,dy1,dy2,dz);
   }
   else ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoTrd2::TGeoTrd2(Double_t *param)
         :TGeoBBox(0,0,0)
{
   // ctor with an array of parameters
   // param[0] = dx1
   // param[1] = dx2
   // param[2] = dy1
   // param[3] = dy2
   // param[4] = dz
   SetBit(kGeoTrd2);
   SetDimensions(param);
   if ((fDx1<0) || (fDx2<0) || (fDy1<0) || (fDy2<0) || (fDz<0)) SetBit(kGeoRunTimeShape);
   else ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoTrd2::~TGeoTrd2()
{
// destructor
}
//-----------------------------------------------------------------------------
void TGeoTrd2::ComputeBBox()
{
// compute bounding box for a trd2
   fDX = TMath::Max(fDx1, fDx2);
   fDY = TMath::Max(fDy1, fDy2);
   fDZ = fDz;
   memset(fOrigin, 0, 3*sizeof(Double_t));
}
//-----------------------------------------------------------------------------
Bool_t TGeoTrd2::Contains(Double_t *point) const
{
// test if point is inside this shape
   // check Z range
   if (TMath::Abs(point[2]) > fDz) return kFALSE;
   // then y
   Double_t dy = 0.5*(fDy2*(point[2]+fDz)+fDy1*(fDz-point[2]))/fDz;
   if (TMath::Abs(point[1]) > dy) return kFALSE;
   // then x
   Double_t dx = 0.5*(fDx2*(point[2]+fDz)+fDx1*(fDz-point[2]))/fDz;
   if (TMath::Abs(point[0]) > dx) return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
Double_t TGeoTrd2::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from inside point to surface of the trd2
   Double_t snxt = kBig;

   Double_t saf[6];
   //--- Compute safety first
   // check Z facettes
   saf[0] = point[2]+fDz;
   saf[1] = fDz-point[2];
   Double_t fx = 0.5*(fDx1-fDx2)/fDz;
   Double_t fy = 0.5*(fDy1-fDy2)/fDz;
   Double_t calfx = 1./TMath::Sqrt(1.0+fx*fx);
   Double_t salfx = calfx*fx;
   Double_t calfy = 1./TMath::Sqrt(1.0+fy*fy);
   Double_t salfy = calfy*fy;
   Double_t s,cn;
   // check X facettes
   Double_t distx = 0.5*(fDx1+fDx2)-fx*point[2];
   Double_t disty = 0.5*(fDy1+fDy2)-fy*point[2];
   saf[2] = (distx+point[0])*calfx;
   saf[3] = (distx-point[0])*calfx;
   // check Y facettes
   saf[4] = (disty+point[1])*calfy;
   saf[5] = (disty-point[1])*calfy;
   if (iact<3 && safe) {
   // compute safe distance
      *safe = saf[TMath::LocMin(6, saf)];
      if (iact==0) return kBig;
      if (iact==1 && step<*safe) return kBig;
   }
   //--- Compute distance to this shape
   // first check if Z facettes are crossed
   cn = -dir[2];
   if (cn>0) {
      gGeoManager->SetNormalChecked(cn);
      snxt = saf[0]/cn;
   } else {
      gGeoManager->SetNormalChecked(-cn);
      snxt = -saf[1]/cn;             
   }
   // now check X facettes
   cn = -calfx*dir[0]+salfx*dir[2];
   if (cn>0) {
      s = saf[2]/cn;
      if (s<snxt) {
         snxt = s;
         gGeoManager->SetNormalChecked(cn);
      }
   }
   cn = calfx*dir[0]+salfx*dir[2];
   if (cn>0) {
      s = saf[3]/cn;
      if (s<snxt) {
         snxt = s;
         gGeoManager->SetNormalChecked(cn);
      }
   }
   // now check Y facettes
   cn = -calfy*dir[1]+salfy*dir[2];
   if (cn>0) {
      s = saf[4]/cn;
      if (s<snxt) {
         snxt = s;
         gGeoManager->SetNormalChecked(cn);
      }
   }
   cn = calfy*dir[1]+salfy*dir[2];
   if (cn>0) {
      s = saf[5]/cn;
      if (s<snxt) {
         gGeoManager->SetNormalChecked(cn);
         return s;
      }
   }
   return snxt;
}
//-----------------------------------------------------------------------------
Double_t TGeoTrd2::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from outside point to surface of the trd2
   Double_t snxt = kBig;
   // find a visible face
   Double_t ptnew[3];
   Double_t saf[6];
   memset(saf, 0, 6*sizeof(Double_t));
   //--- Compute safety first
   // check visibility of Z facettes
   if (point[2]<-fDz) {
      saf[0] = -fDz-point[2];
   } else {
      if (point[2]>fDz) {
         saf[1] = point[2]-fDz;
      }
   }   
   Double_t fx = 0.5*(fDx1-fDx2)/fDz;
   Double_t calfx = 1./TMath::Sqrt(1.0+fx*fx);
   Double_t salfx = calfx*fx;
   Double_t fy = 0.5*(fDy1-fDy2)/fDz;
   Double_t calfy = 1./TMath::Sqrt(1.0+fy*fy);
   Double_t salfy = calfy*fy;
   Double_t cn;
   // check visibility of X faces
   Double_t distx = 0.5*(fDx1+fDx2)-fx*point[2];
   Double_t disty = 0.5*(fDy1+fDy2)-fy*point[2];
   if (point[0]<-distx) {
      saf[2] = (-point[0]-distx)*calfx;
   }
   if (point[0]>distx) {
      saf[3] = (point[0]-distx)*calfx;
   }      
   // check visibility of Y facettes
   if (point[1]<-disty) {
      saf[4] = (-point[1]-disty)*calfy;
   }
   if (point[1]>disty) {
      saf[5] = (point[1]-disty)*calfy;
   }
      
   if (iact<3 && safe) {
   // compute safe distance
      *safe = saf[TMath::LocMax(6, saf)];
      if (iact==0) return kBig;
      if (iact==1 && step<*safe) return kBig;
   }
   //--- Compute distance to this shape
   // first check if Z facettes are crossed
   if (saf[0]>0) {
      cn = -dir[2];
      if (cn<0) {
         snxt = -saf[0]/cn;
         // find extrapolated X and Y
         ptnew[0] = point[0]+snxt*dir[0];
         if (TMath::Abs(ptnew[0]) < fDx1) {
            ptnew[1] = point[1]+snxt*dir[1];
            if (TMath::Abs(ptnew[1]) < fDy1) {
               // bottom Z facette is crossed
               gGeoManager->SetNormalChecked(-cn);
               return snxt;
            }
         }
      }      
   } else {
      if (saf[1]>0) {
         cn = dir[2];
         if (cn<0) {
            snxt = -saf[1]/cn;
            // find extrapolated X and Y
            ptnew[0] = point[0]+snxt*dir[0];
            if (TMath::Abs(ptnew[0]) < fDx2) {
               ptnew[1] = point[1]+snxt*dir[1];
               if (TMath::Abs(ptnew[1]) < fDy2) {
                  // top Z facette is crossed
                  gGeoManager->SetNormalChecked(-cn);
                  return snxt;
               }
            }
         }      
      }
   }      
   // check if X facettes are crossed
   if (saf[2]>0) {
      cn = -calfx*dir[0]+salfx*dir[2];
      if (cn<0) {
         snxt = -saf[2]/cn;
         // find extrapolated Y and Z
         ptnew[2] = point[2]+snxt*dir[2];
         if (TMath::Abs(ptnew[2]) < fDz) {
            disty = 0.5*(fDy1+fDy2)-fy*ptnew[2];
            ptnew[1] = point[1]+snxt*dir[1];
            if (TMath::Abs(ptnew[1]) < disty) {
               // lower X facette is crossed
               gGeoManager->SetNormalChecked(-cn);
               return snxt;
            }
         }
      }
   }            
   if (saf[3]>0) {
      cn = calfx*dir[0]+salfx*dir[2];
      if (cn<0) {
         snxt = -saf[3]/cn;
         // find extrapolated Y and Z
         ptnew[2] = point[2]+snxt*dir[2];
         if (TMath::Abs(ptnew[2]) < fDz) {
            disty = 0.5*(fDy1+fDy2)-fy*ptnew[2];
            ptnew[1] = point[1]+snxt*dir[1];
            if (TMath::Abs(ptnew[1]) < disty) {
               // upper X facette is crossed
               gGeoManager->SetNormalChecked(-cn);
               return snxt;
            }
         }
      }
   }
   // finally check Y facettes
   if (saf[4]>0) {
      cn = -calfy*dir[1]+salfy*dir[2];
      if (cn<0) {
         snxt = -saf[4]/cn;
         // find extrapolated X and Z
         ptnew[2] = point[2]+snxt*dir[2];
         if (TMath::Abs(ptnew[2]) < fDz) {
            distx = 0.5*(fDx1+fDx2)-fx*ptnew[2];
            ptnew[0] = point[0]+snxt*dir[0];
            if (TMath::Abs(ptnew[0]) < distx) {
               // lower Y facette is crossed
               gGeoManager->SetNormalChecked(-cn);
               return snxt;
            }
         }
      }
   }            
   if (saf[5]>0) {
      cn = calfy*dir[1]+salfy*dir[2];
      if (cn<0) {
         snxt = -saf[5]/cn;
         // find extrapolated X and Z
         ptnew[2] = point[2]+snxt*dir[2];
         if (TMath::Abs(ptnew[2]) < fDz) {
            distx = 0.5*(fDx1+fDx2)-fx*ptnew[2];
            ptnew[0] = point[0]+snxt*dir[0];
            if (TMath::Abs(ptnew[0]) < distx) {
               // upper Y facette is crossed
               gGeoManager->SetNormalChecked(-cn);
               return snxt;
            }
         }
      }
   }
   return kBig;
}
//-----------------------------------------------------------------------------
Double_t TGeoTrd2::DistToSurf(Double_t * /*point*/, Double_t * /*dir*/) const
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   return kBig;
}

//-----------------------------------------------------------------------------
Double_t TGeoTrd2::GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const
{
// Get range of shape for a given axis.
   xlo = 0;
   xhi = 0;
   Double_t dx = 0;
   switch (iaxis) {
      case 3:
         xlo = -fDz;
         xhi = fDz;
         dx = xhi-xlo;
         return dx;
   }
   return dx;
}         
            
//-----------------------------------------------------------------------------
void TGeoTrd2::GetVisibleCorner(Double_t *point, Double_t *vertex, Double_t *normals) const
{
// get the most visible corner from outside point and the normals
   Double_t fx = 0.5*(fDx1-fDx2)/fDz;
   Double_t fy = 0.5*(fDy1-fDy2)/fDz;
   Double_t calf = 1./TMath::Sqrt(1.0+fx*fx);
   Double_t salf = calf*fx;
   Double_t cbet = 1./TMath::Sqrt(1.0+fy*fy);
   Double_t sbet = cbet*fy;
   // check visibility of X,Y faces
   Double_t distx = fDx1-fx*(fDz+point[2]);
   Double_t disty = fDy1-fy*(fDz+point[2]);
   memset(normals, 0, 9*sizeof(Double_t));
   TGeoTrd2 *trd2 = (TGeoTrd2*)this;
   if (point[0]>distx) {
   // hi x face visible
      trd2->SetBit(kGeoVisX);
      normals[0]=calf;
      normals[2]=salf;
   } else {   
      trd2->SetBit(kGeoVisX, kFALSE);
      normals[0]=-calf;
      normals[2]=salf;
   }
   if (point[1]>disty) {
   // hi y face visible
      trd2->SetBit(kGeoVisY);
      normals[4]=cbet;
      normals[5]=sbet;
   } else {
      trd2->SetBit(kGeoVisY, kFALSE);
      normals[4]=-cbet; 
      normals[5]=sbet; 
   }   
   if (point[2]>fDz) {
   // hi z face visible
      trd2->SetBit(kGeoVisZ);
      normals[8]=1;
   } else {
      trd2->SetBit(kGeoVisZ, kFALSE);
      normals[8]=-1;  
   }
   SetVertex(vertex);
}
//-----------------------------------------------------------------------------
void TGeoTrd2::GetOppositeCorner(Double_t * /*point*/, Int_t inorm, Double_t *vertex, Double_t *normals) const
{
// get the opposite corner of the intersected face
   TGeoTrd2 *trd2 = (TGeoTrd2*)this;
   if (inorm != 0) {
   // change x face
      trd2->SetBit(kGeoVisX, !TestBit(kGeoVisX));
      normals[0]=-normals[0];
   }
   if (inorm != 1) {
   // change y face
      trd2->SetBit(kGeoVisY, !TestBit(kGeoVisY));
      normals[4]=-normals[4];
   } 
   if (inorm != 2) {
   // hi z face visible
      trd2->SetBit(kGeoVisZ, !TestBit(kGeoVisZ));
      normals[8]=-normals[8];
   } 
   SetVertex(vertex);
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoTrd2::Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                             Double_t start, Double_t step) 
{
//--- Divide this trd2 shape belonging to volume "voldiv" into ndiv volumes
// called divname, from start position with the given step. Only Z divisions
// are supported. For Z divisions just return the pointer to the volume to be 
// divided. In case a wrong division axis is supplied, returns pointer to 
// volume that was divided.
   TGeoShape *shape;           //--- shape to be created
   TGeoVolume *vol;            //--- division volume to be created
   TGeoVolumeMulti *vmulti;    //--- generic divided volume
   TGeoPatternFinder *finder;  //--- finder to be attached 
   TString opt = "";           //--- option to be attached
   Double_t zmin, zmax, dx1n, dx2n, dy1n, dy2n;
   Int_t id;
   Double_t end = start+ndiv*step;
   switch (iaxis) {
      case 1:
         Warning("Divide", "dividing a Trd2 on X not implemented");
         return 0;
      case 2:
         Warning("Divide", "dividing a Trd2 on Y not implemented");
         return 0;
      case 3:
         finder = new TGeoPatternZ(voldiv, ndiv, start, end);
         vmulti = gGeoManager->MakeVolumeMulti(divname, voldiv->GetMedium());
         voldiv->SetFinder(finder);
         finder->SetDivIndex(voldiv->GetNdaughters());            
         for (id=0; id<ndiv; id++) {
            zmin = start+id*step;
            zmax = start+(id+1)*step;
            dx1n = 0.5*(fDx1*(fDz-zmin)+fDx2*(fDz+zmin))/fDz;
            dx2n = 0.5*(fDx1*(fDz-zmax)+fDx2*(fDz+zmax))/fDz;
            dy1n = 0.5*(fDy1*(fDz-zmin)+fDy2*(fDz+zmin))/fDz;
            dy2n = 0.5*(fDy1*(fDz-zmax)+fDy2*(fDz+zmax))/fDz;
            shape = new TGeoTrd2(dx1n, dx2n, dy1n, dy2n, step/2.);
            vol = new TGeoVolume(divname, shape, voldiv->GetMedium()); 
            vmulti->AddVolume(vol);
            opt = "Z";             
            voldiv->AddNodeOffset(vol, id, start+step/2+id*step, opt.Data());
            ((TGeoNodeOffset*)voldiv->GetNodes()->At(voldiv->GetNdaughters()-1))->SetFinder(finder);
         }
         return vmulti;
      default:
         Error("Divide", "Wrong axis type for division");
         return 0;
   }
}
//-----------------------------------------------------------------------------
void TGeoTrd2::GetBoundingCylinder(Double_t *param) const
{
//--- Fill vector param[4] with the bounding cylinder parameters. The order
// is the following : Rmin, Rmax, Phi1, Phi2
   TGeoBBox::GetBoundingCylinder(param);
}   
//-----------------------------------------------------------------------------
TGeoShape *TGeoTrd2::GetMakeRuntimeShape(TGeoShape *mother) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestBit(kGeoRunTimeShape)) return 0;
   if (mother->IsRunTimeShape() || !mother->TestBit(kGeoTrd2)) {
      Error("GetMakeRuntimeShape", "invalid mother");
      return 0;
   }
   Double_t dx1, dx2, dy1, dy2, dz;
   if (fDx1<0) dx1=((TGeoTrd2*)mother)->GetDx1();
   else dx1=fDx1;
   if (fDx2<0) dx2=((TGeoTrd2*)mother)->GetDx2();
   else dx2=fDx2;
   if (fDy1<0) dy1=((TGeoTrd2*)mother)->GetDy1();
   else dy1=fDy1;
   if (fDy2<0) dy2=((TGeoTrd2*)mother)->GetDy2();
   else dy2=fDy2;
   if (fDz<0) dz=((TGeoTrd2*)mother)->GetDz();
   else dz=fDz;

   return (new TGeoTrd2(dx1, dx2, dy1, dy2, dz));
}
//-----------------------------------------------------------------------------
void TGeoTrd2::InspectShape() const
{
// print shape parameters
   printf("*** TGeoTrd2 parameters ***\n");
   printf("    dx1 = %11.5f\n", fDx1);
   printf("    dx2 = %11.5f\n", fDx2);
   printf("    dy1 = %11.5f\n", fDy1);
   printf("    dy2 = %11.5f\n", fDy2);
   printf("    dz  = %11.5f\n", fDz);
   TGeoBBox::InspectShape();
}
//-----------------------------------------------------------------------------
void TGeoTrd2::NextCrossing(TGeoParamCurve * /*c*/, Double_t * /*point*/) const
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoTrd2::Safety(Double_t * /*point*/, Bool_t /*in*/) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return kBig;
}
//-----------------------------------------------------------------------------
void TGeoTrd2::SetDimensions(Double_t *param)
{
// set arb8 params in one step :
   fDx1 = param[0];
   fDx2 = param[1];
   fDy1 = param[2];
   fDy2 = param[3];
   fDz  = param[4];
   ComputeBBox();
}   
//-----------------------------------------------------------------------------
void TGeoTrd2::SetPoints(Double_t *buff) const
{
// create trd2 mesh points
   if (!buff) return;
   buff[ 0] = -fDx1; buff[ 1] = -fDy1; buff[ 2] = -fDz;
   buff[ 3] = -fDx1; buff[ 4] =  fDy1; buff[ 5] = -fDz;
   buff[ 6] =  fDx1; buff[ 7] =  fDy1; buff[ 8] = -fDz;
   buff[ 9] =  fDx1; buff[10] = -fDy1; buff[11] = -fDz;
   buff[12] = -fDx2; buff[13] = -fDy2; buff[14] =  fDz;
   buff[15] = -fDx2; buff[16] =  fDy2; buff[17] =  fDz;
   buff[18] =  fDx2; buff[19] =  fDy2; buff[20] =  fDz;
   buff[21] =  fDx2; buff[22] = -fDy2; buff[23] =  fDz;
}
//-----------------------------------------------------------------------------
void TGeoTrd2::SetPoints(Float_t *buff) const
{
// create trd2 mesh points
   if (!buff) return;
   buff[ 0] = -fDx1; buff[ 1] = -fDy1; buff[ 2] = -fDz;
   buff[ 3] = -fDx1; buff[ 4] =  fDy1; buff[ 5] = -fDz;
   buff[ 6] =  fDx1; buff[ 7] =  fDy1; buff[ 8] = -fDz;
   buff[ 9] =  fDx1; buff[10] = -fDy1; buff[11] = -fDz;
   buff[12] = -fDx2; buff[13] = -fDy2; buff[14] =  fDz;
   buff[15] = -fDx2; buff[16] =  fDy2; buff[17] =  fDz;
   buff[18] =  fDx2; buff[19] =  fDy2; buff[20] =  fDz;
   buff[21] =  fDx2; buff[22] = -fDy2; buff[23] =  fDz;
}
//-----------------------------------------------------------------------------
void TGeoTrd2::SetVertex(Double_t *vertex) const
{
// set vertex of a corner according to visibility flags
   if (TestBit(kGeoVisX)) {
      if (TestBit(kGeoVisZ)) {
         vertex[0] = fDx2;
         vertex[2] = fDz;
         vertex[1] = (TestBit(kGeoVisY))?fDy2:-fDy2;
      } else {   
         vertex[0] = fDx1;
         vertex[2] = -fDz;
         vertex[1] = (TestBit(kGeoVisY))?fDy1:-fDy1;
      }
   } else {
      if (TestBit(kGeoVisZ)) {
         vertex[0] = -fDx2;
         vertex[2] = fDz;
         vertex[1] = (TestBit(kGeoVisY))?fDy2:-fDy2;
      } else {   
         vertex[0] = -fDx1;
         vertex[2] = -fDz;
         vertex[1] = (TestBit(kGeoVisY))?fDy1:-fDy1;
      }
   }            
} 
//-----------------------------------------------------------------------------
void TGeoTrd2::Sizeof3D() const
{
// fill size of this 3-D object
   TGeoBBox::Sizeof3D();
}
