// @(#)root/geom:$Name:  $:$Id: TGeoTrd1.cxx,v 1.11 2003/01/20 14:35:48 brun Exp $
// Author: Andrei Gheata   24/10/01
// TGeoTrd1::Contains() and DistToOut() implemented by Mihaela Gheata

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
#include "TGeoTrd1.h"



 /*************************************************************************
 * TGeoTrd1 - a trapezoid with only x length varying with z. It has 4
 *   parameters, the half length in x at the low z surface, that at the
 *   high z surface, the half length in y, and in z
 *
 *************************************************************************/
//Begin_Html
/*
<img src="gif/TGeoTrd1.gif">
*/
//End_Html

ClassImp(TGeoTrd1)
   
//-----------------------------------------------------------------------------
TGeoTrd1::TGeoTrd1()
{
   // dummy ctor
   fDz = fDx1 = fDx2 = fDy = 0;
   SetBit(kGeoTrd1);
}
//-----------------------------------------------------------------------------
TGeoTrd1::TGeoTrd1(Double_t dx1, Double_t dx2, Double_t dy, Double_t dz)
         :TGeoBBox(0,0,0)
{
// constructor. 
   SetBit(kGeoTrd1);
   fDx1 = dx1;
   fDx2 = dx2;
   fDy = dy;
   fDz = dz;
   if ((dx1<0) || (dx2<0) || (dy<0) || (dz<0)) {
      SetBit(kGeoRunTimeShape);
      printf("trd1 : dx1=%f, dx2=%f, dy=%f, dz=%f\n",
              dx1,dx2,dy,dz);
   }
   else ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoTrd1::TGeoTrd1(const char *name, Double_t dx1, Double_t dx2, Double_t dy, Double_t dz)
         :TGeoBBox(name, 0,0,0)
{
// constructor. 
   SetBit(kGeoTrd1);
   fDx1 = dx1;
   fDx2 = dx2;
   fDy = dy;
   fDz = dz;
   if ((dx1<0) || (dx2<0) || (dy<0) || (dz<0)) {
      SetBit(kGeoRunTimeShape);
      printf("trd1 : dx1=%f, dx2=%f, dy=%f, dz=%f\n",
              dx1,dx2,dy,dz);
   }
   else ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoTrd1::TGeoTrd1(Double_t *param)
         :TGeoBBox(0,0,0)
{
   // ctor with an array of parameters
   // param[0] = dx1
   // param[1] = dx2
   // param[2] = dy
   // param[3] = dz
   SetBit(kGeoTrd1);
   SetDimensions(param);
   if ((fDx1<0) || (fDx2<0) || (fDy<=0) || (fDz<=0)) SetBit(kGeoRunTimeShape);
   else ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoTrd1::~TGeoTrd1()
{
// destructor
}
//-----------------------------------------------------------------------------
void TGeoTrd1::ComputeBBox()
{
// compute bounding box for a trd1
   fDX = TMath::Max(fDx1, fDx2);
   fDY = fDy;
   fDZ = fDz;
   memset(fOrigin, 0, 3*sizeof(Double_t));
}
//-----------------------------------------------------------------------------
Bool_t TGeoTrd1::Contains(Double_t *point) const
{
// test if point is inside this shape
   // check Z range
   if (TMath::Abs(point[2]) > fDz) return kFALSE;
   // then y
   if (TMath::Abs(point[1]) > fDy) return kFALSE;
   // then x
   Double_t dx = 0.5*(fDx2*(point[2]+fDz)+fDx1*(fDz-point[2]))/fDz;
   if (TMath::Abs(point[0]) > dx) return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
Double_t TGeoTrd1::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from inside point to surface of the trd1

   Double_t snxt = kBig;

   Double_t saf[6];
   //--- Compute safety first
   // check Z facettes
   saf[0] = point[2]+fDz;
   saf[1] = fDz-point[2];
   Double_t fx = 0.5*(fDx1-fDx2)/fDz;
   Double_t calf = 1./TMath::Sqrt(1.0+fx*fx);
   Double_t salf = calf*fx;
   Double_t s,cn;
   // check X facettes
   Double_t distx = 0.5*(fDx1+fDx2)-fx*point[2];
   saf[2] = (distx+point[0])*calf;
   saf[3] = (distx-point[0])*calf;
   // check Y facettes
   saf[4] = point[1]+fDy;
   saf[5] = fDy-point[1];
   if (iact<3 && safe) {
   // compute safe distance
      *safe = saf[TMath::LocMin(6, saf)];
      if (iact==0) return kBig;
      if (iact==1 && step<*safe) return kBig;
   }
   //--- Compute distance to this shape
   Double_t *norm = gGeoManager->GetNormalChecked();
   // first check if Z facettes are crossed
   cn = -dir[2];
   if (cn>0) {
      snxt = saf[0]/cn;
      norm[0] = norm[1] = 0;
      norm[2] = 1.;
   } else {
      snxt = -saf[1]/cn;             
      norm[0] = norm[1] = 0;
      norm[2] = -1.;
   }
   // now check X facettes
   cn = -calf*dir[0]+salf*dir[2];
   if (cn>0) {
      s = saf[2]/cn;
      if (s<snxt) {
         snxt = s;
         norm[0] = calf;
         norm[1] = 0;
         norm[2] = -salf;
      }
   }
   cn = calf*dir[0]+salf*dir[2];
   if (cn>0) {
      s = saf[3]/cn;
      if (s<snxt) {
         snxt = s;
         norm[0] = -calf;
         norm[1] = 0;
         norm[2] = -salf;
      }
   }
   // now check Y facettes
   cn = -dir[1];
   if (cn>0) {
      s = saf[4]/cn;
      if (s<snxt) {
         norm[0] = norm[2] = 0;
         norm[1] = 1;
         return s;
      }   
   } else {
      s = -saf[5]/cn;         
      if (s<snxt) {
         norm[0] = norm[2] = 0;
         norm[1] = -1;
         return s;
      }
   }            
   return snxt;
}
//-----------------------------------------------------------------------------
void TGeoTrd1::GetVisibleCorner(Double_t *point, Double_t *vertex, Double_t *normals) const
{
// get the most visible corner from outside point and the normals
   Double_t fx = 0.5*(fDx1-fDx2)/fDz;
   Double_t calf = 1./TMath::Sqrt(1.0+fx*fx);
   Double_t salf = calf*fx;
   // check visibility of X faces
   Double_t distx = 0.5*(fDx1+fDx2)-fx*point[2];
   memset(normals, 0, 9*sizeof(Double_t));
   TGeoTrd1 *trd1 = (TGeoTrd1*)this;
   if (point[0]>distx) {
   // hi x face visible
      trd1->SetBit(kGeoVisX);
      normals[0]=calf;
      normals[2]=salf;
   } else {   
      trd1->SetBit(kGeoVisX, kFALSE);
      normals[0]=-calf;
      normals[2]=salf;
   }
   if (point[1]>fDy) {
   // hi y face visible
      trd1->SetBit(kGeoVisY);
      normals[4]=1;
   } else {
      trd1->SetBit(kGeoVisY, kFALSE);
      normals[4]=-1;  
   }   
   if (point[2]>fDz) {
   // hi z face visible
      trd1->SetBit(kGeoVisZ);
      normals[8]=1;
   } else {
      trd1->SetBit(kGeoVisZ, kFALSE);
      normals[8]=-1;  
   }
   SetVertex(vertex);
}
//-----------------------------------------------------------------------------
void TGeoTrd1::GetOppositeCorner(Double_t * /*point*/, Int_t inorm, Double_t *vertex, Double_t *normals) const
{
// get the opposite corner of the intersected face
   TGeoTrd1 *trd1 = (TGeoTrd1*)this;
   if (inorm != 0) {
   // change x face
      trd1->SetBit(kGeoVisX, !TestBit(kGeoVisX));
      normals[0]=-normals[0];
   }
   if (inorm != 1) {
   // change y face
      trd1->SetBit(kGeoVisY, !TestBit(kGeoVisY));
      normals[4]=-normals[4];
   } 
   if (inorm != 2) {
   // hi z face visible
      trd1->SetBit(kGeoVisZ, !TestBit(kGeoVisZ));
      normals[8]=-normals[8];
   } 
   SetVertex(vertex);
}
//-----------------------------------------------------------------------------
Double_t TGeoTrd1::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from outside point to surface of the trd1
   Double_t snxt = kBig;
   // find a visible face
   Double_t *norm = gGeoManager->GetNormalChecked();
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
   Double_t calf = 1./TMath::Sqrt(1.0+fx*fx);
   Double_t salf = calf*fx;
   Double_t cn;
   // check visibility of X faces
   Double_t distx = 0.5*(fDx1+fDx2)-fx*point[2];
   if (point[0]<-distx) {
      saf[2] = (-point[0]-distx)*calf;
   }
   if (point[0]>distx) {
      saf[3] = (point[0]-distx)*calf;
   }      
   // check visibility of Y facettes
   if (point[1]<-fDy) {
      saf[4] = -fDy-point[1];
   } else {
      if (point[1]>fDy) {
         saf[5] = point[1]-fDy;
      }
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
            if (TMath::Abs(ptnew[1]) < fDy) {
               // bottom Z facette is crossed
               norm[0]=norm[1]=0;
               norm[2] = -1;
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
               if (TMath::Abs(ptnew[1]) < fDy) {
                  // top Z facette is crossed
                  norm[0]=norm[1]=0;
                  norm[2] = 1;
                  return snxt;
               }
            }
         }      
      }
   }      
   // check if X facettes are crossed
   if (saf[2]>0) {
      cn = -calf*dir[0]+salf*dir[2];
      if (cn<0) {
         snxt = -saf[2]/cn;
         // find extrapolated Y and Z
         ptnew[1] = point[1]+snxt*dir[1];
         if (TMath::Abs(ptnew[1]) < fDy) {
            ptnew[2] = point[2]+snxt*dir[2];
            if (TMath::Abs(ptnew[2]) < fDz) {
               // lower X facette is crossed
               norm[0] = -calf;
               norm[1] = 0;
               norm[2] = salf;
               return snxt;
            }
         }
      }
   }            
   if (saf[3]>0) {
      cn = calf*dir[0]+salf*dir[2];
      if (cn<0) {
         snxt = -saf[3]/cn;
         // find extrapolated Y and Z
         ptnew[1] = point[1]+snxt*dir[1];
         if (TMath::Abs(ptnew[1]) < fDy) {
            ptnew[2] = point[2]+snxt*dir[2];
            if (TMath::Abs(ptnew[2]) < fDz) {
               // lower X facette is crossed
               norm[0] = calf;
               norm[1] = 0;
               norm[2] = salf;
               return snxt;
            }
         }
      }
   }
   // finally check Y facettes
   if (saf[4]>0) {
      cn = -dir[1];            
      if (cn<0) {
         snxt = -saf[4]/cn;
         // find extrapolated X and Z
         ptnew[2] = point[2]+snxt*dir[2];
         if (TMath::Abs(ptnew[2]) < fDz) {
            ptnew[0] = point[0]+snxt*dir[0];
            distx = 0.5*(fDx1+fDx2)-fx*ptnew[2];
            if (TMath::Abs(ptnew[0]) < distx) {
               // lower Y facette is crossed
               norm[0] = norm[2] = 0;
               norm[1] = -1;
               return snxt;
            }
         }
      }
   } else {
      if (saf[5]>0) {
         cn = dir[1];            
         if (cn<0) {
            snxt = -saf[5]/cn;
            // find extrapolated X and Z
            ptnew[2] = point[2]+snxt*dir[2];
            if (TMath::Abs(ptnew[2]) < fDz) {
               ptnew[0] = point[0]+snxt*dir[0];
               distx = 0.5*(fDx1+fDx2)-fx*ptnew[2];
               if (TMath::Abs(ptnew[0]) < distx) {
                  // higher Y facette is crossed
                  norm[0] = norm[2] = 0;
                  norm[1] = 1;
                  return snxt;
               }
            }
         }
      }
   }               
   return kBig;
}
//-----------------------------------------------------------------------------
Double_t TGeoTrd1::DistToSurf(Double_t * /*point*/, Double_t * /*dir*/) const
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   return kBig;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoTrd1::Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                             Double_t start, Double_t step) 
{
//--- Divide this trd1 shape belonging to volume "voldiv" into ndiv volumes
// called divname, from start position with the given step. Returns pointer
// to created division cell volume in case of Y divisions. For Z divisions just
// return the pointer to the volume to be divided. In case a wrong 
// division axis is supplied, returns pointer to volume that was divided.
   TGeoShape *shape;           //--- shape to be created
   TGeoVolume *vol;            //--- division volume to be created
   TGeoVolumeMulti *vmulti;    //--- generic divided volume
   TGeoPatternFinder *finder;  //--- finder to be attached 
   TString opt = "";           //--- option to be attached
   Double_t zmin, zmax, dx1n, dx2n;
   Int_t id;
   Double_t end = start+ndiv*step;
   switch (iaxis) {
      case 1:
         Warning("Divide", "dividing a Trd1 on X not implemented");
         return 0;
      case 2:
         finder = new TGeoPatternY(voldiv, ndiv, start, end);
         voldiv->SetFinder(finder);
         finder->SetDivIndex(voldiv->GetNdaughters());            
         shape = new TGeoTrd1(fDx1, fDx2, step/2, fDz);
         vol = new TGeoVolume(divname, shape, voldiv->GetMedium()); 
         vmulti = gGeoManager->MakeVolumeMulti(divname, voldiv->GetMedium());
         vmulti->AddVolume(vol);
         opt = "Y";
         for (id=0; id<ndiv; id++) {
            voldiv->AddNodeOffset(vol, id, start+step/2+id*step, opt.Data());
            ((TGeoNodeOffset*)voldiv->GetNodes()->At(voldiv->GetNdaughters()-1))->SetFinder(finder);
         }
         return vmulti;
      case 3:
         finder = new TGeoPatternZ(voldiv, ndiv, start, end);
         voldiv->SetFinder(finder);
         finder->SetDivIndex(voldiv->GetNdaughters());            
         vmulti = gGeoManager->MakeVolumeMulti(divname, voldiv->GetMedium());
         for (id=0; id<ndiv; id++) {
            zmin = start+id*step;
            zmax = start+(id+1)*step;
            dx1n = 0.5*(fDx1*(fDz-zmin)+fDx2*(fDz+zmin))/fDz;
            dx2n = 0.5*(fDx1*(fDz-zmax)+fDx2*(fDz+zmax))/fDz;
            shape = new TGeoTrd1(dx1n, dx2n, fDy, step/2.);
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
Double_t TGeoTrd1::GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const
{
// Get range of shape for a given axis.
   xlo = 0;
   xhi = 0;
   Double_t dx = 0;
   switch (iaxis) {
      case 2:
         xlo = -fDy;
         xhi = fDy;
         dx = xhi-xlo;
         return dx;
      case 3:
         xlo = -fDz;
         xhi = fDz;
         dx = xhi-xlo;
         return dx;
   }
   return dx;
}         

//-----------------------------------------------------------------------------
void TGeoTrd1::GetBoundingCylinder(Double_t *param) const
{
//--- Fill vector param[4] with the bounding cylinder parameters. The order
// is the following : Rmin, Rmax, Phi1, Phi2
   TGeoBBox::GetBoundingCylinder(param);
}   
//-----------------------------------------------------------------------------
TGeoShape *TGeoTrd1::GetMakeRuntimeShape(TGeoShape *mother) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestBit(kGeoRunTimeShape)) return 0;
   if (mother->IsRunTimeShape() || !mother->TestBit(kGeoTrd1)) {
      Error("GetMakeRuntimeShape", "invalid mother");
      return 0;
   }
   Double_t dx1, dx2, dy, dz;
   if (fDx1<0) dx1=((TGeoTrd1*)mother)->GetDx1();
   else dx1=fDx1;
   if (fDx2<0) dx2=((TGeoTrd1*)mother)->GetDx2();
   else dx2=fDx2;
   if (fDy<0) dy=((TGeoTrd1*)mother)->GetDy();
   else dy=fDy;
   if (fDz<0) dz=((TGeoTrd1*)mother)->GetDz();
   else dz=fDz;

   return (new TGeoTrd1(dx1, dx2, dy, dz));
}
//-----------------------------------------------------------------------------
void TGeoTrd1::InspectShape() const
{
// print shape parameters
   printf("*** TGeoTrd1 parameters ***\n");
   printf("    dx1 = %11.5f\n", fDx1);
   printf("    dx2 = %11.5f\n", fDx2);
   printf("    dy  = %11.5f\n", fDy);
   printf("    dz  = %11.5f\n", fDz);
   TGeoBBox::InspectShape();
}
//-----------------------------------------------------------------------------
void TGeoTrd1::NextCrossing(TGeoParamCurve * /*c*/, Double_t * /*point*/) const
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoTrd1::Safety(Double_t *, Bool_t in) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return kBig;
}
//-----------------------------------------------------------------------------
void TGeoTrd1::SetDimensions(Double_t *param)
{
// set trd1 params in one step :
   fDx1 = param[0];
   fDx2 = param[1];
   fDy  = param[2];
   fDz  = param[3];
   ComputeBBox();
}   
//-----------------------------------------------------------------------------
void TGeoTrd1::SetVertex(Double_t *vertex) const
{
// set vertex of a corner according to visibility flags
   if (TestBit(kGeoVisX)) {
      if (TestBit(kGeoVisZ)) {
         vertex[0] = fDx2;
         vertex[2] = fDz;
         vertex[1] = (TestBit(kGeoVisY))?fDy:-fDy;
      } else {   
         vertex[0] = fDx1;
         vertex[2] = -fDz;
         vertex[1] = (TestBit(kGeoVisY))?fDy:-fDy;
      }
   } else {
      if (TestBit(kGeoVisZ)) {
         vertex[0] = -fDx2;
         vertex[2] = fDz;
         vertex[1] = (TestBit(kGeoVisY))?fDy:-fDy;
      } else {   
         vertex[0] = -fDx1;
         vertex[2] = -fDz;
         vertex[1] = (TestBit(kGeoVisY))?fDy:-fDy;
      }
   }            
} 
//-----------------------------------------------------------------------------
void TGeoTrd1::SetPoints(Double_t *buff) const
{
// create arb8 mesh points
   if (!buff) return;
   buff[ 0] = -fDx1; buff[ 1] = -fDy; buff[ 2] = -fDz;
   buff[ 3] = -fDx1; buff[ 4] =  fDy; buff[ 5] = -fDz;
   buff[ 6] =  fDx1; buff[ 7] =  fDy; buff[ 8] = -fDz;
   buff[ 9] =  fDx1; buff[10] = -fDy; buff[11] = -fDz;
   buff[12] = -fDx2; buff[13] = -fDy; buff[14] =  fDz;
   buff[15] = -fDx2; buff[16] =  fDy; buff[17] =  fDz;
   buff[18] =  fDx2; buff[19] =  fDy; buff[20] =  fDz;
   buff[21] =  fDx2; buff[22] = -fDy; buff[23] =  fDz;
}
//-----------------------------------------------------------------------------
void TGeoTrd1::SetPoints(Float_t *buff) const
{
// create arb8 mesh points
   if (!buff) return;
   buff[ 0] = -fDx1; buff[ 1] = -fDy; buff[ 2] = -fDz;
   buff[ 3] = -fDx1; buff[ 4] =  fDy; buff[ 5] = -fDz;
   buff[ 6] =  fDx1; buff[ 7] =  fDy; buff[ 8] = -fDz;
   buff[ 9] =  fDx1; buff[10] = -fDy; buff[11] = -fDz;
   buff[12] = -fDx2; buff[13] = -fDy; buff[14] =  fDz;
   buff[15] = -fDx2; buff[16] =  fDy; buff[17] =  fDz;
   buff[18] =  fDx2; buff[19] =  fDy; buff[20] =  fDz;
   buff[21] =  fDx2; buff[22] = -fDy; buff[23] =  fDz;
}
//-----------------------------------------------------------------------------
void TGeoTrd1::Sizeof3D() const
{
// fill size of this 3-D object
   TGeoBBox::Sizeof3D();
}
