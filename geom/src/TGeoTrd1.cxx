// @(#)root/geom:$Name:$:$Id:$
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
TGeoTrd1::TGeoTrd1(Double_t *param)
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
   Double_t snxt1 = kBig;
   Double_t close = kBig;
   Double_t close1 = kBig;
   Double_t fx = 0.5*(fDx1-fDx2)/fDz;

   Double_t normals[3*3];
   Int_t inorm, inorm1;
   memset(&normals[0], 0, 9*sizeof(Double_t));
   Double_t vertex[3];
   Double_t cldir[3], cldir1[3];
   // get hi X,Y,Z corner
   normals[0]=1./TMath::Sqrt(1.0+fx*fx);
   normals[2]=normals[0]*fx;
   normals[4]=1;
   normals[8]=1;
   vertex[0] = fDx2;
   vertex[1] = fDy;
   vertex[2] = fDz;

   if ((iact<3) && safe)
      close=TGeoShape::ClosenessToCorner(point, kTRUE, &vertex[0], &normals[0], &cldir[0]);
   if (iact!=0)
      snxt = TGeoShape::DistToCorner(point, dir, kTRUE, &vertex[0],
                                     &normals[0], inorm);
   // get the opposite corner   
   vertex[0] = -fDx1;
   vertex[1] = -fDy;
   vertex[2] = -fDz;
   normals[0]=-normals[0];
   normals[4]=-1;
   normals[8]=-1;
   if ((iact<3) && safe) {
      close1=TGeoShape::ClosenessToCorner(point, kTRUE, &vertex[0], &normals[0], &cldir1[0]);
      if (close1<close) {
         close = close1;
         memcpy(&cldir[0], &cldir1[0], 3*sizeof(Double_t));
      }
   }      
   if (safe) *safe = close;
   if (iact==0) return kBig;
   if ((iact==1) && (step<close)) return kBig;
   // compute distance to shape
   snxt1 = TGeoShape::DistToCorner(point, dir, kTRUE, &vertex[0],
                                   &normals[0], inorm1);
   if (snxt1<snxt) {
      snxt = snxt1;
      inorm = inorm1;
   }
   return snxt;   



/*
   if (iact<3 && safe) {
   // compute safe distance
      saf[1] = fDy-TMath::Abs(point[1]);
      saf[2] = fDz-TMath::Abs(point[2]);
      Double_t distx = fDx1 + fx*(fDz+point[2]) - TMath::Abs(point[0]);
      saf[0] = distx/TMath::Sqrt(1.0+fx*fx);
      *safe = TMath::Min(TMath::Min(saf[0],saf[1]), saf[2]);
      if (iact==0) return kBig;
      if (iact==1 && step<*safe) return step; 
   }
   // compute distance to surface
   //  First check Z
   Double_t zend = (dir[2]<0)?-fDz:fDz;
   if (dir[2]!=0) snxt = (zend-point[2])/dir[2];
   //  Now Y
   Double_t yend = (dir[1]<0)?-fDy:fDy;
   if (dir[1]!=0) snxt = TMath::Min(snxt,(yend-point[1])/dir[1]);
   //  Now X
   Double_t dxm = 0.5*(fDx1+fDx2);
   Double_t anum = dxm+fx*point[2]-point[0];
   Double_t deno = dir[0]-fx*dir[2];
   Double_t quot = kBig;
   if (deno!=0) {
      quot = anum/deno;
      if (quot>0) snxt=TMath::Min(snxt,quot);
   }
   anum = -fx*point[2]-point[0]-dxm;
   deno = dir[0]+fx*dir[2];
   if (deno==0) return snxt;
   quot = anum/deno;
   if (quot>0) snxt=TMath::Min(snxt,quot);
   return snxt;
*/   
}
//-----------------------------------------------------------------------------
void TGeoTrd1::GetVisibleCorner(Double_t *point, Double_t *vertex, Double_t *normals) const
{
// get the most visible corner from outside point and the normals
   Double_t fx = 0.5*(fDx1-fDx2)/fDz;
   Double_t calf = 1./TMath::Sqrt(1.0+fx*fx);
   Double_t salf = calf*fx;
   // check visibility of X faces
   Double_t distx = fDx1-fx*(fDz+point[2]);
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
void TGeoTrd1::GetOppositeCorner(Double_t *point, Int_t inorm, Double_t *vertex, Double_t *normals) const
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
   Double_t normals[3*3];
   Double_t vertex[3];
   Double_t cldir[3];
   GetVisibleCorner(point, &vertex[0], &normals[0]);
//   printf(" ivert=%i (%i %i %i)\n", ivert, (UInt_t)vis[0],(UInt_t)vis[1],(UInt_t)vis[2]); 
   Int_t inorm = -1;
   Int_t inorm1 = -1;
   Double_t close = kBig;
   if ((iact<3) && safe) 
      close=TGeoShape::ClosenessToCorner(point, kFALSE, &vertex[0], &normals[0], &cldir[0]);
   if (safe) *safe = close;
   if (iact==0) return kBig;
   if ((iact==1) && (step<close)) return kBig;
   // compute distance to shape
   snxt = TGeoShape::DistToCorner(point, dir, kFALSE, &vertex[0],
                                  &normals[0], inorm);
   if (inorm<0) 
      return kBig;  
//   return snxt;
   // second step : we have found the intersected face, given by inorm - check
   // if the opposite corner is also hit
   GetOppositeCorner(point, inorm, &vertex[0], &normals[0]);
   snxt = TGeoShape::DistToCorner(point, dir, kFALSE, &vertex[0],
                                  &normals[0], inorm1);
   if (inorm1<0) return kBig;
   if (inorm1!=inorm) {
      GetOppositeCorner(point, inorm1, &vertex[0], &normals[0]);
      snxt = TGeoShape::DistToCorner(point, dir, kFALSE, &vertex[0],
                                     &normals[0], inorm);
      if (inorm!=inorm1) return kBig;
   }
   return snxt;
}
//-----------------------------------------------------------------------------
Double_t TGeoTrd1::DistToSurf(Double_t *point, Double_t *dir) const
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   return kBig;
}
//-----------------------------------------------------------------------------
void TGeoTrd1::Draw(Option_t *option)
{
// draw this shape according to option
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
void TGeoTrd1::Paint(Option_t *option)
{
// paint this shape according to option
   TGeoBBox::Paint(option);
}
//-----------------------------------------------------------------------------
void TGeoTrd1::NextCrossing(TGeoParamCurve *c, Double_t *point) const
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoTrd1::Safety(Double_t *point, Double_t *spoint, Option_t *option) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return 0.0;
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
