/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author :  Andrei Gheata  - date Thu 31 Jan 2002 01:47:40 PM CET
// TGeoTrd2::Contains() and DistToOut() implemented by Mihaela Gheata

#include "TROOT.h"

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoPainter.h"
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
TGeoTrd2::TGeoTrd2(Double_t *param)
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
Bool_t TGeoTrd2::Contains(Double_t *point)
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
Double_t TGeoTrd2::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from inside point to surface of the trd2
   Double_t snxt = kBig;
   Double_t snxt1 = kBig;
   Double_t close = kBig;
   Double_t close1 = kBig;
   Double_t fx = 0.5*(fDx1-fDx2)/fDz;
   Double_t fy = 0.5*(fDy1-fDy2)/fDz;

   Double_t normals[3*3];
   Int_t inorm, inorm1;
   memset(&normals[0], 0, 9*sizeof(Double_t));
   Double_t vertex[3];
   Double_t cldir[3], cldir1[3];
   // get hi X,Y,Z corner
   normals[0]=1./TMath::Sqrt(1.0+fx*fx);
   normals[2]=normals[0]*fx;
   normals[4]=1./TMath::Sqrt(1.0+fy*fy);
   normals[5]=normals[4]*fy; 
   normals[8]=1;
   vertex[0] = fDx2;
   vertex[1] = fDy2;
   vertex[2] = fDz;

   if ((iact<3) && safe)
      close=TGeoShape::ClosenessToCorner(point, kTRUE, &vertex[0], &normals[0], &cldir[0]);
   if (iact!=0)
      snxt = TGeoShape::DistToCorner(point, dir, kTRUE, &vertex[0],
                                     &normals[0], inorm);
   // get the opposite corner   
   vertex[0] = -fDx1;
   vertex[1] = -fDy1;
   vertex[2] = -fDz;
   normals[0]=-normals[0];
   normals[4]=-normals[4];
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
   Double_t snxt = kBig;
   Double_t fx = (fDx2-fDx1)/(2*fDz);
   Double_t fy = (fDy2-fDy1)/(2*fDz);
   Double_t saf[3];
   if (iact<3 && safe) {
   // compute safe distance
      saf[2] = fDz-TMath::Abs(point[2]);
      Double_t distx = fDx1 + fx*(fDz+point[2]) - TMath::Abs(point[0]);
      saf[0] = distx/TMath::Sqrt(1.0+fx*fx);
      Double_t disty = fDy1 + fy*(fDz+point[2]) - TMath::Abs(point[1]);
      saf[1] = disty/TMath::Sqrt(1.0+fy*fy);
      *safe = TMath::Min(TMath::Min(saf[0],saf[1]), saf[2]);
      if (iact==0) return kBig;
      if (iact==1 && step<*safe) return step; 
   }
   // compute distance to surface
   //  First check Z
   Double_t zend = (dir[2]<0)?-fDz:fDz;
   if (dir[2]!=0) snxt = (zend-point[2])/dir[2];
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
   if (deno!=0) {;
      quot = anum/deno;
      if (quot>0) snxt=TMath::Min(snxt,quot);
   }
   //  Now Y
   Double_t dym = 0.5*(fDy1+fDy2);
   anum = dym+fy*point[2]-point[1];
   deno = dir[1]-fy*dir[2];
   if (deno!=0) {
      quot = anum/deno;
      if (quot>0) snxt=TMath::Min(snxt,quot);
   }
   anum = -fy*point[2]-point[1]-dym;
   deno = dir[1]+fy*dir[2];
   if (deno==0) return snxt;
   quot = anum/deno;
   if (quot>0) snxt=TMath::Min(snxt,quot);
   return snxt;
*/
}
//-----------------------------------------------------------------------------
Double_t TGeoTrd2::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from outside point to surface of the trd2
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
Double_t TGeoTrd2::DistToSurf(Double_t *point, Double_t *dir)
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   return 0.0;
}
//-----------------------------------------------------------------------------
void TGeoTrd2::GetVisibleCorner(Double_t *point, Double_t *vertex, Double_t *normals)
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
   if (point[0]>distx) {
   // hi x face visible
      SetBit(kGeoVisX);
      normals[0]=calf;
      normals[2]=salf;
   } else {   
      SetBit(kGeoVisX, kFALSE);
      normals[0]=-calf;
      normals[2]=salf;
   }
   if (point[1]>disty) {
   // hi y face visible
      SetBit(kGeoVisY);
      normals[4]=cbet;
      normals[5]=sbet;
   } else {
      SetBit(kGeoVisY, kFALSE);
      normals[4]=-cbet; 
      normals[5]=sbet; 
   }   
   if (point[2]>fDz) {
   // hi z face visible
      SetBit(kGeoVisZ);
      normals[8]=1;
   } else {
      SetBit(kGeoVisZ, kFALSE);
      normals[8]=-1;  
   }
   SetVertex(vertex);
}
//-----------------------------------------------------------------------------
void TGeoTrd2::GetOppositeCorner(Double_t *point, Int_t inorm, Double_t *vertex, Double_t *normals)
{
// get the opposite corner of the intersected face
   if (inorm != 0) {
   // change x face
      SetBit(kGeoVisX, !TestBit(kGeoVisX));
      normals[0]=-normals[0];
   }
   if (inorm != 1) {
   // change y face
      SetBit(kGeoVisY, !TestBit(kGeoVisY));
      normals[4]=-normals[4];
   } 
   if (inorm != 2) {
   // hi z face visible
      SetBit(kGeoVisZ, !TestBit(kGeoVisZ));
      normals[8]=-normals[8];
   } 
   SetVertex(vertex);
}
//-----------------------------------------------------------------------------
void TGeoTrd2::Draw(Option_t *option)
{
// draw this shape according to option
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
void TGeoTrd2::InspectShape()
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
void TGeoTrd2::Paint(Option_t *option)
{
// paint this shape according to option
   TGeoBBox::Paint(option);
}
//-----------------------------------------------------------------------------
void TGeoTrd2::NextCrossing(TGeoParamCurve *c, Double_t *point)
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoTrd2::Safety(Double_t *point, Double_t *spoint, Option_t *option)
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return 0.0;
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
void TGeoTrd2::SetVertex(Double_t *vertex)
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
