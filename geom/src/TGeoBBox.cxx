// @(#)root/geom:$Name:$:$Id:$
// Author: Andrei Gheata   24/10/01

// Contains() and DistToIn/Out() implemented by Mihaela Gheata

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
#include "TGeoBBox.h"

/*************************************************************************
 * TGeoBBox - box class. All shape primitives inherit from this, their 
 *   constructor filling automatically the parameters of the box that bounds
 *   the given shape. Defined by 6 parameters :
 *      fDX, fDY, fDZ - half lengths on X, Y and Z axis
 *      fOrigin[3]    - position of box origin
 *
 *************************************************************************/
//Begin_Html
/*
<img src="gif/TGeoBBox.gif">
*/
//End_Html

//--- Building boxes
//  ==================
//  Normally a box has to be build only with 3 parameters : dx, dy, dz
// representing the half lengths on X, Y and Z axis. In this case, the origin 
// of the box will match the one of its reference frame. The translation of the
// origin is used only by the constructors of all other shapes in order to
// define their own bounding boxes. Users should be aware that building a
// translated box that will represent a physical shape by itself will affect any
// further positioning of other shapes inside. Therefore in order to build a
// positioned box one should follow the recipe described in class TGeoNode.
//
//   Fast build of box volumes : TGeoManager::MakeBox()
//
//   See also class TGeoShape for utility methods provided by any particular 
// shape.
//------------------------------------------------------------------------------
ClassImp(TGeoBBox)
   
//-----------------------------------------------------------------------------
TGeoBBox::TGeoBBox()
{
// Default constructor
   SetBit(TGeoShape::kGeoBox);
   fDX = fDY = fDZ = 0;
   for (Int_t i=0; i<3; i++)
      fOrigin[i] = 0;
}   
//-----------------------------------------------------------------------------
TGeoBBox::TGeoBBox(Double_t dx, Double_t dy, Double_t dz, Double_t *origin)
         :TGeoShape()
{
// Constructor
   SetBit(TGeoShape::kGeoBox);
   SetBoxDimensions(dx, dy, dz, origin);
}
//-----------------------------------------------------------------------------
TGeoBBox::TGeoBBox(Double_t *param)
         :TGeoShape()
{
// constructor based on the array of parameters
// param[0] - half-length in x
// param[1] - half-length in y
// param[2] - half-length in z
   SetBit(TGeoShape::kGeoBox);
   SetDimensions(param);
}   
//-----------------------------------------------------------------------------
TGeoBBox::~TGeoBBox()
{
// Destructor
}
//-----------------------------------------------------------------------------
Bool_t TGeoBBox::CouldBeCrossed(Double_t *point, Double_t *dir) const
{
// decide fast if the bounding box could be crossed by a vector
   Double_t rmax2 = fDX*fDX+fDY*fDY+fDZ*fDZ;
   Double_t dx = fOrigin[0]-point[0];
   Double_t dy = fOrigin[1]-point[1];
   Double_t dz = fOrigin[2]-point[2];
   Double_t do2 = dx*dx+dy*dy+dz*dz;
   Double_t f2 = rmax2/do2;
   // inside bounding sphere
   if (f2>1) return kTRUE;
   Double_t doct = dx*dir[0]+dy*dir[1]+dz*dir[2];
   // leaving ray
   if (doct<=0) return kFALSE;
   if ((doct*doct)<=(do2-rmax2)) return kTRUE;
   return kFALSE;
}
//-----------------------------------------------------------------------------
Int_t TGeoBBox::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute closest distance from point px,py to each corner
   const Int_t numPoints = 8;
   return ShapeDistancetoPrimitive(numPoints, px, py);
}
//-----------------------------------------------------------------------------   
void TGeoBBox::SetBoxDimensions(Double_t dx, Double_t dy, Double_t dz, Double_t *origin)
{
// set parameters of box
   fDX = dx;
   fDY = dy;
   fDZ = dz;
   for (Int_t i=0; i<3; i++) {
      if (!origin) {
         fOrigin[i] = 0.0;
      } else {
         fOrigin[i] = origin[i];
      }
   }
   if ((fDX==0) && (fDY==0) && (fDZ==0)) return;
   if ((fDX<0) || (fDY<0) || (fDZ<0)) {
      SetBit(kGeoRunTimeShape);
//      printf("box : %f %f %f\n", fDX, fDY, fDZ);
   }
}        
//-----------------------------------------------------------------------------   
void TGeoBBox::ComputeBBox()
{
// compute bounding box - already computed in this case
}   
//-----------------------------------------------------------------------------
Bool_t TGeoBBox::Contains(Double_t *point) const
{
// test if point is inside this shape
   if (TMath::Abs(point[0]-fOrigin[0]) > fDX) return kFALSE;
   if (TMath::Abs(point[1]-fOrigin[1]) > fDY) return kFALSE;
   if (TMath::Abs(point[2]-fOrigin[2]) > fDZ) return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
Double_t TGeoBBox::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from inside point to surface of the box
   Double_t saf[6];
   saf[0] = fDX-point[0];
   saf[1] = fDX+point[0];
   saf[2] = fDY-point[1];
   saf[3] = fDY+point[1];
   saf[4] = fDZ-point[2];
   saf[5] = fDZ+point[2];
   if (iact<3 && safe) {
   // compute safe distance
      *safe = saf[TMath::LocMin(6, &saf[0])];
      if (iact==0) return kBig;
      if (iact==1 && step<*safe) return step; 
   }
                                                   // compute distance to surface
   Double_t s0, s1, s2;
   s0 = s1 = s2 = kBig;
   if (dir[0]>0) s0=saf[0]/dir[0];
   if (dir[0]<0) s0=-saf[1]/dir[0];
   if (dir[1]>0) s1=saf[2]/dir[1];
   if (dir[1]<0) s1=-saf[3]/dir[1];
   if (dir[2]>0) s2=saf[4]/dir[2];
   if (dir[2]<0) s2=-saf[5]/dir[2];
   return TMath::Min(s0, TMath::Min(s1,s2));
}
//-----------------------------------------------------------------------------
Double_t TGeoBBox::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from outside point to surface of the box
   Double_t saf[3];
   Double_t par[3]; 
   par[0] = fDX;
   par[1] = fDY;
   par[2] = fDZ;
   Int_t i;
   for (i=0; i<3; i++)
      saf[i] = TMath::Abs(point[i])-par[i];
   if (safe) {
   // compute minimum distance from point to box
      Int_t iv = 0;
      Double_t safplus[3];
      *safe = kBig;
      for (i=0; i<3; i++)
         if (saf[i]>0) safplus[iv++]=saf[i];
      if (iv==1) *safe=safplus[0];
      else {
         if (iv==2) *safe=TMath::Sqrt(safplus[0]*safplus[0]+safplus[1]*safplus[1]);
         else *safe=TMath::Sqrt(safplus[0]*safplus[0]+safplus[1]*safplus[1]+safplus[2]*safplus[2]);
      }
      if (iact==0) return kBig;
   }
   if (iact==1 && step<*safe) return step; 
   // compute distance from point to box
   for (i=0; i<3; i++) if ((point[i]*dir[i]>0)&&(saf[i]>0)) return kBig;
   Double_t smin[3], smax[3];
   for (i=0; i<3; i++) {
      if (dir[i]==0) return kBig;
      smin[i] = 0;
      smax[i] = kBig;
      if (saf[i]<0) smax[i]=par[i]/TMath::Abs(dir[i]) - point[i]/dir[i];
      else {
         smin[i] = saf[i]/TMath::Abs(dir[i]);
         smax[i] = (par[i]+TMath::Abs(point[i]))/TMath::Abs(dir[i]);
      }
   }
   Double_t smint = TMath::Max(TMath::Max(smin[0], smin[1]), smin[2]);
   Double_t smaxt = TMath::Min(TMath::Min(smax[0], smax[1]), smax[2]);
   if (smaxt<smint) return kBig;
   return smint;
}
//-----------------------------------------------------------------------------
Double_t TGeoBBox::DistToSurf(Double_t *point, Double_t *dir) const
{
// computes the distance to next surface of the box along a ray
// starting from given point to the given direction.
   return 0.0;
}
//-----------------------------------------------------------------------------
void TGeoBBox::Draw(Option_t *option)
{
// draw this shape according to option
}
//-----------------------------------------------------------------------------
TGeoShape *TGeoBBox::GetMakeRuntimeShape(TGeoShape *mother) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestBit(kGeoRunTimeShape)) return 0;
   if (mother->IsRunTimeShape() || !mother->TestBit(kGeoBox)) {
      Error("GetMakeRuntimeShape", "invalid mother");
      return 0;
   }
   Double_t dx, dy, dz;
   if (fDX<0) dx=((TGeoBBox*)mother)->GetDX();
   else dx=fDX;
   if (fDY<0) dy=((TGeoBBox*)mother)->GetDY();
   else dy=fDY;
   if (fDZ<0) dz=((TGeoBBox*)mother)->GetDZ();
   else dz=fDZ;
   return (new TGeoBBox(dx, dy, dz));
}
//-----------------------------------------------------------------------------
void TGeoBBox::InspectShape() const
{
// print shape parameters
   printf("*** TGeoBBox parameters ***\n");
   printf("    dX = %11.5f\n", fDX);
   printf("    dY = %11.5f\n", fDY);
   printf("    dZ = %11.5f\n", fDZ);
   printf("    origin: x=%11.5f y=%11.5f z=%11.5f\n", fOrigin[0], fOrigin[1], fOrigin[2]);
}
//-----------------------------------------------------------------------------
void TGeoBBox::Paint(Option_t *option)
{
// paint this shape according to option
   TVirtualGeoPainter *painter = gGeoManager->GetMakeDefPainter();
   if (!painter) return;
   TGeoVolume *vol = gGeoManager->GetCurrentVolume();
   if (vol->GetShape() != (TGeoShape*)this) return;
   painter->PaintBox(vol, option);
}
//-----------------------------------------------------------------------------
void TGeoBBox::NextCrossing(TGeoParamCurve *c, Double_t *point) const
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoBBox::Safety(Double_t *point, Double_t *spoint, Option_t *option) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return kBig;
}
//-----------------------------------------------------------------------------
void TGeoBBox::SetDimensions(Double_t *param)
{
// constructor based on the array of parameters
// param[0] - half-length in x
// param[1] - half-length in y
// param[2] - half-length in z
   if (!param) {
      Error("ctor", "null parameters");
      return;
   }
   fDX = param[0];
   fDY = param[1];
   fDZ = param[2];
   if ((fDX==0) && (fDY==0) && (fDZ==0)) return;
   if ((fDX<0) || (fDY<0) || (fDZ<0)) {
      SetBit(kGeoRunTimeShape);
//      printf("box : %f %f %f\n", fDX, fDY, fDZ);
   }
}   
//-----------------------------------------------------------------------------
void TGeoBBox::SetBoxPoints(Double_t *buff) const
{
   TGeoBBox::SetPoints(buff);
}
//-----------------------------------------------------------------------------
void TGeoBBox::SetPoints(Double_t *buff) const
{
// create box points
   if (!buff) return;
   buff[ 0] = -fDX+fOrigin[0]; buff[ 1] = -fDY+fOrigin[1]; buff[ 2] = -fDZ+fOrigin[2];
   buff[ 3] = -fDX+fOrigin[0]; buff[ 4] =  fDY+fOrigin[1]; buff[ 5] = -fDZ+fOrigin[2];
   buff[ 6] =  fDX+fOrigin[0]; buff[ 7] =  fDY+fOrigin[1]; buff[ 8] = -fDZ+fOrigin[2];
   buff[ 9] =  fDX+fOrigin[0]; buff[10] = -fDY+fOrigin[1]; buff[11] = -fDZ+fOrigin[2];
   buff[12] = -fDX+fOrigin[0]; buff[13] = -fDY+fOrigin[1]; buff[14] =  fDZ+fOrigin[2];
   buff[15] = -fDX+fOrigin[0]; buff[16] =  fDY+fOrigin[1]; buff[17] =  fDZ+fOrigin[2];
   buff[18] =  fDX+fOrigin[0]; buff[19] =  fDY+fOrigin[1]; buff[20] =  fDZ+fOrigin[2];
   buff[21] =  fDX+fOrigin[0]; buff[22] = -fDY+fOrigin[1]; buff[23] =  fDZ+fOrigin[2];
}
//-----------------------------------------------------------------------------
void TGeoBBox::SetPoints(Float_t *buff) const
{
// create box points
   if (!buff) return;
   buff[ 0] = -fDX+fOrigin[0]; buff[ 1] = -fDY+fOrigin[1]; buff[ 2] = -fDZ+fOrigin[2];
   buff[ 3] = -fDX+fOrigin[0]; buff[ 4] =  fDY+fOrigin[1]; buff[ 5] = -fDZ+fOrigin[2];
   buff[ 6] =  fDX+fOrigin[0]; buff[ 7] =  fDY+fOrigin[1]; buff[ 8] = -fDZ+fOrigin[2];
   buff[ 9] =  fDX+fOrigin[0]; buff[10] = -fDY+fOrigin[1]; buff[11] = -fDZ+fOrigin[2];
   buff[12] = -fDX+fOrigin[0]; buff[13] = -fDY+fOrigin[1]; buff[14] =  fDZ+fOrigin[2];
   buff[15] = -fDX+fOrigin[0]; buff[16] =  fDY+fOrigin[1]; buff[17] =  fDZ+fOrigin[2];
   buff[18] =  fDX+fOrigin[0]; buff[19] =  fDY+fOrigin[1]; buff[20] =  fDZ+fOrigin[2];
   buff[21] =  fDX+fOrigin[0]; buff[22] = -fDY+fOrigin[1]; buff[23] =  fDZ+fOrigin[2];
}
//-----------------------------------------------------------------------------
void TGeoBBox::Sizeof3D() const
{
// fill size of this 3-D object
    gSize3D.numPoints += 8;
    gSize3D.numSegs   += 12;
    gSize3D.numPolys  += 6;
}
