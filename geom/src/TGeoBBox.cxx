// @(#)root/geom:$Name:  $:$Id: TGeoBBox.cxx,v 1.16 2003/01/27 13:16:26 brun Exp $
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
         :TGeoShape("")
{
// Constructor
   SetBit(TGeoShape::kGeoBox);
   SetBoxDimensions(dx, dy, dz, origin);
}
//-----------------------------------------------------------------------------
TGeoBBox::TGeoBBox(const char *name, Double_t dx, Double_t dy, Double_t dz, Double_t *origin)
         :TGeoShape(name)
{
// Constructor
   SetBit(TGeoShape::kGeoBox);
   SetBoxDimensions(dx, dy, dz, origin);
}
//-----------------------------------------------------------------------------
TGeoBBox::TGeoBBox(Double_t *param)
         :TGeoShape("")
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
   Double_t mind = fDX;
   if (fDY<mind) mind=fDY;
   if (fDZ<mind) mind=fDZ;
   Double_t dx = fOrigin[0]-point[0];
   Double_t dy = fOrigin[1]-point[1];
   Double_t dz = fOrigin[2]-point[2];
   Double_t do2 = dx*dx+dy*dy+dz*dz;
   if (do2<=(mind*mind)) return kTRUE;
   Double_t rmax2 = fDX*fDX+fDY*fDY+fDZ*fDZ;
   if (do2<=rmax2) return kTRUE;
   // inside bounding sphere
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
TGeoVolume *TGeoBBox::Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                             Double_t start, Double_t step) 
{
//--- Divide this box shape belonging to volume "voldiv" into ndiv equal volumes
// called divname, from start position with the given step. Returns pointer
// to created division cell volume. In case a wrong division axis is supplied,
// returns pointer to volume to be divided.
   TGeoShape *shape;           //--- shape to be created
   TGeoVolume *vol;            //--- division volume to be created
   TGeoVolumeMulti *vmulti;    //--- generic divided volume
   TGeoPatternFinder *finder;  //--- finder to be attached
   TString opt = "";           //--- option to be attached
   Double_t end = start+ndiv*step;
   switch (iaxis) {
      case 1:                  //--- divide on X
         shape = new TGeoBBox(step/2., fDY, fDZ); 
         finder = new TGeoPatternX(voldiv, ndiv, start, end);
         opt = "X";
         break;
      case 2:                  //--- divide on Y
         shape = new TGeoBBox(fDX, step/2., fDZ); 
         finder = new TGeoPatternY(voldiv, ndiv, start, end);
         opt = "Y";
         break;
      case 3:                  //--- divide on Z
         shape = new TGeoBBox(fDX, fDY, step/2.); 
         finder = new TGeoPatternZ(voldiv, ndiv, start, end);
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
   finder->SetBasicVolume(vol);
   finder->SetDivIndex(voldiv->GetNdaughters());
   for (Int_t ic=0; ic<ndiv; ic++) {
      voldiv->AddNodeOffset(vol, ic, start+step/2.+ic*step, opt.Data());
      ((TGeoNodeOffset*)voldiv->GetNodes()->At(voldiv->GetNdaughters()-1))->SetFinder(finder);
   }
   return vmulti;
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
   Double_t newpt[3];
   memcpy(&newpt[0], point, 3*sizeof(Double_t));
   Int_t i;
   for (i=0; i<3; i++) newpt[i]-=fOrigin[i];
   saf[0] = fDX+newpt[0];
   saf[1] = fDX-newpt[0];
   saf[2] = fDY+newpt[1];
   saf[3] = fDY-newpt[1];
   saf[4] = fDZ+newpt[2];
   saf[5] = fDZ-newpt[2];
   if (iact<3 && safe) {
   // compute safe distance
      *safe = saf[TMath::LocMin(6, &saf[0])];
      if (iact==0) return kBig;
      if (iact==1 && step<*safe) return kBig;
   }
   // compute distance to surface
   Double_t s[3];
   Int_t ipl;
   for (i=0; i<3; i++) {
      if (dir[i]!=0) {
         s[i] = (dir[i]>0)?(saf[(i<<1)+1]/dir[i]):(-saf[i<<1]/dir[i]);
      } else {
         s[i] = kBig;
      }
   }
   ipl = TMath::LocMin(3, s);
   gGeoManager->SetNormalChecked(TMath::Abs(dir[ipl]));
   return s[ipl];
}
//-----------------------------------------------------------------------------
Double_t TGeoBBox::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from outside point to surface of the box
   Double_t saf[3];
   Double_t par[3];
   Double_t newpt[3];
   memcpy(&newpt[0], point, 3*sizeof(Double_t));
   Int_t i;
   for (i=0; i<3; i++) newpt[i]-=fOrigin[i];
   par[0] = fDX;
   par[1] = fDY;
   par[2] = fDZ;
   for (i=0; i<3; i++)
      saf[i] = TMath::Abs(newpt[i])-par[i];
   if (iact<3 && safe) {
      // compute safe distance
      *safe = saf[TMath::LocMax(3, saf)];
      if (iact==0) return kBig;
      if (iact==1 && step<*safe) return kBig;
   }
   // compute distance from point to box
   Double_t coord, snxt=kBig;
   Int_t ibreak=0;
   for (i=0; i<3; i++) {
      if (saf[i]<=0) continue;
      if (newpt[i]*dir[i] >= 0) continue;
      snxt = saf[i]/TMath::Abs(dir[i]);
      ibreak = 0;
      for (Int_t j=0; j<3; j++) {
         if (j==i) continue;
         coord=newpt[j]+snxt*dir[j];
         if (TMath::Abs(coord)>par[j]) {
            ibreak=1;
            break;
         }
      }
      if (!ibreak) {
         gGeoManager->SetNormalChecked(TMath::Abs(dir[i]));
         return snxt;
      }
   }
   return kBig;
}
//-----------------------------------------------------------------------------
Double_t TGeoBBox::DistToSurf(Double_t * /*point*/, Double_t * /*dir*/) const
{
// computes the distance to next surface of the box along a ray
// starting from given point to the given direction.
   return kBig;
}
//-----------------------------------------------------------------------------
const char *TGeoBBox::GetAxisName(Int_t iaxis) const
{
// Returns name of axis IAXIS.
   switch (iaxis) {
      case 1:
         return "X";
      case 2:
         return "Y";
      case 3:
         return "Z";
      default:
         return "UNDEFINED";
   }
}   

//-----------------------------------------------------------------------------
Double_t TGeoBBox::GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const
{
// Get range of shape for a given axis.
   xlo = 0;
   xhi = 0;
   Double_t dx = 0;
   switch (iaxis) {
      case 1:
         xlo = -fDX;
         xhi = fDX;
         dx = xhi-xlo;
         return dx;
      case 2:
         xlo = -fDY;
         xhi = fDY;
         dx = xhi-xlo;
         return dx;
      case 3:
         xlo = -fDZ;
         xhi = fDZ;
         dx = xhi-xlo;
         return dx;
   }
   return dx;
}         
            
//-----------------------------------------------------------------------------
void TGeoBBox::GetBoundingCylinder(Double_t *param) const
{
//--- Fill vector param[4] with the bounding cylinder parameters. The order
// is the following : Rmin, Rmax, Phi1, Phi2
   param[0] = 0.;                  // Rmin
   param[1] = fDX*fDX+fDY*fDY;     // Rmax
   param[2] = 0.;                  // Phi1
   param[3] = 360.;                // Phi2
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
void *TGeoBBox::Make3DBuffer(const TGeoVolume *vol) const
{
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return 0;
   return painter->MakeBox3DBuffer(vol);
}   
//-----------------------------------------------------------------------------
void TGeoBBox::NextCrossing(TGeoParamCurve * /*c*/, Double_t * /*point*/) const
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
void TGeoBBox::Paint(Option_t *option)
{
// paint this shape according to option
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   TGeoVolume *vol = gGeoManager->GetCurrentVolume();
   if (vol->GetShape() != (TGeoShape*)this) return;
   painter->PaintBox(this, option);
}
//-----------------------------------------------------------------------------
void TGeoBBox::PaintNext(TGeoHMatrix *glmat, Option_t *option)
{
// paint this shape according to option
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   painter->PaintBox(this, option, glmat);
}
//-----------------------------------------------------------------------------
Double_t TGeoBBox::Safety(Double_t *point, Bool_t in) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   Double_t safe = kBig;
   Double_t saf[3];
   Double_t par[3];
   par[0] = fDX;
   par[1] = fDY;
   par[2] = fDZ;
   for (Int_t i=0; i<3; i++) {
      saf[i] = par[i] - TMath::Abs(point[i]-fOrigin[i]);
      if (!in) saf[i] = -saf[i];
   }
   safe = (in)?saf[TMath::LocMin(3,saf)]:saf[TMath::LocMax(3,saf)];   
   return safe;
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
    TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
    if (painter) painter->AddSize3D(8, 12, 6);
}
