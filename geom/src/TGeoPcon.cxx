// @(#)root/geom:$Name:  $:$Id: TGeoPcon.cxx,v 1.16 2003/02/07 13:46:47 brun Exp $
// Author: Andrei Gheata   24/10/01
// TGeoPcon::Contains() implemented by Mihaela Gheata

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
#include "TGeoTube.h"
#include "TGeoCone.h"
#include "TGeoPcon.h"


/*************************************************************************
 * TGeoPcon - a polycone. It has at least 9 parameters :
 *            - the lower phi limit;
 *            - the range in phi;
 *            - the number of z planes (at least two) where the inner/outer 
 *              radii are changing;
 *            - z coordinate, inner and outer radius for each z plane
 *
 *************************************************************************/
//Begin_Html
/*
<img src="gif/TGeoPcon.gif">
*/
//End_Html

ClassImp(TGeoPcon)

//-----------------------------------------------------------------------------
TGeoPcon::TGeoPcon()
{
// dummy ctor
   SetBit(TGeoShape::kGeoPcon);
   fRmin = 0;
   fRmax = 0;
   fZ    = 0;
}   
//-----------------------------------------------------------------------------
TGeoPcon::TGeoPcon(Double_t phi, Double_t dphi, Int_t nz)
         :TGeoBBox(0, 0, 0)
{
// Default constructor
   SetBit(TGeoShape::kGeoPcon);
   fPhi1 = phi;
   if (fPhi1<0) fPhi1+=360.;
   fDphi = dphi;
   fNz   = nz;
   fRmin = new Double_t [nz];
   fRmax = new Double_t [nz];
   fZ    = new Double_t [nz];
}
//-----------------------------------------------------------------------------
TGeoPcon::TGeoPcon(const char *name, Double_t phi, Double_t dphi, Int_t nz)
         :TGeoBBox(name, 0, 0, 0)
{
// Default constructor
   SetBit(TGeoShape::kGeoPcon);
   fPhi1 = phi;
   if (fPhi1<0) fPhi1+=360.;
   fDphi = dphi;
   fNz   = nz;
   fRmin = new Double_t [nz];
   fRmax = new Double_t [nz];
   fZ    = new Double_t [nz];
}
//-----------------------------------------------------------------------------
TGeoPcon::TGeoPcon(Double_t *param)
         :TGeoBBox(0, 0, 0)
{
// Default constructor in GEANT3 style
// param[0] = phi1
// param[1] = dphi
// param[2] = nz
//
// param[3] = z1
// param[4] = Rmin1
// param[5] = Rmax1
// ...
   SetBit(TGeoShape::kGeoPcon);
   SetDimensions(param);
   ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoPcon::~TGeoPcon()
{
// destructor
   if (fRmin) {delete[] fRmin; fRmin = 0;}
   if (fRmax) {delete[] fRmax; fRmax = 0;}
   if (fZ)    {delete[] fZ; fZ = 0;}
}
//-----------------------------------------------------------------------------   
void TGeoPcon::ComputeBBox()
{
// compute bounding box of the pcon
   Double_t zmin = TMath::Min(fZ[0], fZ[fNz-1]);
   Double_t zmax = TMath::Max(fZ[0], fZ[fNz-1]);
   // find largest rmax an smallest rmin
   Double_t rmin, rmax;
   rmin = fRmin[TMath::LocMin(fNz, fRmin)];
   rmax = fRmax[TMath::LocMax(fNz, fRmax)];
   Double_t phi1 = fPhi1;
   Double_t phi2 = phi1 + fDphi;
   
   Double_t xc[4];
   Double_t yc[4];
   xc[0] = rmax*TMath::Cos(phi1*kDegRad);
   yc[0] = rmax*TMath::Sin(phi1*kDegRad);
   xc[1] = rmax*TMath::Cos(phi2*kDegRad);
   yc[1] = rmax*TMath::Sin(phi2*kDegRad);
   xc[2] = rmin*TMath::Cos(phi1*kDegRad);
   yc[2] = rmin*TMath::Sin(phi1*kDegRad);
   xc[3] = rmin*TMath::Cos(phi2*kDegRad);
   yc[3] = rmin*TMath::Sin(phi2*kDegRad);

   Double_t xmin = xc[TMath::LocMin(4, &xc[0])];
   Double_t xmax = xc[TMath::LocMax(4, &xc[0])]; 
   Double_t ymin = yc[TMath::LocMin(4, &yc[0])]; 
   Double_t ymax = yc[TMath::LocMax(4, &yc[0])];

   Double_t ddp = -phi1;
   if (ddp<0) ddp+= 360;
   if (ddp<=fDphi) xmax = rmax;
   ddp = 90-phi1;
   if (ddp<0) ddp+= 360;
   if (ddp<=fDphi) ymax = rmax;
   ddp = 180-phi1;
   if (ddp<0) ddp+= 360;
   if (ddp<=fDphi) xmin = -rmax;
   ddp = 270-phi1;
   if (ddp<0) ddp+= 360;
   if (ddp<=fDphi) ymin = -rmax;
   fOrigin[0] = (xmax+xmin)/2;
   fOrigin[1] = (ymax+ymin)/2;
   fOrigin[2] = (zmax+zmin)/2;
   fDX = (xmax-xmin)/2;
   fDY = (ymax-ymin)/2;
   fDZ = (zmax-zmin)/2;
}   
//-----------------------------------------------------------------------------
Bool_t TGeoPcon::Contains(Double_t *point) const
{
// test if point is inside this shape
   // check total z range
   if ((point[2]<fZ[0]) || (point[2]>fZ[fNz-1])) return kFALSE;
   // check R squared
   Double_t r2 = point[0]*point[0]+point[1]*point[1];
   
   Int_t izl = 0;
   Int_t izh = fNz-1;
   Int_t izt = (fNz-1)/2;
   while ((izh-izl)>1) {
      if (point[2] > fZ[izt]) izl = izt;     
      else izh = izt;
      izt = (izl+izh)>>1;
   }
   // the point is in the section bounded by izl and izh Z planes
   
   // compute Rmin and Rmax and test the value of R squared
   Double_t rmin, rmax;  
   if ((fZ[izl]==fZ[izh]) && (point[2]==fZ[izl])) {
      rmin = TMath::Min(fRmin[izl], fRmin[izh]);
      rmax = TMath::Max(fRmax[izl], fRmax[izh]);
   } else {
      Double_t dz = fZ[izh] - fZ[izl];
      Double_t dz1 = point[2] - fZ[izl];
      rmin = (fRmin[izl]*(dz-dz1)+fRmin[izh]*dz1)/dz;
      rmax = (fRmax[izl]*(dz-dz1)+fRmax[izh]*dz1)/dz;
   }
   if ((r2<rmin*rmin) || (r2>rmax*rmax)) return kFALSE;
   // now check phi 
   if (fDphi==360) return kTRUE;
   if (r2<1E-10) return kTRUE;
   Double_t phi = TMath::ATan2(point[1], point[0]) * kRadDeg;
   if (phi < 0) phi+=360.0;
   Double_t ddp = phi-fPhi1;
   if (ddp<0) ddp+=360.;
   if (ddp<=fDphi) return kTRUE;
   return kFALSE;
}
//-----------------------------------------------------------------------------
Int_t TGeoPcon::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute closest distance from point px,py to each corner
   Int_t n = gGeoManager->GetNsegments()+1;
   const Int_t numPoints = 2*n*fNz;
   return ShapeDistancetoPrimitive(numPoints, px, py);
}
//-----------------------------------------------------------------------------
Double_t TGeoPcon::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from inside point to surface of the polycone
   if (iact<3 && safe) {
      *safe = Safety(point, kTRUE);
      if (iact==0) return kBig;
      if ((iact==1) && (*safe>step)) return kBig;
   }
   Double_t snxt = kBig;
   // determine which z segment contains the point
   Int_t ipl = TMath::BinarySearch(fNz, fZ, point[2]);
   if (ipl==(fNz-1)) ipl--;
   Double_t dz = 0.5*(fZ[ipl+1]-fZ[ipl]);
   // determine if the current segment is a tube or a cone
   Bool_t intub = kTRUE;
   if (fRmin[ipl]!=fRmin[ipl+1]) intub=kFALSE;
   else if (fRmax[ipl]!=fRmax[ipl+1]) intub=kFALSE;
   // determine phi segmentation
   Bool_t inphi=kTRUE;
   if (fDphi==360) inphi=kFALSE;     
   Double_t point_new[3];
   memcpy(point_new, point, 2*sizeof(Double_t));
   // new point in reference system of the current segment
   point_new[2] = point[2]-0.5*(fZ[ipl]+fZ[ipl+1]);
   
   Double_t phi1 = fPhi1;
   if (phi1<0) phi1+=360.;
   Double_t phi2 = phi1+fDphi;
   Double_t phim = 0.5*(phi1+phi2);
   Double_t c1 = TMath::Cos(phi1*kDegRad);
   Double_t s1 = TMath::Sin(phi1*kDegRad);
   Double_t c2 = TMath::Cos(phi2*kDegRad);
   Double_t s2 = TMath::Sin(phi2*kDegRad);
   Double_t cm = TMath::Cos(phim*kDegRad);
   Double_t sm = TMath::Sin(phim*kDegRad);
   if (intub) {
      if (inphi) snxt=TGeoTubeSeg::DistToOutS(point_new, dir, fRmin[ipl], fRmax[ipl],dz, c1,s1,c2,s2,cm,sm); 
      else snxt=TGeoTube::DistToOutS(point_new, dir, fRmin[ipl], fRmax[ipl],dz);
   } else {
      if (inphi) snxt=TGeoConeSeg::DistToOutS(point_new, dir, dz, fRmin[ipl], fRmax[ipl], fRmin[ipl+1], fRmax[ipl+1], phi1,phi2);
      else snxt=TGeoCone::DistToOutS(point_new,dir,dz,fRmin[ipl],fRmax[ipl],fRmin[ipl+1], fRmax[ipl+1]);
   }                              

   for (Int_t i=0; i<3; i++) point_new[i]=point[i]+(snxt+1E-6)*dir[i];
   if (!Contains(&point_new[0])) return snxt;
   
   snxt += DistToOut(&point_new[0], dir, 3) + 1E-6;
   return snxt;
}
//-----------------------------------------------------------------------------
Double_t TGeoPcon::DistToSegZ(Double_t *point, Double_t *dir, Int_t &iz, Double_t c1, Double_t s1, 
                              Double_t c2, Double_t s2, Double_t cfio, Double_t sfio, Double_t cdfi) const
{
// compute distance to a pcon Z slice. Segment iz must be valid
   Double_t zmin=fZ[iz];
   Double_t zmax=fZ[iz+1];
   if (zmin==zmax) {
      if (dir[2]==0) return kBig;
      Int_t istep=(dir[2]>0)?1:-1;
      iz+=istep;
      if (iz<0 || iz>(fNz-2)) return kBig;
      return DistToSegZ(point,dir,iz,c1,s1,c2,s2,cfio,sfio,cdfi);
   }
   Double_t dz=0.5*(zmax-zmin);
   Double_t local[3];
   memcpy(&local[0], point, 3*sizeof(Double_t));
   local[2]=point[2]-0.5*(zmin+zmax);
   Double_t snxt;
   Double_t rmin1=fRmin[iz];
   Double_t rmax1=fRmax[iz];
   Double_t rmin2=fRmin[iz+1];
   Double_t rmax2=fRmax[iz+1];
   Bool_t is_seg=(fDphi==360)?kFALSE:kTRUE;
   
   Double_t phi1 = fPhi1;
   if (phi1<0) phi1+=360.;
   Double_t phi2 = phi1+fDphi;

   if ((rmin1==rmin2) && (rmax1==rmax2)) {
      if (!is_seg) snxt=TGeoTube::DistToInS(local, dir, rmin1, rmax1, dz);
      else snxt=TGeoTubeSeg::DistToInS(local, dir, rmin1, rmax1, dz, c1, s1, c2, s2, cfio, sfio, cdfi);
   } else {  
      if (!is_seg) snxt=TGeoCone::DistToInS(local,dir,dz,rmin1, rmax1,rmin2,rmax2);
      else snxt=TGeoConeSeg::DistToInS(local,dir,rmin1, rmax1, rmin2, rmax2, dz, phi1, phi2);
   }
   if (snxt<1E20) return snxt;
   // check next segment
   if (dir[2]==0) return kBig;
   Int_t istep=(dir[2]>0)?1:-1;
   iz+=istep;
   if (iz<0 || iz>(fNz-2)) return kBig;
   return DistToSegZ(point,dir,iz,c1,s1,c2,s2,cfio,sfio,cdfi);
}      
//-----------------------------------------------------------------------------
Double_t TGeoPcon::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from outside point to surface of the tube
   if ((iact<3) && safe) {
      *safe = Safety(point, kFALSE);
      if ((iact==1) && (*safe>step)) return kBig;
      if (iact==0) return kBig;
   }
   // check if ray intersect outscribed cylinder
   if ((point[2]<fZ[0]) && (dir[2]<=0)) return kBig;
   if ((point[2]>fZ[fNz-1]) && (dir[2]>=0)) return kBig;

   Double_t r2 = point[0]*point[0]+point[1]*point[1];
   Double_t radmax=0;
   radmax=fRmax[TMath::LocMax(fNz, fRmax)];
   if (r2>(radmax*radmax)) {
      Double_t rpr=-point[0]*dir[0]-point[1]*dir[1];
      Double_t nxy=dir[0]*dir[0]+dir[1]*dir[1];
      if (rpr<TMath::Sqrt((r2-radmax*radmax)*nxy)) return kBig;
   }

   // find in which Z segment we are
   Int_t ipl = TMath::BinarySearch(fNz, fZ, point[2]);
   Int_t ifirst = ipl;
   if (ifirst<0) {
      ifirst=0;
   } else if (ifirst>=(fNz-1)) ifirst=fNz-2;
   // find if point is in the phi gap
   Double_t phi=0;
   Double_t phi1=0;
   Double_t phi2=0;
   Double_t c1=0., s1=0., c2=0., s2=0., cfio=0., sfio=0., cdfi=0.;
   Bool_t inphi = (fDphi<360)?kTRUE:kFALSE;
   if (inphi) {
      phi1=fPhi1;
      if (phi1<0) phi1+=360;
      phi2=(phi1+fDphi)*kDegRad;
      phi1=phi1*kDegRad;
      phi=TMath::ATan2(point[1], point[0]);
      if (phi<0) phi+=2.*TMath::Pi();
      c1=TMath::Cos(phi1);
      s1=TMath::Sin(phi1);
      c2=TMath::Cos(phi2);
      s2=TMath::Sin(phi2);
      Double_t fio=0.5*(phi1+phi2);
      cfio=TMath::Cos(fio);
      sfio=TMath::Sin(fio);
      cdfi=TMath::Cos(0.5*(phi2-phi1));
   } 

   // compute distance to boundary
   return DistToSegZ(point,dir,ifirst, c1,s1,c2,s2,cfio,sfio,cdfi);
}
//-----------------------------------------------------------------------------
Double_t TGeoPcon::DistToSurf(Double_t * /*point*/, Double_t * /*dir*/) const
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   return kBig;
}
//-----------------------------------------------------------------------------
void TGeoPcon::DefineSection(Int_t snum, Double_t z, Double_t rmin, Double_t rmax)
{
// defines z position of a section plane, rmin and rmax at this z.
   if ((snum<0) || (snum>=fNz)) return;
   fZ[snum]    = z;
   fRmin[snum] = rmin;
   fRmax[snum] = rmax;
   if (rmin>rmax) {
      Warning("DefineSection", "invalid rmin/rmax");
      printf("rmin=%f rmax=%f\n", rmin, rmax);
   }
   if (snum==(fNz-1)) ComputeBBox();
}
//-----------------------------------------------------------------------------
Int_t TGeoPcon::GetNsegments() const
{
   return gGeoManager->GetNsegments();
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoPcon::Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                             Double_t start, Double_t step) 
{
//--- Divide this polycone shape belonging to volume "voldiv" into ndiv volumes
// called divname, from start position with the given step. Returns pointer
// to created division cell volume in case of Z divisions. Z divisions can be
// performed if the divided range is in between two consecutive Z planes.
//  In case a wrong division axis is supplied, returns pointer to 
// volume that was divided.
   TGeoShape *shape;           //--- shape to be created
   TGeoVolume *vol;            //--- division volume to be created
   TGeoVolumeMulti *vmulti;    //--- generic divided volume
   TGeoPatternFinder *finder;  //--- finder to be attached 
   TString opt = "";           //--- option to be attached
   Double_t zmin = start;
   Double_t zmax = start+ndiv*step;            
   Int_t isect = -1;
   Int_t is, id, ipl;
   switch (iaxis) {
      case 1:  //---               R division
         Error("Divide", "cannot divide a pcon on radius");
         return 0;
      case 2:  //---               Phi division
         finder = new TGeoPatternCylPhi(voldiv, ndiv, start, start+ndiv*step);
         vmulti = gGeoManager->MakeVolumeMulti(divname, voldiv->GetMedium());
         voldiv->SetFinder(finder);
         finder->SetDivIndex(voldiv->GetNdaughters());            
         shape = new TGeoPcon(-step/2, step, fNz);
         for (is=0; is<fNz; is++)
            ((TGeoPcon*)shape)->DefineSection(is, fZ[is], fRmin[is], fRmax[is]); 
            vol = new TGeoVolume(divname, shape, voldiv->GetMedium());
            vmulti->AddVolume(vol);
            opt = "Phi";
            for (id=0; id<ndiv; id++) {
               voldiv->AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
               ((TGeoNodeOffset*)voldiv->GetNodes()->At(voldiv->GetNdaughters()-1))->SetFinder(finder);
            }
            return vmulti;
      case 3: //---                Z division
         // find start plane
         for (ipl=0; ipl<fNz-1; ipl++) {
            if (start<fZ[ipl]) continue;
            else {
               if ((start+ndiv*step)>fZ[ipl+1]) continue;
            }
            isect = ipl;
            break;
         }
         if (isect<0) {
            Error("Divide", "cannot divide pcon on Z if divided region is not between 2 planes");
            return 0;
         }
         finder = new TGeoPatternZ(voldiv, ndiv, start, start+ndiv*step);
         vmulti = gGeoManager->MakeVolumeMulti(divname, voldiv->GetMedium());
         voldiv->SetFinder(finder);
         finder->SetDivIndex(voldiv->GetNdaughters());
         opt = "Z";
         for (id=0; id<ndiv; id++) {
            Double_t z1 = start+id*step;
            Double_t z2 = start+(id+1)*step;
            Double_t rmin1 = (fRmin[isect]*(zmax-z1)-fRmin[isect+1]*(zmin-z1))/(zmax-zmin);
            Double_t rmax1 = (fRmax[isect]*(zmax-z1)-fRmax[isect+1]*(zmin-z1))/(zmax-zmin);
            Double_t rmin2 = (fRmin[isect]*(zmax-z2)-fRmin[isect+1]*(zmin-z2))/(zmax-zmin);
            Double_t rmax2 = (fRmax[isect]*(zmax-z2)-fRmax[isect+1]*(zmin-z2))/(zmax-zmin);
            shape = new TGeoConeSeg(step/2, rmin1, rmax1, rmin2, rmax2, fPhi1, fPhi1+fDphi); 
            vol = new TGeoVolume(divname, shape, voldiv->GetMedium());
            vmulti->AddVolume(vol);
            voldiv->AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
            ((TGeoNodeOffset*)voldiv->GetNodes()->At(voldiv->GetNdaughters()-1))->SetFinder(finder);
         }
         return vmulti;
      default:
         Error("Divide", "Wrong axis type for division");
         return 0;            
   }
}

//-----------------------------------------------------------------------------
const char *TGeoPcon::GetAxisName(Int_t iaxis) const
{
// Returns name of axis IAXIS.
   switch (iaxis) {
      case 1:
         return "R";
      case 2:
         return "PHI";
      case 3:
         return "Z";
      default:
         return "UNDEFINED";
   }
}   

//-----------------------------------------------------------------------------
Double_t TGeoPcon::GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const
{
// Get range of shape for a given axis.
   xlo = 0;
   xhi = 0;
   Double_t dx = 0;
   switch (iaxis) {
      case 2:
         xlo = fPhi1;
         xhi = fPhi1 + fDphi;
         dx = fDphi;
         return dx;
      case 3:
         xlo = fZ[0];
         xhi = fZ[fNz-1];
         dx = xhi-xlo;
         return dx;
   }
   return dx;
}         
            
//-----------------------------------------------------------------------------
void TGeoPcon::GetBoundingCylinder(Double_t *param) const
{
//--- Fill vector param[4] with the bounding cylinder parameters. The order
// is the following : Rmin, Rmax, Phi1, Phi2
   param[0] = fRmin[0];           // Rmin
   param[1] = fRmax[0];           // Rmax
   for (Int_t i=1; i<fNz; i++) {
      if (fRmin[i] < param[0]) param[0] = fRmin[i];
      if (fRmax[i] > param[1]) param[1] = fRmax[i];
   }
   param[0] *= param[0];
   param[1] *= param[1];
   if (fDphi==360.) {
      param[2] = 0.;
      param[3] = 360.;
      return;
   }   
   param[2] = (fPhi1<0)?(fPhi1+360.):fPhi1;     // Phi1
   param[3] = param[2]+fDphi;                   // Phi2
}   
//-----------------------------------------------------------------------------
void TGeoPcon::InspectShape() const
{
// print shape parameters
   printf("*** TGeoPcon parameters ***\n");
   printf("    Nz    = %i\n", fNz);
   printf("    phi1  = %11.5f\n", fPhi1);
   printf("    dphi  = %11.5f\n", fDphi);
   for (Int_t ipl=0; ipl<fNz; ipl++)
      printf("     plane %i: z=%11.5f Rmin=%11.5f Rmax=%11.5f\n", ipl, fZ[ipl], fRmin[ipl], fRmax[ipl]);
   TGeoBBox::InspectShape();
}
//-----------------------------------------------------------------------------
void *TGeoPcon::Make3DBuffer(const TGeoVolume *vol) const
{
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return 0;
   return painter->MakePcon3DBuffer(vol);
}   

//-----------------------------------------------------------------------------
void TGeoPcon::Paint(Option_t *option)
{
// paint this shape according to option
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   TGeoVolume *vol = gGeoManager->GetCurrentVolume();
   if (vol->GetShape() != (TGeoShape*)this) return;
   painter->PaintPcon(this, option);
}
//-----------------------------------------------------------------------------
void TGeoPcon::PaintNext(TGeoHMatrix *glmat, Option_t *option)
{
// paint this shape according to option
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   painter->PaintPcon(this, option, glmat);
}
//-----------------------------------------------------------------------------
void TGeoPcon::NextCrossing(TGeoParamCurve * /*c*/, Double_t * /*point*/) const
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoPcon::Safety(Double_t *point, Bool_t in) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   //---> localize the Z segment
   
   Double_t safe;
   Double_t ptnew[3];
   Double_t dz, rmin1, rmax1, rmin2, rmax2;
   Bool_t is_tube, is_seg;
   Double_t phi1=0, phi2=0, c1=0, s1=0, c2=0, s2=0;
   Int_t skipz;
   Double_t saf[2];
   saf[0] = saf[1] = kBig;
   if (in) {
   //---> point is inside pcon
      Int_t ipl = TMath::BinarySearch(fNz, fZ, point[2]);
      if (ipl==(fNz-1)) return 0;   // point on last Z boundary
      if (ipl<0) return 0;          // point on first Z boundary
      dz = 0.5*(fZ[ipl+1]-fZ[ipl]);
      if (dz<1E-10) return 0;
      skipz = 3; // skip z checks
      if (ipl==0) {
         saf[0] = point[2]-fZ[0];
         if (saf[0]<1E-4) return saf[0];
      }
      if (ipl==fNz-2) {
         saf[1] = fZ[fNz-1]-point[2];
         if (saf[1]<1E-4) return saf[1];
      }
      if (ipl>1) {
         if (fZ[ipl]==fZ[ipl-1]) {
            if (fRmin[ipl]<fRmin[ipl-1] || fRmax[ipl]>fRmax[ipl-1]) {
               saf[0] = point[2]-fZ[ipl];      
               if (saf[0]<1E-4) return saf[0];
            }
         }
      }
      if (ipl<fNz-3) {
         if (fZ[ipl+1]==fZ[ipl+2]) {
            if (fRmin[ipl+1]<fRmin[ipl+2] || fRmax[ipl+1]>fRmax[ipl+2]) {
               saf[1] = fZ[ipl+1]-point[2];
               if (saf[1]<1E-4) return saf[1];
            }
         }
      }
      if (saf[0]<1E10) {
         if (saf[1]<1E10) skipz=0; // check both Z planes
         else             skipz=2; // skip upper Z
      } else {
         if (saf[1]<1E10) skipz=1; // skip lower Z
         else             skipz=3; // skip both Z planes
      }   
      //---> Check shape type
      memcpy(ptnew, point, 3*sizeof(Double_t));
      ptnew[2] -= 0.5*(fZ[ipl]+fZ[ipl+1]);
      rmin1 = fRmin[ipl];
      rmax1 = fRmax[ipl];
      rmin2 = fRmin[ipl+1];
      rmax2 = fRmax[ipl+1];
      is_tube = ((rmin1==rmin2) && (rmax1==rmax2))?kTRUE:kFALSE;
      is_seg  = (fDphi<360)?kTRUE:kFALSE;
      if (is_seg) {
         phi1 = fPhi1;
         if (phi1<0) phi1+=360;
         phi2 = phi1 + fDphi;
         phi1 *= kDegRad;
         phi2 *= kDegRad;
         c1 = TMath::Cos(phi1);
         s1 = TMath::Sin(phi1);
         c2 = TMath::Cos(phi2);
         s2 = TMath::Sin(phi2);
         if (is_tube) safe = TGeoTubeSeg::SafetyS(ptnew,in,rmin1,rmax1, dz,c1,s1,c2,s2,skipz);
         else         safe = TGeoConeSeg::SafetyS(ptnew,in,dz,rmin1,rmax1,rmin2,rmax2,c1,s1,c2,s2,skipz);
      } else {
         if (is_tube) safe = TGeoTube::SafetyS(ptnew,in,rmin1,rmax1,dz,skipz);
         else         safe = TGeoCone::SafetyS(ptnew,in,dz,rmin1,rmax1,rmin2,rmax2,skipz);
      }
      return safe;
   }
   //---> point is outside pcon
   Int_t ipl = TMath::BinarySearch(fNz, fZ, point[2]);
   is_seg  = (fDphi<360)?kTRUE:kFALSE;
   if (is_seg) {
      phi1 = fPhi1;
      if (phi1<0) phi1+=360;
      phi2 = phi1 + fDphi;
      phi1 *= kDegRad;
      phi2 *= kDegRad;
      c1 = TMath::Cos(phi1);
      s1 = TMath::Sin(phi1);
      c2 = TMath::Cos(phi2);
      s2 = TMath::Sin(phi2);
   }         
   Bool_t outz = kFALSE;
   skipz = 3;
   if (ipl<0) {
      ipl++;
      skipz = 0;
      outz = kTRUE;
   } else if (ipl==fNz-1) {
      ipl--;
      skipz = 0;
      outz = kTRUE;
   }
   dz = 0.5*(fZ[ipl+1]-fZ[ipl]);
   if (dz==0) {
      ipl++;
      dz = 0.5*(fZ[ipl+1]-fZ[ipl]);
   }   
   rmin1 = fRmin[ipl];
   rmax1 = fRmax[ipl];
   rmin2 = fRmin[ipl+1];
   rmax2 = fRmax[ipl+1];
   is_tube = ((rmin1==rmin2) && (rmax1==rmax2))?kTRUE:kFALSE;
   is_seg  = (fDphi<360)?kTRUE:kFALSE;
   memcpy(ptnew, point, 2*sizeof(Double_t));
   ptnew[2] = point[2] - 0.5*(fZ[ipl]+fZ[ipl+1]);
   if (is_seg) {
      if (is_tube) safe = TGeoTubeSeg::SafetyS(ptnew,in,rmin1,rmax1, dz,c1,s1,c2,s2,skipz);
      else         safe = TGeoConeSeg::SafetyS(ptnew,in,dz,rmin1,rmax1,rmin2,rmax2,c1,s1,c2,s2,skipz);
   } else {
      if (is_tube) safe = TGeoTube::SafetyS(ptnew,in,rmin1,rmax1,dz,skipz);
      else         safe = TGeoCone::SafetyS(ptnew,in,dz,rmin1,rmax1,rmin2,rmax2,skipz);
   }
   if (outz) return safe;
   skipz = 0;
   Double_t safup   = kBig;
   Double_t safdown = kBig;
   Double_t rpt;
   Int_t ipnew = -1;
   if (ipl>1) {
      if (fZ[ipl]==fZ[ipl-1]) {
         rpt = TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
         if (rpt<fRmin[ipl] && fRmin[ipl-1]<fRmin[ipl]) {
            ipnew = ipl-2;
         } else if (rpt>fRmax[ipl] && fRmax[ipl-1]>fRmax[ipl]) ipnew=ipl-2;
      }
      if (ipnew>=0) {
      // fully check slice at index ipnew
         rmin1 = fRmin[ipnew];
         rmax1 = fRmax[ipnew];
         rmin2 = fRmin[ipnew+1];
         rmax2 = fRmax[ipnew+1];
         is_tube = ((rmin1==rmin2) && (rmax1==rmax2))?kTRUE:kFALSE;
         ptnew[2] = point[2] - 0.5*(fZ[ipnew]+fZ[ipnew+1]);
         dz = 0.5*(fZ[ipnew+1]-fZ[ipnew]);
         if (is_seg) {
            if (is_tube) safdown = TGeoTubeSeg::SafetyS(ptnew,in,rmin1,rmax1, dz,c1,s1,c2,s2,skipz);
            else         safdown = TGeoConeSeg::SafetyS(ptnew,in,dz,rmin1,rmax1,rmin2,rmax2,c1,s1,c2,s2,skipz);
         } else {
            if (is_tube) safdown = TGeoTube::SafetyS(ptnew,in,rmin1,rmax1,dz,skipz);
            else         safdown = TGeoCone::SafetyS(ptnew,in,dz,rmin1,rmax1,rmin2,rmax2,skipz);
         }
      }
   }   
   ipnew = -1;   
   if (ipl<fNz-3) {
      if (fZ[ipl+1]==fZ[ipl+2]) {
         rpt = TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
         if (rpt<fRmin[ipl+1] && fRmin[ipl+2]<fRmin[ipl+1]) {
            ipnew = ipl+2;
         } else if (rpt>fRmax[ipl+1] && fRmax[ipl+2]>fRmax[ipl+1]) ipnew=ipl+2;
      }
      if (ipnew>=0) {
      // fully check slice at index ipnew
         rmin1 = fRmin[ipnew];
         rmax1 = fRmax[ipnew];
         rmin2 = fRmin[ipnew+1];
         rmax2 = fRmax[ipnew+1];
         is_tube = ((rmin1==rmin2) && (rmax1==rmax2))?kTRUE:kFALSE;
         ptnew[2] = point[2] - 0.5*(fZ[ipnew]+fZ[ipnew+1]);
         dz = 0.5*(fZ[ipnew+1]-fZ[ipnew]);
         if (is_seg) {
            if (is_tube) safup = TGeoTubeSeg::SafetyS(ptnew,in,rmin1,rmax1, dz,c1,s1,c2,s2,skipz);
            else         safup = TGeoConeSeg::SafetyS(ptnew,in,dz,rmin1,rmax1,rmin2,rmax2,c1,s1,c2,s2,skipz);
         } else {
            if (is_tube) safup = TGeoTube::SafetyS(ptnew,in,rmin1,rmax1,dz,skipz);
            else         safup = TGeoCone::SafetyS(ptnew,in,dz,rmin1,rmax1,rmin2,rmax2,skipz);
         }
      }
   }   
         
   safe = TMath::Min(safe, TMath::Min(safdown, safup));
   return safe;
}
//-----------------------------------------------------------------------------
void TGeoPcon::SetDimensions(Double_t *param)
{
   fPhi1    = param[0];
   fDphi    = param[1];
   fNz      = (Int_t)param[2];
   if (!fRmin) fRmin = new Double_t [fNz];
   if (!fRmax) fRmax = new Double_t [fNz];
   if (!fZ)    fZ    = new Double_t [fNz];
   for (Int_t i=0; i<fNz; i++) 
      DefineSection(i, param[3+3*i], param[4+3*i], param[5+3*i]);
}   
//-----------------------------------------------------------------------------
void TGeoPcon::SetPoints(Double_t *buff) const
{
// create polycone mesh points
    Double_t phi, dphi;
    Int_t n = gGeoManager->GetNsegments() + 1;
    dphi = fDphi/(n-1);
    Int_t i, j;
    Int_t indx = 0;

    if (buff) {
        for (i = 0; i < fNz; i++)
        {
            for (j = 0; j < n; j++)
            {
                phi = (fPhi1+j*dphi)*kDegRad;
                buff[indx++] = fRmin[i] * TMath::Cos(phi);
                buff[indx++] = fRmin[i] * TMath::Sin(phi);
                buff[indx++] = fZ[i];
            }
            for (j = 0; j < n; j++)
            {
                phi = (fPhi1+j*dphi)*kDegRad;
                buff[indx++] = fRmax[i] * TMath::Cos(phi);
                buff[indx++] = fRmax[i] * TMath::Sin(phi);
                buff[indx++] = fZ[i];
            }
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoPcon::SetPoints(Float_t *buff) const
{
// create polycone mesh points
    Double_t phi, dphi;
    Int_t n = gGeoManager->GetNsegments() + 1;
    dphi = fDphi/(n-1);
    Int_t i, j;
    Int_t indx = 0;

    if (buff) {
        for (i = 0; i < fNz; i++)
        {
            for (j = 0; j < n; j++)
            {
                phi = (fPhi1+j*dphi)*kDegRad;
                buff[indx++] = fRmin[i] * TMath::Cos(phi);
                buff[indx++] = fRmin[i] * TMath::Sin(phi);
                buff[indx++] = fZ[i];
            }
            for (j = 0; j < n; j++)
            {
                phi = (fPhi1+j*dphi)*kDegRad;
                buff[indx++] = fRmax[i] * TMath::Cos(phi);
                buff[indx++] = fRmax[i] * TMath::Sin(phi);
                buff[indx++] = fZ[i];
            }
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoPcon::Sizeof3D() const
{
// fill size of this 3-D object
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
    Int_t n;

    n = gGeoManager->GetNsegments()+1;

    Int_t numPoints = fNz*2*n;
    Int_t numSegs   = 4*(fNz*n-1+(fDphi == 360));
    Int_t numPolys  = 2*(fNz*n-1+(fDphi == 360));
    painter->AddSize3D(numPoints, numSegs, numPolys);
}

