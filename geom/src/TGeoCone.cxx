// @(#)root/geom:$Name:  $:$Id: TGeoCone.cxx,v 1.29 2004/06/25 11:59:55 brun Exp $
// Author: Andrei Gheata   31/01/02
// TGeoCone::Contains() and DistToOut() implemented by Mihaela Gheata

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//--------------------------------------------------------------------------
// TGeoCone - conical tube  class. It has 5 parameters :
//            dz - half length in z
//            Rmin1, Rmax1 - inside and outside radii at -dz
//            Rmin2, Rmax2 - inside and outside radii at +dz
//
//--------------------------------------------------------------------------
//Begin_Html
/*
<img src="gif/t_cone.gif">
*/
//End_Html
//
//Begin_Html
/*
<img src="gif/t_conedivR.gif">
*/
//End_Html
//
//Begin_Html
/*
<img src="gif/t_conedivPHI.gif">
*/
//End_Html
//Begin_Html
/*
<img src="gif/t_conedivZ.gif">
*/
//End_Html

//--------------------------------------------------------------------------
// TGeoConeSeg - a phi segment of a conical tube. Has 7 parameters :
//            - the same 5 as a cone;
//            - first phi limit (in degrees)
//            - second phi limit 
//
//--------------------------------------------------------------------------
//
//Begin_Html
/*
<img src="gif/t_coneseg.gif">
*/
//End_Html
//
//Begin_Html
/*
<img src="gif/t_conesegdivstepZ.gif">
*/
//End_Html

#include "TROOT.h"

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TVirtualGeoPainter.h"
#include "TGeoCone.h"
#include "TVirtualPad.h"
#include "TBuffer3D.h"

ClassImp(TGeoCone)
   
//_____________________________________________________________________________
TGeoCone::TGeoCone()
{
// Default constructor
   SetShapeBit(TGeoShape::kGeoCone);
   fDz    = 0.0;
   fRmin1 = 0.0;
   fRmax1 = 0.0;
   fRmin2 = 0.0;
   fRmax2 = 0.0;
}   

//_____________________________________________________________________________
TGeoCone::TGeoCone(Double_t dz, Double_t rmin1, Double_t rmax1,
                   Double_t rmin2, Double_t rmax2)
         :TGeoBBox(0, 0, 0)
{
// Default constructor specifying minimum and maximum radius
   SetShapeBit(TGeoShape::kGeoCone);
   SetConeDimensions(dz, rmin1, rmax1, rmin2, rmax2);
   if ((dz<0) || (rmin1<0) || (rmax1<0) || (rmin2<0) || (rmax2<0)) {
      SetShapeBit(kGeoRunTimeShape);
   }
   else ComputeBBox();
}

//_____________________________________________________________________________
TGeoCone::TGeoCone(const char *name, Double_t dz, Double_t rmin1, Double_t rmax1,
                   Double_t rmin2, Double_t rmax2)
         :TGeoBBox(name, 0, 0, 0)
{
// Default constructor specifying minimum and maximum radius
   SetShapeBit(TGeoShape::kGeoCone);
   SetConeDimensions(dz, rmin1, rmax1, rmin2, rmax2);
   if ((dz<0) || (rmin1<0) || (rmax1<0) || (rmin2<0) || (rmax2<0)) {
      SetShapeBit(kGeoRunTimeShape);
   }
   else ComputeBBox();
}

//_____________________________________________________________________________
TGeoCone::TGeoCone(Double_t *param)
         :TGeoBBox(0, 0, 0)
{
// Default constructor specifying minimum and maximum radius
// param[0] = dz
// param[1] = Rmin1
// param[2] = Rmax1
// param[3] = Rmin2
// param[4] = Rmax2
   SetShapeBit(TGeoShape::kGeoCone);
   SetDimensions(param);
   if ((fDz<0) || (fRmin1<0) || (fRmax1<0) || (fRmin2<0) || (fRmax2<0))
      SetShapeBit(kGeoRunTimeShape);
   else ComputeBBox();
}

//_____________________________________________________________________________
TGeoCone::~TGeoCone()
{
// destructor
}

//_____________________________________________________________________________
void TGeoCone::ComputeBBox()
{
// compute bounding box of the sphere
   TGeoBBox *box = (TGeoBBox*)this;
   box->SetBoxDimensions(TMath::Max(fRmax1, fRmax2), TMath::Max(fRmax1, fRmax2), fDz);
   memset(fOrigin, 0, 3*sizeof(Double_t));
}   

//_____________________________________________________________________________
void TGeoCone::ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm)
{
// Compute normal to closest surface from POINT. 
   Double_t safr,safe,phi;
   memset(norm,0,3*sizeof(Double_t));
   phi = TMath::ATan2(point[1],point[0]);
   Double_t cphi = TMath::Cos(phi);
   Double_t sphi = TMath::Sin(phi);
   Double_t ro1 = 0.5*(fRmin1+fRmin2);
   Double_t tg1 = 0.5*(fRmin2-fRmin1)/fDz;
   Double_t cr1 = 1./TMath::Sqrt(1.+tg1*tg1);
   Double_t ro2 = 0.5*(fRmax1+fRmax2);
   Double_t tg2 = 0.5*(fRmax2-fRmax1)/fDz;
   Double_t cr2 = 1./TMath::Sqrt(1.+tg2*tg2);
   
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t rin = tg1*point[2]+ro1;
   Double_t rout = tg2*point[2]+ro2;
   safe = TMath::Abs(fDz-TMath::Abs(point[2]));
   norm[2] = 1;
   
   safr = (ro1>0)?(TMath::Abs((r-rin)*cr1)):TGeoShape::Big();
   if (safr<safe) {
      safe = safr;
      norm[0] = cr1*cphi;
      norm[1] = cr1*sphi;
      norm[2] = tg1*cr1;
   }      
   safr = TMath::Abs((rout-r)*cr2);
   if (safr<safe) {
      norm[0] = cr2*cphi;
      norm[1] = cr2*sphi;
      norm[2] = tg2*cr2;
   }      
   if (norm[0]*dir[0]+norm[1]*dir[1]+norm[2]*dir[2]<0) {
      norm[0] = -norm[0];
      norm[1] = -norm[1];
      norm[2] = -norm[2];
   }
}

//_____________________________________________________________________________
void TGeoCone::ComputeNormalS(Double_t *point, Double_t *dir, Double_t *norm,
                              Double_t dz, Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2)
{
// Compute normal to closest surface from POINT. 
   Double_t safe,phi;
   memset(norm,0,3*sizeof(Double_t));
   phi = TMath::ATan2(point[1],point[0]);
   Double_t cphi = TMath::Cos(phi);
   Double_t sphi = TMath::Sin(phi);
   Double_t ro1 = 0.5*(rmin1+rmin2);
   Double_t tg1 = 0.5*(rmin2-rmin1)/dz;
   Double_t cr1 = 1./TMath::Sqrt(1.+tg1*tg1);
   Double_t ro2 = 0.5*(rmax1+rmax2);
   Double_t tg2 = 0.5*(rmax2-rmax1)/dz;
   Double_t cr2 = 1./TMath::Sqrt(1.+tg2*tg2);
   
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t rin = tg1*point[2]+ro1;
   Double_t rout = tg2*point[2]+ro2;
   safe = (ro1>0)?(TMath::Abs((r-rin)*cr1)):TGeoShape::Big();
   norm[0] = cr1*cphi;
   norm[1] = cr1*sphi;
   norm[2] = tg1*cr1;
   if (TMath::Abs((rout-r)*cr2)<safe) {
      norm[0] = cr2*cphi;
      norm[1] = cr2*sphi;
      norm[2] = tg2*cr2;
   }      
   if (norm[0]*dir[0]+norm[1]*dir[1]+norm[2]*dir[2]<0) {
      norm[0] = -norm[0];
      norm[1] = -norm[1];
      norm[2] = -norm[2];
   }
}

//_____________________________________________________________________________
Bool_t TGeoCone::Contains(Double_t *point) const
{
// test if point is inside this cone
   if (TMath::Abs(point[2]) > fDz) return kFALSE;
   Double_t r2 = point[0]*point[0]+point[1]*point[1];
   Double_t rl = 0.5*(fRmin2*(point[2]+fDz)+fRmin1*(fDz-point[2]))/fDz;
   Double_t rh = 0.5*(fRmax2*(point[2]+fDz)+fRmax1*(fDz-point[2]))/fDz;
   if ((r2<rl*rl) || (r2>rh*rh)) return kFALSE;
   return kTRUE;
}

//_____________________________________________________________________________
Double_t TGeoCone::DistToOutS(Double_t *point, Double_t *dir, Double_t dz, 
                              Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2)
{
// compute distance from inside point to surface of the cone (static)
   if (dz<=0) return TGeoShape::Big();
   // compute distance to surface 
   // Do Z
   Double_t sz = TGeoShape::Big();
   if (dir[2]>0) {
      sz = (dz-point[2])/dir[2];
      if (sz<=0) return 0.;
   } else {
      if (dir[2]<0) {
         sz = -(dz+point[2])/dir[2];
         if (sz<=0) return 0.;
      }
   }      
   // Do Rmin
   Double_t sr1=TGeoShape::Big(), sr2=TGeoShape::Big();
   Double_t b,delta, znew;
   Bool_t found = kFALSE;
   if ((rmin1+rmin2)>0) {
      TGeoCone::DistToCone(point, dir, rmin1, -dz, rmin2, dz, b, delta);
      if (delta>0) {
         sr1 = -b-delta;
         if (sr1>0) {
            znew = point[2]+sr1*dir[2];
            if (TMath::Abs(znew)<dz) found=kTRUE;
         }
         if (!found) {
            sr1 = -b+delta;
            if (sr1>0) {
               znew = point[2]+sr1*dir[2];
               if (TMath::Abs(znew)>=dz) sr1=TGeoShape::Big();
            } else {
               sr1 = TGeoShape::Big();
            }   
         }
      }
   }
   // Do Rmax     
   found = kFALSE;          
   TGeoCone::DistToCone(point, dir, rmax1, -dz, rmax2, dz, b, delta);
   if (delta>0) {
      sr2 = -b-delta;
      if (sr2>0) {
         znew = point[2]+sr2*dir[2];
         if (TMath::Abs(znew)<dz) found=kTRUE;
      }
      if (!found) {
         sr2 = -b+delta;
         if (sr2>0) {
            znew = point[2]+sr2*dir[2];
            if (TMath::Abs(znew)>=dz) sr2=TGeoShape::Big();
         } else {
            sr2 = TGeoShape::Big();
         }   
      }
   }
   
   return TMath::Min(TMath::Min(sr1, sr2), sz);                   
}

//_____________________________________________________________________________
Double_t TGeoCone::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from inside point to surface of the cone
   
   if (iact<3 && safe) {
      *safe = Safety(point, kTRUE);
      if (iact==0) return TGeoShape::Big();
      if ((iact==1) && (*safe>step)) return TGeoShape::Big();
   }
   // compute distance to surface 
   return TGeoCone::DistToOutS(point, dir, fDz, fRmin1, fRmax1, fRmin2, fRmax2);
}

//_____________________________________________________________________________
Double_t TGeoCone::DistToInS(Double_t *point, Double_t *dir, Double_t dz, 
                             Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2)
{
// compute distance from outside point to surface of the tube
   // compute distance to Z planes
   if (dz<=0) return TGeoShape::Big();
   Double_t snxt = TGeoShape::Big();
   Double_t ro1=0.5*(rmin1+rmin2);
   Bool_t hasrmin = (ro1>0)?kTRUE:kFALSE;
   Double_t ro2=0.5*(rmax1+rmax2);
   Double_t tg2=0.5*(rmax2-rmax1)/dz;
   Double_t r2=point[0]*point[0]+point[1]*point[1];
   Double_t r=TMath::Sqrt(r2);
   Double_t rout=tg2*point[2]+ro2;
   Double_t xp, yp;
   
   if ((point[2]<=-dz) && (dir[2]>0)) {
      snxt = (-dz-point[2])/dir[2];
      xp = point[0]+snxt*dir[0];
      yp = point[1]+snxt*dir[1];
      r2 = xp*xp+yp*yp;
      if ((r2>=rmin1*rmin1) && (r2<=rmax1*rmax1)) return snxt;
   } else {
      if ((point[2]>=dz) && (dir[2]<0)) {
         snxt = (dz-point[2])/dir[2];    
         xp = point[0]+snxt*dir[0];
         yp = point[1]+snxt*dir[1];
         r2 = xp*xp+yp*yp;
         if ((r2>=rmin2*rmin2) && (r2<=rmax2*rmax2)) return snxt;
      }
   }           
   
   // compute distance to inner cone
   Double_t din=TGeoShape::Big(), dout=TGeoShape::Big();
   Double_t b,delta,znew;
   Bool_t found = kFALSE;
   snxt = TGeoShape::Big();
   if (hasrmin) {
      TGeoCone::DistToCone(point, dir, rmin1, -dz, rmin2, dz, b, delta);
      if (delta>0) {
         din = -b-delta;
         if (din>0) {
            znew = point[2]+din*dir[2];
            if (TMath::Abs(znew)<dz) found=kTRUE;
         }
         if (!found) {
            din = -b+delta;
            if (din>0) {
               znew = point[2]+din*dir[2];
               if (TMath::Abs(znew)>=dz) din=TGeoShape::Big();
            } else {
               din = TGeoShape::Big();
            }   
         }
      }
   }

   // compute distance to outer cone      
   if (r>=rout) {
      found = kFALSE;          
      TGeoCone::DistToCone(point, dir, rmax1, -dz, rmax2, dz, b, delta);
      if (delta>0) {
         dout = -b-delta;
         if (dout>0) {
            znew = point[2]+dout*dir[2];
            if (TMath::Abs(znew)<dz) found=kTRUE;
         }
         if (!found) {
            dout = -b+delta;
            if (dout>0) {
               znew = point[2]+dout*dir[2];
               if (TMath::Abs(znew)>=dz) dout=TGeoShape::Big();
            } else {
               dout = TGeoShape::Big();
            }   
         }
      }
   }
//   printf("din=%f  dout=%f\n", din, dout);
   snxt = TMath::Min(din, dout);
   return snxt;
}

//_____________________________________________________________________________
Double_t TGeoCone::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from outside point to surface of the tube
   // compute safe radius
   if (iact<3 && safe) {
      *safe = Safety(point, kFALSE);
      if (iact==0) return TGeoShape::Big();
      if ((iact==1) && (*safe>step)) return TGeoShape::Big();
   }
   // compute distance to Z planes
   return TGeoCone::DistToInS(point, dir, fDz, fRmin1, fRmax1, fRmin2, fRmax2);
}

//_____________________________________________________________________________
void TGeoCone::DistToCone(Double_t *point, Double_t *dir, Double_t r1, Double_t z1, Double_t r2, Double_t z2,
                              Double_t &b, Double_t &delta)
{
   // Static method to compute distance to a conical surface with : 
   // - r1, z1 - radius and Z position of lower base
   // - r2, z2 - radius and Z position of upper base
   Double_t dz = z2-z1;
   delta = -1.;
   if (dz<0) return;
   Double_t ro0 = 0.5*(r1+r2);
   Double_t fz  = (r2-r1)/dz;
   Double_t r0sq = point[0]*point[0] + point[1]*point[1];
   Double_t rc = ro0 + fz*(point[2]-0.5*(z1+z2));
   
   Double_t a = dir[0]*dir[0] + dir[1]*dir[1] - fz*fz*dir[2]*dir[2];
   b = point[0]*dir[0] + point[1]*dir[1] - fz*rc*dir[2];
   Double_t c = r0sq - rc*rc;
   
   if (a==0) return;
   a = 1./a;
   b *= a;
   c *= a;
   delta = b*b - c;
   if (delta>0) {
      delta = TMath::Sqrt(delta);
   } else {   
      delta = -1.;
   }   
}

//_____________________________________________________________________________
Int_t TGeoCone::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute closest distance from point px,py to each corner
   Int_t n = gGeoManager->GetNsegments();
   const Int_t numPoints = 4*n;
   return ShapeDistancetoPrimitive(numPoints, px, py);
}

//_____________________________________________________________________________
TGeoVolume *TGeoCone::Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                             Double_t start, Double_t step) 
{
//--- Divide this cone shape belonging to volume "voldiv" into ndiv volumes
// called divname, from start position with the given step. Returns pointer
// to created division cell volume in case of Z divisions. For Z division 
// creates all volumes with different shapes and returns pointer to volume that
// was divided. In case a wrong division axis is supplied, returns pointer to 
// volume that was divided.
   TGeoShape *shape;           //--- shape to be created
   TGeoVolume *vol;            //--- division volume to be created
   TGeoVolumeMulti *vmulti;    //--- generic divided volume
   TGeoPatternFinder *finder;  //--- finder to be attached 
   TString opt = "";           //--- option to be attached
   Int_t id;
   Double_t end = start+ndiv*step;
   switch (iaxis) {
      case 1:  //---              R division
         Error("Divide","division of a cone on R not implemented");
         return 0;
      case 2:  // ---             Phi division
         finder = new TGeoPatternCylPhi(voldiv, ndiv, start, end);
         voldiv->SetFinder(finder);
         finder->SetDivIndex(voldiv->GetNdaughters());            
         shape = new TGeoConeSeg(fDz, fRmin1, fRmax1, fRmin2, fRmax2, -step/2, step/2);
         vol = new TGeoVolume(divname, shape, voldiv->GetMedium());
         vmulti = gGeoManager->MakeVolumeMulti(divname, voldiv->GetMedium());
         vmulti->AddVolume(vol);
         opt = "Phi";
         for (id=0; id<ndiv; id++) {
            voldiv->AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
            ((TGeoNodeOffset*)voldiv->GetNodes()->At(voldiv->GetNdaughters()-1))->SetFinder(finder);
         }
         return vmulti;
      case 3: //---               Z division
         vmulti = gGeoManager->MakeVolumeMulti(divname, voldiv->GetMedium());
         finder = new TGeoPatternZ(voldiv, ndiv, start, end);
         voldiv->SetFinder(finder);
         finder->SetDivIndex(voldiv->GetNdaughters());            
         for (id=0; id<ndiv; id++) {
            Double_t z1 = start+id*step;
            Double_t z2 = start+(id+1)*step;
            Double_t rmin1n = 0.5*(fRmin1*(fDz-z1)+fRmin2*(fDz+z1))/fDz;
            Double_t rmax1n = 0.5*(fRmax1*(fDz-z1)+fRmax2*(fDz+z1))/fDz;
            Double_t rmin2n = 0.5*(fRmin1*(fDz-z2)+fRmin2*(fDz+z2))/fDz;
            Double_t rmax2n = 0.5*(fRmax1*(fDz-z2)+fRmax2*(fDz+z2))/fDz;
            shape = new TGeoCone(0.5*step,rmin1n, rmax1n, rmin2n, rmax2n); 
            vol = new TGeoVolume(divname, shape, voldiv->GetMedium());
            vmulti->AddVolume(vol);
            opt = "Z";
            voldiv->AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
            ((TGeoNodeOffset*)voldiv->GetNodes()->At(voldiv->GetNdaughters()-1))->SetFinder(finder);
         }
         return vmulti;
      default:
         Error("Divide", "Wrong axis type for division");
         return 0;            
   }
}

//_____________________________________________________________________________
const char *TGeoCone::GetAxisName(Int_t iaxis) const
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
         return "undefined";
   }
}   

//_____________________________________________________________________________
Double_t TGeoCone::GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const
{
// Get range of shape for a given axis.
   xlo = 0;
   xhi = 0;
   Double_t dx = 0;
   switch (iaxis) {
      case 2:
         xlo = 0.;
         xhi = 360.;
         return 360.;
      case 3:
         xlo = -fDz;
         xhi = fDz;
         dx = xhi-xlo;
         return dx;
   }
   return dx;
}         

//_____________________________________________________________________________
void TGeoCone::GetBoundingCylinder(Double_t *param) const
{
//--- Fill vector param[4] with the bounding cylinder parameters. The order
// is the following : Rmin, Rmax, Phi1, Phi2, dZ
   param[0] = TMath::Min(fRmin1, fRmin2); // Rmin
   param[0] *= param[0];
   param[1] = TMath::Max(fRmax1, fRmax2); // Rmax
   param[1] *= param[1];
   param[2] = 0.;                         // Phi1
   param[3] = 360.;                       // Phi1
}

//_____________________________________________________________________________
TGeoShape *TGeoCone::GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix * /*mat*/) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestShapeBit(kGeoRunTimeShape)) return 0;
   if (!mother->TestShapeBit(kGeoCone)) {
      Error("GetMakeRuntimeShape", "invalid mother");
      return 0;
   }
   Double_t rmin1, rmax1, rmin2, rmax2, dz;
   rmin1 = fRmin1;
   rmax1 = fRmax1;
   rmin2 = fRmin2;
   rmax2 = fRmax2;
   dz = fDz;
   if (fDz<0) dz=((TGeoCone*)mother)->GetDz();
   if (fRmin1<0) 
      rmin1 = ((TGeoCone*)mother)->GetRmin1();
   if (fRmax1<0)
      rmax1 = ((TGeoCone*)mother)->GetRmax1();
   if (fRmin2<0) 
      rmin2 = ((TGeoCone*)mother)->GetRmin2();
   if (fRmax2<0)
      rmax2 = ((TGeoCone*)mother)->GetRmax2();

   return (new TGeoCone(rmin1, rmax1, rmin2, rmax2, dz));
}

//_____________________________________________________________________________
void TGeoCone::InspectShape() const
{
// print shape parameters
   printf("*** Shape %s TGeoCone ***\n", GetName());
   printf("    dz    =: %11.5f\n", fDz);
   printf("    Rmin1 = %11.5f\n", fRmin1);
   printf("    Rmax1 = %11.5f\n", fRmax1);
   printf("    Rmin2 = %11.5f\n", fRmin2);
   printf("    Rmax2 = %11.5f\n", fRmax2);
   printf(" Bounding box:\n");
   TGeoBBox::InspectShape();
}

//_____________________________________________________________________________
void *TGeoCone::Make3DBuffer(const TGeoVolume *vol) const
{
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return 0;
   return painter->MakeTube3DBuffer(vol);
}   

//_____________________________________________________________________________
void TGeoCone::Paint(Option_t *option)
{
   // Paint this shape according to option

   // Allocate the necessary spage in gPad->fBuffer3D to store this shape
   Int_t i, j, n = 20;
   if (gGeoManager) n = gGeoManager->GetNsegments();
   Int_t NbPnts = 4*n;
   Int_t NbSegs = 8*n;
   Int_t NbPols = 4*n; 
   TBuffer3D *buff = gPad->AllocateBuffer3D(3*NbPnts, 3*NbSegs, 6*NbPols);
   if (!buff) return;

   buff->fType = TBuffer3D::kTUBE;
   buff->fId   = this;

   // Fill gPad->fBuffer3D. Points coordinates are in Master space
   buff->fNbPnts = NbPnts;
   buff->fNbSegs = NbSegs;
   buff->fNbPols = NbPols;
   // In case of option "size" it is not necessary to fill the buffer
   if (strstr(option,"size")) {
      buff->Paint(option);
      return;
   }

   SetPoints(buff->fPnts);

   TransformPoints(buff);

   // Basic colors: 0, 1, ... 7
   Int_t c = ((gGeoManager->GetCurrentVolume()->GetLineColor() % 8) - 1) * 4;
   if (c < 0) c = 0;

   for (i = 0; i < 4; i++) {
      for (j = 0; j < n; j++) {
         buff->fSegs[(i*n+j)*3  ] = c;
         buff->fSegs[(i*n+j)*3+1] = i*n+j;
         buff->fSegs[(i*n+j)*3+2] = i*n+j+1;
      }
      buff->fSegs[(i*n+j-1)*3+2] = i*n;
   }
   for (i = 4; i < 6; i++) {
      for (j = 0; j < n; j++) {
         buff->fSegs[(i*n+j)*3  ] = c+1;
         buff->fSegs[(i*n+j)*3+1] = (i-4)*n+j;
         buff->fSegs[(i*n+j)*3+2] = (i-2)*n+j;
      }
   }
   for (i = 6; i < 8; i++) {
      for (j = 0; j < n; j++) {
         buff->fSegs[(i*n+j)*3  ] = c;
         buff->fSegs[(i*n+j)*3+1] = 2*(i-6)*n+j;
         buff->fSegs[(i*n+j)*3+2] = (2*(i-6)+1)*n+j;
      }
   }

   Int_t indx = 0;
   i=0;
   for (j = 0; j < n; j++) {
      indx = 6*(i*n+j);
      buff->fPols[indx  ] = c;
      buff->fPols[indx+1] = 4;
      buff->fPols[indx+5] = i*n+j;
      buff->fPols[indx+4] = (4+i)*n+j;
      buff->fPols[indx+3] = (2+i)*n+j;
      buff->fPols[indx+2] = (4+i)*n+j+1;
   }
   buff->fPols[indx+2] = (4+i)*n;
   i=1;
   for (j = 0; j < n; j++) {
      indx = 6*(i*n+j);
      buff->fPols[indx  ] = c;
      buff->fPols[indx+1] = 4;
      buff->fPols[indx+2] = i*n+j;
      buff->fPols[indx+3] = (4+i)*n+j;
      buff->fPols[indx+4] = (2+i)*n+j;
      buff->fPols[indx+5] = (4+i)*n+j+1;
   }
   buff->fPols[indx+5] = (4+i)*n;
   i=2;
   for (j = 0; j < n; j++) {
      indx = 6*(i*n+j);
      buff->fPols[indx  ] = c+i;
      buff->fPols[indx+1] = 4;
      buff->fPols[indx+2] = (i-2)*2*n+j;
      buff->fPols[indx+3] = (4+i)*n+j;
      buff->fPols[indx+4] = ((i-2)*2+1)*n+j;
      buff->fPols[indx+5] = (4+i)*n+j+1;
   }
   buff->fPols[indx+5] = (4+i)*n;
   i=3;
   for (j = 0; j < n; j++) {
      indx = 6*(i*n+j);
      buff->fPols[indx  ] = c+i;
      buff->fPols[indx+1] = 4;
      buff->fPols[indx+5] = (i-2)*2*n+j;
      buff->fPols[indx+4] = (4+i)*n+j;
      buff->fPols[indx+3] = ((i-2)*2+1)*n+j;
      buff->fPols[indx+2] = (4+i)*n+j+1;
   }
   buff->fPols[indx+2] = (4+i)*n;

   // Paint gPad->fBuffer3D
   buff->Paint(option);
}

//_____________________________________________________________________________
Double_t TGeoCone::Safety(Double_t *point, Bool_t in) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   Double_t saf[3];
   Double_t ro1 = 0.5*(fRmin1+fRmin2);
   Double_t tg1 = 0.5*(fRmin2-fRmin1)/fDz;
   Double_t cr1 = 1./TMath::Sqrt(1.+tg1*tg1);
   Double_t ro2 = 0.5*(fRmax1+fRmax2);
   Double_t tg2 = 0.5*(fRmax2-fRmax1)/fDz;
   Double_t cr2 = 1./TMath::Sqrt(1.+tg2*tg2);
   
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t rin = tg1*point[2]+ro1;
   Double_t rout = tg2*point[2]+ro2;
   saf[0] = fDz-TMath::Abs(point[2]);
   saf[1] = (ro1>0)?((r-rin)*cr1):TGeoShape::Big();
   saf[2] = (rout-r)*cr2;
   if (in) return saf[TMath::LocMin(3,saf)];
   for (Int_t i=0; i<3; i++) saf[i]=-saf[i];
   return saf[TMath::LocMax(3,saf)];
}

//_____________________________________________________________________________
Double_t TGeoCone::SafetyS(Double_t *point, Bool_t in, Double_t dz, Double_t rmin1, Double_t rmax1,
                           Double_t rmin2, Double_t rmax2, Int_t skipz)
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   Double_t saf[3];
   Double_t ro1 = 0.5*(rmin1+rmin2);
   Double_t tg1 = 0.5*(rmin2-rmin1)/dz;
   Double_t cr1 = 1./TMath::Sqrt(1.+tg1*tg1);
   Double_t ro2 = 0.5*(rmax1+rmax2);
   Double_t tg2 = 0.5*(rmax2-rmax1)/dz;
   Double_t cr2 = 1./TMath::Sqrt(1.+tg2*tg2);
   
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t rin = tg1*point[2]+ro1;
   Double_t rout = tg2*point[2]+ro2;
   switch (skipz) {
     case 1: // skip lower Z plane
        saf[0] = dz - point[2];
        break;
     case 2: // skip upper Z plane
        saf[0] = dz + point[2];
        break;
     case 3: // skip both
        saf[0] = TGeoShape::Big();
        break;   
     default:
        saf[0] = dz-TMath::Abs(point[2]);         
   }
   saf[1] = (ro1>0)?((r-rin)*cr1):TGeoShape::Big();
   saf[2] = (rout-r)*cr2;
   if (in) return saf[TMath::LocMin(3,saf)];
   for (Int_t i=0; i<3; i++) saf[i]=-saf[i];
   return saf[TMath::LocMax(3,saf)];
}

//_____________________________________________________________________________
void TGeoCone::SetConeDimensions(Double_t dz, Double_t rmin1, Double_t rmax1,
                             Double_t rmin2, Double_t rmax2)
{
   if (rmin1>=0) {
      if (rmax1>0) {
         if (rmin1<=rmax1) {
         // normal rmin/rmax
            fRmin1 = rmin1;
            fRmax1 = rmax1;
         } else {
            fRmin1 = rmax1;
            fRmax1 = rmin1;
            Warning("SetConeDimensions", "rmin1>rmax1 Switch rmin1<->rmax1");
            SetShapeBit(TGeoShape::kGeoBad);
         }
      } else {
         // run-time
         fRmin1 = rmin1;
         fRmax1 = rmax1;
      }
   } else {
      // run-time
      fRmin1 = rmin1;
      fRmax1 = rmax1;
   }               
   if (rmin2>=0) {
      if (rmax2>0) {
         if (rmin2<=rmax2) {
         // normal rmin/rmax
            fRmin2 = rmin2;
            fRmax2 = rmax2;
         } else {
            fRmin2 = rmax2;
            fRmax2 = rmin2;
            Warning("SetConeDimensions", "rmin2>rmax2 Switch rmin2<->rmax2");
            SetShapeBit(TGeoShape::kGeoBad);
         }
      } else {
         // run-time
         fRmin2 = rmin2;
         fRmax2 = rmax2;
      }
   } else {
      // run-time
      fRmin2 = rmin2;
      fRmax2 = rmax2;
   }               
   
   fDz   = dz;
}   

//_____________________________________________________________________________
void TGeoCone::SetDimensions(Double_t *param)
{
   Double_t dz    = param[0];
   Double_t rmin1 = param[1];
   Double_t rmax1 = param[2];
   Double_t rmin2 = param[3];
   Double_t rmax2 = param[4];
   SetConeDimensions(dz, rmin1, rmax1, rmin2, rmax2);
}   

//_____________________________________________________________________________
void TGeoCone::SetPoints(Double_t *buff) const
{
// create cone mesh points
    Double_t dz, phi, dphi;
    Int_t j, n;

    n = gGeoManager->GetNsegments();
    dphi = 360./n;
    dz    = fDz;
    Int_t indx = 0;

    if (buff) {
        for (j = 0; j < n; j++) {
            phi = j*dphi*TMath::DegToRad();
            buff[indx++] = fRmin1 * TMath::Cos(phi);
            buff[indx++] = fRmin1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }
        for (j = 0; j < n; j++) {
            phi = j*dphi*TMath::DegToRad();
            buff[indx++] = fRmax1 * TMath::Cos(phi);
            buff[indx++] = fRmax1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }

        for (j = 0; j < n; j++) {
            phi = j*dphi*TMath::DegToRad();
            buff[indx++] = fRmin2 * TMath::Cos(phi);
            buff[indx++] = fRmin2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }

        for (j = 0; j < n; j++) {
            phi = j*dphi*TMath::DegToRad();
            buff[indx++] = fRmax2 * TMath::Cos(phi);
            buff[indx++] = fRmax2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }
    }
}

//_____________________________________________________________________________
void TGeoCone::SetPoints(Float_t *buff) const
{
// create cone mesh points
    Double_t dz, phi, dphi;
    Int_t j, n;

    n = gGeoManager->GetNsegments();
    dphi = 360./n;
    dz    = fDz;
    Int_t indx = 0;

    if (buff) {
        for (j = 0; j < n; j++) {
            phi = j*dphi*TMath::DegToRad();
            buff[indx++] = fRmin1 * TMath::Cos(phi);
            buff[indx++] = fRmin1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }
        for (j = 0; j < n; j++) {
            phi = j*dphi*TMath::DegToRad();
            buff[indx++] = fRmax1 * TMath::Cos(phi);
            buff[indx++] = fRmax1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }

        for (j = 0; j < n; j++) {
            phi = j*dphi*TMath::DegToRad();
            buff[indx++] = fRmin2 * TMath::Cos(phi);
            buff[indx++] = fRmin2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }

        for (j = 0; j < n; j++) {
            phi = j*dphi*TMath::DegToRad();
            buff[indx++] = fRmax2 * TMath::Cos(phi);
            buff[indx++] = fRmax2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }
    }
}

//_____________________________________________________________________________
Int_t TGeoCone::GetNmeshVertices() const
{
// Return number of vertices of the mesh representation
    Int_t n = gGeoManager->GetNsegments();
    Int_t numPoints = n*4;
    return numPoints;
}

//_____________________________________________________________________________
void TGeoCone::Sizeof3D() const
{
///// fill size of this 3-D object
///    TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
///    if (!painter) return;
///    Int_t n = gGeoManager->GetNsegments();
///    Int_t numPoints = n*4;
///    Int_t numSegs   = n*8;
///    Int_t numPolys  = n*4;
///    painter->AddSize3D(numPoints, numSegs, numPolys);
}


ClassImp(TGeoConeSeg)
   
//_____________________________________________________________________________
TGeoConeSeg::TGeoConeSeg()
{
// Default constructor
   SetShapeBit(TGeoShape::kGeoConeSeg);
   fPhi1 = fPhi2 = 0.0;
}   

//_____________________________________________________________________________
TGeoConeSeg::TGeoConeSeg(Double_t dz, Double_t rmin1, Double_t rmax1, 
                          Double_t rmin2, Double_t rmax2, Double_t phi1, Double_t phi2)
            :TGeoCone(dz, rmin1, rmax1, rmin2, rmax2)
{
// Default constructor specifying minimum and maximum radius
   SetShapeBit(TGeoShape::kGeoConeSeg);
   SetConsDimensions(dz, rmin1, rmax1, rmin2, rmax2, phi1, phi2);
   ComputeBBox();
}

//_____________________________________________________________________________
TGeoConeSeg::TGeoConeSeg(const char *name, Double_t dz, Double_t rmin1, Double_t rmax1, 
                          Double_t rmin2, Double_t rmax2, Double_t phi1, Double_t phi2)
            :TGeoCone(name, dz, rmin1, rmax1, rmin2, rmax2)
{
// Default constructor specifying minimum and maximum radius
   SetShapeBit(TGeoShape::kGeoConeSeg);
   SetConsDimensions(dz, rmin1, rmax1, rmin2, rmax2, phi1, phi2);
   ComputeBBox();
}

//_____________________________________________________________________________
TGeoConeSeg::TGeoConeSeg(Double_t *param)
            :TGeoCone(0,0,0,0,0)
{
// Default constructor specifying minimum and maximum radius
// param[0] = dz
// param[1] = Rmin1
// param[2] = Rmax1
// param[3] = Rmin2
// param[4] = Rmax2
// param[5] = phi1
// param[6] = phi2
   SetShapeBit(TGeoShape::kGeoConeSeg);
   SetDimensions(param);
   ComputeBBox();
}

//_____________________________________________________________________________
TGeoConeSeg::~TGeoConeSeg()
{
// destructor
}

//_____________________________________________________________________________
void TGeoConeSeg::ComputeBBox()
{
// compute bounding box of the tube segment
   Double_t rmin, rmax;
   rmin = TMath::Min(fRmin1, fRmin2);
   rmax = TMath::Max(fRmax1, fRmax2);

   Double_t xc[4];
   Double_t yc[4];
   xc[0] = rmax*TMath::Cos(fPhi1*TMath::DegToRad());
   yc[0] = rmax*TMath::Sin(fPhi1*TMath::DegToRad());
   xc[1] = rmax*TMath::Cos(fPhi2*TMath::DegToRad());
   yc[1] = rmax*TMath::Sin(fPhi2*TMath::DegToRad());
   xc[2] = rmin*TMath::Cos(fPhi1*TMath::DegToRad());
   yc[2] = rmin*TMath::Sin(fPhi1*TMath::DegToRad());
   xc[3] = rmin*TMath::Cos(fPhi2*TMath::DegToRad());
   yc[3] = rmin*TMath::Sin(fPhi2*TMath::DegToRad());

   Double_t xmin = xc[TMath::LocMin(4, &xc[0])];
   Double_t xmax = xc[TMath::LocMax(4, &xc[0])]; 
   Double_t ymin = yc[TMath::LocMin(4, &yc[0])]; 
   Double_t ymax = yc[TMath::LocMax(4, &yc[0])];

   Double_t dp = fPhi2-fPhi1;
   Double_t ddp = -fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp<=dp) xmax = rmax;
   ddp = 90-fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp<=dp) ymax = rmax;
   ddp = 180-fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp<=dp) xmin = -rmax;
   ddp = 270-fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp<=dp) ymin = -rmax;
   fOrigin[0] = (xmax+xmin)/2;
   fOrigin[1] = (ymax+ymin)/2;
   fOrigin[2] = 0;
   fDX = (xmax-xmin)/2;
   fDY = (ymax-ymin)/2;
   fDZ = fDz;
}   

//_____________________________________________________________________________
void TGeoConeSeg::ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm)
{
// Compute normal to closest surface from POINT. 
   Double_t saf[3];
   Double_t ro1 = 0.5*(fRmin1+fRmin2);
   Double_t tg1 = 0.5*(fRmin2-fRmin1)/fDz;
   Double_t cr1 = 1./TMath::Sqrt(1.+tg1*tg1);
   Double_t ro2 = 0.5*(fRmax1+fRmax2);
   Double_t tg2 = 0.5*(fRmax2-fRmax1)/fDz;
   Double_t cr2 = 1./TMath::Sqrt(1.+tg2*tg2);
   
   Double_t c1 = TMath::Cos(fPhi1*TMath::DegToRad());
   Double_t s1 = TMath::Sin(fPhi1*TMath::DegToRad());
   Double_t c2 = TMath::Cos(fPhi2*TMath::DegToRad());
   Double_t s2 = TMath::Sin(fPhi2*TMath::DegToRad());

   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t rin = tg1*point[2]+ro1;
   Double_t rout = tg2*point[2]+ro2;
   saf[0] = TMath::Abs(fDz-TMath::Abs(point[2]));
   saf[1] = (ro1>0)?(TMath::Abs((r-rin)*cr1)):TGeoShape::Big();
   saf[2] = TMath::Abs((rout-r)*cr2);
   Int_t i = TMath::LocMin(3,saf);
   if (TGeoShape::IsCloseToPhi(saf[i], point,c1,s1,c2,s2)) {
      TGeoShape::NormalPhi(point,dir,norm,c1,s1,c2,s2);
      return;
   }   
   if (i==0) {
      norm[0] = norm[1] = 0.;
      norm[2] = TMath::Sign(1.,dir[2]);
      return;
   }

   Double_t phi = TMath::ATan2(point[1],point[0]);
   Double_t cphi = TMath::Cos(phi);
   Double_t sphi = TMath::Sin(phi);
   
   if (i==1) {
      norm[0] = cr1*cphi;
      norm[1] = cr1*sphi;
      norm[2] = tg1*cr1;
   } else {   
      norm[0] = cr2*cphi;
      norm[1] = cr2*sphi;
      norm[2] = tg2*cr2;
   }
        
   if (norm[0]*dir[0]+norm[1]*dir[1]+norm[2]*dir[2]<0) {
      norm[0] = -norm[0];
      norm[1] = -norm[1];
      norm[2] = -norm[2];
   }
}

//_____________________________________________________________________________
void TGeoConeSeg::ComputeNormalS(Double_t *point, Double_t *dir, Double_t *norm,
                                 Double_t dz, Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2,
                                 Double_t c1, Double_t s1, Double_t c2, Double_t s2)
{
// Compute normal to closest surface from POINT. 
   Double_t saf[2];
   Double_t ro1 = 0.5*(rmin1+rmin2);
   Double_t tg1 = 0.5*(rmin2-rmin1)/dz;
   Double_t cr1 = 1./TMath::Sqrt(1.+tg1*tg1);
   Double_t ro2 = 0.5*(rmax1+rmax2);
   Double_t tg2 = 0.5*(rmax2-rmax1)/dz;
   Double_t cr2 = 1./TMath::Sqrt(1.+tg2*tg2);
   
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t rin = tg1*point[2]+ro1;
   Double_t rout = tg2*point[2]+ro2;
   saf[0] = (ro1>0)?(TMath::Abs((r-rin)*cr1)):TGeoShape::Big();
   saf[1] = TMath::Abs((rout-r)*cr2);
   Int_t i = TMath::LocMin(2,saf);
   if (TGeoShape::IsCloseToPhi(saf[i], point,c1,s1,c2,s2)) {
      TGeoShape::NormalPhi(point,dir,norm,c1,s1,c2,s2);
      return;
   }   

   Double_t phi = TMath::ATan2(point[1],point[0]);
   Double_t cphi = TMath::Cos(phi);
   Double_t sphi = TMath::Sin(phi);
   
   if (i==0) {
      norm[0] = cr1*cphi;
      norm[1] = cr1*sphi;
      norm[2] = tg1*cr1;
   } else {   
      norm[0] = cr2*cphi;
      norm[1] = cr2*sphi;
      norm[2] = tg2*cr2;
   }
        
   if (norm[0]*dir[0]+norm[1]*dir[1]+norm[2]*dir[2]<0) {
      norm[0] = -norm[0];
      norm[1] = -norm[1];
      norm[2] = -norm[2];
   }
}

//_____________________________________________________________________________
Bool_t TGeoConeSeg::Contains(Double_t *point) const
{
// test if point is inside this sphere
   if (!TGeoCone::Contains(point)) return kFALSE;
   Double_t phi = TMath::ATan2(point[1], point[0]) * TMath::RadToDeg();
   if (phi < 0 ) phi+=360.;
   Double_t dphi = fPhi2 - fPhi1;
   Double_t ddp = phi-fPhi1;
   if (ddp < 0) ddp+=360.; 
//   if (ddp > 360) ddp-=360;
   if (ddp > dphi) return kFALSE;
   return kTRUE;
}

//_____________________________________________________________________________
Double_t TGeoConeSeg::DistToCons(Double_t *point, Double_t *dir, Double_t r1, Double_t z1, Double_t r2, Double_t z2, Double_t phi1, Double_t phi2)
{
   // Static method to compute distance to a conical surface with : 
   // - r1, z1 - radius and Z position of lower base
   // - r2, z2 - radius and Z position of upper base
   // - phi1, phi2 - phi limits
   Double_t dz = z2-z1;
   if (dz<=0) {
      return TGeoShape::Big();
   }      

   Double_t dphi = phi2 - phi1;
   if (dphi < 0) dphi+=360.;
//   printf("phi1=%f phi2=%f dphi=%f\n", phi1, phi2, dphi);

   Double_t ro0 = 0.5*(r1+r2);
   Double_t fz  = (r2-r1)/dz;
   Double_t r0sq = point[0]*point[0] + point[1]*point[1];
   Double_t rc = ro0 + fz*(point[2]-0.5*(z1+z2));
   
   Double_t a = dir[0]*dir[0] + dir[1]*dir[1] - fz*fz*dir[2]*dir[2];
   Double_t b = point[0]*dir[0] + point[1]*dir[1] - fz*rc*dir[2];
   Double_t c = r0sq - rc*rc;
   
   if (a==0) return TGeoShape::Big();
   a = 1./a;
   b *= a;
   c *= a;
   Double_t delta = b*b - c;
   if (delta<0) return TGeoShape::Big();
   delta = TMath::Sqrt(delta);
   
   Double_t snxt = -b-delta;
   Double_t ptnew[3];
   Double_t ddp, phi;
   if (snxt>0) {
      // check Z range
      ptnew[2] = point[2] + snxt*dir[2];
      if (((ptnew[2]-z1)*(ptnew[2]-z2)) < 0) {
      // check phi range
         ptnew[0] = point[0] + snxt*dir[0];
         ptnew[1] = point[1] + snxt*dir[1];
         phi = TMath::ATan2(ptnew[1], ptnew[0]) * TMath::RadToDeg();
         if (phi < 0 ) phi+=360.;
         ddp = phi-phi1;
         if (ddp < 0) ddp+=360.; 
//	 printf("snxt1=%f phi=%f ddp=%f\n", snxt, phi, ddp);
         if (ddp<=dphi) return snxt;
      }	 
   }
   snxt = -b+delta;       
   if (snxt>0) {
      // check Z range	
      ptnew[2] = point[2] + snxt*dir[2];
      if (((ptnew[2]-z1)*(ptnew[2]-z2)) < 0) {
      // check phi range
         ptnew[0] = point[0] + snxt*dir[0];
         ptnew[1] = point[1] + snxt*dir[1];
         phi = TMath::ATan2(ptnew[1], ptnew[0]) * TMath::RadToDeg();
         if (phi < 0 ) phi+=360.;
         ddp = phi-phi1;
         if (ddp < 0) ddp+=360.; 
//	 printf("snxt2=%f phi=%f ddp=%f\n", snxt, phi, ddp);
	 if (ddp<=dphi) return snxt;
      }	 
   }
   return TGeoShape::Big();
}

//_____________________________________________________________________________
Double_t TGeoConeSeg::DistToPhiMin(Double_t *point, Double_t *dir, Double_t s1, Double_t c1,
                                   Double_t s2, Double_t c2, Double_t sm, Double_t cm)
{
// compute distance from poin to both phi planes. Return minimum.
   Double_t sfi1=TGeoShape::Big();
   Double_t sfi2=TGeoShape::Big();
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

//_____________________________________________________________________________
Double_t TGeoConeSeg::DistToOutS(Double_t *point, Double_t *dir, Double_t dz, Double_t rmin1, Double_t rmax1, 
                                 Double_t rmin2, Double_t rmax2, Double_t phi1, Double_t phi2)
{
// compute distance from inside point to surface of the tube segment
   if (dz<=0) return TGeoShape::Big();
   
   Double_t ph1 = phi1*TMath::DegToRad();
   Double_t ph2 = phi2*TMath::DegToRad();
   if (ph2<ph1) ph2+=2.*TMath::Pi();
   Double_t phim = 0.5*(ph1+ph2);
   Double_t cm = TMath::Cos(phim);
   Double_t sm = TMath::Sin(phim);
   Double_t c1 = TMath::Cos(ph1);
   Double_t c2 = TMath::Cos(ph2);
   Double_t s1 = TMath::Sin(ph1);
   Double_t s2 = TMath::Sin(ph2);
   
   // compute distance to surface 
   // Do Z
   Double_t sz = TGeoShape::Big();
   if (dir[2]>0) {
      sz = (dz-point[2])/dir[2];
      if (sz<=0) return 0.;
   } else {
      if (dir[2]<0) {
         sz = -(dz+point[2])/dir[2];
         if (sz<=0) return 0.;
      }   
   }
   // check conical surfaces
   Double_t sr1 = TGeoConeSeg::DistToCons(point, dir, rmin1, -dz, rmin2, dz, phi1, phi2);
   Double_t sr2 = TGeoConeSeg::DistToCons(point, dir, rmax1, -dz, rmax2, dz, phi1, phi2);
   Double_t sr = TMath::Min(sr1, sr2);
   // phi planes

   Double_t sfmin=DistToPhiMin(point, dir, s1, c1, s2, c2, sm, cm);
   return TMath::Min(TMath::Min(sz,sr), sfmin);      
}

//_____________________________________________________________________________
Double_t TGeoConeSeg::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from inside point to surface of the tube segment
   Double_t phi1 = fPhi1*TMath::DegToRad();
   Double_t phi2 = fPhi2*TMath::DegToRad();
   Double_t c1 = TMath::Cos(phi1);
   Double_t c2 = TMath::Cos(phi2);
   Double_t s1 = TMath::Sin(phi1);
   Double_t s2 = TMath::Sin(phi2);
   
   if (iact<3 && safe) {
      *safe = TGeoConeSeg::SafetyS(point, kTRUE, fDz,fRmin1,fRmax1,fRmin2,fRmax2,fPhi1,fPhi2);
      if (iact==0) return TGeoShape::Big();
      if ((iact==1) && (*safe>step)) return TGeoShape::Big();
   }
   // compute distance to surface 
   // Do Z
   Double_t sz = TGeoShape::Big();
   if (dir[2]>0) {
      sz = (fDz-point[2])/dir[2];
      if (sz<=0) return 0.;
   } else {
      if (dir[2]<0) {
         sz = -(fDz+point[2])/dir[2];
         if (sz<=0) return 0.;
      }   
   }
   // check conical surfaces
   Double_t sr1 = TGeoConeSeg::DistToCons(point, dir, fRmin1, -fDz, fRmin2, fDz, fPhi1, fPhi2);
   Double_t sr2 = TGeoConeSeg::DistToCons(point, dir, fRmax1, -fDz, fRmax2, fDz, fPhi1, fPhi2);
   Double_t sr = TMath::Min(sr1, sr2);
   // phi planes

   Double_t phim = 0.5*(phi1+phi2);
   Double_t cm = TMath::Cos(phim);
   Double_t sm = TMath::Sin(phim);
   Double_t sfmin=DistToPhiMin(point, dir, s1, c1, s2, c2, sm, cm);
   return TMath::Min(TMath::Min(sz,sr), sfmin);      
}

//_____________________________________________________________________________
Double_t TGeoConeSeg::DistToInS(Double_t *point, Double_t *dir, Double_t rmin1, Double_t rmax1, 
                                Double_t rmin2, Double_t rmax2, Double_t dz, Double_t phi1, Double_t phi2)
{
// compute distance from outside point to surface of arbitrary tube
   Double_t snxt=TGeoShape::Big();
   if (dz<=0) return TGeoShape::Big();
   Double_t ro1=0.5*(rmin1+rmin2);
   Double_t tg1=0.5*(rmin2-rmin1)/dz;
   Double_t ro2=0.5*(rmax1+rmax2);
   Double_t tg2=0.5*(rmax2-rmax1)/dz;

   Double_t ph1 = phi1*TMath::DegToRad();
   Double_t ph2 = phi2*TMath::DegToRad();
   Double_t c1 = TMath::Cos(ph1);
   Double_t s1 = TMath::Sin(ph1);
   Double_t c2 = TMath::Cos(ph2);
   Double_t s2 = TMath::Sin(ph2);
   Double_t fio = 0.5*(ph1+ph2);
   Double_t cfio = TMath::Cos(fio);
   Double_t sfio = TMath::Sin(fio);
   Double_t dfi = 0.5*(ph2-ph1);
   Double_t cdfi = TMath::Cos(dfi);
   Double_t cpsi;
   
   // intersection with Z planes
   Double_t s, xi, yi, zi, riq, r1q, r2q;
   if (TMath::Abs(point[2])>=dz) {
      if ((point[2]*dir[2])<0) {
         s=(TMath::Abs(point[2])-dz)/TMath::Abs(dir[2]);
         xi=point[0]+s*dir[0];
         yi=point[1]+s*dir[1];
         riq=xi*xi+yi*yi;
         if (point[2]<0) {
            r1q=rmin1*rmin1;
            r2q=rmax1*rmax1;
         } else {
            r1q=rmin2*rmin2;
            r2q=rmax2*rmax2;
         }      
         if ((r1q<=riq) && (riq<=r2q)) {
//            gGeoManager->SetNormalChecked(TMath::Abs(dir[2]));
            if (riq==0) return s;
            cpsi=(xi*cfio+yi*sfio)/TMath::Sqrt(riq);
            if (cpsi>=cdfi) return s;
         }   
      }   
   }
   // intersection with cones
   Double_t sr1 = TGeoConeSeg::DistToCons(point, dir, rmin1, -dz, rmin2, dz, phi1, phi2);
   Double_t sr2 = TGeoConeSeg::DistToCons(point, dir, rmax1, -dz, rmax2, dz, phi1, phi2);
   snxt = TMath::Min(sr1, sr2);
   // check phi planes
   Double_t un;
   un=dir[0]*s1-dir[1]*c1;
   if (un!=0) {
      s=(point[1]*c1-point[0]*s1)/un;
      if ((s>=0) && (s<snxt)) {
         zi=point[2]+s*dir[2];
         if (TMath::Abs(zi)<=dz) {
            xi=point[0]+s*dir[0];
            yi=point[1]+s*dir[1];
            riq=xi*xi+yi*yi;
            r1q=(tg1*zi+ro1)*(tg1*zi+ro1);
            r2q=(tg2*zi+ro2)*(tg2*zi+ro2);
            if ((r1q<=riq) && (riq<=r2q)) {
               if ((yi*cfio-xi*sfio)<=0) {
                  snxt = s;
//                  gGeoManager->SetNormalChecked(TMath::Abs(un));
               }
            }
         }
      }
   }               
   un=dir[0]*s2-dir[1]*c2;
   if (un!=0) {
      s=(point[1]*c2-point[0]*s2)/un;
      if ((s>=0) && (s<snxt)) {
         zi=point[2]+s*dir[2];
         if (TMath::Abs(zi)<=dz) {
            xi=point[0]+s*dir[0];
            yi=point[1]+s*dir[1];
            riq=xi*xi+yi*yi;
            r1q=(tg1*zi+ro1)*(tg1*zi+ro1);
            r2q=(tg2*zi+ro2)*(tg2*zi+ro2);
            if ((r1q<=riq) && (riq<=r2q)) {
               if ((yi*cfio-xi*sfio)>=0) {
//                  gGeoManager->SetNormalChecked(TMath::Abs(un));
                  snxt = s;
               }
            }
         }
      }
   }    
   return snxt;               
}

//_____________________________________________________________________________
Double_t TGeoConeSeg::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from outside point to surface of the tube
   // compute safe radius
   if (iact<3 && safe) {
      *safe = Safety(point, kFALSE);
      if (iact==0) return TGeoShape::Big();
      if ((iact==1) && (*safe>step)) return TGeoShape::Big();
   }
   return TGeoConeSeg::DistToInS(point, dir,fRmin1,fRmax1,fRmin2,fRmax2,fDz, fPhi1, fPhi2);
}

//_____________________________________________________________________________
Int_t TGeoConeSeg::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute closest distance from point px,py to each corner
   Int_t n = gGeoManager->GetNsegments()+1;
   const Int_t numPoints = 4*n;
   return ShapeDistancetoPrimitive(numPoints, px, py);
}

//_____________________________________________________________________________
TGeoVolume *TGeoConeSeg::Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                             Double_t start, Double_t step) 
{
//--- Divide this cone segment shape belonging to volume "voldiv" into ndiv volumes
// called divname, from start position with the given step. Returns pointer
// to created division cell volume in case of Z divisions. For Z division 
// creates all volumes with different shapes and returns pointer to volume that
// was divided. In case a wrong division axis is supplied, returns pointer to 
// volume that was divided.
   TGeoShape *shape;           //--- shape to be created
   TGeoVolume *vol;            //--- division volume to be created
   TGeoVolumeMulti *vmulti;    //--- generic divided volume
   TGeoPatternFinder *finder;  //--- finder to be attached 
   TString opt = "";           //--- option to be attached
   Double_t dphi;
   Int_t id;
   Double_t end = start+ndiv*step;
   switch (iaxis) {
      case 1:  //---               R division
         Error("Divide","division of a cone segment on R not implemented");
         return 0;
      case 2:  //---               Phi division
         dphi = fPhi2-fPhi1;
         if (dphi<0) dphi+=360.;
         finder = new TGeoPatternCylPhi(voldiv, ndiv, start, end);
         voldiv->SetFinder(finder);
         finder->SetDivIndex(voldiv->GetNdaughters());            
         shape = new TGeoConeSeg(fDz, fRmin1, fRmax1, fRmin2, fRmax2, -step/2, step/2);
         vol = new TGeoVolume(divname, shape, voldiv->GetMedium());
         vmulti = gGeoManager->MakeVolumeMulti(divname, voldiv->GetMedium());
         vmulti->AddVolume(vol);
         opt = "Phi";
         for (id=0; id<ndiv; id++) {
            voldiv->AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
            ((TGeoNodeOffset*)voldiv->GetNodes()->At(voldiv->GetNdaughters()-1))->SetFinder(finder);
         }
         return vmulti;
      case 3: //---                 Z division
         finder = new TGeoPatternZ(voldiv, ndiv, start, end);
         vmulti = gGeoManager->MakeVolumeMulti(divname, voldiv->GetMedium());
         voldiv->SetFinder(finder);
         finder->SetDivIndex(voldiv->GetNdaughters());            
         for (id=0; id<ndiv; id++) {
            Double_t z1 = start+id*step;
            Double_t z2 = start+(id+1)*step;
            Double_t rmin1n = 0.5*(fRmin1*(fDz-z1)+fRmin2*(fDz+z1))/fDz;
            Double_t rmax1n = 0.5*(fRmax1*(fDz-z1)+fRmax2*(fDz+z1))/fDz;
            Double_t rmin2n = 0.5*(fRmin1*(fDz-z2)+fRmin2*(fDz+z2))/fDz;
            Double_t rmax2n = 0.5*(fRmax1*(fDz-z2)+fRmax2*(fDz+z2))/fDz;
            shape = new TGeoConeSeg(step/2, rmin1n, rmax1n, rmin2n, rmax2n, fPhi1, fPhi2); 
            vol = new TGeoVolume(divname, shape, voldiv->GetMedium());
            vmulti->AddVolume(vol);
            opt = "Z";
            voldiv->AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
            ((TGeoNodeOffset*)voldiv->GetNodes()->At(voldiv->GetNdaughters()-1))->SetFinder(finder);
          }
          return vmulti;
      default:
         Error("Divide", "Wrong axis type for division");
         return 0;            
   }
}

//_____________________________________________________________________________
Double_t TGeoConeSeg::GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const
{
// Get range of shape for a given axis.
   xlo = 0;
   xhi = 0;
   Double_t dx = 0;
   switch (iaxis) {
      case 2:
         xlo = fPhi1;
         xhi = fPhi2;
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

//_____________________________________________________________________________
void TGeoConeSeg::GetBoundingCylinder(Double_t *param) const
{
//--- Fill vector param[4] with the bounding cylinder parameters. The order
// is the following : Rmin, Rmax, Phi1, Phi2
   param[0] = TMath::Min(fRmin1, fRmin2); // Rmin
   param[0] *= param[0];
   param[1] = TMath::Max(fRmax1, fRmax2); // Rmax
   param[1] *= param[1];
   param[2] = (fPhi1<0)?(fPhi1+360.):fPhi1; // Phi1
   param[3] = fPhi2;                        // Phi2
   while (param[3]<param[2]) param[3]+=360.;
}

//_____________________________________________________________________________
TGeoShape *TGeoConeSeg::GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix * /*mat*/) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestShapeBit(kGeoRunTimeShape)) return 0;
   if (!mother->TestShapeBit(kGeoConeSeg)) {
      Error("GetMakeRuntimeShape", "invalid mother");
      return 0;
   }
   Double_t rmin1, rmax1, rmin2, rmax2, dz;
   rmin1 = fRmin1;
   rmax1 = fRmax1;
   rmin2 = fRmin2;
   rmax2 = fRmax2;
   dz = fDz;
   if (fDz<0) dz=((TGeoCone*)mother)->GetDz();
   if (fRmin1<0) 
      rmin1 = ((TGeoCone*)mother)->GetRmin1();
   if ((fRmax1<0) || (fRmax1<fRmin1))
      rmax1 = ((TGeoCone*)mother)->GetRmax1();
   if (fRmin2<0) 
      rmin2 = ((TGeoCone*)mother)->GetRmin2();
   if ((fRmax2<0) || (fRmax2<fRmin2))
      rmax2 = ((TGeoCone*)mother)->GetRmax2();

   return (new TGeoConeSeg(rmin1, rmax1, rmin2, rmax2, dz, fPhi1, fPhi2));
}

//_____________________________________________________________________________
void TGeoConeSeg::InspectShape() const
{
// print shape parameters
   printf("*** Shape %s: TGeoConeSeg ***\n", GetName());
   printf("    dz    = %11.5f\n", fDz);
   printf("    Rmin1 = %11.5f\n", fRmin1);
   printf("    Rmax1 = %11.5f\n", fRmax1);
   printf("    Rmin2 = %11.5f\n", fRmin2);
   printf("    Rmax2 = %11.5f\n", fRmax2);
   printf("    phi1  = %11.5f\n", fPhi1);
   printf("    phi2  = %11.5f\n", fPhi2);
   printf(" Bounding box:\n");
   TGeoBBox::InspectShape();
}

//_____________________________________________________________________________
void *TGeoConeSeg::Make3DBuffer(const TGeoVolume *vol) const
{
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return 0;
   return painter->MakeTubs3DBuffer(vol);
}   

//_____________________________________________________________________________
void TGeoConeSeg::Paint(Option_t *option)
{
   // Paint this shape according to option

   // Allocate the necessary spage in gPad->fBuffer3D to store this shape
   Int_t i, j, n = 20;
   if (gGeoManager) n = gGeoManager->GetNsegments()+1;
   Int_t NbPnts = 4*n;
   Int_t NbSegs = 2*NbPnts;
   Int_t NbPols = NbPnts-2; 
   TBuffer3D *buff = gPad->AllocateBuffer3D(3*NbPnts, 3*NbSegs, 6*NbPols);
   if (!buff) return;

   buff->fType = TBuffer3D::kTUBS;
   buff->fId   = this;

   // Fill gPad->fBuffer3D. Points coordinates are in Master space
   buff->fNbPnts = NbPnts;
   buff->fNbSegs = NbSegs;
   buff->fNbPols = NbPols;
   // In case of option "size" it is not necessary to fill the buffer
   if (strstr(option,"size")) {
      buff->Paint(option);
      return;
   }

   SetPoints(buff->fPnts);

   TransformPoints(buff);

   // Basic colors: 0, 1, ... 7
   Int_t c = ((gGeoManager->GetCurrentVolume()->GetLineColor() % 8) - 1) * 4;
   if (c < 0) c = 0;

   memset(buff->fSegs, 0, buff->fNbSegs*3*sizeof(Int_t));
   for (i = 0; i < 4; i++) {
      for (j = 1; j < n; j++) {
         buff->fSegs[(i*n+j-1)*3  ] = c;
         buff->fSegs[(i*n+j-1)*3+1] = i*n+j-1;
         buff->fSegs[(i*n+j-1)*3+2] = i*n+j;
      }
   }
   for (i = 4; i < 6; i++) {
      for (j = 0; j < n; j++) {
         buff->fSegs[(i*n+j)*3  ] = c+1;
         buff->fSegs[(i*n+j)*3+1] = (i-4)*n+j;
         buff->fSegs[(i*n+j)*3+2] = (i-2)*n+j;
      }
   }
   for (i = 6; i < 8; i++) {
      for (j = 0; j < n; j++) {
         buff->fSegs[(i*n+j)*3  ] = c;
         buff->fSegs[(i*n+j)*3+1] = 2*(i-6)*n+j;
         buff->fSegs[(i*n+j)*3+2] = (2*(i-6)+1)*n+j;
      }
   }

   Int_t indx = 0;
   memset(buff->fPols, 0, buff->fNbPols*6*sizeof(Int_t));
   i = 0;
   for (j = 0; j < n-1; j++) {
      buff->fPols[indx++] = c;
      buff->fPols[indx++] = 4;
      buff->fPols[indx++] = (4+i)*n+j+1;
      buff->fPols[indx++] = (2+i)*n+j;
      buff->fPols[indx++] = (4+i)*n+j;
      buff->fPols[indx++] = i*n+j;
   }
   i = 1;
   for (j = 0; j < n-1; j++) {
      buff->fPols[indx++] = c;
      buff->fPols[indx++] = 4;
      buff->fPols[indx++] = i*n+j;
      buff->fPols[indx++] = (4+i)*n+j;
      buff->fPols[indx++] = (2+i)*n+j;
      buff->fPols[indx++] = (4+i)*n+j+1;
   }
   i = 2;
   for (j = 0; j < n-1; j++) {
      buff->fPols[indx++] = c+i;
      buff->fPols[indx++] = 4;
      buff->fPols[indx++] = (i-2)*2*n+j;
      buff->fPols[indx++] = (4+i)*n+j;
      buff->fPols[indx++] = ((i-2)*2+1)*n+j;
      buff->fPols[indx++] = (4+i)*n+j+1;
   }
   i = 3;
   for (j = 0; j < n-1; j++) {
      buff->fPols[indx++] = c+i;
      buff->fPols[indx++] = 4;
      buff->fPols[indx++] = (4+i)*n+j+1;
      buff->fPols[indx++] = ((i-2)*2+1)*n+j;
      buff->fPols[indx++] = (4+i)*n+j;
      buff->fPols[indx++] = (i-2)*2*n+j;
   }
   buff->fPols[indx++] = c+2;
   buff->fPols[indx++] = 4;
   buff->fPols[indx++] = 6*n;
   buff->fPols[indx++] = 4*n;
   buff->fPols[indx++] = 7*n;
   buff->fPols[indx++] = 5*n;
   buff->fPols[indx++] = c+2;
   buff->fPols[indx++] = 4;
   buff->fPols[indx++] = 6*n-1;
   buff->fPols[indx++] = 8*n-1;
   buff->fPols[indx++] = 5*n-1;
   buff->fPols[indx++] = 7*n-1;

   // Paint gPad->fBuffer3D
   buff->Paint(option);
}


//_____________________________________________________________________________
Double_t TGeoConeSeg::Safety(Double_t *point, Bool_t in) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.

   Double_t saf[3];
   Double_t ro1 = 0.5*(fRmin1+fRmin2);
   Double_t tg1 = 0.5*(fRmin2-fRmin1)/fDz;
   Double_t cr1 = 1./TMath::Sqrt(1.+tg1*tg1);
   Double_t ro2 = 0.5*(fRmax1+fRmax2);
   Double_t tg2 = 0.5*(fRmax2-fRmax1)/fDz;
   Double_t cr2 = 1./TMath::Sqrt(1.+tg2*tg2);
   
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t rin = tg1*point[2]+ro1;
   Double_t rout = tg2*point[2]+ro2;
   Double_t safe = TGeoShape::Big();
   if (in) {
      saf[0] = fDz-TMath::Abs(point[2]);
      saf[1] = (r-rin)*cr1;
      saf[2] = (rout-r)*cr2;
      safe = saf[TMath::LocMin(3,saf)];
   } else {
      saf[0] = TMath::Abs(point[2])-fDz; // positive if inside
      saf[1] = (rin-r)*cr1;
      saf[2] = (r-rout)*cr2;
      safe = saf[TMath::LocMax(3,saf)];
   }   
   Double_t safphi = TGeoShape::SafetyPhi(point, in, fPhi1, fPhi2);

   if (in) return TMath::Min(safe, safphi);
   return TMath::Max(safe, safphi);
}

//_____________________________________________________________________________
Double_t TGeoConeSeg::SafetyS(Double_t *point, Bool_t in, Double_t dz, Double_t rmin1, Double_t rmax1,
                              Double_t rmin2, Double_t rmax2, Double_t phi1, Double_t phi2, Int_t skipz)
{
// Static method to compute the closest distance from given point to this shape.
   Double_t saf[3];
   Double_t ro1 = 0.5*(rmin1+rmin2);
   Double_t tg1 = 0.5*(rmin2-rmin1)/dz;
   Double_t cr1 = 1./TMath::Sqrt(1.+tg1*tg1);
   Double_t ro2 = 0.5*(rmax1+rmax2);
   Double_t tg2 = 0.5*(rmax2-rmax1)/dz;
   Double_t cr2 = 1./TMath::Sqrt(1.+tg2*tg2);
   
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t rin = tg1*point[2]+ro1;
   Double_t rout = tg2*point[2]+ro2;

   Double_t safe = TGeoShape::Big();
   switch (skipz) {
      case 1: // skip lower Z plane
         saf[0] = dz - point[2];
         break;
      case 2: // skip upper Z plane
         saf[0] = dz + point[2];
         break;
      case 3: // skip both
        saf[0] = TGeoShape::Big();   
      default:
         saf[0] = dz-TMath::Abs(point[2]);         
   }
   saf[1] = (r-rin)*cr1;
   saf[2] = (rout-r)*cr2;
   Double_t safphi = TGeoShape::SafetyPhi(point,in,phi1,phi2);
   if (in) {
      safe = saf[TMath::LocMin(3,saf)];
      return TMath::Min(safe,safphi);
   }   
   for (Int_t i=0; i<3; i++) saf[i]=-saf[i];
   safe = saf[TMath::LocMax(3,saf)];
   return TMath::Max(safe,safphi);
}

//_____________________________________________________________________________
void TGeoConeSeg::SetConsDimensions(Double_t dz, Double_t rmin1, Double_t rmax1,
                   Double_t rmin2, Double_t rmax2, Double_t phi1, Double_t phi2)
{
   fDz   = dz;
   fRmin1 = rmin1;
   fRmax1 = rmax1;
   fRmin2 = rmin2;
   fRmax2 = rmax2;
   fPhi1 = phi1;
   if (fPhi1<0) fPhi1+=360.;
   fPhi2 = phi2;
   while (fPhi2<fPhi1) fPhi2+=360.;
}   

//_____________________________________________________________________________
void TGeoConeSeg::SetDimensions(Double_t *param)
{
   Double_t dz    = param[0];
   Double_t rmin1 = param[1];
   Double_t rmax1 = param[2];
   Double_t rmin2 = param[3];
   Double_t rmax2 = param[4];
   Double_t phi1  = param[5];
   Double_t phi2  = param[6];
   SetConsDimensions(dz, rmin1, rmax1,rmin2, rmax2, phi1, phi2);
}   

//_____________________________________________________________________________
void TGeoConeSeg::SetPoints(Double_t *buff) const
{
// create cone segment mesh points
    Int_t j, n;
    Float_t dphi,phi,phi1, phi2,dz;

    n = gGeoManager->GetNsegments()+1;
    dz    = fDz;
    phi1 = fPhi1;
    phi2 = fPhi2;

    dphi = (phi2-phi1)/(n-1);

    Int_t indx = 0;

    if (buff) {
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*TMath::DegToRad();
            buff[indx++] = fRmin1 * TMath::Cos(phi);
            buff[indx++] = fRmin1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*TMath::DegToRad();
            buff[indx++] = fRmax1 * TMath::Cos(phi);
            buff[indx++] = fRmax1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*TMath::DegToRad();
            buff[indx++] = fRmin2 * TMath::Cos(phi);
            buff[indx++] = fRmin2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*TMath::DegToRad();
            buff[indx++] = fRmax2 * TMath::Cos(phi);
            buff[indx++] = fRmax2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }
    }
}

//_____________________________________________________________________________
void TGeoConeSeg::SetPoints(Float_t *buff) const
{
// create cone segment mesh points
    Int_t j, n;
    Float_t dphi,phi,phi1, phi2,dz;

    n = gGeoManager->GetNsegments()+1;
    dz    = fDz;
    phi1 = fPhi1;
    phi2 = fPhi2;

    dphi = (phi2-phi1)/(n-1);

    Int_t indx = 0;

    if (buff) {
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*TMath::DegToRad();
            buff[indx++] = fRmin1 * TMath::Cos(phi);
            buff[indx++] = fRmin1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*TMath::DegToRad();
            buff[indx++] = fRmax1 * TMath::Cos(phi);
            buff[indx++] = fRmax1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*TMath::DegToRad();
            buff[indx++] = fRmin2 * TMath::Cos(phi);
            buff[indx++] = fRmin2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*TMath::DegToRad();
            buff[indx++] = fRmax2 * TMath::Cos(phi);
            buff[indx++] = fRmax2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }
    }
}
//_____________________________________________________________________________
Int_t TGeoConeSeg::GetNmeshVertices() const
{
// Return number of vertices of the mesh representation
    Int_t n = gGeoManager->GetNsegments()+1;
    Int_t numPoints = n*4;
    return numPoints;
}

//_____________________________________________________________________________
void TGeoConeSeg::Sizeof3D() const
{
///// fill size of this 3-D object
///    TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
///    if (!painter) return;
///
///    Int_t n = gGeoManager->GetNsegments()+1;
///
///    Int_t numPoints = n*4;
///    Int_t numSegs   = n*8;
///    Int_t numPolys  = n*4-2;
///    painter->AddSize3D(numPoints, numSegs, numPolys);
}
