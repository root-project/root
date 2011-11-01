// @(#)root/geom:$Id$
// Author: Andrei Gheata   31/01/02
// TGeoTrd2::Contains() and DistFromInside() implemented by Mihaela Gheata

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_____________________________________________________________________________
// TGeoTrd2 - a trapezoid with both x and y lengths varying with z. It 
//   has 5 parameters, the half lengths in x at -dz and +dz, the half
//  lengths in y at -dz and +dz, and the half length in z (dz).
//
//_____________________________________________________________________________
//Begin_Html
/*
<img src="gif/t_trd2.gif">
*/
//End_Html

//Begin_Html
/*
<img src="gif/t_trd2divZ.gif">
*/
//End_Html

//Begin_Html
/*
<img src="gif/t_trd2divstepZ.gif">
*/
//End_Html

#include "Riostream.h"

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"
#include "TGeoTrd2.h"
#include "TMath.h"

ClassImp(TGeoTrd2)
   
//_____________________________________________________________________________
TGeoTrd2::TGeoTrd2()
{
   // dummy ctor
   SetShapeBit(kGeoTrd2);
   fDz = fDx1 = fDx2 = fDy1 = fDy2 = 0;
}

//_____________________________________________________________________________
TGeoTrd2::TGeoTrd2(Double_t dx1, Double_t dx2, Double_t dy1, Double_t dy2, Double_t dz)
         :TGeoBBox(0,0,0)
{
// constructor. 
   SetShapeBit(kGeoTrd2);
   fDx1 = dx1;
   fDx2 = dx2;
   fDy1 = dy1;
   fDy2 = dy2;
   fDz = dz;
   if ((fDx1<0) || (fDx2<0) || (fDy1<0) || (fDy2<0) || (fDz<0)) {
      SetShapeBit(kGeoRunTimeShape);
      printf("trd2 : dx1=%f, dx2=%f, dy1=%f, dy2=%f, dz=%f\n",
              dx1,dx2,dy1,dy2,dz);
   }
   else ComputeBBox();
}

//_____________________________________________________________________________
TGeoTrd2::TGeoTrd2(const char * name, Double_t dx1, Double_t dx2, Double_t dy1, Double_t dy2, Double_t dz)
         :TGeoBBox(name, 0,0,0)
{
// constructor. 
   SetShapeBit(kGeoTrd2);
   fDx1 = dx1;
   fDx2 = dx2;
   fDy1 = dy1;
   fDy2 = dy2;
   fDz = dz;
   if ((fDx1<0) || (fDx2<0) || (fDy1<0) || (fDy2<0) || (fDz<0)) {
      SetShapeBit(kGeoRunTimeShape);
      printf("trd2 : dx1=%f, dx2=%f, dy1=%f, dy2=%f, dz=%f\n",
              dx1,dx2,dy1,dy2,dz);
   }
   else ComputeBBox();
}

//_____________________________________________________________________________
TGeoTrd2::TGeoTrd2(Double_t *param)
         :TGeoBBox(0,0,0)
{
   // ctor with an array of parameters
   // param[0] = dx1
   // param[1] = dx2
   // param[2] = dy1
   // param[3] = dy2
   // param[4] = dz
   SetShapeBit(kGeoTrd2);
   SetDimensions(param);
   if ((fDx1<0) || (fDx2<0) || (fDy1<0) || (fDy2<0) || (fDz<0)) SetShapeBit(kGeoRunTimeShape);
   else ComputeBBox();
}

//_____________________________________________________________________________
TGeoTrd2::~TGeoTrd2()
{
// destructor
}

//_____________________________________________________________________________
Double_t TGeoTrd2::Capacity() const
{
// Computes capacity of the shape in [length^3]
   Double_t capacity = 2*(fDx1+fDx2)*(fDy1+fDy2)*fDz + 
                      (2./3.)*(fDx1-fDx2)*(fDy1-fDy2)*fDz;
   return capacity;
}   

//_____________________________________________________________________________
void TGeoTrd2::ComputeBBox()
{
// compute bounding box for a trd2
   fDX = TMath::Max(fDx1, fDx2);
   fDY = TMath::Max(fDy1, fDy2);
   fDZ = fDz;
   memset(fOrigin, 0, 3*sizeof(Double_t));
}

//_____________________________________________________________________________   
void TGeoTrd2::ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm)
{
// Compute normal to closest surface from POINT. 
   Double_t safe, safemin;
   Double_t fx = 0.5*(fDx1-fDx2)/fDz;
   Double_t calf = 1./TMath::Sqrt(1.0+fx*fx);
   // check Z facettes
   safe = safemin = TMath::Abs(fDz-TMath::Abs(point[2]));
   norm[0] = norm[1] = 0;
   norm[2] = (dir[2]>=0)?1:-1;
   if (safe<TGeoShape::Tolerance()) return;
   // check X facettes
   Double_t distx = 0.5*(fDx1+fDx2)-fx*point[2];
   if (distx>=0) {
      safe=TMath::Abs(distx-TMath::Abs(point[0]))*calf;
      if (safe<safemin) {
         safemin = safe;
         norm[0] = (point[0]>0)?calf:(-calf);
         norm[1] = 0;
         norm[2] = calf*fx;
         Double_t dot = norm[0]*dir[0]+norm[1]*dir[1]+norm[2]*dir[2];
         if (dot<0) {
            norm[0]=-norm[0];
            norm[2]=-norm[2];
         }   
         if (safe<TGeoShape::Tolerance()) return;
      }
   }
   
   Double_t fy = 0.5*(fDy1-fDy2)/fDz;
   calf = 1./TMath::Sqrt(1.0+fy*fy);

   // check Y facettes
   distx = 0.5*(fDy1+fDy2)-fy*point[2];
   if (distx>=0) {
      safe=TMath::Abs(distx-TMath::Abs(point[1]))*calf;
      if (safe<safemin) {
         norm[0] = 0;
         norm[1] = (point[1]>0)?calf:(-calf);
         norm[2] = calf*fy;
         Double_t dot = norm[0]*dir[0]+norm[1]*dir[1]+norm[2]*dir[2];
         if (dot<0) {
            norm[1]=-norm[1];
            norm[2]=-norm[2];
         }   
      }
   }
}

//_____________________________________________________________________________
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

//_____________________________________________________________________________
Double_t TGeoTrd2::DistFromInside(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// Compute distance from inside point to surface of the trd2
// Boundary safe algorithm
   Double_t snxt = TGeoShape::Big();
   if (iact<3 && safe) {
   // compute safe distance
      *safe = Safety(point, kTRUE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }

   Double_t fx = 0.5*(fDx1-fDx2)/fDz;
   Double_t fy = 0.5*(fDy1-fDy2)/fDz;
   Double_t cn;

   Double_t distx = 0.5*(fDx1+fDx2)-fx*point[2];
   Double_t disty = 0.5*(fDy1+fDy2)-fy*point[2];

   //--- Compute distance to this shape
   // first check if Z facettes are crossed
   Double_t dist[3];
   for (Int_t i=0; i<3; i++) dist[i]=TGeoShape::Big();
   if (dir[2]<0) {
      dist[0]=-(point[2]+fDz)/dir[2];
   } else if (dir[2]>0) {
      dist[0]=(fDz-point[2])/dir[2];
   }      
   if (dist[0]<=0) return 0.0;     
   // now check X facettes
   cn = -dir[0]+fx*dir[2];
   if (cn>0) {
      dist[1] = point[0]+distx;
      if (dist[1]<=0) return 0.0;
      dist[1] /= cn;
   }   
   cn = dir[0]+fx*dir[2];
   if (cn>0) {
      Double_t s = distx-point[0];
      if (s<=0) return 0.0;
      s /= cn;
      if (s<dist[1]) dist[1] = s;
   }
   // now check Y facettes
   cn = -dir[1]+fy*dir[2];
   if (cn>0) {
      dist[2] = point[1]+disty;
      if (dist[2]<=0) return 0.0;
      dist[2] /= cn;
   }
   cn = dir[1]+fy*dir[2];
   if (cn>0) {
      Double_t s = disty-point[1];
      if (s<=0) return 0.0;
      s /= cn;
      if (s<dist[2]) dist[2] = s;
   }
   snxt = dist[TMath::LocMin(3,dist)];
   return snxt;
}

//_____________________________________________________________________________
Double_t TGeoTrd2::DistFromOutside(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// Compute distance from outside point to surface of the trd2
// Boundary safe algorithm
   Double_t snxt = TGeoShape::Big();
   if (iact<3 && safe) {
   // compute safe distance
      *safe = Safety(point, kFALSE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }
   // find a visible face
   Double_t xnew,ynew,znew;
   Double_t fx = 0.5*(fDx1-fDx2)/fDz;
   Double_t fy = 0.5*(fDy1-fDy2)/fDz;
   Double_t cn;
   // check visibility of X faces
   Double_t distx = 0.5*(fDx1+fDx2)-fx*point[2];
   Double_t disty = 0.5*(fDy1+fDy2)-fy*point[2];
   Bool_t in = kTRUE;
   Double_t safx = distx-TMath::Abs(point[0]);
   Double_t safy = disty-TMath::Abs(point[1]);
   Double_t safz = fDz-TMath::Abs(point[2]);
   //--- Compute distance to this shape
   // first check if Z facettes are crossed
   if (point[2]<=-fDz) {
      cn = -dir[2];
      if (cn>=0) return TGeoShape::Big();
      in = kFALSE;
      snxt = (fDz+point[2])/cn;
      // find extrapolated X and Y
      xnew = point[0]+snxt*dir[0];
      if (TMath::Abs(xnew) < fDx1) {
         ynew = point[1]+snxt*dir[1];
         if (TMath::Abs(ynew) < fDy1) return snxt;
      }
   } else if (point[2]>=fDz) {
      cn = dir[2];
      if (cn>=0) return TGeoShape::Big();
      in = kFALSE;
      snxt = (fDz-point[2])/cn;
      // find extrapolated X and Y
      xnew = point[0]+snxt*dir[0];
      if (TMath::Abs(xnew) < fDx2) {
         ynew = point[1]+snxt*dir[1];
         if (TMath::Abs(ynew) < fDy2) return snxt;
      }
   }
   // check if X facettes are crossed
   if (point[0]<=-distx) {
      cn = -dir[0]+fx*dir[2];
      if (cn>=0) return TGeoShape::Big();
      in = kFALSE;
      snxt = (point[0]+distx)/cn;
      // find extrapolated Y and Z
      znew = point[2]+snxt*dir[2];
      if (TMath::Abs(znew) < fDz) {
         Double_t dy = 0.5*(fDy1+fDy2)-fy*znew;
         ynew = point[1]+snxt*dir[1];
         if (TMath::Abs(ynew) < dy) return snxt;
      }
   }            
   if (point[0]>=distx) {
      cn = dir[0]+fx*dir[2];
      if (cn>=0) return TGeoShape::Big();
      in = kFALSE;
      snxt = (distx-point[0])/cn;
      // find extrapolated Y and Z
      znew = point[2]+snxt*dir[2];
      if (TMath::Abs(znew) < fDz) {
         Double_t dy = 0.5*(fDy1+fDy2)-fy*znew;
         ynew = point[1]+snxt*dir[1];
         if (TMath::Abs(ynew) < dy) return snxt;
      }
   }
   // finally check Y facettes
   if (point[1]<=-disty) {
      cn = -dir[1]+fy*dir[2];
      in = kFALSE;
      if (cn>=0) return TGeoShape::Big();
      snxt = (point[1]+disty)/cn;
      // find extrapolated X and Z
      znew = point[2]+snxt*dir[2];
      if (TMath::Abs(znew) < fDz) {
         Double_t dx = 0.5*(fDx1+fDx2)-fx*znew;
         xnew = point[0]+snxt*dir[0];
         if (TMath::Abs(xnew) < dx) return snxt;
      }
   }            
   if (point[1]>=disty) {
      cn = dir[1]+fy*dir[2];
      if (cn>=0) return TGeoShape::Big();
      in = kFALSE;
      snxt = (disty-point[1])/cn;
      // find extrapolated X and Z
      znew = point[2]+snxt*dir[2];
      if (TMath::Abs(znew) < fDz) {
         Double_t dx = 0.5*(fDx1+fDx2)-fx*znew;
         xnew = point[0]+snxt*dir[0];
         if (TMath::Abs(xnew) < dx) return snxt;
      }
   }
   if (!in) return TGeoShape::Big();
   // Point actually inside
   if (safz<safx && safz<safy) {
      if (point[2]*dir[2]>=0) return TGeoShape::Big();
      return 0.0;
   }
   if (safy<safx) {
      cn = TMath::Sign(1.0,point[1])*dir[1]+fy*dir[2];     
      if (cn>=0) return TGeoShape::Big();
      return 0.0;
   }   
   cn = TMath::Sign(1.0,point[0])*dir[0]+fx*dir[2];     
   if (cn>=0) return TGeoShape::Big();
   return 0.0;      
}

//_____________________________________________________________________________
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
            
//_____________________________________________________________________________
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
      trd2->SetShapeBit(kGeoVisX);
      normals[0]=calf;
      normals[2]=salf;
   } else {   
      trd2->SetShapeBit(kGeoVisX, kFALSE);
      normals[0]=-calf;
      normals[2]=salf;
   }
   if (point[1]>disty) {
   // hi y face visible
      trd2->SetShapeBit(kGeoVisY);
      normals[4]=cbet;
      normals[5]=sbet;
   } else {
      trd2->SetShapeBit(kGeoVisY, kFALSE);
      normals[4]=-cbet; 
      normals[5]=sbet; 
   }   
   if (point[2]>fDz) {
   // hi z face visible
      trd2->SetShapeBit(kGeoVisZ);
      normals[8]=1;
   } else {
      trd2->SetShapeBit(kGeoVisZ, kFALSE);
      normals[8]=-1;  
   }
   SetVertex(vertex);
}

//_____________________________________________________________________________
void TGeoTrd2::GetOppositeCorner(Double_t * /*point*/, Int_t inorm, Double_t *vertex, Double_t *normals) const
{
// get the opposite corner of the intersected face
   TGeoTrd2 *trd2 = (TGeoTrd2*)this;
   if (inorm != 0) {
   // change x face
      trd2->SetShapeBit(kGeoVisX, !TestShapeBit(kGeoVisX));
      normals[0]=-normals[0];
   }
   if (inorm != 1) {
   // change y face
      trd2->SetShapeBit(kGeoVisY, !TestShapeBit(kGeoVisY));
      normals[4]=-normals[4];
   } 
   if (inorm != 2) {
   // hi z face visible
      trd2->SetShapeBit(kGeoVisZ, !TestShapeBit(kGeoVisZ));
      normals[8]=-normals[8];
   } 
   SetVertex(vertex);
}

//_____________________________________________________________________________
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

//_____________________________________________________________________________
void TGeoTrd2::GetBoundingCylinder(Double_t *param) const
{
//--- Fill vector param[4] with the bounding cylinder parameters. The order
// is the following : Rmin, Rmax, Phi1, Phi2
   TGeoBBox::GetBoundingCylinder(param);
}   

//_____________________________________________________________________________
Int_t TGeoTrd2::GetFittingBox(const TGeoBBox *parambox, TGeoMatrix *mat, Double_t &dx, Double_t &dy, Double_t &dz) const
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
      dd[2] = TMath::Min(origin[2]+fDz, fDz-origin[2]); 
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
   Double_t fx = 0.5*(fDx1-fDx2)/fDz;
   Double_t fy = 0.5*(fDy1-fDy2)/fDz;
   Double_t dx0 = 0.5*(fDx1+fDx2);
   Double_t dy0 = 0.5*(fDy1+fDy2);
   Double_t z=origin[2]-dd[2];
   dd[0] = dx0-fx*z-origin[0]; 
   dd[1] = dy0-fy*z-origin[1]; 
   z=origin[2]+dd[2];
   dd[0] = TMath::Min(dd[0], dx0-fx*z-origin[0]);
   dd[1] = TMath::Min(dd[1], dy0-fy*z-origin[1]);
   if (dd[0]<0 || dd[1]<0) {
      Error("GetFittingBox", "wrong matrix");
      return 1;
   }   
   dx = dd[0];
   dy = dd[1];
   dz = dd[2];
   return 0;
}   

//_____________________________________________________________________________
TGeoShape *TGeoTrd2::GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix * /*mat*/) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestShapeBit(kGeoRunTimeShape)) return 0;
   if (!mother->TestShapeBit(kGeoTrd2)) {
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

//_____________________________________________________________________________
void TGeoTrd2::InspectShape() const
{
// print shape parameters
   printf("*** Shape %s: TGeoTrd2 ***\n", GetName());
   printf("    dx1 = %11.5f\n", fDx1);
   printf("    dx2 = %11.5f\n", fDx2);
   printf("    dy1 = %11.5f\n", fDy1);
   printf("    dy2 = %11.5f\n", fDy2);
   printf("    dz  = %11.5f\n", fDz);
   printf(" Bounding box:\n");
   TGeoBBox::InspectShape();
}

//_____________________________________________________________________________
Double_t TGeoTrd2::Safety(Double_t *point, Bool_t in) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   Double_t saf[3];
   //--- Compute safety first
   // check Z facettes
   saf[0] = fDz-TMath::Abs(point[2]);
   Double_t fx = 0.5*(fDx1-fDx2)/fDz;
   Double_t calf = 1./TMath::Sqrt(1.0+fx*fx);
   // check X facettes
   Double_t distx = 0.5*(fDx1+fDx2)-fx*point[2];
   if (distx<0) saf[1]=TGeoShape::Big();
   else         saf[1]=(distx-TMath::Abs(point[0]))*calf;

   Double_t fy = 0.5*(fDy1-fDy2)/fDz;
   calf = 1./TMath::Sqrt(1.0+fy*fy);
   // check Y facettes
   distx = 0.5*(fDy1+fDy2)-fy*point[2];
   if (distx<0) saf[2]=TGeoShape::Big();
   else         saf[2]=(distx-TMath::Abs(point[1]))*calf;
   
   if (in) return saf[TMath::LocMin(3,saf)];
   for (Int_t i=0; i<3; i++) saf[i]=-saf[i];
   return saf[TMath::LocMax(3,saf)];
}

//_____________________________________________________________________________
void TGeoTrd2::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
// Save a primitive as a C++ statement(s) on output stream "out".
   if (TObject::TestBit(kGeoSavePrimitive)) return;
   out << "   // Shape: " << GetName() << " type: " << ClassName() << endl;
   out << "   dx1 = " << fDx1 << ";" << endl;
   out << "   dx2 = " << fDx2 << ";" << endl;
   out << "   dy1 = " << fDy1 << ";" << endl;
   out << "   dy2 = " << fDy2 << ";" << endl;
   out << "   dz  = " << fDZ  << ";" << endl;
   out << "   TGeoShape *" << GetPointerName() << " = new TGeoTrd2(\"" << GetName() << "\", dx1,dx2,dy1,dy2,dz);" << endl;  
   TObject::SetBit(TGeoShape::kGeoSavePrimitive);
} 
        
//_____________________________________________________________________________
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

//_____________________________________________________________________________
void TGeoTrd2::SetPoints(Double_t *points) const
{
// create trd2 mesh points
   if (!points) return;
   points[ 0] = -fDx1; points[ 1] = -fDy1; points[ 2] = -fDz;
   points[ 3] = -fDx1; points[ 4] =  fDy1; points[ 5] = -fDz;
   points[ 6] =  fDx1; points[ 7] =  fDy1; points[ 8] = -fDz;
   points[ 9] =  fDx1; points[10] = -fDy1; points[11] = -fDz;
   points[12] = -fDx2; points[13] = -fDy2; points[14] =  fDz;
   points[15] = -fDx2; points[16] =  fDy2; points[17] =  fDz;
   points[18] =  fDx2; points[19] =  fDy2; points[20] =  fDz;
   points[21] =  fDx2; points[22] = -fDy2; points[23] =  fDz;
}

//_____________________________________________________________________________
void TGeoTrd2::SetPoints(Float_t *points) const
{
// create trd2 mesh points
   if (!points) return;
   points[ 0] = -fDx1; points[ 1] = -fDy1; points[ 2] = -fDz;
   points[ 3] = -fDx1; points[ 4] =  fDy1; points[ 5] = -fDz;
   points[ 6] =  fDx1; points[ 7] =  fDy1; points[ 8] = -fDz;
   points[ 9] =  fDx1; points[10] = -fDy1; points[11] = -fDz;
   points[12] = -fDx2; points[13] = -fDy2; points[14] =  fDz;
   points[15] = -fDx2; points[16] =  fDy2; points[17] =  fDz;
   points[18] =  fDx2; points[19] =  fDy2; points[20] =  fDz;
   points[21] =  fDx2; points[22] = -fDy2; points[23] =  fDz;
}

//_____________________________________________________________________________
void TGeoTrd2::SetVertex(Double_t *vertex) const
{
// set vertex of a corner according to visibility flags
   if (TestShapeBit(kGeoVisX)) {
      if (TestShapeBit(kGeoVisZ)) {
         vertex[0] = fDx2;
         vertex[2] = fDz;
         vertex[1] = (TestShapeBit(kGeoVisY))?fDy2:-fDy2;
      } else {   
         vertex[0] = fDx1;
         vertex[2] = -fDz;
         vertex[1] = (TestShapeBit(kGeoVisY))?fDy1:-fDy1;
      }
   } else {
      if (TestShapeBit(kGeoVisZ)) {
         vertex[0] = -fDx2;
         vertex[2] = fDz;
         vertex[1] = (TestShapeBit(kGeoVisY))?fDy2:-fDy2;
      } else {   
         vertex[0] = -fDx1;
         vertex[2] = -fDz;
         vertex[1] = (TestShapeBit(kGeoVisY))?fDy1:-fDy1;
      }
   }            
} 

//_____________________________________________________________________________
void TGeoTrd2::Sizeof3D() const
{
// fill size of this 3-D object
   TGeoBBox::Sizeof3D();
}

