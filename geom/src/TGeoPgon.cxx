/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author :  Andrei Gheata  - date Thu 31 Jan 2002 01:47:40 PM CET
// TGeoPgon::Contains() implemented by Mihaela Gheata

#include "TROOT.h"

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoPainter.h"
#include "TGeoPgon.h"


 /*************************************************************************
 * TGeoPgon - a polygone. It has at least 10 parameters :
 *            - the lower phi limit;
 *            - the range in phi;
 *            - the number of edges on each z plane;
 *            - the number of z planes (at least two) where the inner/outer 
 *              radii are changing;
 *            - z coordinate, inner and outer radius for each z plane
 *
 *************************************************************************/
//Begin_Html
/*
<img src="gif/TGeoPgon.gif">
*/
//End_Html
   
ClassImp(TGeoPgon)

//-----------------------------------------------------------------------------
TGeoPgon::TGeoPgon()
{
// dummy ctor
   SetBit(TGeoShape::kGeoPgon);
   fNedges = 0;
}   
//-----------------------------------------------------------------------------
TGeoPgon::TGeoPgon(Double_t phi, Double_t dphi, Int_t nedges, Int_t nz)
         :TGeoPcon(phi, dphi, nz) 
{
// Default constructor
   SetBit(TGeoShape::kGeoPgon);
   fNedges = nedges;
   ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoPgon::TGeoPgon(Double_t *param)
{
// Default constructor in GEANT3 style
// param[0] = phi1
// param[1] = dphi
// param[2] = nedges
// param[3] = nz
//
// param[4] = z1
// param[5] = Rmin1
// param[6] = Rmax1
// ...
   SetBit(TGeoShape::kGeoPgon);
   SetDimensions(param);
   ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoPgon::~TGeoPgon()
{
// destructor
}
//-----------------------------------------------------------------------------
void TGeoPgon::ComputeBBox()
{
// compute bounding box for a polygone
   Double_t zmin = TMath::Min(fZ[0], fZ[fNz-1]);
   Double_t zmax = TMath::Max(fZ[0], fZ[fNz-1]);
   // find largest rmax an smallest rmin
   Double_t rmin, rmax;
   Double_t divphi = fDphi/fNedges;
   // find the radius of the outscribed circle
   rmin = fRmin[TMath::LocMin(fNz, fRmin)];   
   rmax = fRmax[TMath::LocMax(fNz, fRmax)];
   rmax = rmax/TMath::Cos(0.5*divphi*kDegRad);
   Double_t phi1 = fPhi1;
   Double_t phi2 = phi1 + fDphi;
   if (phi2 > 360) phi2-=360;
   
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
   if (ddp>360) ddp-=360;
   if (ddp<=fDphi) xmax = rmax;
   ddp = 90-phi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=fDphi) ymax = rmax;
   ddp = 180-phi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=fDphi) xmin = -rmax;
   ddp = 270-phi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=fDphi) ymin = -rmax;
   fOrigin[0] = (xmax+xmin)/2;
   fOrigin[1] = (ymax+ymin)/2;
   fOrigin[2] = (zmax+zmin)/2;
   fDX = (xmax-xmin)/2;
   fDY = (ymax-ymin)/2;
   fDZ = (zmax-zmin)/2;
}
//-----------------------------------------------------------------------------
Bool_t TGeoPgon::Contains(Double_t *point)
{
// test if point is inside this shape
   // check total z range
   if ((point[2]<fZ[0]) || (point[2]>fZ[fNz-1])) return kFALSE;
   // find smallest Rmin and largest Rmax
   Double_t rmin = fRmin[0];
   Double_t rmax = fRmax[0];
   for (Int_t i=1; i<fNz; i++) {
      if (fRmin[i] < rmin) rmin = fRmin[i];
      if (fRmax[i] > rmax) rmax = fRmax[i];
   }
   // check R against rmin
   Double_t r = TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   if (r < rmin) return kFALSE;
   Double_t divphi = fDphi/fNedges;
   // find the radius of the outscribed circle
   rmax = rmax/TMath::Cos(0.5*divphi*kDegRad);
   // check R against rmax
   if (r > rmax) return kFALSE;
   // now check phi
   Double_t phi = TMath::ATan2(point[1], point[0])*kRadDeg;
   if (phi < fPhi1) phi += 360.0;
   if ((phi<fPhi1) || ((phi-fPhi1)>fDphi)) return kFALSE;
   // now find phi division
   Int_t ipsec = (Int_t)TMath::Min((phi-fPhi1)/divphi+1., (Double_t)fNedges);
   Double_t ph0 = (fPhi1+divphi*(ipsec-0.5))*kDegRad;
   // now check projected distance
   r = point[0]*TMath::Cos(ph0) + point[1]*TMath::Sin(ph0);
   // find in which Z section the point is in
   Int_t izl = 0;
   Int_t izh = fNz-1;
   Int_t izt = 0;
   while ((izh-izl)>1) {
      izt = (izl+izh)/2;
      if (point[2] < fZ[izt]) izh = izt;
      else izl=izt;
   }
   // now compute rmin and rmax and test the value of r
   Double_t dzrat = (point[2]-fZ[izl])/(fZ[izh]-fZ[izl]);
   rmin = fRmin[izl]+dzrat*(fRmin[izh]-fRmin[izl]);
   // is the point inside the 'hole' at the center of the volume ?
   if (r < rmin) return kFALSE;
   rmax = fRmax[izl]+dzrat*(fRmax[izh]-fRmax[izl]);
   if (r > rmax) return kFALSE;
   
   return kTRUE;
}
//-----------------------------------------------------------------------------
void TGeoPgon::DefineSection(Int_t snum, Double_t z, Double_t rmin, Double_t rmax)
{
// defines z position of a section plane, rmin and rmax at this z.
   if ((snum<0) || (snum>=fNz)) return;
   fZ[snum]    = z;
   fRmin[snum] = rmin;
   fRmax[snum] = rmax;
   if (snum==(fNz-1)) ComputeBBox();
}
//-----------------------------------------------------------------------------
Double_t TGeoPgon::DistToOutSect(Double_t *point, Double_t *dir, Int_t &iz, Int_t &isect)
{
// compute distance to outside from a  pgon phi trapezoid
//   printf("Checking sector : iz=%i isect=%i\n", iz, isect);
//   printf(" point is : %f %f %f\n", point[0], point[1], point[2]);
//   printf(" dir   is : %f %f %f\n", dir[0], dir[1], dir[2]);
   Double_t saf, dist;
   Double_t zmin = fZ[iz];
   Double_t zmax = fZ[iz+1];
   if (zmax==zmin) return 0;
   Double_t divphi = fDphi/fNedges;
   Double_t phi1 = (fPhi1 + divphi*(isect-1))*kDegRad;
   Double_t phi2 = phi1 + divphi*kDegRad;
//   printf(" phi1=%f phi2=%f\n", phi1*kRadDeg, phi2*kRadDeg);
   Double_t phim = 0.5*(phi1+phi2);
   Double_t cphim = TMath::Cos(phim);
   Double_t sphim = TMath::Sin(phim);
   Double_t *norm = gGeoManager->GetNormalChecked();
   Double_t *snorm = gGeoManager->GetCldirChecked();
   Double_t minsafe = 0;
   Double_t dmin = kBig;
   Double_t no[3];
   Bool_t intersect = kFALSE;
   // check outer slanted face
   Double_t ct, st;
   Double_t fz = (fRmax[iz+1]-fRmax[iz])/(zmax-zmin);
//   printf("alfa=%f\n", TMath::ATan(fzo)*kRadDeg);
   st = 1./TMath::Sqrt(1.+fz*fz);
   ct = -fz*st;
   if (st<0) st=-st;
//   printf("theta outer : cto=%f sto=%f\n", cto, sto);
   // normal
   no[0] = st*cphim;
   no[1] = st*sphim;
   no[2] = ct;
//   printf("normal to outer : %f %f %f\n", no[0], no[1], no[2]);
   saf = (fRmax[iz]*cphim-point[0])*no[0]+
         (fRmax[iz]*sphim-point[1])*no[1]+
         (fZ[iz]-point[2])*no[2];
   minsafe = saf;
   memcpy(snorm, &no[0], 3*sizeof(Double_t));         
//   printf("safe to outer : %f\n", saf[0]);
   Double_t calf = dir[0]*no[0]+dir[1]*no[1]+dir[2]*no[2];
   if (calf>0) {
      dmin = saf/calf;
      memcpy(norm, &no[0], 3*sizeof(Double_t));
   }   
//   printf("out = %f\n", dist[0]);

   // check inner slanted face
   fz = (fRmin[iz+1]-fRmin[iz])/(zmax-zmin);
//   printf("alfa=%f\n", TMath::ATan(fzi)*kRadDeg);
   st = -1./TMath::Sqrt(1.+fz*fz);
   ct = -fz*st;
   if (st<0) st=-st;
//   printf("theta inner : cto=%f sto=%f\n", cti, sti);
   // normal
   no[0] = -st*cphim;
   no[1] = -st*sphim;
   no[2] = ct;
//   printf("normal to inner : %f %f %f\n", ni[0], ni[1], ni[2]);
   saf = (fRmin[iz]*cphim-point[0])*no[0]+
         (fRmin[iz]*sphim-point[1])*no[1]+
         (fZ[iz]-point[2])*no[2];
   if (saf<minsafe) {
      minsafe = saf;
      memcpy(snorm, &no[0], 3*sizeof(Double_t));
   }            
//   printf("safe to inner : %f\n", saf[1]);
   calf = dir[0]*no[0]+dir[1]*no[1]+dir[2]*no[2];
   if (calf>0) {
      dist = saf/calf;
      if (dist<dmin) {
         dmin = dist;
         memcpy(norm, &no[0], 3*sizeof(Double_t));
      }   
   }   
//   printf("in  = %f\n", dist[1]);
               
   // check upper and lower Z planes
   saf = point[2]-fZ[iz];
   if (saf<minsafe) {
      minsafe = saf;
      memset(snorm, 0, 3*sizeof(Int_t));
      snorm[2] = -1;
   }            
//   printf("safe to down : %f\n", saf[2]);
   if (dir[2]<0) {
      intersect = kTRUE;
      dist=-saf/dir[2];
      if (dist<dmin) {
         dmin = dist;
         memset(norm, 0, 3*sizeof(Double_t));
         norm[2] = -1;
      }   
   }   
//   printf("down= %f\n", dist[2]);
   saf = fZ[iz+1]-point[2];
   if (saf<minsafe) {
      minsafe = saf;
      memset(snorm, 0, 3*sizeof(Int_t));
      snorm[2] = 1;
   }            
//   printf("safe to up : %f\n", saf[3]);
   if (!intersect) {
      if (dir[2]>0) {
         dist=saf/dir[2];
         if (dist<dmin) {
            dmin = dist;
            memset(norm, 0, 3*sizeof(Double_t));
            norm[2] = 1;
         }   
      }
   } else {
      intersect = kFALSE;
   }         
//   printf("up  = %f\n", dist[3]);
   // check phi1 and phi2 walls
   Double_t r = TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t phi = TMath::ATan2(point[1], point[0]);
   if (phi<phi1) phi+=2*TMath::Pi();
//   printf("phi point : %f\n", phi*kRadDeg);
   no[0] = TMath::Sin(phi1);
   no[1] = -TMath::Cos(phi1);
   no[2] = 0;
//   printf("normal to phi1 : %f %f %f\n", nph1[0], nph1[1], nph1[2]);
   saf = TMath::Abs(r*TMath::Sin(phi-phi1));
   calf = dir[0]*no[0]+dir[1]*no[1]+dir[2]*no[2];
//   printf(" angle between dir and N: %f\n", TMath::ACos(calf)*kRadDeg);
   if (calf>0) {
      dist = saf/calf;
      if (dist<dmin) {
         // we have to check the other wall too
         Double_t no2[3];
         no2[0] = -TMath::Sin(phi2);
         no2[1] = TMath::Cos(phi2);
         no2[2] = 0;
         Double_t saf2 = TMath::Abs(r*TMath::Sin(phi2-phi));
         calf = dir[0]*no2[0]+dir[1]*no2[1]+dir[2]*no2[2];
         if (calf>0) {
            Double_t dist2 = saf2/calf;
            if (dist2<dist) {
               isect++;
               // check if phi1 wall is real
               if (isect>fNedges) {
                  if (fDphi==360) {
                     isect=1; 
                  } else {
                  // this was last sector
                     dmin = dist2;
                     memcpy(norm, &no2[0], 3*sizeof(Double_t));
                     if (saf2<minsafe) {
                        minsafe = saf2;
                        memcpy(snorm, &no2[0], 3*sizeof(Double_t));
                     }
                     if ((fNedges==1) && (saf<saf2)) {
                        minsafe = saf;
                        memcpy(snorm, &no[0], 3*sizeof(Double_t));
                     }
                     return dmin;            
                  }
               } 
               // propagate to next sector
               dmin = dist + 1E-12;  // be sure to propagate INSIDE next sector
               for (Int_t i=0; i<3; i++) point[i]+=dmin*dir[i];
               dmin += DistToOutSect(point, dir, iz, isect);
               return dmin;   
            }   
         }
         isect--;
         // check if phi1 wall is real
         if (isect==0) {
            if (fDphi==360) {
               isect=fNedges; 
            } else {
            // this was last sector
               dmin = dist;
               memcpy(norm, &no[0], 3*sizeof(Double_t));
               if (saf<minsafe) {
                  minsafe = saf;
                  memcpy(snorm, &no[0], 3*sizeof(Double_t));
               } 
               return dmin;            
            }
         } 
         // propagate to next sector
         dmin = dist + 1E-12;  // be sure to propagate INSIDE next sector
         for (Int_t i=0; i<3; i++) point[i]+=dmin*dir[i];
         dmin += DistToOutSect(point, dir, iz, isect);
         return dmin;   
      }   
   }   
   if (fDphi != 360) {
      if ((isect==1) && (saf<minsafe)) {
         minsafe = saf;
         memcpy(snorm, &no[0], 3*sizeof(Double_t));
      }   
   }            
//   printf("phi1= %f\n", dist[4]);

//   printf("safe to phi1 : %f\n", saf[4]);
   no[0] = -TMath::Sin(phi2);
   no[1] = TMath::Cos(phi2);
   no[2] = 0;
//   printf("normal to phi2 : %f %f %f\n", nph2[0], nph2[1], nph2[2]);
   saf = TMath::Abs(r*TMath::Sin(phi2-phi));
//   printf("safe to phi2 : %f\n", saf[5]);
   calf = dir[0]*no[0]+dir[1]*no[1]+dir[2]*no[2];
//   printf(" angle between dir and N: %f\n", TMath::ACos(calf)*kRadDeg);
   if (calf>0) {
      dist = saf/calf;
      if (dist<dmin) {
         isect++;
         // check if phi1 wall is real
         if (isect>fNedges) {
            if (fDphi==360) {
               isect=1; 
            } else {
            // this was last sector
               dmin = dist;
               memcpy(norm, &no[0], 3*sizeof(Double_t));
               if (saf<minsafe) {
                  minsafe = saf;
                  memcpy(snorm, &no[0], 3*sizeof(Double_t));
               } 
               return dmin;            
            }
         } 
         // propagate to next sector
         dmin = dist + 1E-12;  // be sure to propagate INSIDE next sector
         for (Int_t i=0; i<3; i++) point[i]+=dmin*dir[i];
         dmin += DistToOutSect(point, dir, iz, isect);
         return dmin;   
      }   
   }
   if (fDphi != 360) {
      if ((isect==fNedges) && (saf<minsafe)) {
         minsafe = saf;
         memcpy(snorm, &no[0], 3*sizeof(Double_t));
      }   
   }            
//   printf("phi2= %f\n", dist[5]);
   return dmin; 
//   if (imin==0)
}
//-----------------------------------------------------------------------------
Double_t TGeoPgon::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from inside point to surface of the polygone
   // first find out in which Z section the point is in
   Int_t ipl = TMath::BinarySearch(fNz, fZ, point[2]);
   if ((ipl==(fNz-1)) || (ipl<0)) {
      // point out
      Warning("DistToOut", "point is outside Z range");
      return kBig;
   }
//   Double_t dz = 0.5*(fZ[ipl+1]-fZ[ipl]);

   // now find out in which phi section the point is in
   Double_t divphi = fDphi/fNedges;
   Double_t phi = TMath::ATan2(point[1], point[0])*kRadDeg;
   if (phi < fPhi1) phi += 360.0;
   if ((phi<fPhi1) || ((phi-fPhi1)>fDphi)) return kBig;
   // now find phi division [1..fNedges]
   Double_t pt[3];
   memcpy(&pt[0], point, 3*sizeof(Double_t));
   Int_t ipsec = (Int_t)TMath::Min((phi-fPhi1)/divphi+1., (Double_t)fNedges);
   Double_t dsec = DistToOutSect(pt, dir, ipl, ipsec);
//   Double_t ph0 = (fPhi1+divphi*(ipsec-0.5))*kDegRad;
   
   // find the radius of the outscribed circle
//   rmin = fRmin[TMath::LocMin(fNz, fRmin)];   
//   rmax = fRmax[TMath::LocMax(fNz, fRmax)];
//   rmax = rmax/TMath::Cos(0.5*divphi*kDegRad);


//   Double_t saf[4];
/*
   Double_t snxt = kBig;
   Double_t r2 = point[0]*point[0] + point[1]*point[1];
   Double_t r = TMath::Sqrt(r2);
   
   Double_t pdiv = fDphi/fNedges;
   Double_t delphi = pdiv*kDegRad;
   Double_t dphi2 = 0.5*delphi;
   Double_t csdph2 = TMath::Cos(dphi2);
   Double_t zmin = fZ[0];
   Double_t zmax = fZ[fNz-1];
   Double_t safz1 = point[2]-zmin;
   Double_t safz2 = zmax-point[2];
   Double_t safz = TMath::Min(safz1, safz2);
   // find the segment containing the point
   Int_t ipl = TMath::BinarySearch(fNz, fZ, point[2]);
   if (ipl==(fNz-1)) {
      // point on end z plane
      if (safe) *safe=0;
      return 0;
   }
   Double_t dz = 0.5*(fZ[ipl+1]-fZ[ipl]);
*/
   return dsec;   
}   
//-----------------------------------------------------------------------------
Double_t TGeoPgon::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from outside point to surface of the polygone
   // first find in which segment we are
   UChar_t bits=0;
   const UChar_t kUp = 0x01;
   const UChar_t kDown = 0x02;
   const UChar_t kOut  = kUp | kDown;
   const UChar_t kInhole = 0x04;
   const UChar_t kOuthole = 0x08;
   const UChar_t kInphi = 0x10;
   Bool_t cross=kTRUE;
   // check if ray may intersect outscribed cylinder
   if ((point[2]<fZ[0]) && (dir[2]<=0)) {
      if (iact==3) return kBig; 
      cross=kFALSE;
   }
   if (cross) {
      if ((point[2]>fZ[fNz-1]) && (dir[2]>=0)) {
         if (iact==3) return kBig; 
         cross=kFALSE;
      }
   }   
   Double_t r2 = point[0]*point[0]+point[1]*point[1];
   Double_t radmax=0;
   Double_t divphi=fDphi/fNedges;
   if (cross) {
      radmax = fRmax[TMath::LocMax(fNz, fRmax)];
      radmax = radmax/TMath::Cos(0.5*divphi*kDegRad);
      if (r2>(radmax*radmax)) {
         Double_t rpr=point[0]*dir[0]+point[1]*dir[1];
         if (rpr>TMath::Sqrt(r2-radmax*radmax)) {
            if (iact==3) return kBig;
            cross=kFALSE;
         }
      }
   }        

   Double_t r = TMath::Sqrt(r2);
   Double_t saf[8];


   Int_t ipl = TMath::BinarySearch(fNz, fZ, point[2]);
   Int_t ifirst = ipl;
   if (ifirst<0) {
      ifirst=0;
      bits |= kDown;
   } else {
      if (ifirst==(fNz-1)) {
         ifirst=fNz-2;
         bits |= kUp;
      } 
   }      
   if (!(bits & kOut)) {
      saf[0]=point[2]-fZ[ifirst];
      saf[1]=fZ[ifirst+1]-point[2];
   } else {
      if (ipl<0) {
         saf[0]=fZ[ifirst]-point[2];
         saf[1]=-kBig;
      } else {
         saf[0]=-kBig;
         saf[1]=point[2]-fZ[ifirst+1];
      }   
   }    
   // find out if point is in the hole of current segment or outside
   Double_t phi = TMath::ATan2(point[1], point[0])*kRadDeg;
   Double_t phi1, phi2;
   if (phi<fPhi1) phi+=360.;
   Int_t ipsec = Int_t((phi-fPhi1)/divphi+1.);
   if (ipsec>fNedges) {
   // point in gap mellon slice
      ipsec = -1;
      saf[2]=saf[3]=-kBig;
      phi1=saf[6]=fPhi1;
      phi2=saf[7]=fPhi1+fDphi;
   } else {
      bits |= kInphi;
      Double_t ph0=(fPhi1+divphi*(ipsec-0.5))*kDegRad;
      phi1=saf[6]=fPhi1+(ipsec-1)*divphi;
      phi2=saf[7]=phi1+divphi;
      // projected distance
      Double_t rproj=point[0]*TMath::Cos(ph0)+point[1]*TMath::Sin(ph0);
//   Double_t r2=point[0]*point[0]+point[1]*point[1];
      Double_t dzrat=(point[2]-fZ[ifirst])/(fZ[ifirst+1]-fZ[ifirst]);
      // rmin and rmax at Z coordinate of the point
      Double_t rmin=fRmin[ifirst]+(fRmin[ifirst+1]-fRmin[ifirst])*dzrat;
      Double_t rmax=fRmax[ifirst]+(fRmax[ifirst+1]-fRmax[ifirst])*dzrat;
      if ((rmin>0) && (rproj<rmin)) bits |= kInhole;
      if (rproj>rmax) bits |= kOuthole;
      Double_t tin=(fRmin[ifirst+1]-fRmin[ifirst])/(fZ[ifirst+1]-fZ[ifirst]);
      Double_t cin=1./TMath::Sqrt(1.0+tin*tin);
      Double_t tou=(fRmax[ifirst+1]-fRmax[ifirst])/(fZ[ifirst+1]-fZ[ifirst]);
      Double_t cou=1./TMath::Sqrt(1.0+tou*tou);
      saf[2] = (bits & kInhole)?((rmin-rproj)*cin):-kBig;
      saf[3] = (bits & kOuthole)?((rproj-rmax)*cou):-kBig;
   }
   // find closest distance to phi walls
   Double_t dph1=(bits & kInphi)?(phi-phi1):(phi1-phi);
   Double_t dph2=(bits & kInphi)?(phi2-phi):(phi-phi2);
   saf[4]=r*TMath::Sin(dph1*kDegRad);
   saf[5]=r*TMath::Sin(dph2*kDegRad);   
/*
   if (bits & kUp) printf("UP\n");
   if (bits & kDown) printf("DOWN\n");
   if (!(bits & kOut)) printf("IN Z sector %i\n", ifirst);
   printf("SECTOR: %i\n",ipsec);
   printf("safz=%f safz=%f\n", saf[0], saf[1]);
   if (bits & kInhole) printf("INHOLE safin=%f\n", saf[2]);
   if (bits & kOuthole) printf("OUTHOLE safout=%f\n", saf[3]);
   if (bits & kInphi) printf("INPHI\n");
   printf("phi1=%f phi2=%f phi=%f r=%f\n", phi1, phi2, phi, r);
   printf("safphi1=%f  safphi2=%f\n", saf[4], saf[5]);
   printf("DIRECTION : nx=%f ny=%f nz=%f\n", dir[0], dir[1], dir[2]);
*/
   if ((iact<3) && safe) {
      *safe = saf[TMath::LocMax(6, &saf[0])];
      if ((iact==1) && (*safe>step)) return step;
      if (iact==0) return kBig;
   }
   // compute distance to boundary   
   if (!cross) return kBig;
   Double_t pt[3];
   memcpy(&pt[0], point, 3*sizeof(Double_t));
   Double_t snxt=DistToInSect(pt, dir, ifirst, ipsec, bits, &saf[0]);
   return snxt;


}
//-----------------------------------------------------------------------------
Double_t TGeoPgon::DistToInSect(Double_t *point, Double_t *dir, Int_t &iz, Int_t &ipsec, 
                                UChar_t &bits, Double_t *saf) 
{
   // propagate to next Z plane
   const UChar_t kUp = 0x01;
   const UChar_t kDown = 0x02;
//   const UChar_t kOut  = kUp | kDown;
   const UChar_t kInhole = 0x04;
   const UChar_t kOuthole = 0x08;
   const UChar_t kInphi = 0x10;
   Double_t nwall[3];
   Double_t *norm = gGeoManager->GetNormalChecked();
   Double_t s=kBig;
   Double_t snxt=kBig;
   memset(norm, 0, 2*sizeof(Double_t));
   if (bits & kUp) {
      if (dir[2]>=0) return kBig;
      norm[2]=1;
      snxt=-saf[1]/dir[2];
   } else {   
      if (bits & kDown) {
         if (dir[2]<=0) return kBig;
         norm[2]=-1;
         snxt=saf[0]/dir[2];
      } else {
         if (dir[2]>0) {
            snxt=saf[1]/dir[2];
            norm[2]=-1;
         } else {
            if (dir[2]<0) {
               snxt=-saf[0]/dir[2];
               norm[2]=1;
            }
         }         
      }
   }      
//   printf("dist to Z : %f\n", snxt);
   // propagate to closest wall
   Double_t calf,tz, st, ct, sp, cp;
//   Double_t divphi=fDphi/fNedges;
   Double_t phi1 = saf[6];
   Double_t phi2 = saf[7];
   Double_t ph0;
   if (bits & kInphi) {
      ph0=0.5*(phi1+phi2)*kDegRad;
      sp = -TMath::Sin(ph0);
      cp = -TMath::Cos(ph0);
      if (bits & kInhole) {
         tz = (fRmin[iz+1]-fRmin[iz])/(fZ[iz+1]-fZ[iz]);
         st = 1./TMath::Sqrt(1.0+tz*tz);
         ct = st*tz;
//         printf("norm to inner : st=%f ct=%f sp=%f cp=%f\n", st,ct,sp,cp);
         nwall[0]=st*cp;
         nwall[1]=st*sp;
         nwall[2]=ct;
         calf = nwall[0]*dir[0]+nwall[1]*dir[1]+nwall[2]*dir[2];
         if (calf<0) {
            s=-saf[2]/calf;
//            printf("dist to inner : %f\n", s);
            if (s<snxt) {
               memcpy(norm, &nwall[0], 3*sizeof(Double_t));
               snxt=s;
            }
         }
      }
      if (bits & kOuthole) {
          sp = -sp;
         cp = -cp;
         tz = (fRmax[iz]-fRmax[iz+1])/(fZ[iz+1]-fZ[iz]);
         st = 1./TMath::Sqrt(1.0+tz*tz);
         ct = st*tz;
//         printf("norm to outer : st=%f ct=%f sp=%f cp=%f\n", st,ct,sp,cp);
         nwall[0]=st*cp;
         nwall[1]=st*sp;
         nwall[2]=ct;
         calf = nwall[0]*dir[0]+nwall[1]*dir[1]+nwall[2]*dir[2];
         if (calf<0) {
            s=-saf[3]/calf;
//            printf("dist to outer : %f\n", s);
            if (s<snxt) {
               memcpy(norm, &nwall[0], 3*sizeof(Double_t));
               snxt=s;
            }
         }
      }
   }   
   // propagate to phi planes
   if (saf[4]>0) {
      nwall[0]=-TMath::Sin(phi1*kDegRad);
      nwall[1]=TMath::Cos(phi1*kDegRad);
      nwall[2]=0;
      if (!(bits & kInphi)) {
         nwall[0] = -nwall[0];
         nwall[1] = -nwall[1];
      }   
//      printf("norm to phi1 : nx=%f ny=%f\n", nwall[0], nwall[1]);
      calf= nwall[0]*dir[0]+nwall[1]*dir[1]+nwall[2]*dir[2];
      if (calf<0) {
         s=-saf[4]/calf;
//         printf("dist to phi1 : %f\n", s);
         if (s<snxt) {
            memcpy(norm, &nwall[0], 3*sizeof(Double_t));
            snxt=s;
         }
      }
   }      

   if (saf[5]>0) {
      nwall[0]=TMath::Sin(phi2*kDegRad);
      nwall[1]=-TMath::Cos(phi2*kDegRad);
      nwall[2]=0;
      if (!(bits & kInphi)) {
         nwall[0] = -nwall[0];
         nwall[1] = -nwall[1];
      }   
//      printf("norm to phi2 : nx=%f ny=%f\n", nwall[0], nwall[1]);
      calf= nwall[0]*dir[0]+nwall[1]*dir[1]+nwall[2]*dir[2];
      if (calf<0) {
         s=-saf[5]/calf;
//         printf("dist to phi2 : %f\n", s);
         if (s<snxt) {
            memcpy(norm, &nwall[0], 3*sizeof(Double_t));
            snxt=s;
         }
      }
   }      
   for (Int_t i=0; i<3; i++) point[i]+=dir[i]*(snxt+1E-9);
   if (Contains(point)) return snxt;
   snxt += DistToIn(point, dir, 3);
   return snxt;        
}
//-----------------------------------------------------------------------------
Int_t TGeoPgon::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute closest distance from point px,py to each corner
   Int_t n = fNedges+1;
   const Int_t numPoints = 2*n*fNz;
   return ShapeDistancetoPrimitive(numPoints, px, py);
}
//-----------------------------------------------------------------------------
Double_t TGeoPgon::DistToSurf(Double_t *point, Double_t *dir)
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   return kBig;
}
//-----------------------------------------------------------------------------
void TGeoPgon::Draw(Option_t *option)
{
// draw this shape according to option
}
//-----------------------------------------------------------------------------
void TGeoPgon::InspectShape()
{
   printf("*** TGeoPgon parameters ***\n");
   printf("    Nedges = %i\n", fNedges);
   TGeoPcon::InspectShape();
}
//-----------------------------------------------------------------------------
void TGeoPgon::Paint(Option_t *option)
{
// paint this shape according to option
   TGeoPainter *painter = (TGeoPainter*)gGeoManager->GetMakeDefPainter();
   if (!painter) return;
   TGeoVolume *vol = gGeoManager->GetCurrentVolume();
   if (vol->GetShape() != (TGeoShape*)this) return;
   painter->PaintPcon(vol, option);
}
//-----------------------------------------------------------------------------
void TGeoPgon::NextCrossing(TGeoParamCurve *c, Double_t *point)
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoPgon::Safety(Double_t *point, Double_t *spoint, Option_t *option)
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return 0.0;
}
//-----------------------------------------------------------------------------
void TGeoPgon::SetDimensions(Double_t *param)
{
   fPhi1    = param[0];
   fDphi    = param[1];
   fNedges  = (Int_t)param[2];
   fNz      = (Int_t)param[3];
   if (!fRmin) fRmin = new Double_t [fNz];
   if (!fRmax) fRmax = new Double_t [fNz];
   if (!fZ)    fZ    = new Double_t [fNz];
   for (Int_t i=0; i<fNz; i++) 
      DefineSection(i, param[4+3*i], param[5+3*i], param[6+3*i]);
}   
//-----------------------------------------------------------------------------
void TGeoPgon::SetPoints(Double_t *buff) const
{
// create polygone mesh points
    Double_t phi, dphi;
    Int_t n = fNedges + 1;
    dphi = fDphi/(n-1);
    Double_t factor = 1./TMath::Cos(kDegRad*dphi/2);
    Int_t i, j;
    Int_t indx = 0;

    if (buff) {
        for (i = 0; i < fNz; i++)
        {
            for (j = 0; j < n; j++)
            {
                phi = (fPhi1+j*dphi)*kDegRad;
                buff[indx++] = factor * fRmin[i] * TMath::Cos(phi);
                buff[indx++] = factor * fRmin[i] * TMath::Sin(phi);
                buff[indx++] = fZ[i];
            }
            for (j = 0; j < n; j++)
            {
                phi = (fPhi1+j*dphi)*kDegRad;
                buff[indx++] = factor * fRmax[i] * TMath::Cos(phi);
                buff[indx++] = factor * fRmax[i] * TMath::Sin(phi);
                buff[indx++] = fZ[i];
            }
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoPgon::SetPoints(Float_t *buff) const
{
// create polygone mesh points
    Double_t phi, dphi;
    Int_t n = fNedges + 1;
    dphi = fDphi/(n-1);
    Double_t factor = 1./TMath::Cos(kDegRad*dphi/2);
    Int_t i, j;
    Int_t indx = 0;

    if (buff) {
        for (i = 0; i < fNz; i++)
        {
            for (j = 0; j < n; j++)
            {
                phi = (fPhi1+j*dphi)*kDegRad;
                buff[indx++] = factor * fRmin[i] * TMath::Cos(phi);
                buff[indx++] = factor * fRmin[i] * TMath::Sin(phi);
                buff[indx++] = fZ[i];
            }
            for (j = 0; j < n; j++)
            {
                phi = (fPhi1+j*dphi)*kDegRad;
                buff[indx++] = factor * fRmax[i] * TMath::Cos(phi);
                buff[indx++] = factor * fRmax[i] * TMath::Sin(phi);
                buff[indx++] = fZ[i];
            }
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoPgon::Sizeof3D() const
{
// fill size of this 3-D object
    Int_t n;

    n = fNedges+1;

    gSize3D.numPoints += fNz*2*n;
    gSize3D.numSegs   += 4*(fNz*n-1+(fDphi == 360));
    gSize3D.numPolys  += 2*(fNz*n-1+(fDphi == 360));
}
