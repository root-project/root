// @(#)root/geom:$Id$
// Author: Mihaela Gheata   24/01/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_____________________________________________________________________________
// TGeoXtru 
//==========
//   An extrusion with fixed outline shape in x-y and a sequence
// of z extents (segments).  The overall scale of the outline scales
// linearly between z points and the center can have an x-y offset.
//
// Creation of TGeoXtru shape
//=============================
//   A TGeoXtru represents a polygonal extrusion. It is defined by the:
// a. 'Blueprint' of the arbitrary polygon representing any Z section. This
//    is an arbytrary polygon (convex or not) defined by the X/Y positions of
//    its vertices.
// b. A sequence of Z sections ordered on the Z axis. Each section defines the
//   'actual' parameters of the polygon at a given Z. The sections may be 
//    translated with respect to the blueprint and/or scaled. The TGeoXtru
//   segment in between 2 Z sections is a solid represented by the linear 
//   extrusion between the 2 polygons. Two consecutive sections may be defined
//   at same Z position.
//
// 1. TGeoXtru *xtru = TGeoXtru(Int_t nz);
//   where nz=number of Z planes
// 2. Double_t x[nvertices]; // array of X positions of blueprint polygon vertices
//    Double_t y[nvertices]; // array of Y positions of blueprint polygon vertices
// 3. xtru->DefinePolygon(nvertices,x,y);
// 4. DefineSection(0, z0, x0, y0, scale0); // Z position, offset and scale for first section
//    DefineSection(1, z1, x1, y1, scale1); // -''- secons section
//    ....
//    DefineSection(nz-1, zn, xn, yn, scalen); // parameters for last section
//
// *NOTES*
// Currently navigation functionality not fully implemented (only Contains()).
// Decomposition in concave polygons not implemented - drawing in solid mode
// within x3d produces incorrect end-faces
//_____________________________________________________________________________

#include "Riostream.h"

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoPolygon.h"
#include "TVirtualGeoPainter.h"
#include "TGeoXtru.h"
#include "TVirtualPad.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TClass.h"
#include "TMath.h"

ClassImp(TGeoXtru)

//_____________________________________________________________________________
TGeoXtru::TGeoXtru()
{
// dummy ctor
   SetShapeBit(TGeoShape::kGeoXtru);
   fNvert = 0;
   fNz = 0;
   fZcurrent = 0.;
   fPoly = 0;
   fX = 0;
   fY = 0;
   fXc = 0;
   fYc = 0;
   fZ = 0;
   fScale = 0;
   fX0 = 0;
   fY0 = 0;
   fSeg = 0;
   fIz = 0;
}   

//_____________________________________________________________________________
TGeoXtru::TGeoXtru(Int_t nz)
         :TGeoBBox(0, 0, 0)
{
// Default constructor
   SetShapeBit(TGeoShape::kGeoXtru);
   if (nz<2) {
      Error("ctor", "Cannot create TGeoXtru %s with less than 2 Z planes", GetName());
      SetShapeBit(TGeoShape::kGeoBad);
      return;
   }
   fNvert = 0;
   fNz = nz;   
   fZcurrent = 0.;
   fPoly = 0;
   fX = 0;
   fY = 0;
   fXc = 0;
   fYc = 0;
   fZ = new Double_t[nz];
   fScale = new Double_t[nz];
   fX0 = new Double_t[nz];
   fY0 = new Double_t[nz];
   fSeg = 0;
   fIz = 0;
}

//_____________________________________________________________________________
TGeoXtru::TGeoXtru(Double_t *param)
         :TGeoBBox(0, 0, 0)
{
// Default constructor in GEANT3 style
// param[0] = nz  // number of z planes
//
// param[1] = z1  // Z position of first plane
// param[2] = x1  // X position of first plane
// param[3] = y1  // Y position of first plane
// param[4] = scale1  // scale factor for first plane
// ...
// param[4*(nz-1]+1] = zn
// param[4*(nz-1)+2] = xn
// param[4*(nz-1)+3] = yn
// param[4*(nz-1)+4] = scalen
   SetShapeBit(TGeoShape::kGeoXtru);
   fNvert = 0;
   fNz = 0;
   fZcurrent = 0.;
   fPoly = 0;
   fX = 0;
   fY = 0;
   fXc = 0;
   fYc = 0;
   fZ = 0;
   fScale = 0;
   fX0 = 0;
   fY0 = 0;
   fSeg = 0;
   fIz = 0;
   SetDimensions(param);
}

//_____________________________________________________________________________
TGeoXtru::TGeoXtru(const TGeoXtru& xt) :
  TGeoBBox(xt),
  fNvert(xt.fNvert),
  fNz(xt.fNz),
  fZcurrent(xt.fZcurrent),
  fPoly(xt.fPoly),
  fX(xt.fX),
  fY(xt.fY),
  fXc(xt.fXc),
  fYc(xt.fYc),
  fZ(xt.fZ),
  fScale(xt.fScale),
  fX0(xt.fX0),
  fY0(xt.fY0),
  fSeg(xt.fSeg),
  fIz(xt.fIz)
{ 
   //copy constructor
}

//_____________________________________________________________________________
TGeoXtru& TGeoXtru::operator=(const TGeoXtru& xt)
{
   //assignment operator
   if(this!=&xt) {
      TGeoBBox::operator=(xt);
      fNvert=xt.fNvert;
      fNz=xt.fNz;
      fZcurrent=xt.fZcurrent;
      fPoly=xt.fPoly;
      fX=xt.fX;
      fY=xt.fY;
      fXc=xt.fXc;
      fYc=xt.fYc;
      fZ=xt.fZ;
      fScale=xt.fScale;
      fX0=xt.fX0;
      fY0=xt.fY0;
      fSeg=xt.fSeg;
      fIz=xt.fIz;
   } 
   return *this;
}

//_____________________________________________________________________________
TGeoXtru::~TGeoXtru()
{
// destructor
   if (fX)  {delete[] fX; fX = 0;}
   if (fY)  {delete[] fY; fY = 0;}
   if (fXc) {delete[] fXc; fXc = 0;}
   if (fYc) {delete[] fYc; fYc = 0;}
   if (fZ)  {delete[] fZ; fZ = 0;}
   if (fScale) {delete[] fScale; fScale = 0;}
   if (fX0)  {delete[] fX0; fX0 = 0;}
   if (fY0)  {delete[] fY0; fY0 = 0;}
}

//_____________________________________________________________________________   
Double_t TGeoXtru::Capacity() const
{
// Compute capacity [length^3] of this shape.
   Int_t iz;
   Double_t capacity = 0;
   Double_t area, dz, sc1, sc2;
   TGeoXtru *xtru = (TGeoXtru*)this;
   xtru->SetCurrentVertices(0.,0.,1.);  
   area = fPoly->Area();
   for (iz=0; iz<fNz-1; iz++) {
      dz = fZ[iz+1]-fZ[iz];
      if (TGeoShape::IsSameWithinTolerance(dz,0)) continue;
      sc1 = fScale[iz];
      sc2 = fScale[iz+1];
      capacity += (area*dz/3.)*(sc1*sc1+sc1*sc2+sc2*sc2);
   }
   return capacity;
}      

//_____________________________________________________________________________   
void TGeoXtru::ComputeBBox()
{
// compute bounding box of the pcon
   if (!fX || !fZ || !fNvert) {
      Error("ComputeBBox", "In shape %s polygon not defined", GetName());
      SetShapeBit(TGeoShape::kGeoBad);
      return;
   }   
   Double_t zmin = fZ[0];
   Double_t zmax = fZ[fNz-1];
   Double_t xmin = TGeoShape::Big();
   Double_t xmax = -TGeoShape::Big();
   Double_t ymin = TGeoShape::Big();
   Double_t ymax = -TGeoShape::Big();
   for (Int_t i=0; i<fNz; i++) {
      SetCurrentVertices(fX0[i], fY0[i], fScale[i]);
      for (Int_t j=0; j<fNvert; j++) {
         if (fXc[j]<xmin) xmin=fXc[j];
         if (fXc[j]>xmax) xmax=fXc[j];
         if (fYc[j]<ymin) ymin=fYc[j];
         if (fYc[j]>ymax) ymax=fYc[j];
      }
   }
   fOrigin[0] = 0.5*(xmin+xmax);      
   fOrigin[1] = 0.5*(ymin+ymax);      
   fOrigin[2] = 0.5*(zmin+zmax);      
   fDX = 0.5*(xmax-xmin);
   fDY = 0.5*(ymax-ymin);
   fDZ = 0.5*(zmax-zmin);
}   

//_____________________________________________________________________________   
void TGeoXtru::ComputeNormal(Double_t * /*point*/, Double_t *dir, Double_t *norm)
{
// Compute normal to closest surface from POINT. 
   if (fIz<0) {  
      memset(norm,0,3*sizeof(Double_t));
      norm[2] = (dir[2]>0)?1:-1;
      return;
   }
   Double_t vert[12];      
   GetPlaneVertices(fIz, fSeg, vert);
   GetPlaneNormal(vert, norm);
   Double_t ndotd = norm[0]*dir[0]+norm[1]*dir[1]+norm[2]*dir[2];
   if (ndotd<0) {
      norm[0] = -norm[0];
      norm[1] = -norm[1];
      norm[2] = -norm[2];
   }   
}

//_____________________________________________________________________________
Bool_t TGeoXtru::Contains(Double_t *point) const
{
// test if point is inside this shape
   // Check Z range
   TGeoXtru *xtru = (TGeoXtru*)this;
   if (point[2]<fZ[0]) return kFALSE;   
   if (point[2]>fZ[fNz-1]) return kFALSE; 
   Int_t iz = TMath::BinarySearch(fNz, fZ, point[2]);
   if (iz<0 || iz==fNz-1) return kFALSE;
   if (TGeoShape::IsSameWithinTolerance(point[2],fZ[iz])) {
      xtru->SetIz(-1);
      xtru->SetCurrentVertices(fX0[iz],fY0[iz], fScale[iz]);
      if (fPoly->Contains(point)) return kTRUE;
      if (iz>1 && TGeoShape::IsSameWithinTolerance(fZ[iz],fZ[iz-1])) {
         xtru->SetCurrentVertices(fX0[iz-1],fY0[iz-1], fScale[iz-1]);
         return fPoly->Contains(point);
      } else if (iz<fNz-2 && TGeoShape::IsSameWithinTolerance(fZ[iz],fZ[iz+1])) {
         xtru->SetCurrentVertices(fX0[iz+1],fY0[iz+1], fScale[iz+1]);
         return fPoly->Contains(point);
      }      
   }      
   xtru->SetCurrentZ(point[2], iz);
   if (TMath::Abs(point[2]-fZ[iz])<TGeoShape::Tolerance() ||
       TMath::Abs(fZ[iz+1]-point[2])<TGeoShape::Tolerance())  xtru->SetIz(-1);
   // Now fXc,fYc represent the vertices of the section at point[2]
   return fPoly->Contains(point);
}

//_____________________________________________________________________________
Int_t TGeoXtru::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute closest distance from point px,py to each corner
   const Int_t numPoints = fNvert*fNz;
   return ShapeDistancetoPrimitive(numPoints, px, py);
}
//_____________________________________________________________________________
Double_t TGeoXtru::DistToPlane(Double_t *point, Double_t *dir, Int_t iz, Int_t ivert, Double_t stepmax, Bool_t in) const
{
// Compute distance to a Xtru lateral surface.
   Double_t snext;
   Double_t vert[12];
   Double_t norm[3];
   Double_t znew;
   Double_t pt[3];
   Double_t safe;
   if (TGeoShape::IsSameWithinTolerance(fZ[iz],fZ[iz+1]) && !in) {
      TGeoXtru *xtru = (TGeoXtru*)this;
      snext = (fZ[iz]-point[2])/dir[2];
      if (snext<0) return TGeoShape::Big();
      pt[0] = point[0]+snext*dir[0];
      pt[1] = point[1]+snext*dir[1];
      pt[2] = point[2]+snext*dir[2];
      if (dir[2] < 0.) xtru->SetCurrentVertices(fX0[iz], fY0[iz], fScale[iz]);
      else             xtru->SetCurrentVertices(fX0[iz+1], fY0[iz+1], fScale[iz+1]);
      if (!fPoly->Contains(pt)) return TGeoShape::Big();
      return snext;
   }      
   GetPlaneVertices(iz, ivert, vert);
   GetPlaneNormal(vert, norm);
   Double_t ndotd = norm[0]*dir[0]+norm[1]*dir[1]+norm[2]*dir[2];
   if (in) {
      if (ndotd<=0) return TGeoShape::Big();
      safe = (vert[0]-point[0])*norm[0]+
             (vert[1]-point[1])*norm[1]+
             (vert[2]-point[2])*norm[2];
      if (safe<0) return TGeoShape::Big(); // direction outwards plane
   } else {
      ndotd = -ndotd;
      if (ndotd<=0) return TGeoShape::Big(); 
      safe = (point[0]-vert[0])*norm[0]+
             (point[1]-vert[1])*norm[1]+
             (point[2]-vert[2])*norm[2];
      if (safe<0) return TGeoShape::Big(); // direction outwards plane
   }      
   snext = safe/ndotd;
   if (snext>stepmax) return TGeoShape::Big();
   if (fZ[iz]<fZ[iz+1]) {
      znew = point[2] + snext*dir[2];
      if (znew<fZ[iz]) return TGeoShape::Big();
      if (znew>fZ[iz+1]) return TGeoShape::Big();
   }
   pt[0] = point[0]+snext*dir[0];
   pt[1] = point[1]+snext*dir[1];
   pt[2] = point[2]+snext*dir[2];
   if (!IsPointInsidePlane(pt, vert, norm)) return TGeoShape::Big();
   return snext;         
}

//_____________________________________________________________________________
Double_t TGeoXtru::DistFromInside(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from inside point to surface of the polycone
   // locate Z segment
   if (iact<3 && safe) {
      *safe = Safety(point, kTRUE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }   
   TGeoXtru *xtru = (TGeoXtru*)this;
   Int_t iz = TMath::BinarySearch(fNz, fZ, point[2]);
   if (iz < 0) {
      if (dir[2]<=0) {
         xtru->SetIz(-1);
         return 0.;
      }
      iz = 0;
   }            
   if (iz==fNz-1) {
      if (dir[2]>=0) {
         xtru->SetIz(-1);
         return 0.;
      }   
      iz--;
   } else {   
      if (iz>0) {
         if (TGeoShape::IsSameWithinTolerance(point[2],fZ[iz])) {
            if (TGeoShape::IsSameWithinTolerance(fZ[iz],fZ[iz+1]) && dir[2]<0) iz++;
            else if (TGeoShape::IsSameWithinTolerance(fZ[iz],fZ[iz-1]) && dir[2]>0) iz--;
         }   
      }
   }   
   Bool_t convex = fPoly->IsConvex();
//   Double_t stepmax = step;
//   if (stepmax>TGeoShape::Big()) stepmax = TGeoShape::Big();
   Double_t snext = TGeoShape::Big();
   Double_t dist, sz;
   Double_t pt[3];
   Int_t iv, ipl, inext;
   // we treat the special case when dir[2]=0
   if (TGeoShape::IsSameWithinTolerance(dir[2],0)) {
      for (iv=0; iv<fNvert; iv++) {
         xtru->SetIz(-1);
         dist = DistToPlane(point,dir,iz,iv,TGeoShape::Big(),kTRUE);
         if (dist<snext) {
            snext = dist;
            xtru->SetSeg(iv);
            if (convex) return snext;
         }   
      }
      return TGeoShape::Tolerance();
   }      
   
   // normal case   
   Int_t incseg = (dir[2]>0)?1:-1; 
   Int_t iznext = iz;
   Bool_t zexit = kFALSE;  
   while (iz>=0 && iz<fNz-1) {
      // find the distance  to current segment end Z surface
      ipl = iz+((incseg+1)>>1); // next plane
      inext = ipl+incseg; // next next plane
      sz = (fZ[ipl]-point[2])/dir[2];
      if (sz<snext) {
         iznext += incseg;
         // we cross the next Z section before stepmax
         pt[0] = point[0]+sz*dir[0];
         pt[1] = point[1]+sz*dir[1];
         xtru->SetCurrentVertices(fX0[ipl],fY0[ipl],fScale[ipl]);
         if (fPoly->Contains(pt)) {
            // ray gets through next polygon - is it the last one?
            if (ipl==0 || ipl==fNz-1) {
               xtru->SetIz(-1);
               if (convex) return sz;
               zexit = kTRUE;
               snext = sz;
            }   
            // maybe a Z discontinuity - check this
            if (!zexit && TGeoShape::IsSameWithinTolerance(fZ[ipl],fZ[inext])) {
               xtru->SetCurrentVertices(fX0[inext],fY0[inext],fScale[inext]);
               // if we do not cross the next polygone, we are out
               if (!fPoly->Contains(pt)) {
                  xtru->SetIz(-1);
                  if (convex) return sz;
                  zexit = kTRUE;
                  snext = sz;
               } else {  
                  iznext = inext;
               }   
            } 
         }
      } else {
         iznext = fNz-1;   // stop
      }   
      // ray may cross the lateral surfaces of section iz      
      xtru->SetIz(iz);
      for (iv=0; iv<fNvert; iv++) {
         dist = DistToPlane(point,dir,iz,iv,TGeoShape::Big(),kTRUE); 
         if (dist<snext) {
            xtru->SetSeg(iv);
            snext = dist;
            if (convex) return snext;
            zexit = kTRUE;
         }   
      }
      if (zexit) return snext;
      iz = iznext;
   }
   return TGeoShape::Tolerance();
}

//_____________________________________________________________________________
Double_t TGeoXtru::DistFromOutside(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from outside point to surface of the tube
//   Warning("DistFromOutside", "not implemented");
   if (iact<3 && safe) {
      *safe = Safety(point, kTRUE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }   
// Check if the bounding box is crossed within the requested distance
   Double_t sdist = TGeoBBox::DistFromOutside(point,dir, fDX, fDY, fDZ, fOrigin, step);
   if (sdist>=step) return TGeoShape::Big();
   Double_t stepmax = step;
   if (stepmax>TGeoShape::Big()) stepmax = TGeoShape::Big();
   Double_t snext = 0.;
   Double_t dist = TGeoShape::Big();
   Int_t i, iv;
   Double_t pt[3];
   memcpy(pt,point,3*sizeof(Double_t));
   TGeoXtru *xtru = (TGeoXtru*)this;
   // We might get out easy with Z checks
   Int_t iz = TMath::BinarySearch(fNz, fZ, point[2]);
   if (iz<0) {
      if (dir[2]<=0) return TGeoShape::Big();
      // propagate to first Z plane
      snext = (fZ[0] - point[2])/dir[2];
      if (snext>stepmax) return TGeoShape::Big();
      for (i=0; i<3; i++) pt[i] = point[i] + snext*dir[i];
      xtru->SetCurrentVertices(fX0[0],fY0[0],fScale[0]);
      if (fPoly->Contains(pt)) {
         xtru->SetIz(-1);
         return snext;
      }   
      iz=0; // valid starting value = first segment
      stepmax -= snext;
   } else {
      if (iz==fNz-1) {
         if (dir[2]>=0) return TGeoShape::Big();
         // propagate to last Z plane
         snext = (fZ[fNz-1] - point[2])/dir[2];
         if (snext>stepmax) return TGeoShape::Big();
         for (i=0; i<3; i++) pt[i] = point[i] + snext*dir[i];
         xtru->SetCurrentVertices(fX0[fNz-1],fY0[fNz-1],fScale[fNz-1]);
         if (fPoly->Contains(pt)) {
            xtru->SetIz(-1);
            return snext;
         }   
         iz = fNz-2; // valid value = last segment
         stepmax -= snext;
      }
   }      
   // Check if the bounding box is missed by the track
   if (!TGeoBBox::Contains(pt)) {
      dist = TGeoBBox::DistFromOutside(pt,dir,3);
      if (dist>stepmax) return TGeoShape::Big();
      if (dist>1E-6) dist-=1E-6; // decrease snext to make sure we do not cross the xtru
      else dist = 0;
      for (i=0; i<3; i++) pt[i] += dist*dir[i]; // we are now closer
      iz = TMath::BinarySearch(fNz, fZ, pt[2]);      
      if (iz<0) iz=0;
      else if (iz==fNz-1) iz = fNz-2;
      snext += dist;
      stepmax -= dist;
   }   
   // not the case - we have to do some work...
   // Start trackink from current iz
   // - first solve particular case dir[2]=0
   Bool_t convex = fPoly->IsConvex();
   Bool_t hit = kFALSE;
   if (TGeoShape::IsSameWithinTolerance(dir[2],0)) {
      // loop lateral planes to see if we cross something
      xtru->SetIz(iz);
      for (iv=0; iv<fNvert; iv++) {
         dist = DistToPlane(pt,dir,iz,iv,stepmax,kFALSE);
         if (dist<stepmax) {
            xtru->SetSeg(iv);
            if (convex) return (snext+dist);
            stepmax = dist;
            hit = kTRUE;
         }   
      }
      if (hit) return (snext+stepmax);
      return TGeoShape::Big();
   }   
   // general case
   Int_t incseg = (dir[2]>0)?1:-1;
   while (iz>=0 && iz<fNz-1) {
      // compute distance to lateral planes
      xtru->SetIz(iz);
      if (TGeoShape::IsSameWithinTolerance(fZ[iz],fZ[iz+1])) xtru->SetIz(-1);
      for (iv=0; iv<fNvert; iv++) {
         dist = DistToPlane(pt,dir,iz,iv,stepmax,kFALSE);
         if (dist<stepmax) {
            // HIT
            xtru->SetSeg(iv);
            if (convex) return (snext+dist);
            stepmax = dist;
            hit = kTRUE;
         }   
      }
      if (hit) return (snext+stepmax);
      iz += incseg;
   }   
   return TGeoShape::Big();  
}

//_____________________________________________________________________________
Bool_t TGeoXtru::DefinePolygon(Int_t nvert, const Double_t *xv, const Double_t *yv)
{
// Creates the polygon representing the blueprint of any Xtru section.
//   nvert     = number of vertices >2
//   xv[nvert] = array of X vertex positions 
//   yv[nvert] = array of Y vertex positions 
// *NOTE* should be called before DefineSection or ctor with 'param'
   if (nvert<3) {
      Error("DefinePolygon","In shape %s cannot create polygon with less than 3 vertices", GetName());
      SetShapeBit(TGeoShape::kGeoBad);
      return kFALSE;
   }
   for (Int_t i=0; i<nvert-1; i++) {
      for (Int_t j=i+1; j<nvert; j++) {
         if (TMath::Abs(xv[i]-xv[j])<TGeoShape::Tolerance() &&
             TMath::Abs(yv[i]-yv[j])<TGeoShape::Tolerance()) {
             Error("DefinePolygon","In shape %s 2 vertices cannot be identical",GetName());
             SetShapeBit(TGeoShape::kGeoBad);
//             return kFALSE;
          }   
      }
   }
   fNvert = nvert;
   if (fX) delete [] fX;
   fX = new Double_t[nvert];
   if (fY) delete [] fY;
   fY = new Double_t[nvert];
   if (fXc) delete [] fXc;
   fXc = new Double_t[nvert];
   if (fYc) delete [] fYc;
   fYc = new Double_t[nvert];
   memcpy(fX,xv,nvert*sizeof(Double_t));
   memcpy(fXc,xv,nvert*sizeof(Double_t));
   memcpy(fY,yv,nvert*sizeof(Double_t));
   memcpy(fYc,yv,nvert*sizeof(Double_t));
   
   if (fPoly) delete fPoly;
   fPoly = new TGeoPolygon(nvert);
   fPoly->SetXY(fXc,fYc); // initialize with current coordinates
   fPoly->FinishPolygon();
   if (fPoly->IsIllegalCheck()) {
      Error("DefinePolygon", "Shape %s of type XTRU has an illegal polygon.", GetName());
   }   
   return kTRUE;
}

//_____________________________________________________________________________
void TGeoXtru::DefineSection(Int_t snum, Double_t z, Double_t x0, Double_t y0, Double_t scale)
{
// defines z position of a section plane, rmin and rmax at this z.
   if ((snum<0) || (snum>=fNz)) return;
   fZ[snum]  = z;
   fX0[snum] = x0;
   fY0[snum] = y0;
   fScale[snum] = scale;
   if (snum) {
      if (fZ[snum]<fZ[snum-1]) {
         Warning("DefineSection", "In shape: %s, Z position of section "
                 "%i, z=%e, not in increasing order, %i, z=%e",
                 GetName(),snum,fZ[snum],snum-1,fZ[snum-1]);
         return;
      }   
   }
   if (snum==(fNz-1)) {
      ComputeBBox();
      if (TestShapeBit(TGeoShape::kGeoBad)) InspectShape();
   }   
}
            
//_____________________________________________________________________________
Double_t TGeoXtru::GetZ(Int_t ipl) const
{
// Return the Z coordinate for segment ipl.
   if (ipl<0 || ipl>(fNz-1)) {
      Error("GetZ","In shape %s, ipl=%i out of range (0,%i)",GetName(),ipl,fNz-1);
      return 0.;
   }
   return fZ[ipl];
}      
//_____________________________________________________________________________
void TGeoXtru::GetPlaneNormal(const Double_t *vert, Double_t *norm) const
{
// Returns normal vector to the planar quadrilateral defined by vector VERT.
// The normal points outwards the xtru.
   Double_t cross = 0.;
   Double_t v1[3], v2[3];
   v1[0] = vert[9]-vert[0];
   v1[1] = vert[10]-vert[1];
   v1[2] = vert[11]-vert[2];
   v2[0] = vert[3]-vert[0];
   v2[1] = vert[4]-vert[1];
   v2[2] = vert[5]-vert[2];
   norm[0] = v1[1]*v2[2]-v1[2]*v2[1];
   cross += norm[0]*norm[0];
   norm[1] = v1[2]*v2[0]-v1[0]*v2[2];
   cross += norm[1]*norm[1];
   norm[2] = v1[0]*v2[1]-v1[1]*v2[0];
   cross += norm[2]*norm[2];
   if (cross < TGeoShape::Tolerance()) return;
   cross = 1./TMath::Sqrt(cross);
   for (Int_t i=0; i<3; i++) norm[i] *= cross;
}   

//_____________________________________________________________________________
void TGeoXtru::GetPlaneVertices(Int_t iz, Int_t ivert, Double_t *vert) const
{
// Returns (x,y,z) of 3 vertices of the surface defined by Z sections (iz, iz+1)
// and polygon vertices (ivert, ivert+1). No range check.
   Double_t x,y,z1,z2;
   Int_t iv1 = (ivert+1)%fNvert;
   Int_t icrt = 0;
   z1 = fZ[iz];
   z2 = fZ[iz+1];
   if (fPoly->IsClockwise()) {
      x = fX[ivert]*fScale[iz]+fX0[iz];
      y = fY[ivert]*fScale[iz]+fY0[iz];
      vert[icrt++] = x;
      vert[icrt++] = y;
      vert[icrt++] = z1;
      x = fX[iv1]*fScale[iz]+fX0[iz];
      y = fY[iv1]*fScale[iz]+fY0[iz];
      vert[icrt++] = x;
      vert[icrt++] = y;
      vert[icrt++] = z1;
      x = fX[iv1]*fScale[iz+1]+fX0[iz+1];
      y = fY[iv1]*fScale[iz+1]+fY0[iz+1];
      vert[icrt++] = x;
      vert[icrt++] = y;
      vert[icrt++] = z2;
      x = fX[ivert]*fScale[iz+1]+fX0[iz+1];
      y = fY[ivert]*fScale[iz+1]+fY0[iz+1];
      vert[icrt++] = x;
      vert[icrt++] = y;
      vert[icrt++] = z2;
   } else {
      x = fX[iv1]*fScale[iz]+fX0[iz];
      y = fY[iv1]*fScale[iz]+fY0[iz];
      vert[icrt++] = x;
      vert[icrt++] = y;
      vert[icrt++] = z1;
      x = fX[ivert]*fScale[iz]+fX0[iz];
      y = fY[ivert]*fScale[iz]+fY0[iz];
      vert[icrt++] = x;
      vert[icrt++] = y;
      vert[icrt++] = z1;
      x = fX[ivert]*fScale[iz+1]+fX0[iz+1];
      y = fY[ivert]*fScale[iz+1]+fY0[iz+1];
      vert[icrt++] = x;
      vert[icrt++] = y;
      vert[icrt++] = z2;
      x = fX[iv1]*fScale[iz+1]+fX0[iz+1];
      y = fY[iv1]*fScale[iz+1]+fY0[iz+1];
      vert[icrt++] = x;
      vert[icrt++] = y;
      vert[icrt++] = z2;
   }
}
//_____________________________________________________________________________
Bool_t TGeoXtru::IsPointInsidePlane(Double_t *point, Double_t *vert, Double_t *norm) const
{
// Check if the quadrilateral defined by VERT contains a coplanar POINT.
   Double_t v1[3], v2[3];
   Double_t cross;
   Int_t j,k;
   for (Int_t i=0; i<4; i++) { // loop vertices
      j = 3*i;
      k = 3*((i+1)%4);
      v1[0] = point[0]-vert[j];
      v1[1] = point[1]-vert[j+1];
      v1[2] = point[2]-vert[j+2];
      v2[0] = vert[k]-vert[j];
      v2[1] = vert[k+1]-vert[j+1];
      v2[2] = vert[k+2]-vert[j+2];
      cross = (v1[1]*v2[2]-v1[2]*v2[1])*norm[0]+
              (v1[2]*v2[0]-v1[0]*v2[2])*norm[1]+
              (v1[0]*v2[1]-v1[1]*v2[0])*norm[2];
      if (cross<0) return kFALSE;
   }
   return kTRUE;   
}

//_____________________________________________________________________________
void TGeoXtru::InspectShape() const
{
// Print actual Xtru parameters.
   printf("*** Shape %s: TGeoXtru ***\n", GetName());
   printf("    Nz    = %i\n", fNz);
   printf("    List of (x,y) of polygon vertices:\n");
   for (Int_t ivert = 0; ivert<fNvert; ivert++)
      printf("    x = %11.5f  y = %11.5f\n", fX[ivert],fY[ivert]);
   for (Int_t ipl=0; ipl<fNz; ipl++)
      printf("     plane %i: z=%11.5f x0=%11.5f y0=%11.5f scale=%11.5f\n", ipl, fZ[ipl], fX0[ipl], fY0[ipl], fScale[ipl]);
   printf(" Bounding box:\n");
   TGeoBBox::InspectShape();
}

//_____________________________________________________________________________
TBuffer3D *TGeoXtru::MakeBuffer3D() const
{ 
   // Creates a TBuffer3D describing *this* shape.
   // Coordinates are in local reference frame.
   Int_t nz = GetNz();
   Int_t nvert = GetNvert();
   Int_t nbPnts = nz*nvert;
   Int_t nbSegs = nvert*(2*nz-1);
   Int_t nbPols = nvert*(nz-1)+2;

   TBuffer3D* buff = new TBuffer3D(TBuffer3DTypes::kGeneric,
                                   nbPnts, 3*nbPnts, nbSegs, 3*nbSegs, nbPols, 6*(nbPols-2)+2*(2+nvert));
   if (buff)
   {
      SetPoints(buff->fPnts);   
      SetSegsAndPols(*buff);
   }

   return buff; 
}

//_____________________________________________________________________________
void TGeoXtru::SetSegsAndPols(TBuffer3D &buff) const
{
// Fill TBuffer3D structure for segments and polygons.
   Int_t nz = GetNz();
   Int_t nvert = GetNvert();
   Int_t c = GetBasicColor();

   Int_t i,j;
   Int_t indx, indx2, k;
   indx = indx2 = 0;
   for (i=0; i<nz; i++) {
      // loop Z planes
      indx2 = i*nvert;
      // loop polygon segments
      for (j=0; j<nvert; j++) {
         k = (j+1)%nvert;
         buff.fSegs[indx++] = c;
         buff.fSegs[indx++] = indx2+j;
         buff.fSegs[indx++] = indx2+k;
      }
   } // total: nz*nvert polygon segments
   for (i=0; i<nz-1; i++) {
      // loop Z planes
      indx2 = i*nvert;
      // loop polygon segments
      for (j=0; j<nvert; j++) {
         k = j + nvert;
         buff.fSegs[indx++] = c;
         buff.fSegs[indx++] = indx2+j;
         buff.fSegs[indx++] = indx2+k;
      }
   } // total (nz-1)*nvert lateral segments

   indx = 0;

   // fill lateral polygons
   for (i=0; i<nz-1; i++) {
      indx2 = i*nvert;
      for (j=0; j<nvert; j++) {
      k = (j+1)%nvert;
      buff.fPols[indx++] = c+j%3;
      buff.fPols[indx++] = 4;
      buff.fPols[indx++] = indx2+j;
      buff.fPols[indx++] = nz*nvert+indx2+k;
      buff.fPols[indx++] = indx2+nvert+j;
      buff.fPols[indx++] = nz*nvert+indx2+j;
      }
   } // total (nz-1)*nvert polys
   buff.fPols[indx++] = c+2;
   buff.fPols[indx++] = nvert;
   indx2 = 0;
   for (j = nvert - 1; j >= 0; --j) {
      buff.fPols[indx++] = indx2+j;
   }

   buff.fPols[indx++] = c;
   buff.fPols[indx++] = nvert;
   indx2 = (nz-1)*nvert;
 
   for (j=0; j<nvert; j++) {
      buff.fPols[indx++] = indx2+j;
   }
}

//_____________________________________________________________________________
Double_t TGeoXtru::SafetyToSector(Double_t *point, Int_t iz, Double_t safmin)
{
// Compute safety to sector iz, returning also the closest segment index.
   Double_t safz = TGeoShape::Big();
   Double_t saf1, saf2;
   Bool_t in1, in2;
   Int_t iseg;
   Double_t safe = TGeoShape::Big();
   // segment-break case
   if (TGeoShape::IsSameWithinTolerance(fZ[iz],fZ[iz+1])) {
      safz = TMath::Abs(point[2]-fZ[iz]);
      if (safz>safmin) return TGeoShape::Big();
      SetCurrentVertices(fX0[iz], fY0[iz], fScale[iz]);
      saf1 = fPoly->Safety(point, iseg);
      in1 = fPoly->Contains(point);
      if (!in1 && saf1>safmin) return TGeoShape::Big(); 
      SetCurrentVertices(fX0[iz+1], fY0[iz+1], fScale[iz+1]);
      saf2 = fPoly->Safety(point, iseg);
      in2 = fPoly->Contains(point);
      if ((in1&!in2)|(in2&!in1)) {
         safe = safz; 
      } else {
         safe = TMath::Min(saf1,saf2);
         safe = TMath::Max(safe, safz);
      }
      if (safe>safmin) return TGeoShape::Big();
      return safe;
   }      
   // normal case
   safz = fZ[iz]-point[2];
   if (safz>safmin) return TGeoShape::Big();
   if (safz<0) {
      saf1 = point[2]-fZ[iz+1];
      if (saf1>safmin) return TGeoShape::Big(); 
      if (saf1<0) {
         safz = TMath::Max(safz, saf1); // we are in between the 2 Z segments - we ignore safz
      } else {
         safz = saf1;
      }
   }         
   SetCurrentZ(point[2],iz);
   saf1 = fPoly->Safety(point, iseg);
   Double_t vert[12];
   Double_t norm[3];
   GetPlaneVertices(iz,iseg,vert);
   GetPlaneNormal(vert, norm);
   saf1 = saf1*TMath::Sqrt(1.-norm[2]*norm[2]);
   if (fPoly->Contains(point)) saf1 = -saf1;
   safe = TMath::Max(safz, saf1);
   safe = TMath::Abs(safe);
   if (safe>safmin) return TGeoShape::Big();
   return safe;
}

//_____________________________________________________________________________
Double_t TGeoXtru::Safety(Double_t *point, Bool_t in) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   //---> localize the Z segment
   Double_t safmin = TGeoShape::Big();
   Double_t safe;
   Double_t safz = TGeoShape::Big();
   TGeoXtru *xtru = (TGeoXtru*)this;
   Int_t iz;
   if (in) {
      safmin = TMath::Min(point[2]-fZ[0], fZ[fNz-1]-point[2]);
      for (iz=0; iz<fNz-1; iz++) {
         safe = xtru->SafetyToSector(point, iz, safmin);
         if (safe<safmin) safmin = safe;
      }
      return safmin;
   }
   iz = TMath::BinarySearch(fNz, fZ, point[2]);
   if (iz<0) {
      iz = 0;
      safz = fZ[0] - point[2];
   } else {
      if (iz==fNz-1) {
         iz = fNz-2;
         safz = point[2] - fZ[fNz-1];
      }
   }
   // loop segments from iz up
   Int_t i;
   for (i=iz; i<fNz-1; i++) {
      safe = xtru->SafetyToSector(point,i,safmin);
      if (safe<safmin) safmin=safe;
   }
   // loop segments from iz-1 down
   for (i=iz-1; i>0; i--) {            
      safe = xtru->SafetyToSector(point,i,safmin);
      if (safe<safmin) safmin=safe;
   }
   safe = TMath::Min(safmin, safz);
   return safe;
}

//_____________________________________________________________________________
void TGeoXtru::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
// Save a primitive as a C++ statement(s) on output stream "out".
   if (TObject::TestBit(kGeoSavePrimitive)) return;   
   out << "   // Shape: " << GetName() << " type: " << ClassName() << endl;
   out << "   nz       = " << fNz << ";" << endl;
   out << "   nvert    = " << fNvert << ";" << endl;
   out << "   TGeoXtru *xtru = new TGeoXtru(nz);" << endl;
   out << "   xtru->SetName(\"" << GetName() << "\");" << endl;
   Int_t i;
   for (i=0; i<fNvert; i++) {
      out << "   xvert[" << i << "] = " << fX[i] << ";   yvert[" << i << "] = " << fY[i] << ";" << endl;
   }
   out << "   xtru->DefinePolygon(nvert,xvert,yvert);" << endl;
   for (i=0; i<fNz; i++) {
      out << "   zsect  = " << fZ[i] << ";" << endl; 
      out << "   x0     = " << fX0[i] << ";" << endl; 
      out << "   y0     = " << fY0[i] << ";" << endl; 
      out << "   scale0 = " << fScale[i] << ";" << endl; 
      out << "   xtru->DefineSection(" << i << ",zsect,x0,y0,scale0);" << endl;
   }
   out << "   TGeoShape *" << GetPointerName() << " = xtru;" << endl;
   TObject::SetBit(TGeoShape::kGeoSavePrimitive);
}         

//_____________________________________________________________________________
void TGeoXtru::SetCurrentZ(Double_t z, Int_t iz)
{
// Recompute current section vertices for a given Z position within range of section iz.
   Double_t x0, y0, scale, a, b;
   Int_t ind1, ind2;
   ind1 = iz;
   ind2 = iz+1;   
   Double_t invdz = 1./(fZ[ind2]-fZ[ind1]);
   a = (fX0[ind1]*fZ[ind2]-fX0[ind2]*fZ[ind1])*invdz;
   b = (fX0[ind2]-fX0[ind1])*invdz;
   x0 = a+b*z;
   a = (fY0[ind1]*fZ[ind2]-fY0[ind2]*fZ[ind1])*invdz;
   b = (fY0[ind2]-fY0[ind1])*invdz;
   y0 = a+b*z;
   a = (fScale[ind1]*fZ[ind2]-fScale[ind2]*fZ[ind1])*invdz;
   b = (fScale[ind2]-fScale[ind1])*invdz;
   scale = a+b*z;
   SetCurrentVertices(x0,y0,scale);
}
      
//_____________________________________________________________________________
void TGeoXtru::SetCurrentVertices(Double_t x0, Double_t y0, Double_t scale)      
{
// Set current vertex coordinates according X0, Y0 and SCALE.
   for (Int_t i=0; i<fNvert; i++) {
      fXc[i] = scale*fX[i] + x0;
      fYc[i] = scale*fY[i] + y0;
   }   
}

//_____________________________________________________________________________
void TGeoXtru::SetDimensions(Double_t *param)
{
// param[0] = nz  // number of z planes
//
// param[1] = z1  // Z position of first plane
// param[2] = x1  // X position of first plane
// param[3] = y1  // Y position of first plane
// param[4] = scale1  // scale factor for first plane
// ...
// param[4*(nz-1]+1] = zn
// param[4*(nz-1)+2] = xn
// param[4*(nz-1)+3] = yn
// param[4*(nz-1)+4] = scalen
   fNz = (Int_t)param[0];   
   if (fNz<2) {
      Error("SetDimensions","Cannot create TGeoXtru %s with less than 2 Z planes",GetName());
      SetShapeBit(TGeoShape::kGeoBad);
      return;
   }
   if (fZ) delete [] fZ;
   if (fScale) delete [] fScale;
   if (fX0) delete [] fX0;
   if (fY0) delete [] fY0;
   fZ = new Double_t[fNz];
   fScale = new Double_t[fNz];
   fX0 = new Double_t[fNz];
   fY0 = new Double_t[fNz];
   
   for (Int_t i=0; i<fNz; i++) 
      DefineSection(i, param[1+4*i], param[2+4*i], param[3+4*i], param[4+4*i]);
}   

//_____________________________________________________________________________
void TGeoXtru::SetPoints(Double_t *points) const
{
// create polycone mesh points
   Int_t i, j;
   Int_t indx = 0;
   TGeoXtru *xtru = (TGeoXtru*)this;
   if (points) {
      for (i = 0; i < fNz; i++) {
         xtru->SetCurrentVertices(fX0[i], fY0[i], fScale[i]);
         if (fPoly->IsClockwise()) {
            for (j = 0; j < fNvert; j++) {
               points[indx++] = fXc[j];
               points[indx++] = fYc[j];
               points[indx++] = fZ[i];
            }
         } else {
            for (j = 0; j < fNvert; j++) {
               points[indx++] = fXc[fNvert-1-j];
               points[indx++] = fYc[fNvert-1-j];
               points[indx++] = fZ[i];
            }
         }   
      }
   }
}

//_____________________________________________________________________________
void TGeoXtru::SetPoints(Float_t *points) const
{
// create polycone mesh points
   Int_t i, j;
   Int_t indx = 0;
   TGeoXtru *xtru = (TGeoXtru*)this;
   if (points) {
      for (i = 0; i < fNz; i++) {
         xtru->SetCurrentVertices(fX0[i], fY0[i], fScale[i]);
         if (fPoly->IsClockwise()) {
            for (j = 0; j < fNvert; j++) {
               points[indx++] = fXc[j];
               points[indx++] = fYc[j];
               points[indx++] = fZ[i];
            }
         } else {
            for (j = 0; j < fNvert; j++) {
               points[indx++] = fXc[fNvert-1-j];
               points[indx++] = fYc[fNvert-1-j];
               points[indx++] = fZ[i];
            }
         }   
      }
   }
}

//_____________________________________________________________________________
void TGeoXtru::GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const
{
// Returns numbers of vertices, segments and polygons composing the shape mesh.
   Int_t nz = GetNz();
   Int_t nv = GetNvert();
   nvert = nz*nv;
   nsegs = nv*(2*nz-1);
   npols = nv*(nz-1)+2;
}

//_____________________________________________________________________________
Int_t TGeoXtru::GetNmeshVertices() const
{
// Return number of vertices of the mesh representation
   Int_t numPoints = fNz*fNvert;
   return numPoints;
}

//_____________________________________________________________________________
void TGeoXtru::Sizeof3D() const
{
///// fill size of this 3-D object
///   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
///   if (!painter) return;
///
///   Int_t numPoints = fNz*fNvert;
///   Int_t numSegs   = fNvert*(2*fNz-1);
///   Int_t numPolys  = fNvert*(fNz-1)+2;
///   painter->AddSize3D(numPoints, numSegs, numPolys);
}

//_____________________________________________________________________________
void TGeoXtru::Streamer(TBuffer &R__b)
{
   // Stream an object of class TGeoVolume.
   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(TGeoXtru::Class(), this);
      if (fPoly) fPoly->SetXY(fXc,fYc); // initialize with current coordinates   
   } else {
      R__b.WriteClassBuffer(TGeoXtru::Class(), this);
   }
}

//_____________________________________________________________________________
const TBuffer3D & TGeoXtru::GetBuffer3D(Int_t reqSections, Bool_t localFrame) const
{
// Fills a static 3D buffer and returns a reference.
   static TBuffer3D buffer(TBuffer3DTypes::kGeneric);

   TGeoBBox::FillBuffer3D(buffer, reqSections, localFrame);

   if (reqSections & TBuffer3D::kRawSizes) {
      Int_t nz = GetNz();
      Int_t nvert = GetNvert();
      Int_t nbPnts = nz*nvert;
      Int_t nbSegs = nvert*(2*nz-1);
      Int_t nbPols = nvert*(nz-1)+2;            
      if (buffer.SetRawSizes(nbPnts, 3*nbPnts, nbSegs, 3*nbSegs, nbPols, 6*(nbPols-2)+2*(2+nvert))) {
         buffer.SetSectionsValid(TBuffer3D::kRawSizes);
      }
   }
   // TODO: Push down to TGeoShape?
   if ((reqSections & TBuffer3D::kRaw) && buffer.SectionsValid(TBuffer3D::kRawSizes)) {
      SetPoints(buffer.fPnts);
      if (!buffer.fLocalFrame) {
         TransformPoints(buffer.fPnts, buffer.NbPnts());
      }

      SetSegsAndPols(buffer);      
      buffer.SetSectionsValid(TBuffer3D::kRaw);
   }
      
   return buffer;
}
