// @(#)root/geom:$Id$// Author: Andrei Gheata   24/10/01

// Contains() and DistFromOutside/Out() implemented by Mihaela Gheata

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//--------------------------------------------------------------------------
// TGeoBBox - box class. All shape primitives inherit from this, their 
//   constructor filling automatically the parameters of the box that bounds
//   the given shape. Defined by 6 parameters :
//      fDX, fDY, fDZ - half lengths on X, Y and Z axis
//      fOrigin[3]    - position of box origin
//
//--------------------------------------------------------------------------
//
//
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
// Creation of boxes
// 1.   TGeoBBox *box = new TGeoBBox("BOX", 20, 30, 40);
//Begin_Html
/*
<img src="gif/t_box.gif">
*/
//End_Html
//
// 2. A volume having a box shape can be built in one step:
//      TGeoVolume *vbox = gGeoManager->MakeBox("vbox", ptrMed, 20,30,40);
//
// Divisions of boxes.
//
//   Volumes having box shape can be divided with equal-length slices on 
// X, Y or Z axis. The following options are supported:
// a) Dividing the full range of one axis in N slices
//      TGeoVolume *divx = vbox->Divide("SLICEX", 1, N);
//   - here 1 stands for the division axis (1-X, 2-Y, 3-Z)
//Begin_Html
/*
<img src="gif/t_boxdivX.gif">
*/
//End_Html
//
// b) Dividing in a limited range - general case.
//      TGeoVolume *divy = vbox->Divide("SLICEY",2,N,start,step);
//   - start = starting offset within (-fDY, fDY)
//   - step  = slicing step
//
//Begin_Html
/*
<img src="gif/t_boxdivstepZ.gif">
*/
//End_Html
//
// Both cases are supported by all shapes.
//   See also class TGeoShape for utility methods provided by any particular 
// shape.
//_____________________________________________________________________________

#include "Riostream.h"

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"
#include "TVirtualGeoPainter.h"
#include "TGeoBBox.h"
#include "TVirtualPad.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TMath.h"
#include "TRandom.h"

ClassImp(TGeoBBox)
   
//_____________________________________________________________________________
TGeoBBox::TGeoBBox()
{
// Default constructor
   SetShapeBit(TGeoShape::kGeoBox);
   fDX = fDY = fDZ = 0;
   fOrigin[0] = fOrigin[1] = fOrigin[2] = 0.0;
}   

//_____________________________________________________________________________
TGeoBBox::TGeoBBox(Double_t dx, Double_t dy, Double_t dz, Double_t *origin)
         :TGeoShape("")
{
// Constructor where half-lengths are provided.
   SetShapeBit(TGeoShape::kGeoBox);
   fOrigin[0] = fOrigin[1] = fOrigin[2] = 0.0;
   SetBoxDimensions(dx, dy, dz, origin);
}

//_____________________________________________________________________________
TGeoBBox::TGeoBBox(const char *name, Double_t dx, Double_t dy, Double_t dz, Double_t *origin)
         :TGeoShape(name)
{
// Constructor with shape name.
   SetShapeBit(TGeoShape::kGeoBox);
   fOrigin[0] = fOrigin[1] = fOrigin[2] = 0.0;
   SetBoxDimensions(dx, dy, dz, origin);
}

//_____________________________________________________________________________
TGeoBBox::TGeoBBox(Double_t *param)
         :TGeoShape("")
{
// Constructor based on the array of parameters
// param[0] - half-length in x
// param[1] - half-length in y
// param[2] - half-length in z
   SetShapeBit(TGeoShape::kGeoBox);
   fOrigin[0] = fOrigin[1] = fOrigin[2] = 0.0;
   SetDimensions(param);
}   

//_____________________________________________________________________________
TGeoBBox::~TGeoBBox()
{
// Destructor
}

//_____________________________________________________________________________
Bool_t TGeoBBox::AreOverlapping(const TGeoBBox *box1, const TGeoMatrix *mat1, const TGeoBBox *box2, const TGeoMatrix *mat2)
{
// Check if 2 positioned boxes overlap.
   Double_t master[3];
   Double_t local[3];
   Double_t ldir1[3], ldir2[3];
   const Double_t *o1 = box1->GetOrigin();
   const Double_t *o2 = box2->GetOrigin();
   // Convert center of first box to the local frame of second
   mat1->LocalToMaster(o1, master);
   mat2->MasterToLocal(master, local);
   if (TGeoBBox::Contains(local,box2->GetDX(),box2->GetDY(),box2->GetDZ(),o2)) return kTRUE;
   Double_t distsq = (local[0]-o2[0])*(local[0]-o2[0]) +
                     (local[1]-o2[1])*(local[1]-o2[1]) +
                     (local[2]-o2[2])*(local[2]-o2[2]);
   // Compute distance between box centers and compare with max value
   Double_t rmaxsq = (box1->GetDX()+box2->GetDX())*(box1->GetDX()+box2->GetDX()) +
                     (box1->GetDY()+box2->GetDY())*(box1->GetDY()+box2->GetDY()) +
                     (box1->GetDZ()+box2->GetDZ())*(box1->GetDZ()+box2->GetDZ());
   if (distsq > rmaxsq + TGeoShape::Tolerance()) return kFALSE;
   // We are still not sure: shoot a ray from the center of "1" towards the
   // center of 2.
   Double_t dir[3];
   mat1->LocalToMaster(o1, ldir1);
   mat2->LocalToMaster(o2, ldir2);
   distsq = 1./TMath::Sqrt(distsq);
   dir[0] = (ldir2[0]-ldir1[0])*distsq;
   dir[1] = (ldir2[1]-ldir1[1])*distsq;
   dir[2] = (ldir2[2]-ldir1[2])*distsq;
   mat1->MasterToLocalVect(dir, ldir1);
   mat2->MasterToLocalVect(dir, ldir2);
   // Distance to exit from o1
   Double_t dist1 = TGeoBBox::DistFromInside(o1,ldir1,box1->GetDX(),box1->GetDY(),box1->GetDZ(),o1);
   // Distance to enter from o2
   Double_t dist2 = TGeoBBox::DistFromOutside(local,ldir2,box2->GetDX(),box2->GetDY(),box2->GetDZ(),o2);
   if (dist1 > dist2) return kTRUE;
   return kFALSE;
}

//_____________________________________________________________________________
Double_t TGeoBBox::Capacity() const
{
// Computes capacity of the shape in [length^3].
   return (8.*fDX*fDY*fDZ);
}

//_____________________________________________________________________________
void TGeoBBox::ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm)
{
// Computes normal to closest surface from POINT. 
   memset(norm,0,3*sizeof(Double_t));
   Double_t saf[3];
   Int_t i;
   saf[0]=TMath::Abs(TMath::Abs(point[0]-fOrigin[0])-fDX);
   saf[1]=TMath::Abs(TMath::Abs(point[1]-fOrigin[1])-fDY);
   saf[2]=TMath::Abs(TMath::Abs(point[2]-fOrigin[2])-fDZ);
   i = TMath::LocMin(3,saf);
   norm[i] = (dir[i]>0)?1:(-1);
}

//_____________________________________________________________________________
Bool_t TGeoBBox::CouldBeCrossed(Double_t *point, Double_t *dir) const
{
// Decides fast if the bounding box could be crossed by a vector.
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
   Double_t dirnorm=dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2];   
   if ((doct*doct)>=(do2-rmax2)*dirnorm) return kTRUE;
   return kFALSE;
}

//_____________________________________________________________________________
Int_t TGeoBBox::DistancetoPrimitive(Int_t px, Int_t py)
{
// Compute closest distance from point px,py to each corner.
   const Int_t numPoints = 8;
   return ShapeDistancetoPrimitive(numPoints, px, py);
}

//_____________________________________________________________________________
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
   finder->SetDivIndex(voldiv->GetNdaughters());
   for (Int_t ic=0; ic<ndiv; ic++) {
      voldiv->AddNodeOffset(vol, ic, start+step/2.+ic*step, opt.Data());
      ((TGeoNodeOffset*)voldiv->GetNodes()->At(voldiv->GetNdaughters()-1))->SetFinder(finder);
   }
   return vmulti;
}     

//_____________________________________________________________________________
void TGeoBBox::ComputeBBox()
{
// Compute bounding box - nothing to do in this case.
}   

//_____________________________________________________________________________
Bool_t TGeoBBox::Contains(Double_t *point) const
{
// Test if point is inside this shape.
   if (TMath::Abs(point[2]-fOrigin[2]) > fDZ) return kFALSE;
   if (TMath::Abs(point[0]-fOrigin[0]) > fDX) return kFALSE;
   if (TMath::Abs(point[1]-fOrigin[1]) > fDY) return kFALSE;
   return kTRUE;
}

//_____________________________________________________________________________
Bool_t TGeoBBox::Contains(const Double_t *point, Double_t dx, Double_t dy, Double_t dz, const Double_t *origin)
{
// Test if point is inside this shape.
   if (TMath::Abs(point[2]-origin[2]) > dz) return kFALSE;
   if (TMath::Abs(point[0]-origin[0]) > dx) return kFALSE;
   if (TMath::Abs(point[1]-origin[1]) > dy) return kFALSE;
   return kTRUE;
}

//_____________________________________________________________________________
Double_t TGeoBBox::DistFromInside(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// Compute distance from inside point to surface of the box.
// Boundary safe algorithm.
   Double_t s,smin,saf[6];
   Double_t newpt[3];
   Int_t i;
   for (i=0; i<3; i++) newpt[i] = point[i] - fOrigin[i];
   saf[0] = fDX+newpt[0];
   saf[1] = fDX-newpt[0];
   saf[2] = fDY+newpt[1];
   saf[3] = fDY-newpt[1];
   saf[4] = fDZ+newpt[2];
   saf[5] = fDZ-newpt[2];
   if (iact<3 && safe) {
      smin = saf[0];
      // compute safe distance
      for (i=1;i<6;i++) if (saf[i] < smin) smin = saf[i];
      *safe = smin;
      if (smin<0) *safe = 0.0;
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }
   // compute distance to surface
   smin=TGeoShape::Big();
   for (i=0; i<3; i++) {
      if (dir[i]!=0) {
         s = (dir[i]>0)?(saf[(i<<1)+1]/dir[i]):(-saf[i<<1]/dir[i]);
         if (s < 0) return 0.0;
         if (s < smin) smin = s;
      }
   }
   return smin;
}

//_____________________________________________________________________________
Double_t TGeoBBox::DistFromInside(const Double_t *point,const Double_t *dir, 
                                  Double_t dx, Double_t dy, Double_t dz, const Double_t *origin, Double_t /*stepmax*/)
{
// Compute distance from inside point to surface of the box.
// Boundary safe algorithm.
   Double_t s,smin,saf[6];
   Double_t newpt[3];
   Int_t i;
   for (i=0; i<3; i++) newpt[i] = point[i] - origin[i];
   saf[0] = dx+newpt[0];
   saf[1] = dx-newpt[0];
   saf[2] = dy+newpt[1];
   saf[3] = dy-newpt[1];
   saf[4] = dz+newpt[2];
   saf[5] = dz-newpt[2];
   // compute distance to surface
   smin=TGeoShape::Big();
   for (i=0; i<3; i++) {
      if (dir[i]!=0) {
         s = (dir[i]>0)?(saf[(i<<1)+1]/dir[i]):(-saf[i<<1]/dir[i]);
         if (s < 0) return 0.0;
         if (s < smin) smin = s;
      }
   }
   return smin;
}

//_____________________________________________________________________________
Double_t TGeoBBox::DistFromOutside(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// Compute distance from outside point to surface of the box.
// Boundary safe algorithm.
   Bool_t in = kTRUE;
   Double_t saf[3];
   Double_t par[3];
   Double_t newpt[3];
   Int_t i,j;
   for (i=0; i<3; i++) newpt[i] = point[i] - fOrigin[i];
   par[0] = fDX;
   par[1] = fDY;
   par[2] = fDZ;
   for (i=0; i<3; i++) {
      saf[i] = TMath::Abs(newpt[i])-par[i];
      if (saf[i]>=step) return TGeoShape::Big();
      if (in && saf[i]>0) in=kFALSE;
   }   
   if (iact<3 && safe) {
      // compute safe distance
      if (in) {
         *safe = 0.0;
      } else {   
         *safe = saf[0];
         if (saf[1] > *safe) *safe = saf[1];
         if (saf[2] > *safe) *safe = saf[2];
      }   
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }
   // compute distance from point to box
   Double_t coord, snxt=TGeoShape::Big();
   Int_t ibreak=0;
   // protection in case point is actually inside box
   if (in) {
      j = 0;
      Double_t ss = saf[0];
      if (saf[1]>ss) {
         ss = saf[1];
         j = 1;
      }
      if (saf[2]>ss) j = 2;
      if (newpt[j]*dir[j]>0) return TGeoShape::Big(); // in fact exiting
      return 0.0;   
   }
   for (i=0; i<3; i++) {
      if (saf[i]<0) continue;
      if (newpt[i]*dir[i] >= 0) continue;
      snxt = saf[i]/TMath::Abs(dir[i]);
      ibreak = 0;
      for (j=0; j<3; j++) {
         if (j==i) continue;
         coord=newpt[j]+snxt*dir[j];
         if (TMath::Abs(coord)>par[j]) {
            ibreak=1;
            break;
         }
      }
      if (!ibreak) return snxt;
   }
   return TGeoShape::Big();
}

//_____________________________________________________________________________
Double_t TGeoBBox::DistFromOutside(const Double_t *point,const Double_t *dir, 
                                   Double_t dx, Double_t dy, Double_t dz, const Double_t *origin, Double_t stepmax)
{
// Compute distance from outside point to surface of the box.
// Boundary safe algorithm.
   Bool_t in = kTRUE;
   Double_t saf[3];
   Double_t par[3];
   Double_t newpt[3];
   Int_t i,j;
   for (i=0; i<3; i++) newpt[i] = point[i] - origin[i];
   par[0] = dx;
   par[1] = dy;
   par[2] = dz;
   for (i=0; i<3; i++) {
      saf[i] = TMath::Abs(newpt[i])-par[i];
      if (saf[i]>=stepmax) return TGeoShape::Big();
      if (in && saf[i]>0) in=kFALSE;
   }   
   // In case point is inside return ZERO
   if (in) return 0.0;
   Double_t coord, snxt=TGeoShape::Big();
   Int_t ibreak=0;
   for (i=0; i<3; i++) {
      if (saf[i]<0) continue;
      if (newpt[i]*dir[i] >= 0) continue;
      snxt = saf[i]/TMath::Abs(dir[i]);
      ibreak = 0;
      for (j=0; j<3; j++) {
         if (j==i) continue;
         coord=newpt[j]+snxt*dir[j];
         if (TMath::Abs(coord)>par[j]) {
            ibreak=1;
            break;
         }
      }
      if (!ibreak) return snxt;
   }
   return TGeoShape::Big();
}

//_____________________________________________________________________________
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

//_____________________________________________________________________________
Double_t TGeoBBox::GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const
{
// Get range of shape for a given axis.
   xlo = 0;
   xhi = 0;
   Double_t dx = 0;
   switch (iaxis) {
      case 1:
         xlo = fOrigin[0]-fDX;
         xhi = fOrigin[0]+fDX;
         dx = 2*fDX;
         return dx;
      case 2:
         xlo = fOrigin[1]-fDY;
         xhi = fOrigin[1]+fDY;
         dx = 2*fDY;
         return dx;
      case 3:
         xlo = fOrigin[2]-fDZ;
         xhi = fOrigin[2]+fDZ;
         dx = 2*fDZ;
         return dx;
   }
   return dx;
}         
            
//_____________________________________________________________________________
void TGeoBBox::GetBoundingCylinder(Double_t *param) const
{
// Fill vector param[4] with the bounding cylinder parameters. The order
// is the following : Rmin, Rmax, Phi1, Phi2
   param[0] = 0.;                  // Rmin
   param[1] = fDX*fDX+fDY*fDY;     // Rmax
   param[2] = 0.;                  // Phi1
   param[3] = 360.;                // Phi2
}

//_____________________________________________________________________________
Double_t TGeoBBox::GetFacetArea(Int_t index) const
{
// Get area in internal units of the facet with a given index. 
// Possible index values:
//    0 - all facets togeather
//    1 to 6 - facet index from bottom to top Z
   Double_t area = 0.;
   switch (index) {
      case 0:
         area = 8.*(fDX*fDY + fDX*fDZ + fDY*fDZ);
         return area;
      case 1:
      case 6:
         area = 4.*fDX*fDY;
         return area;
      case 2:
      case 4:
         area = 4.*fDX*fDZ;
         return area;
      case 3:
      case 5:
         area = 4.*fDY*fDZ;
         return area;
   }
   return area;
}         

//_____________________________________________________________________________
Bool_t TGeoBBox::GetPointsOnFacet(Int_t index, Int_t npoints, Double_t *array) const
{
// Fills array with n random points located on the surface of indexed facet.
// The output array must be provided with a length of minimum 3*npoints. Returns
// true if operation succeeded.
// Possible index values:
//    0 - all facets togeather
//    1 to 6 - facet index from bottom to top Z
   if (index<0 || index>6) return kFALSE;
   Double_t surf[6];
   Double_t area = 0.;
   if (index==0) {
      for (Int_t isurf=0; isurf<6; isurf++) {
         surf[isurf] = TGeoBBox::GetFacetArea(isurf+1);
         if (isurf>0) surf[isurf] += surf[isurf-1];
      }   
      area = surf[5];
   }
   
   for (Int_t i=0; i<npoints; i++) {
   // Generate randomly a surface index if needed.
      Double_t *point = &array[3*i];
      Int_t surfindex = index;
      if (surfindex==0) {
         Double_t val = area*gRandom->Rndm();
         surfindex = 2+TMath::BinarySearch(6, surf, val);
         if (surfindex>6) surfindex=6;
      } 
      switch (surfindex) {
         case 1:
            point[0] = -fDX + 2*fDX*gRandom->Rndm();
            point[1] = -fDY + 2*fDY*gRandom->Rndm();
            point[2] = -fDZ;
            break;
         case 2:
            point[0] = -fDX + 2*fDX*gRandom->Rndm();
            point[1] = -fDY;
            point[2] = -fDZ + 2*fDZ*gRandom->Rndm();
            break;
         case 3:
            point[0] = -fDX;
            point[1] = -fDY + 2*fDY*gRandom->Rndm();
            point[2] = -fDZ + 2*fDZ*gRandom->Rndm();
            break;
         case 4:
            point[0] = -fDX + 2*fDX*gRandom->Rndm();
            point[1] = fDY;
            point[2] = -fDZ + 2*fDZ*gRandom->Rndm();
            break;
         case 5:
            point[0] = fDX;
            point[1] = -fDY + 2*fDY*gRandom->Rndm();
            point[2] = -fDZ + 2*fDZ*gRandom->Rndm();
            break;
         case 6:
            point[0] = -fDX + 2*fDX*gRandom->Rndm();
            point[1] = -fDY + 2*fDY*gRandom->Rndm();
            point[2] = fDZ;
            break;
      }
   }
   return kTRUE;
}      

//_____________________________________________________________________________
Bool_t TGeoBBox::GetPointsOnSegments(Int_t npoints, Double_t *array) const
{
// Fills array with n random points located on the line segments of the shape mesh.
// The output array must be provided with a length of minimum 3*npoints. Returns
// true if operation is implemented.
   if (npoints<GetNmeshVertices()) {
      Error("GetPointsOnSegments", "You should require at least %d points", GetNmeshVertices());
      return kFALSE;
   }
   TBuffer3D &buff = (TBuffer3D &)GetBuffer3D(TBuffer3D::kRawSizes | TBuffer3D::kRaw, kTRUE);
   Int_t npnts = buff.NbPnts();
   Int_t nsegs = buff.NbSegs();
   // Copy buffered points  in the array
   memcpy(array, buff.fPnts, 3*npnts*sizeof(Double_t));
   Int_t ipoints = npoints - npnts;
   Int_t icrt = 3*npnts;
   Int_t nperseg = (Int_t)(Double_t(ipoints)/nsegs);
   Double_t *p0, *p1;
   Double_t x,y,z, dx,dy,dz;
   for (Int_t i=0; i<nsegs; i++) {
      p0 = &array[3*buff.fSegs[3*i+1]];
      p1 = &array[3*buff.fSegs[3*i+2]];
      if (i==(nsegs-1)) nperseg = ipoints;
      dx = (p1[0]-p0[0])/(nperseg+1);
      dy = (p1[1]-p0[1])/(nperseg+1);
      dz = (p1[2]-p0[2])/(nperseg+1);
      for (Int_t j=0; j<nperseg; j++) {
         x = p0[0] + (j+1)*dx;
         y = p0[1] + (j+1)*dy;
         z = p0[2] + (j+1)*dz;
         array[icrt++] = x; array[icrt++] = y; array[icrt++] = z;
         ipoints--;
      }
   }
   return kTRUE;
}      
            
//_____________________________________________________________________________
Int_t TGeoBBox::GetFittingBox(const TGeoBBox *parambox, TGeoMatrix *mat, Double_t &dx, Double_t &dy, Double_t &dz) const
{
// Fills real parameters of a positioned box inside this one. Returns 0 if successfull.
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
   Double_t xlo=0, xhi=0;
   Double_t dd[3];
   dd[0] = parambox->GetDX();
   dd[1] = parambox->GetDY();
   dd[2] = parambox->GetDZ();
   for (Int_t iaxis=0; iaxis<3; iaxis++) {
      if (dd[iaxis]>=0) continue;
      TGeoBBox::GetAxisRange(iaxis+1, xlo, xhi);
      //-> compute best fitting parameter
      dd[iaxis] = TMath::Min(origin[iaxis]-xlo, xhi-origin[iaxis]); 
      if (dd[iaxis]<0) {
         Error("GetFittingBox", "wrong matrix");
         return 1;
      }   
   }
   dx = dd[0];
   dy = dd[1];
   dz = dd[2];
   return 0;
}

//_____________________________________________________________________________
TGeoShape *TGeoBBox::GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const
{
// In case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestShapeBit(kGeoRunTimeShape)) return 0;
   Double_t dx, dy, dz;
   Int_t ierr = mother->GetFittingBox(this, mat, dx, dy, dz);
   if (ierr) {
      Error("GetMakeRuntimeShape", "cannot fit this to mother");
      return 0;
   }   
   return (new TGeoBBox(dx, dy, dz));
}

//_____________________________________________________________________________
void TGeoBBox::GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const
{
// Returns numbers of vertices, segments and polygons composing the shape mesh.
   nvert = 8;
   nsegs = 12;
   npols = 6;
}

//_____________________________________________________________________________
void TGeoBBox::InspectShape() const
{
// Prints shape parameters
   printf("*** Shape %s: TGeoBBox ***\n", GetName());
   printf("    dX = %11.5f\n", fDX);
   printf("    dY = %11.5f\n", fDY);
   printf("    dZ = %11.5f\n", fDZ);
   printf("    origin: x=%11.5f y=%11.5f z=%11.5f\n", fOrigin[0], fOrigin[1], fOrigin[2]);
}

//_____________________________________________________________________________
TBuffer3D *TGeoBBox::MakeBuffer3D() const
{
// Creates a TBuffer3D describing *this* shape.
// Coordinates are in local reference frame.   
   TBuffer3D* buff = new TBuffer3D(TBuffer3DTypes::kGeneric, 8, 24, 12, 36, 6, 36);
   if (buff)
   {
      SetPoints(buff->fPnts);
      SetSegsAndPols(*buff);
   }

   return buff;   
}

//_____________________________________________________________________________
void TGeoBBox::SetSegsAndPols(TBuffer3D &buff) const
{
// Fills TBuffer3D structure for segments and polygons.
   Int_t c = GetBasicColor();
   
   buff.fSegs[ 0] = c   ; buff.fSegs[ 1] = 0   ; buff.fSegs[ 2] = 1   ;
   buff.fSegs[ 3] = c+1 ; buff.fSegs[ 4] = 1   ; buff.fSegs[ 5] = 2   ;
   buff.fSegs[ 6] = c+1 ; buff.fSegs[ 7] = 2   ; buff.fSegs[ 8] = 3   ;
   buff.fSegs[ 9] = c   ; buff.fSegs[10] = 3   ; buff.fSegs[11] = 0   ;
   buff.fSegs[12] = c+2 ; buff.fSegs[13] = 4   ; buff.fSegs[14] = 5   ;
   buff.fSegs[15] = c+2 ; buff.fSegs[16] = 5   ; buff.fSegs[17] = 6   ;
   buff.fSegs[18] = c+3 ; buff.fSegs[19] = 6   ; buff.fSegs[20] = 7   ;
   buff.fSegs[21] = c+3 ; buff.fSegs[22] = 7   ; buff.fSegs[23] = 4   ;
   buff.fSegs[24] = c   ; buff.fSegs[25] = 0   ; buff.fSegs[26] = 4   ;
   buff.fSegs[27] = c+2 ; buff.fSegs[28] = 1   ; buff.fSegs[29] = 5   ;
   buff.fSegs[30] = c+1 ; buff.fSegs[31] = 2   ; buff.fSegs[32] = 6   ;
   buff.fSegs[33] = c+3 ; buff.fSegs[34] = 3   ; buff.fSegs[35] = 7   ;
   
   buff.fPols[ 0] = c   ; buff.fPols[ 1] = 4   ;  buff.fPols[ 2] = 0  ;
   buff.fPols[ 3] = 9   ; buff.fPols[ 4] = 4   ;  buff.fPols[ 5] = 8  ;
   buff.fPols[ 6] = c+1 ; buff.fPols[ 7] = 4   ;  buff.fPols[ 8] = 1  ;
   buff.fPols[ 9] = 10  ; buff.fPols[10] = 5   ;  buff.fPols[11] = 9  ;
   buff.fPols[12] = c   ; buff.fPols[13] = 4   ;  buff.fPols[14] = 2  ;
   buff.fPols[15] = 11  ; buff.fPols[16] = 6   ;  buff.fPols[17] = 10 ;
   buff.fPols[18] = c+1 ; buff.fPols[19] = 4   ;  buff.fPols[20] = 3  ;
   buff.fPols[21] = 8   ; buff.fPols[22] = 7   ;  buff.fPols[23] = 11 ;
   buff.fPols[24] = c+2 ; buff.fPols[25] = 4   ;  buff.fPols[26] = 0  ;
   buff.fPols[27] = 3   ; buff.fPols[28] = 2   ;  buff.fPols[29] = 1  ;
   buff.fPols[30] = c+3 ; buff.fPols[31] = 4   ;  buff.fPols[32] = 4  ;
   buff.fPols[33] = 5   ; buff.fPols[34] = 6   ;  buff.fPols[35] = 7  ;
}

//_____________________________________________________________________________
Double_t TGeoBBox::Safety(Double_t *point, Bool_t in) const
{
// Computes the closest distance from given point to this shape.

   Double_t safe, safy, safz;
   if (in) {
      safe = fDX - TMath::Abs(point[0]-fOrigin[0]);
      safy = fDY - TMath::Abs(point[1]-fOrigin[1]);
      safz = fDZ - TMath::Abs(point[2]-fOrigin[2]);
      if (safy < safe) safe = safy;
      if (safz < safe) safe = safz;
   } else {
      safe = -fDX + TMath::Abs(point[0]-fOrigin[0]);
      safy = -fDY + TMath::Abs(point[1]-fOrigin[1]);
      safz = -fDZ + TMath::Abs(point[2]-fOrigin[2]);
      if (safy > safe) safe = safy;
      if (safz > safe) safe = safz;
   }
   return safe;
}

//_____________________________________________________________________________
void TGeoBBox::SavePrimitive(std::ostream &out, Option_t * /*option*/ /*= ""*/)
{
// Save a primitive as a C++ statement(s) on output stream "out".
   if (TObject::TestBit(kGeoSavePrimitive)) return;
   out << "   // Shape: " << GetName() << " type: " << ClassName() << std::endl;
   out << "   dx = " << fDX << ";" << std::endl;
   out << "   dy = " << fDY << ";" << std::endl;
   out << "   dz = " << fDZ << ";" << std::endl;
   if (!TGeoShape::IsSameWithinTolerance(fOrigin[0],0) || 
       !TGeoShape::IsSameWithinTolerance(fOrigin[1],0) || 
       !TGeoShape::IsSameWithinTolerance(fOrigin[2],0)) { 
      out << "   origin[0] = " << fOrigin[0] << ";" << std::endl;
      out << "   origin[1] = " << fOrigin[1] << ";" << std::endl;
      out << "   origin[2] = " << fOrigin[2] << ";" << std::endl;
      out << "   TGeoShape *" << GetPointerName() << " = new TGeoBBox(\"" << GetName() << "\", dx,dy,dz,origin);" << std::endl;
   } else {   
      out << "   TGeoShape *" << GetPointerName() << " = new TGeoBBox(\"" << GetName() << "\", dx,dy,dz);" << std::endl;
   }
   TObject::SetBit(TGeoShape::kGeoSavePrimitive);
}         

//_____________________________________________________________________________
void TGeoBBox::SetBoxDimensions(Double_t dx, Double_t dy, Double_t dz, Double_t *origin)
{
// Set parameters of the box.
   fDX = dx;
   fDY = dy;
   fDZ = dz;
   if (origin) {
      fOrigin[0] = origin[0];
      fOrigin[1] = origin[1];
      fOrigin[2] = origin[2];
   }   
   if (TMath::Abs(fDX)<TGeoShape::Tolerance() && 
       TMath::Abs(fDY)<TGeoShape::Tolerance() &&
       TMath::Abs(fDZ)<TGeoShape::Tolerance()) return;
   if ((fDX<0) || (fDY<0) || (fDZ<0)) SetShapeBit(kGeoRunTimeShape);
}        

//_____________________________________________________________________________
void TGeoBBox::SetDimensions(Double_t *param)
{
// Set dimensions based on the array of parameters
// param[0] - half-length in x
// param[1] - half-length in y
// param[2] - half-length in z
   if (!param) {
      Error("SetDimensions", "null parameters");
      return;
   }
   fDX = param[0];
   fDY = param[1];
   fDZ = param[2];
   if (TMath::Abs(fDX)<TGeoShape::Tolerance() && 
       TMath::Abs(fDY)<TGeoShape::Tolerance() &&
       TMath::Abs(fDZ)<TGeoShape::Tolerance()) return;
   if ((fDX<0) || (fDY<0) || (fDZ<0)) SetShapeBit(kGeoRunTimeShape);
}   

//_____________________________________________________________________________
void TGeoBBox::SetBoxPoints(Double_t *points) const
{
// Fill box vertices to an array.
   TGeoBBox::SetPoints(points);
}

//_____________________________________________________________________________
void TGeoBBox::SetPoints(Double_t *points) const
{
// Fill box points.
   if (!points) return;
   Double_t xmin,xmax,ymin,ymax,zmin,zmax;
   xmin = -fDX+fOrigin[0];
   xmax =  fDX+fOrigin[0];
   ymin = -fDY+fOrigin[1];
   ymax =  fDY+fOrigin[1];
   zmin = -fDZ+fOrigin[2];
   zmax =  fDZ+fOrigin[2];
   points[ 0] = xmin; points[ 1] = ymin; points[ 2] = zmin;
   points[ 3] = xmin; points[ 4] = ymax; points[ 5] = zmin;
   points[ 6] = xmax; points[ 7] = ymax; points[ 8] = zmin;
   points[ 9] = xmax; points[10] = ymin; points[11] = zmin;
   points[12] = xmin; points[13] = ymin; points[14] = zmax;
   points[15] = xmin; points[16] = ymax; points[17] = zmax;
   points[18] = xmax; points[19] = ymax; points[20] = zmax;
   points[21] = xmax; points[22] = ymin; points[23] = zmax;
}

//_____________________________________________________________________________
void TGeoBBox::SetPoints(Float_t *points) const
{
// Fill box points.
   if (!points) return;
   Double_t xmin,xmax,ymin,ymax,zmin,zmax;
   xmin = -fDX+fOrigin[0];
   xmax =  fDX+fOrigin[0];
   ymin = -fDY+fOrigin[1];
   ymax =  fDY+fOrigin[1];
   zmin = -fDZ+fOrigin[2];
   zmax =  fDZ+fOrigin[2];
   points[ 0] = xmin; points[ 1] = ymin; points[ 2] = zmin;
   points[ 3] = xmin; points[ 4] = ymax; points[ 5] = zmin;
   points[ 6] = xmax; points[ 7] = ymax; points[ 8] = zmin;
   points[ 9] = xmax; points[10] = ymin; points[11] = zmin;
   points[12] = xmin; points[13] = ymin; points[14] = zmax;
   points[15] = xmin; points[16] = ymax; points[17] = zmax;
   points[18] = xmax; points[19] = ymax; points[20] = zmax;
   points[21] = xmax; points[22] = ymin; points[23] = zmax;
}

//_____________________________________________________________________________
void TGeoBBox::Sizeof3D() const
{
///// fill size of this 3-D object
///    TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
///    if (painter) painter->AddSize3D(8, 12, 6);
}

//_____________________________________________________________________________
const TBuffer3D & TGeoBBox::GetBuffer3D(Int_t reqSections, Bool_t localFrame) const
{
// Fills a static 3D buffer and returns a reference.
   static TBuffer3D buffer(TBuffer3DTypes::kGeneric);

   FillBuffer3D(buffer, reqSections, localFrame);

   // TODO: A box itself has has nothing more as already described
   // by bounding box. How will viewer interpret?
   if (reqSections & TBuffer3D::kRawSizes) {
      if (buffer.SetRawSizes(8, 3*8, 12, 3*12, 6, 6*6)) {
         buffer.SetSectionsValid(TBuffer3D::kRawSizes);
      }
   }
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

//_____________________________________________________________________________
void TGeoBBox::FillBuffer3D(TBuffer3D & buffer, Int_t reqSections, Bool_t localFrame) const
{
// Fills the supplied buffer, with sections in desired frame
// See TBuffer3D.h for explanation of sections, frame etc.
   TGeoShape::FillBuffer3D(buffer, reqSections, localFrame);

   if (reqSections & TBuffer3D::kBoundingBox) {
      Double_t halfLengths[3] = { fDX, fDY, fDZ };
      buffer.SetAABoundingBox(fOrigin, halfLengths);

      if (!buffer.fLocalFrame) {
         TransformPoints(buffer.fBBVertex[0], 8);
      }
      buffer.SetSectionsValid(TBuffer3D::kBoundingBox);
   }
}
