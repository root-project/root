// @(#)root/geom:$Id$
// Author: Andrei Gheata   02/06/05

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "Riostream.h"

#include "TGeoManager.h"
#include "TGeoVoxelFinder.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"
#include "TGeoNode.h"
#include "TGeoShapeAssembly.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TMath.h"

//_____________________________________________________________________________
// TGeoShapeAssembly - The shape encapsulating an assembly (union) of volumes.
//    
//_____________________________________________________________________________


ClassImp(TGeoShapeAssembly)
   
//_____________________________________________________________________________
TGeoShapeAssembly::TGeoShapeAssembly()
{
// Default constructor
   fCurrent = 0;
   fNext = 0;
   fVolume  = 0;
   fBBoxOK = kFALSE;
}   


//_____________________________________________________________________________
TGeoShapeAssembly::TGeoShapeAssembly(TGeoVolumeAssembly *vol)
{
// Constructor specifying hyperboloid parameters.
   fVolume  = vol;
   fCurrent = 0;
   fNext = 0;
   fBBoxOK = kFALSE;
}


//_____________________________________________________________________________
TGeoShapeAssembly::~TGeoShapeAssembly()
{
// destructor
}

//_____________________________________________________________________________   
void TGeoShapeAssembly::ComputeBBox()
{
// Compute bounding box of the assembly
   if (!fVolume) {
      Fatal("ComputeBBox", "Assembly shape %s without volume", GetName());
      return;
   } 
   // Make sure bbox is computed only once or recomputed only if invalidated (by alignment)
   if (fBBoxOK) return;
   Int_t nd = fVolume->GetNdaughters();
   if (!nd) {fBBoxOK = kTRUE; return;}
   TGeoNode *node;
   TGeoBBox *box;
   Double_t xmin, xmax, ymin, ymax, zmin, zmax;
   xmin = ymin = zmin = TGeoShape::Big();
   xmax = ymax = zmax = -TGeoShape::Big();
   Double_t vert[24];
   Double_t pt[3];
   for (Int_t i=0; i<nd; i++) {
      node = fVolume->GetNode(i);
      // Make sure that all assembly daughters have computed their bboxes
      if (node->GetVolume()->IsAssembly()) node->GetVolume()->GetShape()->ComputeBBox();
      box = (TGeoBBox*)node->GetVolume()->GetShape();
      box->SetBoxPoints(vert);
      for (Int_t ipt=0; ipt<8; ipt++) {
         node->LocalToMaster(&vert[3*ipt], pt);
         if (pt[0]<xmin) xmin=pt[0];
         if (pt[0]>xmax) xmax=pt[0];
         if (pt[1]<ymin) ymin=pt[1];
         if (pt[1]>ymax) ymax=pt[1];
         if (pt[2]<zmin) zmin=pt[2];
         if (pt[2]>zmax) zmax=pt[2];
      }
   }
   fDX = 0.5*(xmax-xmin);
   fOrigin[0] = 0.5*(xmin+xmax);
   fDY = 0.5*(ymax-ymin);
   fOrigin[1] = 0.5*(ymin+ymax);
   fDZ = 0.5*(zmax-zmin);
   fOrigin[2] = 0.5*(zmin+zmax);   
   fBBoxOK = kTRUE;      
}   

//_____________________________________________________________________________   
void TGeoShapeAssembly::RecomputeBoxLast()
{
// Recompute bounding box of the assembly after adding a node.
   Int_t nd = fVolume->GetNdaughters();
   if (!nd) {
      Warning("RecomputeBoxLast", "No daughters for volume %s yet", fVolume->GetName());
      return;
   }   
   TGeoNode *node = fVolume->GetNode(nd-1);
   Double_t xmin, xmax, ymin, ymax, zmin, zmax;
   if (nd==1) {
      xmin = ymin = zmin = TGeoShape::Big();
      xmax = ymax = zmax = -TGeoShape::Big();
   } else {
      xmin = fOrigin[0]-fDX;
      xmax = fOrigin[0]+fDX;
      ymin = fOrigin[1]-fDY;
      ymax = fOrigin[1]+fDY;
      zmin = fOrigin[2]-fDZ;
      zmax = fOrigin[2]+fDZ;
   }      
   Double_t vert[24];
   Double_t pt[3];
   TGeoBBox *box = (TGeoBBox*)node->GetVolume()->GetShape();
   if (TGeoShape::IsSameWithinTolerance(box->GetDX(), 0) ||
       node->GetVolume()->IsAssembly()) node->GetVolume()->GetShape()->ComputeBBox();
   box->SetBoxPoints(vert);
   for (Int_t ipt=0; ipt<8; ipt++) {
      node->LocalToMaster(&vert[3*ipt], pt);
      if (pt[0]<xmin) xmin=pt[0];
      if (pt[0]>xmax) xmax=pt[0];
      if (pt[1]<ymin) ymin=pt[1];
      if (pt[1]>ymax) ymax=pt[1];
      if (pt[2]<zmin) zmin=pt[2];
      if (pt[2]>zmax) zmax=pt[2];
   }
   fDX = 0.5*(xmax-xmin);
   fOrigin[0] = 0.5*(xmin+xmax);
   fDY = 0.5*(ymax-ymin);
   fOrigin[1] = 0.5*(ymin+ymax);
   fDZ = 0.5*(zmax-zmin);
   fOrigin[2] = 0.5*(zmin+zmax);   
   fBBoxOK = kTRUE;      
}  

//_____________________________________________________________________________   
void TGeoShapeAssembly::ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm)
{
// Compute normal to closest surface from POINT. Should not be called.
   if (!fBBoxOK) ((TGeoShapeAssembly*)this)->ComputeBBox();
   Int_t inext = fVolume->GetNextNodeIndex();
   if (inext<0) {
      DistFromOutside(point,dir,3);
      inext = fVolume->GetNextNodeIndex();
      if (inext<0) {
         Error("ComputeNormal","Invalid inext=%i (Ncomponents=%i)",inext,fVolume->GetNdaughters());
         return;
      }   
   }   
   TGeoNode *node = fVolume->GetNode(inext);
   Double_t local[3],ldir[3],lnorm[3];
   node->MasterToLocal(point,local);
   node->MasterToLocalVect(dir,ldir);
   node->GetVolume()->GetShape()->ComputeNormal(local,ldir,lnorm);
   node->LocalToMasterVect(lnorm,norm);
}

//_____________________________________________________________________________
Bool_t TGeoShapeAssembly::Contains(Double_t *point) const
{
// Test if point is inside the assembly
   if (!fBBoxOK) ((TGeoShapeAssembly*)this)->ComputeBBox();
   if (!TGeoBBox::Contains(point)) return kFALSE;
   TGeoVoxelFinder *voxels = fVolume->GetVoxels();
   TGeoNode *node;
   TGeoShape *shape;
   Int_t *check_list = 0;
   Int_t ncheck, id;
   Double_t local[3];
   if (voxels) {
      // get the list of nodes passing thorough the current voxel
      Int_t tid = TGeoManager::ThreadId();
      check_list = voxels->GetCheckList(&point[0], ncheck, tid);
      if (!check_list) return kFALSE;
      for (id=0; id<ncheck; id++) {
         node = fVolume->GetNode(check_list[id]);
         shape = node->GetVolume()->GetShape();
         node->MasterToLocal(point,local);
         if (shape->Contains(local)) {
            fVolume->SetCurrentNodeIndex(check_list[id]);
            fVolume->SetNextNodeIndex(check_list[id]);
            return kTRUE;
         }   
      }
      return kFALSE;
   }      
   Int_t nd = fVolume->GetNdaughters();
   for (id=0; id<nd; id++) {
      node = fVolume->GetNode(id);
      shape = node->GetVolume()->GetShape();
      node->MasterToLocal(point,local);
      if (shape->Contains(local)) {
         fVolume->SetCurrentNodeIndex(id);
         fVolume->SetNextNodeIndex(id);      
         return kTRUE;
      }   
   }
   return kFALSE;   
}

//_____________________________________________________________________________
Int_t TGeoShapeAssembly::DistancetoPrimitive(Int_t /*px*/, Int_t /*py*/)
{
// compute closest distance from point px,py to each vertex. Should not be called.
   return 9999;
}

//_____________________________________________________________________________
Double_t TGeoShapeAssembly::DistFromInside(Double_t * /*point*/, Double_t * /*dir*/, Int_t /*iact*/, Double_t /*step*/, Double_t * /*safe*/) const
{
// Compute distance from inside point to surface of the hyperboloid.
   Info("DistFromInside", "Cannot compute distance from inside the assembly (but from a component)"); 
   return TGeoShape::Big();
}


//_____________________________________________________________________________
Double_t TGeoShapeAssembly::DistFromOutside(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from outside point to surface of the hyperboloid.
//   fVolume->SetNextNodeIndex(-1);
   if (!fBBoxOK) ((TGeoShapeAssembly*)this)->ComputeBBox();
   if (iact<3 && safe) {
      *safe = Safety(point, kFALSE);
      if (iact==0) return TGeoShape::Big();
      if ((iact==1) && (step<=*safe)) return TGeoShape::Big();
   }
   // find distance to assembly
   Double_t snext = 0.0;
   Double_t dist;
   Double_t stepmax = step;
   Double_t pt[3];
   Int_t i;
   Bool_t found = kFALSE;
   memcpy(pt,point,3*sizeof(Double_t));
   if (!TGeoBBox::Contains(point)) {
      snext = TGeoBBox::DistFromOutside(point, dir, 3, stepmax);
      if (snext > stepmax) return TGeoShape::Big();
      for (i=0; i<3; i++) pt[i] += (snext+TGeoShape::Tolerance())*dir[i];
      if (Contains(pt)) {
         fVolume->SetNextNodeIndex(fVolume->GetCurrentNodeIndex());
         return snext;
      }   
      snext += TGeoShape::Tolerance();
      stepmax -= snext;
   }
   // Point represented by pt is now inside the bounding box - find distance to components   
   Int_t nd = fVolume->GetNdaughters();
   TGeoNode *node;
   Double_t lpoint[3],ldir[3];
   TGeoVoxelFinder *voxels = fVolume->GetVoxels();
   if (nd<5 || !voxels) {
      for (i=0; i<nd; i++) {
         node = fVolume->GetNode(i);
         if (voxels && voxels->IsSafeVoxel(pt, i, stepmax)) continue;
         node->MasterToLocal(pt, lpoint);
         node->MasterToLocalVect(dir, ldir);
         dist = node->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, stepmax);
         if (dist<stepmax) {
            stepmax = dist;
            fVolume->SetNextNodeIndex(i);
            found = kTRUE;
         }
      }
      if (found) {
         snext += stepmax;
         return snext;
      }
      return TGeoShape::Big();   
   }
   // current volume is voxelized, first get current voxel
   Int_t ncheck = 0;
   Int_t *vlist = 0;
   Int_t tid = TGeoManager::ThreadId();
   voxels->SortCrossedVoxels(pt, dir, tid);
   while ((vlist=voxels->GetNextVoxel(pt, dir, ncheck, tid))) {
      for (i=0; i<ncheck; i++) {
         node = fVolume->GetNode(vlist[i]);
         node->MasterToLocal(pt, lpoint);
         node->MasterToLocalVect(dir, ldir);
         dist = node->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, stepmax);
         if (dist<stepmax) {
            stepmax = dist;
            fVolume->SetNextNodeIndex(vlist[i]);
            found = kTRUE;
         }
      }
   }
   if (found) {
      snext += stepmax;
      return snext;
   }
   return TGeoShape::Big();      
}
   
//_____________________________________________________________________________
TGeoVolume *TGeoShapeAssembly::Divide(TGeoVolume * /*voldiv*/, const char *divname, Int_t /*iaxis*/, Int_t /*ndiv*/, 
                             Double_t /*start*/, Double_t /*step*/) 
{
// Cannot divide assemblies.
   Error("Divide", "Assemblies cannot be divided. Division volume %s not created", divname);
   return 0;
}   

//_____________________________________________________________________________
TGeoShape *TGeoShapeAssembly::GetMakeRuntimeShape(TGeoShape * /*mother*/, TGeoMatrix * /*mat*/) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   Error("GetMakeRuntimeShape", "Assemblies cannot be parametrized.");
   return NULL;
}

//_____________________________________________________________________________
void TGeoShapeAssembly::InspectShape() const
{
// print shape parameters
   printf("*** Shape %s: TGeoShapeAssembly ***\n", GetName());
   printf("    Volume assembly %s with %i nodes\n", fVolume->GetName(), fVolume->GetNdaughters());   
   printf(" Bounding box:\n");
   if (!fBBoxOK) ((TGeoShapeAssembly*)this)->ComputeBBox();
   TGeoBBox::InspectShape();
}

//_____________________________________________________________________________
void TGeoShapeAssembly::SetSegsAndPols(TBuffer3D & /*buff*/) const
{
// Fill TBuffer3D structure for segments and polygons.
   Error("SetSegsAndPols", "Drawing functions should not be called for assemblies, but rather for their content");   
}

//_____________________________________________________________________________
Double_t TGeoShapeAssembly::Safety(Double_t *point, Bool_t in) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   Double_t safety = TGeoShape::Big();
   Double_t pt[3], loc[3];
   if (!fBBoxOK) ((TGeoShapeAssembly*)this)->ComputeBBox();
   if (in) {
      Int_t index = fVolume->GetCurrentNodeIndex();
      TGeoVolume *vol = fVolume;
      TGeoNode *node;
      memcpy(loc, point, 3*sizeof(Double_t));
      while (index>=0) {
         memcpy(pt, loc, 3*sizeof(Double_t));
         node = vol->GetNode(index);
         node->GetMatrix()->MasterToLocal(pt,loc);
         vol = node->GetVolume();
         index = vol->GetCurrentNodeIndex();
         if (index<0) {
            safety = vol->GetShape()->Safety(loc, in);
            return safety;
         }
      }
      return TGeoShape::Big();
   }         
   Double_t safe;
   TGeoVoxelFinder *voxels = fVolume->GetVoxels();
   Int_t nd = fVolume->GetNdaughters();
   Double_t *boxes = 0;
   if (voxels) boxes = voxels->GetBoxes();   
   TGeoNode *node;
   for (Int_t id=0; id<nd; id++) {
      if (boxes && id>0) {
         Int_t ist = 6*id;
         Double_t dxyz = 0.;
         Double_t dxyz0 = TMath::Abs(point[0]-boxes[ist+3])-boxes[ist];
         if (dxyz0 > safety) continue;
         Double_t dxyz1 = TMath::Abs(point[1]-boxes[ist+4])-boxes[ist+1];
         if (dxyz1 > safety) continue;
         Double_t dxyz2 = TMath::Abs(point[2]-boxes[ist+5])-boxes[ist+2];
         if (dxyz2 > safety) continue;
         if (dxyz0>0) dxyz+=dxyz0*dxyz0;
         if (dxyz1>0) dxyz+=dxyz1*dxyz1;
         if (dxyz2>0) dxyz+=dxyz2*dxyz2;
         if (dxyz >= safety*safety) continue;
      }
      node = fVolume->GetNode(id);
      safe = node->Safety(point, kFALSE);
      if (safe<=0.0) return 0.0;
      if (safe<safety) safety = safe;
   }   
   return safety;        
}

//_____________________________________________________________________________
void TGeoShapeAssembly::SavePrimitive(ostream & /*out*/, Option_t * /*option*/ /*= ""*/)
{
// Save a primitive as a C++ statement(s) on output stream "out".
}

//_____________________________________________________________________________
void TGeoShapeAssembly::SetPoints(Double_t * /*points*/) const
{
// No mesh for assemblies.
   Error("SetPoints", "Drawing functions should not be called for assemblies, but rather for their content");
}

//_____________________________________________________________________________
void TGeoShapeAssembly::SetPoints(Float_t * /*points*/) const
{
// No mesh for assemblies.
   Error("SetPoints", "Drawing functions should not be called for assemblies, but rather for their content");
}

//_____________________________________________________________________________
void TGeoShapeAssembly::GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const
{
// Returns numbers of vertices, segments and polygons composing the shape mesh.
   nvert = 0;
   nsegs = 0;
   npols = 0;
}

