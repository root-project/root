// @(#):$Name:  $:$Id: TGeoBoolNode.cxx,v 1.12 2004/06/25 11:59:55 brun Exp $
// Author: Andrei Gheata   30/05/02
// TGeoBoolNode::Contains and parser implemented by Mihaela Gheata

   
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGeoCompositeShape.h"
#include "TGeoMatrix.h"
#include "TGeoManager.h"

#include "TGeoBoolNode.h"

// statics and globals

ClassImp(TGeoBoolNode)

//-----------------------------------------------------------------------------
TGeoBoolNode::TGeoBoolNode()
{
// Default constructor
   fLeft     = 0;
   fRight    = 0;
   fLeftMat  = 0;
   fRightMat = 0;
}
//-----------------------------------------------------------------------------
TGeoBoolNode::TGeoBoolNode(const char *expr1, const char *expr2)
{
// constructor
   fLeft     = 0;
   fRight    = 0;
   fLeftMat  = 0;
   fRightMat = 0;
   if (!MakeBranch(expr1, kTRUE)) {
      return;
   }
   if (!MakeBranch(expr2, kFALSE)) {
      return;
   }
}

//-----------------------------------------------------------------------------
TGeoBoolNode::TGeoBoolNode(TGeoShape *left, TGeoShape *right, TGeoMatrix *lmat, TGeoMatrix *rmat)
{
   fLeft = left;
   if (!fLeft) {
      Error("ctor", "left shape is NULL");
      return;
   }   
   fRight = right;
   if (!fRight) {
      Error("ctor", "right shape is NULL");
      return;
   }   
   fLeftMat = lmat;
   if (!fLeftMat) fLeftMat = gGeoIdentity;
   fRightMat = rmat;
   if (!fRightMat) fRightMat = gGeoIdentity;
}

//-----------------------------------------------------------------------------
TGeoBoolNode::~TGeoBoolNode()
{
// Destructor
// --- deletion of components handled by TGeoManager class.
}
//-----------------------------------------------------------------------------
Bool_t TGeoBoolNode::MakeBranch(const char *expr, Bool_t left)
{
// Expands the boolean expression either on left or right branch, creating
// component elements (composite shapes and boolean nodes). Returns true on success.
   TString sleft, sright, stransf;
   Int_t boolop = TGeoManager::Parse(expr, sleft, sright, stransf);
   if (boolop<0) {
      Error("MakeBranch", "invalid expresion");
      return kFALSE;
   }
   TGeoShape *shape = 0;
   TGeoMatrix *mat;
   TString newshape;

   if (stransf.Length() == 0) {
      mat = gGeoIdentity;
   } else {   
      mat = (TGeoMatrix*)gGeoManager->GetListOfMatrices()->FindObject(stransf.Data());    
   }
   if (!mat) {
      Error("MakeBranch", "transformation %s not found", stransf.Data());
      return kFALSE;
   }
   switch (boolop) {
      case 0:
         // elementary shape
         shape = (TGeoShape*)gGeoManager->GetListOfShapes()->FindObject(sleft.Data()); 
         if (!shape) {
            Error("MakeBranch", "shape %s not found", sleft.Data());
            return kFALSE;
         }
         break;
      case 1:
         // composite shape - union
         newshape = sleft;
         newshape += "+";
         newshape += sright;
         shape = new TGeoCompositeShape(newshape.Data());
         break;
      case 2:
         // composite shape - difference
         newshape = sleft;
         newshape += "-";
         newshape += sright;
         shape = new TGeoCompositeShape(newshape.Data());
         break;
      case 3:
         // composite shape - intersection
         newshape = sleft;
         newshape += "*";
         newshape += sright;
         shape = new TGeoCompositeShape(newshape.Data());
         break;
   }      
   if (boolop && !shape->IsValid()) {
      Error("MakeBranch", "Shape %s not valid", newshape.Data());
      delete shape;
      return kFALSE;
   }      
   if (left) {
      fLeft = shape;
      fLeftMat = mat;
   } else {
      fRight = shape;
      fRightMat = mat;
   }
   return kTRUE;                  
}
//-----------------------------------------------------------------------------
void TGeoBoolNode::Paint(Option_t *option)
{
   TGeoHMatrix *glmat = gGeoManager->GetGLMatrix();
   TGeoHMatrix mat;
   mat = glmat; // keep a copy
   glmat->Multiply(fLeftMat);
   fLeft->Paint(option);
   *glmat = &mat;
   glmat->Multiply(fRightMat);
   fRight->Paint(option);
   *glmat = &mat;   
}
//-----------------------------------------------------------------------------
void TGeoBoolNode::Sizeof3D() const
{
// register size of this 3D object
   fLeft->Sizeof3D();
   fRight->Sizeof3D();
}
ClassImp(TGeoUnion)

//-----------------------------------------------------------------------------
TGeoUnion::TGeoUnion()
{
// Default constructor
}
//-----------------------------------------------------------------------------
TGeoUnion::TGeoUnion(const char *expr1, const char *expr2)
          :TGeoBoolNode(expr1, expr2)
{
// Constructor
}

//-----------------------------------------------------------------------------
TGeoUnion::TGeoUnion(TGeoShape *left, TGeoShape *right, TGeoMatrix *lmat, TGeoMatrix *rmat)
          :TGeoBoolNode(left,right,lmat,rmat)
{
// Constructor providing pointers to components
   if (left->TestShapeBit(TGeoShape::kGeoHalfSpace) || right->TestShapeBit(TGeoShape::kGeoHalfSpace)) {
      Fatal("TGeoUnion", "Unions with a half-space (%s + %s) not allowed", left->GetName(), right->GetName());
   }
}

//-----------------------------------------------------------------------------
TGeoUnion::~TGeoUnion()
{
// Destructor
// --- deletion of components handled by TGeoManager class.
}
//-----------------------------------------------------------------------------
void TGeoUnion::ComputeBBox(Double_t &dx, Double_t &dy, Double_t &dz, Double_t *origin)
{
// Compute bounding box corresponding to a union of two shapes.
   if (((TGeoBBox*)fLeft)->IsNullBox()) fLeft->ComputeBBox();
   if (((TGeoBBox*)fRight)->IsNullBox()) fRight->ComputeBBox();
   Double_t vert[48];
   Double_t pt[3];
   Int_t i;
   Double_t xmin, xmax, ymin, ymax, zmin, zmax;
   xmin = ymin = zmin = TGeoShape::Big();
   xmax = ymax = zmax = -TGeoShape::Big();
   ((TGeoBBox*)fLeft)->SetBoxPoints(&vert[0]);
   ((TGeoBBox*)fRight)->SetBoxPoints(&vert[24]);
   for (i=0; i<8; i++) {
      fLeftMat->LocalToMaster(&vert[3*i], &pt[0]);
      if (pt[0]<xmin) xmin=pt[0];
      if (pt[0]>xmax) xmax=pt[0];
      if (pt[1]<ymin) ymin=pt[1];
      if (pt[1]>ymax) ymax=pt[1];
      if (pt[2]<zmin) zmin=pt[2];
      if (pt[2]>zmax) zmax=pt[2];
   }   
   for (i=8; i<16; i++) {
      fRightMat->LocalToMaster(&vert[3*i], &pt[0]);
      if (pt[0]<xmin) xmin=pt[0];
      if (pt[0]>xmax) xmax=pt[0];
      if (pt[1]<ymin) ymin=pt[1];
      if (pt[1]>ymax) ymax=pt[1];
      if (pt[2]<zmin) zmin=pt[2];
      if (pt[2]>zmax) zmax=pt[2];
   }   
   dx = 0.5*(xmax-xmin);
   origin[0] = 0.5*(xmin+xmax);
   dy = 0.5*(ymax-ymin);
   origin[1] = 0.5*(ymin+ymax);
   dz = 0.5*(zmax-zmin);
   origin[2] = 0.5*(zmin+zmax);
}   
//-----------------------------------------------------------------------------
Bool_t TGeoUnion::Contains(Double_t *point) const
{
// Find if a union of two shapes contains a given point
   Double_t local[3];
   fLeftMat->MasterToLocal(point, &local[0]);
   Bool_t inside = fLeft->Contains(&local[0]);
   if (inside) return kTRUE;
   fRightMat->MasterToLocal(point, &local[0]);
   inside = fRight->Contains(&local[0]);
   return inside;
}

//_____________________________________________________________________________
void TGeoUnion::ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm)
{
   norm[0] = norm[1] = 0.;
   norm[2] = 1.;
   Double_t local[3];
   Double_t ldir[3], lnorm[3];
   fLeftMat->MasterToLocal(point, local);
   if (fLeft->Contains(local)) {
      fLeftMat->MasterToLocalVect(dir, ldir);
      fLeft->ComputeNormal(local,ldir,lnorm);
      fLeftMat->LocalToMasterVect(lnorm, norm);
      return;
   }   
   fRightMat->MasterToLocal(point, local);
   if (fRight->Contains(local)) {
      fRightMat->MasterToLocalVect(dir, ldir);
      fRight->ComputeNormal(local,ldir,lnorm);
      fRightMat->LocalToMasterVect(lnorm, norm);
      return;
   }   
   // Propagate forward/backward to see which of the components is intersected first
   local[0] = point[0] + 1E-5*dir[0];
   local[1] = point[1] + 1E-5*dir[1];
   local[2] = point[2] + 1E-5*dir[2];

   if (!Contains(local)) {
      local[0] = point[0] - 1E-5*dir[0];
      local[1] = point[1] - 1E-5*dir[1];
      local[2] = point[2] - 1E-5*dir[2];
      if (!Contains(local)) return;
   }
   ComputeNormal(local,dir,norm);   
}

//-----------------------------------------------------------------------------
Int_t TGeoUnion::DistanceToPrimitive(Int_t /*px*/, Int_t /*py*/)
{
// Compute minimum distance to shape vertices
   return 9999;
}
//-----------------------------------------------------------------------------
Double_t TGeoUnion::DistToOut(Double_t *point, Double_t *dir, Int_t iact,
                              Double_t step, Double_t *safe) const
{
// Compute distance from a given point to outside.
//   printf("Point is : %g, %g, %g\n", point[0], point[1], point[2]);
   if (iact<3 && safe) {
      // compute safe distance
      *safe = Safety(point,kTRUE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }

   Double_t local[3], master[3], ldir[3], rdir[3];
   memcpy(master, point, 3*sizeof(Double_t));
   Int_t i;
   Double_t d1=0., d2=0., snxt=0.;
   fLeftMat->MasterToLocalVect(dir, &ldir[0]);
   fRightMat->MasterToLocalVect(dir, &rdir[0]);
   fLeftMat->MasterToLocal(point, &local[0]);
   Bool_t inside1 = fLeft->Contains(&local[0]);
   if (inside1) d1 = fLeft->DistToOut(&local[0], &ldir[0], iact, step, safe);
   fRightMat->MasterToLocal(point, &local[0]);
   Bool_t inside2 = fRight->Contains(&local[0]);
   if (inside2) d2 = fRight->DistToOut(&local[0], &rdir[0], iact, step, safe);
   if (inside1 && inside2) {
      snxt = TMath::Min(d1, d2);
      for (i=0; i<3; i++) master[i] += (snxt+1E-8)*dir[i];
      snxt += DistToOut(&master[0], dir, iact, step-snxt, safe)+1E-8;
      return snxt;
   }   
   if (inside1) {
      snxt = d1;
      for (i=0; i<3; i++) master[i] += (snxt+1E-8)*dir[i];
      snxt += DistToOut(&master[0], dir, iact, step-snxt, safe)+1E-8;
      return snxt;
   }   
   if (inside2) {
      snxt = d2;
      for (i=0; i<3; i++) master[i] += (snxt+1E-8)*dir[i];
      snxt += DistToOut(&master[0], dir, iact, step-snxt, safe)+1E-8;
      return snxt;
   }
   return 0;   
}
//-----------------------------------------------------------------------------
Double_t TGeoUnion::DistToIn(Double_t *point, Double_t *dir, Int_t iact,
                              Double_t step, Double_t *safe) const
{
// Compute distance from a given point to inside.
   if (iact<3 && safe) {
      // compute safe distance
      *safe = Safety(point,kFALSE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }
   Double_t local[3], ldir[3], rdir[3];
   Double_t d1, d2, snxt;
   fLeftMat->MasterToLocal(point, &local[0]);
   fLeftMat->MasterToLocalVect(dir, &ldir[0]);
   fRightMat->MasterToLocalVect(dir, &rdir[0]);
   d1 = fLeft->DistToIn(&local[0], &ldir[0], iact, step, safe);
   fRightMat->MasterToLocal(point, &local[0]);
   d2 = fRight->DistToIn(&local[0], &rdir[0], iact, step, safe);
   snxt = TMath::Min(d1, d2);
   return snxt;
}
//-----------------------------------------------------------------------------
Int_t TGeoUnion::GetNpoints() const
{
// Returns number of vertices for the composite shape described by this union.
   return 0;
}
//-----------------------------------------------------------------------------
void TGeoUnion::SetPoints(Double_t * /*buff*/) const
{
// Fill buffer with shape vertices.
}

//-----------------------------------------------------------------------------
Double_t TGeoUnion::Safety(Double_t *point, Bool_t) const
{
// Compute safety distance for a union node;
   Double_t local1[3], local2[3];
   fLeftMat->MasterToLocal(point,local1);
   Bool_t in1 = fLeft->Contains(local1);
   fRightMat->MasterToLocal(point,local2);
   Bool_t in2 = fRight->Contains(local2);
   Double_t saf1 = fLeft->Safety(local1, in1);
   Double_t saf2 = fRight->Safety(local2, in2);
   if (in1 && in2) return TMath::Min(saf1, saf2);
   if (in1)        return saf1;
   if (in2)        return saf2;
   return TMath::Min(saf1,saf2);
}   

//-----------------------------------------------------------------------------
void TGeoUnion::SetPoints(Float_t * /*buff*/) const
{
// Fill buffer with shape vertices.
}
//-----------------------------------------------------------------------------
void TGeoUnion::Sizeof3D() const
{
// Register 3D size of this shape.
   TGeoBoolNode::Sizeof3D();
}
   

ClassImp(TGeoSubtraction)

//-----------------------------------------------------------------------------
TGeoSubtraction::TGeoSubtraction()
{
// Default constructor
}

//-----------------------------------------------------------------------------
TGeoSubtraction::TGeoSubtraction(const char *expr1, const char *expr2)
          :TGeoBoolNode(expr1, expr2)
{
// Constructor
}

//-----------------------------------------------------------------------------
TGeoSubtraction::TGeoSubtraction(TGeoShape *left, TGeoShape *right, TGeoMatrix *lmat, TGeoMatrix *rmat)
                :TGeoBoolNode(left,right,lmat,rmat)
{
// Constructor providing pointers to components
   if (left->TestShapeBit(TGeoShape::kGeoHalfSpace)) {
      Fatal("TGeoSubstraction", "Substractions from a half-space (%s) not allowed", left->GetName());
   }
}

//-----------------------------------------------------------------------------
TGeoSubtraction::~TGeoSubtraction()
{
// Destructor
// --- deletion of components handled by TGeoManager class.
}

//-----------------------------------------------------------------------------
void TGeoSubtraction::ComputeBBox(Double_t &dx, Double_t &dy, Double_t &dz, Double_t *origin)
{
// Compute bounding box corresponding to a subtraction of two shapes.
   TGeoBBox *box = (TGeoBBox*)fLeft;
   if (box->IsNullBox()) fLeft->ComputeBBox();
   Double_t vert[24];
   Double_t pt[3];
   Int_t i;
   Double_t xmin, xmax, ymin, ymax, zmin, zmax;
   xmin = ymin = zmin = TGeoShape::Big();
   xmax = ymax = zmax = -TGeoShape::Big();
   box->SetBoxPoints(&vert[0]);
   for (i=0; i<8; i++) {
      fLeftMat->LocalToMaster(&vert[3*i], &pt[0]);
      if (pt[0]<xmin) xmin=pt[0];
      if (pt[0]>xmax) xmax=pt[0];
      if (pt[1]<ymin) ymin=pt[1];
      if (pt[1]>ymax) ymax=pt[1];
      if (pt[2]<zmin) zmin=pt[2];
      if (pt[2]>zmax) zmax=pt[2];
   }   
   dx = 0.5*(xmax-xmin);
   origin[0] = 0.5*(xmin+xmax);
   dy = 0.5*(ymax-ymin);
   origin[1] = 0.5*(ymin+ymax);
   dz = 0.5*(zmax-zmin);
   origin[2] = 0.5*(zmin+zmax);
}   

//_____________________________________________________________________________
void TGeoSubtraction::ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm)
{
   norm[0] = norm[1] = 0.;
   norm[2] = 1.;
   Double_t local[3], ldir[3], lnorm[3];
   fRightMat->MasterToLocal(point,local);
   if (fRight->Contains(local)) {
      fRightMat->MasterToLocalVect(dir,ldir);
      fRight->ComputeNormal(local,ldir, lnorm);
      fRightMat->LocalToMasterVect(lnorm,norm);
      return;
   }   
   fLeftMat->MasterToLocal(point,local);
   if (!fLeft->Contains(local)) {
      fLeftMat->MasterToLocalVect(dir,ldir);
      fLeft->ComputeNormal(local,ldir, lnorm);
      fLeftMat->LocalToMasterVect(lnorm,norm);
      return;
   }
   // point is inside left shape, but not inside the right
   local[0] = point[0]+1E-5*dir[0];
   local[1] = point[1]+1E-5*dir[1];
   local[2] = point[2]+1E-5*dir[2];
   if (Contains(local)) {
      local[0] = point[0]-1E-5*dir[0];
      local[1] = point[1]-1E-5*dir[1];
      local[2] = point[2]-1E-5*dir[2];
      if (Contains(local)) return;
   }  
   ComputeNormal(local,dir,norm);
}

//-----------------------------------------------------------------------------
Bool_t TGeoSubtraction::Contains(Double_t *point) const
{
// Find if a subtraction of two shapes contains a given point
   Double_t local[3];
   fLeftMat->MasterToLocal(point, &local[0]);
   Bool_t inside = fLeft->Contains(&local[0]);
   if (!inside) return kFALSE;
   fRightMat->MasterToLocal(point, &local[0]);
   inside = !fRight->Contains(&local[0]);
   return inside;
}
//-----------------------------------------------------------------------------
Int_t TGeoSubtraction::DistanceToPrimitive(Int_t /*px*/, Int_t /*py*/)
{
// Compute minimum distance to shape vertices
   return 9999;
}
//-----------------------------------------------------------------------------
Double_t TGeoSubtraction::DistToOut(Double_t *point, Double_t *dir, Int_t iact,
                              Double_t step, Double_t *safe) const
{
// Compute distance from a given point to outside.
   if (iact<3 && safe) {
      // compute safe distance
      *safe = Safety(point,kTRUE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }
   Double_t local[3], ldir[3], rdir[3];
   Double_t d1, d2, snxt=0.;
   fLeftMat->MasterToLocal(point, &local[0]);
   fLeftMat->MasterToLocalVect(dir, &ldir[0]);
   fRightMat->MasterToLocalVect(dir, &rdir[0]);
   d1 = fLeft->DistToOut(&local[0], &ldir[0], iact, step, safe);
   fRightMat->MasterToLocal(point, &local[0]);
   d2 = fRight->DistToIn(&local[0], &rdir[0], iact, step, safe);
   snxt = TMath::Min(d1, d2);
   return snxt;
}   
//-----------------------------------------------------------------------------
Double_t TGeoSubtraction::DistToIn(Double_t *point, Double_t *dir, Int_t iact,
                              Double_t step, Double_t *safe) const
{
// Compute distance from a given point to inside.
   if (iact<3 && safe) {
      // compute safe distance
      *safe = Safety(point,kFALSE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }
   Double_t local[3], master[3], ldir[3], rdir[3];
   memcpy(&master[0], point, 3*sizeof(Double_t));
   Int_t i;
   Double_t d1, d2, snxt=0.;
   fRightMat->MasterToLocal(point, &local[0]);
   fLeftMat->MasterToLocalVect(dir, &ldir[0]);
   fRightMat->MasterToLocalVect(dir, &rdir[0]);
   // check if inside '-'
   Bool_t inside = fRight->Contains(&local[0]);
   Double_t epsil = 0;
   while (1) {
      if (inside) {
         // propagate to outside of '-'
         d1 = fRight->DistToOut(&local[0], &rdir[0], iact, step, safe);
         snxt += d1+epsil;
         for (i=0; i<3; i++) master[i] += (d1+1E-8)*dir[i];
         // now master outside '-'; check if inside '+'
         fLeftMat->MasterToLocal(&master[0], &local[0]);
         if (fLeft->Contains(&local[0])) return snxt;
         epsil = 1E-8;
      } 
      // master outside '-' and outside '+' ;  find distances to both
      fLeftMat->MasterToLocal(&master[0], &local[0]);
      d2 = fLeft->DistToIn(&local[0], &ldir[0], iact, step, safe);
      if (d2>1E20) return TGeoShape::Big();
      fRightMat->MasterToLocal(&master[0], &local[0]);
      d1 = fRight->DistToIn(&local[0], &rdir[0], iact, step, safe);
      if (d2<d1) {
         snxt += d2+epsil;
         return snxt;
      }   
      // propagate to '-'
      for (i=0; i<3; i++) master[i] += (d1+1E-8)*dir[i];
      epsil = 1E-8;
      snxt += d1+epsil;
      // now inside '-' and not inside '+'
      fRightMat->MasterToLocal(&master[0], &local[0]);
      inside = kTRUE;
   }
}
//-----------------------------------------------------------------------------
Int_t TGeoSubtraction::GetNpoints() const
{
// Returns number of vertices for the composite shape described by this subtraction.
   return 0;
}
//-----------------------------------------------------------------------------
Double_t TGeoSubtraction::Safety(Double_t *point, Bool_t) const
{
// Compute safety distance for a union node;
   Double_t local1[3], local2[3];
   fLeftMat->MasterToLocal(point,local1);
   Bool_t in1 = fLeft->Contains(local1);
   fRightMat->MasterToLocal(point,local2);
   Bool_t in2 = fRight->Contains(local2);
   Double_t saf1 = fLeft->Safety(local1, in1);
   Double_t saf2 = fRight->Safety(local2, in2);
   if (in1 && in2) return saf2;
   if (in1)        return TMath::Min(saf1,saf2);
   if (in2)        return TMath::Max(saf1,saf2);
   return saf1;
}   
//-----------------------------------------------------------------------------
void TGeoSubtraction::SetPoints(Double_t * /*buff*/) const
{
// Fill buffer with shape vertices.
}
//-----------------------------------------------------------------------------
void TGeoSubtraction::SetPoints(Float_t * /*buff*/) const
{
// Fill buffer with shape vertices.
}
//-----------------------------------------------------------------------------
void TGeoSubtraction::Sizeof3D() const
{
// Register 3D size of this shape.
   TGeoBoolNode::Sizeof3D();
}
   


ClassImp(TGeoIntersection)

//-----------------------------------------------------------------------------
TGeoIntersection::TGeoIntersection()
{
// Default constructor
}

//-----------------------------------------------------------------------------
TGeoIntersection::TGeoIntersection(const char *expr1, const char *expr2)
          :TGeoBoolNode(expr1, expr2)
{
// Constructor
}

//-----------------------------------------------------------------------------
TGeoIntersection::TGeoIntersection(TGeoShape *left, TGeoShape *right, TGeoMatrix *lmat, TGeoMatrix *rmat)
                 :TGeoBoolNode(left,right,lmat,rmat)
{
// Constructor providing pointers to components
   Bool_t hs1 = (fLeft->TestShapeBit(TGeoShape::kGeoHalfSpace))?kTRUE:kFALSE;
   Bool_t hs2 = (fRight->TestShapeBit(TGeoShape::kGeoHalfSpace))?kTRUE:kFALSE;
   if (hs1 && hs2) Fatal("ctor", "cannot intersect two half-spaces: %s * %s", left->GetName(), right->GetName());
}

//-----------------------------------------------------------------------------
TGeoIntersection::~TGeoIntersection()
{
// Destructor
// --- deletion of components handled by TGeoManager class.
}

//-----------------------------------------------------------------------------
void TGeoIntersection::ComputeBBox(Double_t &dx, Double_t &dy, Double_t &dz, Double_t *origin)
{
// Compute bounding box corresponding to a intersection of two shapes.
   Bool_t hs1 = (fLeft->TestShapeBit(TGeoShape::kGeoHalfSpace))?kTRUE:kFALSE;
   Bool_t hs2 = (fRight->TestShapeBit(TGeoShape::kGeoHalfSpace))?kTRUE:kFALSE;
   Double_t vert[48];
   Double_t pt[3];
   Int_t i;
   Double_t xmin1, xmax1, ymin1, ymax1, zmin1, zmax1;
   Double_t xmin2, xmax2, ymin2, ymax2, zmin2, zmax2;
   xmin1 = ymin1 = zmin1 = xmin2 = ymin2 = zmin2 = TGeoShape::Big();
   xmax1 = ymax1 = zmax1 = xmax2 = ymax2 = zmax2 =  -TGeoShape::Big();
   if (!hs1) {
      if (((TGeoBBox*)fLeft)->IsNullBox()) fLeft->ComputeBBox();
      ((TGeoBBox*)fLeft)->SetBoxPoints(&vert[0]);
      for (i=0; i<8; i++) {
         fLeftMat->LocalToMaster(&vert[3*i], &pt[0]);
         if (pt[0]<xmin1) xmin1=pt[0];
         if (pt[0]>xmax1) xmax1=pt[0];
         if (pt[1]<ymin1) ymin1=pt[1];
         if (pt[1]>ymax1) ymax1=pt[1];
         if (pt[2]<zmin1) zmin1=pt[2];
         if (pt[2]>zmax1) zmax1=pt[2];
      }   
   }   
   if (!hs2) {
      if (((TGeoBBox*)fRight)->IsNullBox()) fRight->ComputeBBox();
      ((TGeoBBox*)fRight)->SetBoxPoints(&vert[24]);
      for (i=8; i<16; i++) {
         fRightMat->LocalToMaster(&vert[3*i], &pt[0]);
         if (pt[0]<xmin2) xmin2=pt[0];
         if (pt[0]>xmax2) xmax2=pt[0];
         if (pt[1]<ymin2) ymin2=pt[1];
         if (pt[1]>ymax2) ymax2=pt[1];
         if (pt[2]<zmin2) zmin2=pt[2];
         if (pt[2]>zmax2) zmax2=pt[2];
      }
   }      
   if (hs1) {
      dx = 0.5*(xmax2-xmin2);
      origin[0] = 0.5*(xmax2+xmin2);   
      dy = 0.5*(ymax2-ymin2);
      origin[1] = 0.5*(ymax2+ymin2);   
      dz = 0.5*(zmax2-zmin2);
      origin[2] = 0.5*(zmax2+zmin2);   
      return;
   }            
   if (hs2) {
      dx = 0.5*(xmax1-xmin1);
      origin[0] = 0.5*(xmax1+xmin1);   
      dy = 0.5*(ymax1-ymin1);
      origin[1] = 0.5*(ymax1+ymin1);   
      dz = 0.5*(zmax1-zmin1);
      origin[2] = 0.5*(zmax1+zmin1);   
      return;
   }   
   Double_t sort[4];
   Int_t isort[4];
   sort[0] = xmin1;
   sort[1] = xmax1;
   sort[2] = xmin2;
   sort[3] = xmax2;
   TMath::Sort(4, &sort[0], &isort[0], kFALSE);
   if (isort[1]%2) {
      Warning("ComputeBBox", "shapes %s and %s do not intersect", fLeft->GetName(), fRight->GetName());
      dx = dy = dz = 0;
      memset(origin, 0, 3*sizeof(Double_t));
      return;
   }
   dx = 0.5*(sort[isort[2]]-sort[isort[1]]);
   origin[0] = 0.5*(sort[isort[1]]+sort[isort[2]]);   
   sort[0] = ymin1;
   sort[1] = ymax1;
   sort[2] = ymin2;
   sort[3] = ymax2;
   TMath::Sort(4, &sort[0], &isort[0], kFALSE);
   if (isort[1]%2) {
      Warning("ComputeBBox", "shapes %s and %s do not intersect", fLeft->GetName(), fRight->GetName());
      dx = dy = dz = 0;
      memset(origin, 0, 3*sizeof(Double_t));
      return;
   }
   dy = 0.5*(sort[isort[2]]-sort[isort[1]]);
   origin[1] = 0.5*(sort[isort[1]]+sort[isort[2]]);   
   sort[0] = zmin1;
   sort[1] = zmax1;
   sort[2] = zmin2;
   sort[3] = zmax2;
   TMath::Sort(4, &sort[0], &isort[0], kFALSE);
   if (isort[1]%2) {
      Warning("ComputeBBox", "shapes %s and %s do not intersect", fLeft->GetName(), fRight->GetName());
      dx = dy = dz = 0;
      memset(origin, 0, 3*sizeof(Double_t));
      return;
   }
   dz = 0.5*(sort[isort[2]]-sort[isort[1]]);
   origin[2] = 0.5*(sort[isort[1]]+sort[isort[2]]);   
}   

//_____________________________________________________________________________
void TGeoIntersection::ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm)
{
   Double_t local[3], ldir[3], lnorm[3];
   norm[0] = norm[1] = 0.;
   norm[2] = 1.;
   fLeftMat->MasterToLocal(point,local);
   if (!fLeft->Contains(local)) {
      fLeftMat->MasterToLocalVect(dir,ldir);
      fLeft->ComputeNormal(local,ldir,lnorm);
      fLeftMat->LocalToMasterVect(lnorm,norm);
      return;
   }
   fRightMat->MasterToLocal(point,local);
   if (!fRight->Contains(local)) {
      fRightMat->MasterToLocalVect(dir,ldir);
      fRight->ComputeNormal(local,ldir,lnorm);
      fRightMat->LocalToMasterVect(lnorm,norm);
      return;
   }
   // point is inside intersection.
   local[0] = point[0] + 1E-5*dir[0];   
   local[1] = point[1] + 1E-5*dir[1];   
   local[2] = point[2] + 1E-5*dir[2];
   if (Contains(local)) {
      local[0] = point[0] - 1E-5*dir[0];   
      local[1] = point[1] - 1E-5*dir[1];   
      local[2] = point[2] - 1E-5*dir[2];
      if (Contains(local)) return;
   }
   ComputeNormal(local,dir,norm);   
}

//-----------------------------------------------------------------------------
Bool_t TGeoIntersection::Contains(Double_t *point) const
{
// Find if a intersection of two shapes contains a given point
   Double_t local[3];
   fLeftMat->MasterToLocal(point, &local[0]);
   Bool_t inside = fLeft->Contains(&local[0]);
   if (!inside) return kFALSE;
   fRightMat->MasterToLocal(point, &local[0]);
   inside = fRight->Contains(&local[0]);
   return inside;
}
//-----------------------------------------------------------------------------
Int_t TGeoIntersection::DistanceToPrimitive(Int_t /*px*/, Int_t /*py*/)
{
// Compute minimum distance to shape vertices
   return 9999;
}
//-----------------------------------------------------------------------------
Double_t TGeoIntersection::DistToOut(Double_t *point, Double_t *dir, Int_t iact,
                              Double_t step, Double_t *safe) const
{
// Compute distance from a given point to outside.
   if (iact<3 && safe) {
      // compute safe distance
      *safe = Safety(point,kTRUE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }
   Double_t local[3], ldir[3], rdir[3];
   Double_t d1, d2, snxt=0.;
   fLeftMat->MasterToLocal(point, &local[0]);
   fLeftMat->MasterToLocalVect(dir, &ldir[0]);
   fRightMat->MasterToLocalVect(dir, &rdir[0]);
   d1 = fLeft->DistToOut(&local[0], &ldir[0], iact, step, safe);
   fRightMat->MasterToLocal(point, &local[0]);
   d2 = fRight->DistToOut(&local[0], &rdir[0], iact, step, safe);
   snxt = TMath::Min(d1, d2);
   return snxt;
}   
//-----------------------------------------------------------------------------
Double_t TGeoIntersection::DistToIn(Double_t *point, Double_t *dir, Int_t iact,
                              Double_t step, Double_t *safe) const
{
// Compute distance from a given point to inside.
   if (iact<3 && safe) {
      // compute safe distance
      *safe = Safety(point,kFALSE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }
   Double_t lpt[3], rpt[3], master[3], ldir[3], rdir[3];
   memcpy(master, point, 3*sizeof(Double_t));
   Int_t i;
   Double_t d1 = 0;
   Double_t d2 = 0;
   fLeftMat->MasterToLocal(point, lpt);
   fRightMat->MasterToLocal(point, rpt);
   fLeftMat->MasterToLocalVect(dir, ldir);
   fRightMat->MasterToLocalVect(dir, rdir);
   Bool_t inleft = fLeft->Contains(lpt);
   Bool_t inright = fRight->Contains(rpt);
   if (inleft && inright) return 0.;
   if (!inleft)  {
      d1 = fLeft->DistToIn(lpt,ldir,iact,step,safe);
      if (d1 > 1E20) return TGeoShape::Big();
   }   
   if (!inright) {
      d2 = fRight->DistToIn(rpt,rdir,iact,step,safe);
      if (d2>1E20) return TGeoShape::Big();
   }
   Double_t snext = TMath::Max(d1,d2);   
   for (i=0; i<3; i++) master[i] += (snext+1E-6)*dir[i];
   if (Contains(master)) return snext;
   snext += DistToIn(master,dir,iact,step,safe)+1E-6;
   return snext;
}      

//-----------------------------------------------------------------------------
Int_t TGeoIntersection::GetNpoints() const
{
// Returns number of vertices for the composite shape described by this intersection.
   return 0;
}
//-----------------------------------------------------------------------------
Double_t TGeoIntersection::Safety(Double_t *point, Bool_t) const
{
// Compute safety distance for a union node;
   Double_t local1[3], local2[3];
   fLeftMat->MasterToLocal(point,local1);
   Bool_t in1 = fLeft->Contains(local1);
   fRightMat->MasterToLocal(point,local2);
   Bool_t in2 = fRight->Contains(local2);
   Double_t saf1 = fLeft->Safety(local1, in1);
   Double_t saf2 = fRight->Safety(local2, in2);
   if (in1 && in2) return TMath::Min(saf1, saf2);
   if (in1)        return saf2;
   if (in2)        return saf1;
   return TMath::Max(saf1,saf2);
}   
//-----------------------------------------------------------------------------
void TGeoIntersection::SetPoints(Double_t * /*buff*/) const
{
// Fill buffer with shape vertices.
}
//-----------------------------------------------------------------------------
void TGeoIntersection::SetPoints(Float_t * /*buff*/) const
{
// Fill buffer with shape vertices.
}
//-----------------------------------------------------------------------------
void TGeoIntersection::Sizeof3D() const
{
// Register 3D size of this shape.
   TGeoBoolNode::Sizeof3D();
}
