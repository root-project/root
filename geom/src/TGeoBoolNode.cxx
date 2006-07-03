// @(#):$Name:  $:$Id: TGeoBoolNode.cxx,v 1.26 2006/03/24 15:11:23 brun Exp $
// Author: Andrei Gheata   30/05/02
// TGeoBoolNode::Contains and parser implemented by Mihaela Gheata

   
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"

#include "TGeoCompositeShape.h"
#include "TGeoMatrix.h"
#include "TGeoManager.h"

#include "TGeoBoolNode.h"

#include "TVirtualPad.h"
#include "TVirtualViewer3D.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

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
   fSelected = 0;
}
//-----------------------------------------------------------------------------
TGeoBoolNode::TGeoBoolNode(const char *expr1, const char *expr2)
{
// Constructor called by TGeoCompositeShape providing 2 subexpressions for the 2 branches.
   fLeft     = 0;
   fRight    = 0;
   fLeftMat  = 0;
   fRightMat = 0;
   fSelected = 0;
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
// Constructor providing left and right shapes and matrices (in the Boolean operation).
   fSelected = 0;
   fLeft = left;
   fRight = right;
   fLeftMat = lmat;
   if (!fLeftMat) fLeftMat = gGeoIdentity;
   else fLeftMat->RegisterYourself();
   fRightMat = rmat;
   if (!fRightMat) fRightMat = gGeoIdentity;
   else fRightMat->RegisterYourself();
   if (!fLeft) {
      Error("ctor", "left shape is NULL");
      return;
   }   
   if (!fRight) {
      Error("ctor", "right shape is NULL");
      return;
   }   
}

//-----------------------------------------------------------------------------
TGeoBoolNode::~TGeoBoolNode()
{
// Destructor.
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
void TGeoBoolNode::Paint(Option_t * option)
{
// Special schema for feeding the 3D buffers to the painter client.
   TVirtualViewer3D * viewer = gPad->GetViewer3D();
   if (!viewer) return;

   // Components of composite shape hierarchies for local frame viewers are painted 
   // in coordinate frame of the top level composite shape. So we force 
   // conversion to this.  See TGeoPainter::PaintNode for loading of GLMatrix.
   Bool_t localFrame = kFALSE; //viewer->PreferLocalFrame();

   TGeoHMatrix *glmat = (TGeoHMatrix*)TGeoShape::GetTransform();
   TGeoHMatrix mat;
   mat = glmat; // keep a copy

   // Now perform fetch and add of the two components buffers.
   // Note we assume that composite shapes are always completely added
   // so don't bother to get addDaughters flag from viewer->AddObject()

   // Setup matrix and fetch/add the left component buffer
   glmat->Multiply(fLeftMat);
   //fLeft->Paint(option);
   if (TGeoCompositeShape *left = dynamic_cast<TGeoCompositeShape *>(fLeft)) left->PaintComposite(option);
   else {
      const TBuffer3D & leftBuffer = fLeft->GetBuffer3D(TBuffer3D::kAll, localFrame);
      viewer->AddObject(leftBuffer);
   }

   // Setup matrix and fetch/add the right component buffer
   *glmat = &mat;
   glmat->Multiply(fRightMat);
   //fRight->Paint(option);
   if (TGeoCompositeShape *right = dynamic_cast<TGeoCompositeShape *>(fRight)) right->PaintComposite(option);
   else {
      const TBuffer3D & rightBuffer = fRight->GetBuffer3D(TBuffer3D::kAll, localFrame);
      viewer->AddObject(rightBuffer);
   }

   *glmat = &mat;   
}
//_____________________________________________________________________________
void TGeoBoolNode::RegisterMatrices()
{
// Register all matrices of the boolean node and descendents.
   if (!fLeftMat->IsIdentity()) fLeftMat->RegisterYourself();   
   if (!fRightMat->IsIdentity()) fRightMat->RegisterYourself();   
   if (fLeft->IsComposite()) ((TGeoCompositeShape*)fLeft)->GetBoolNode()->RegisterMatrices();
   if (fRight->IsComposite()) ((TGeoCompositeShape*)fRight)->GetBoolNode()->RegisterMatrices();
}
//_____________________________________________________________________________
void TGeoBoolNode::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
// Save a primitive as a C++ statement(s) on output stream "out".
   fLeft->SavePrimitive(out,option);
   fRight->SavePrimitive(out,option);
   if (!fLeftMat->IsIdentity()) {
      fLeftMat->RegisterYourself();
      fLeftMat->SavePrimitive(out,option);
   }      
   if (!fRightMat->IsIdentity()) {
      fRightMat->RegisterYourself();
      fRightMat->SavePrimitive(out,option);
   }      
}
//-----------------------------------------------------------------------------
void TGeoBoolNode::Sizeof3D() const
{
// Register size of this 3D object
   fLeft->Sizeof3D();
   fRight->Sizeof3D();
}
ClassImp(TGeoUnion)

//-----------------------------------------------------------------------------
void TGeoUnion::Paint(Option_t *option)
{
// Paint method.
   TVirtualViewer3D *viewer = gPad->GetViewer3D();

   if (!viewer) {
      Error("Paint", "gPad->GetViewer3D() returned 0, cannot work with composite!\n");
      return;
   }

   viewer->AddCompositeOp(TBuffer3D::kCSUnion);

   TGeoBoolNode::Paint(option);
}

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
   TGeoBoolNode *node = (TGeoBoolNode*)this;
   fLeftMat->MasterToLocal(point, &local[0]);
   Bool_t inside = fLeft->Contains(&local[0]);
   if (inside) {
      node->SetSelected(1);
      return kTRUE;
   }   
   fRightMat->MasterToLocal(point, &local[0]);
   inside = fRight->Contains(&local[0]);
   if (inside) node->SetSelected(2);
   return inside;
}

//_____________________________________________________________________________
void TGeoUnion::ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm)
{
// Normal computation in POINT. The orientation is chosen so that DIR.dot.NORM>0.
   norm[0] = norm[1] = 0.;
   norm[2] = 1.;
   Double_t local[3];
   Double_t ldir[3], lnorm[3];
   if (fSelected == 1) {
      fLeftMat->MasterToLocal(point, local);
      fLeftMat->MasterToLocalVect(dir, ldir);
      fLeft->ComputeNormal(local,ldir,lnorm);
      fLeftMat->LocalToMasterVect(lnorm, norm);
      return;
   }
   if (fSelected == 2) {
      fRightMat->MasterToLocal(point, local);
      fRightMat->MasterToLocalVect(dir, ldir);
      fRight->ComputeNormal(local,ldir,lnorm);
      fRightMat->LocalToMasterVect(lnorm, norm);
      return;
   }            
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
// Compute minimum distance to shape vertices.
   return 9999;
}
//-----------------------------------------------------------------------------
Double_t TGeoUnion::DistFromInside(Double_t *point, Double_t *dir, Int_t iact,
                              Double_t step, Double_t *safe) const
{
// Computes distance from a given point inside the shape to its boundary.
   if (iact<3 && safe) {
      // compute safe distance
      *safe = Safety(point,kTRUE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }

   Double_t local[3], master[3], ldir[3], rdir[3];
   memcpy(master, point, 3*sizeof(Double_t));
   Int_t i;
   TGeoBoolNode *node = (TGeoBoolNode*)this;
   Double_t d1=0., d2=0., snxt=0.;
   fLeftMat->MasterToLocalVect(dir, ldir);
   fRightMat->MasterToLocalVect(dir, rdir);
   fLeftMat->MasterToLocal(point, local);
   Bool_t inside1 = fLeft->Contains(local);
   if (inside1) d1 = fLeft->DistFromInside(local, ldir, 3);
   fRightMat->MasterToLocal(point, local);
   Bool_t inside2 = fRight->Contains(local);
   if (inside2) d2 = fRight->DistFromInside(local, rdir, 3);

   while (inside1 || inside2) {
      if (inside1 && inside2) {
         if (d1<d2) {      
            snxt += d1;
            node->SetSelected(1);
            // propagate to exit of left shape
            inside1 = kFALSE;
            for (i=0; i<3; i++) master[i] += d1*dir[i];
            // check if propagated point is in right shape        
            fRightMat->MasterToLocal(master, local);
            inside2 = fRight->Contains(local);
            if (!inside2) return snxt;
            d2 = fRight->DistFromInside(local, rdir, 3);
            if (d2 < TGeoShape::Tolerance()) return snxt;
         } else {
            snxt += d2;
            node->SetSelected(2);
            // propagate to exit of right shape
            inside2 = kFALSE;
            for (i=0; i<3; i++) master[i] += d2*dir[i];
            // check if propagated point is in left shape        
            fLeftMat->MasterToLocal(master, local);
            inside1 = fLeft->Contains(local);
            if (!inside1) return snxt;
            d1 = fLeft->DistFromInside(local, ldir, 3);
            if (d1 < TGeoShape::Tolerance()) return snxt;
         }
      } 
      if (inside1) {
         snxt += d1;   
         node->SetSelected(1);
         // propagate to exit of left shape
         inside1 = kFALSE;
         for (i=0; i<3; i++) master[i] += d1*dir[i];
         // check if propagated point is in right shape        
         fRightMat->MasterToLocal(master, local);
         inside2 = fRight->Contains(local);
         if (!inside2) return snxt;
         d2 = fRight->DistFromInside(local, rdir, 3);
         if (d2 < TGeoShape::Tolerance()) return snxt;
      }   
      if (inside2) {
         snxt += d2;
         node->SetSelected(2);
         // propagate to exit of right shape
         inside2 = kFALSE;
         for (i=0; i<3; i++) master[i] += d2*dir[i];
         // check if propagated point is in left shape        
         fLeftMat->MasterToLocal(master, local);
         inside1 = fLeft->Contains(local);
         if (!inside1) return snxt;
         d1 = fLeft->DistFromInside(local, ldir, 3);
         if (d1 < TGeoShape::Tolerance()) return snxt;
      }
   }      
   return snxt;
}
//-----------------------------------------------------------------------------
Double_t TGeoUnion::DistFromOutside(Double_t *point, Double_t *dir, Int_t iact,
                              Double_t step, Double_t *safe) const
{
// Compute distance from a given outside point to the shape.
   if (iact<3 && safe) {
      // compute safe distance
      *safe = Safety(point,kFALSE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }
   TGeoBoolNode *node = (TGeoBoolNode*)this;
   Double_t local[3], ldir[3], rdir[3];
   Double_t d1, d2, snxt;
   fLeftMat->MasterToLocal(point, &local[0]);
   fLeftMat->MasterToLocalVect(dir, &ldir[0]);
   fRightMat->MasterToLocalVect(dir, &rdir[0]);
   d1 = fLeft->DistFromOutside(&local[0], &ldir[0], iact, step, safe);
   fRightMat->MasterToLocal(point, &local[0]);
   d2 = fRight->DistFromOutside(&local[0], &rdir[0], iact, step, safe);
   if (d1<d2) {
      snxt = d1;
      node->SetSelected(1);
   } else {
      snxt = d2;
      node->SetSelected(2);
   }      
   return snxt;
}
//-----------------------------------------------------------------------------
Int_t TGeoUnion::GetNpoints() const
{
// Returns number of vertices for the composite shape described by this union.
   return 0;
}
//-----------------------------------------------------------------------------
void TGeoUnion::SetPoints(Double_t * /*points*/) const
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

//_____________________________________________________________________________
void TGeoUnion::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
// Save a primitive as a C++ statement(s) on output stream "out".
   TGeoBoolNode::SavePrimitive(out,option);
   out << "   pBoolNode = new TGeoUnion(";
   out << fLeft->GetPointerName() << ",";
   out << fRight->GetPointerName() << ",";
   if (!fLeftMat->IsIdentity()) out << fLeftMat->GetPointerName() << ",";
   else                         out << "0,";
   if (!fRightMat->IsIdentity()) out << fRightMat->GetPointerName() << ");" << endl;
   else                         out << "0);" << endl;
}   

//-----------------------------------------------------------------------------
void TGeoUnion::SetPoints(Float_t * /*points*/) const
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
void TGeoSubtraction::Paint(Option_t *option)
{
// Paint method.
   TVirtualViewer3D *viewer = gPad->GetViewer3D();

   if (!viewer) {
      Error("Paint", "gPad->GetViewer3D() returned 0, cannot work with composite!\n");
      return;
   }

   viewer->AddCompositeOp(TBuffer3D::kCSDifference);

   TGeoBoolNode::Paint(option);
}

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
// Normal computation in POINT. The orientation is chosen so that DIR.dot.NORM>0.
   norm[0] = norm[1] = 0.;
   norm[2] = 1.;
   Double_t local[3], ldir[3], lnorm[3];
   if (fSelected == 1) {
      fLeftMat->MasterToLocal(point, local);
      fLeftMat->MasterToLocalVect(dir, ldir);
      fLeft->ComputeNormal(local,ldir,lnorm);
      fLeftMat->LocalToMasterVect(lnorm, norm);
      return;
   }
   if (fSelected == 2) {
      fRightMat->MasterToLocal(point, local);
      fRightMat->MasterToLocalVect(dir, ldir);
      fRight->ComputeNormal(local,ldir,lnorm);
      fRightMat->LocalToMasterVect(lnorm, norm);
      return;
   }
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
   TGeoBoolNode *node = (TGeoBoolNode*)this;
   fLeftMat->MasterToLocal(point, &local[0]);
   Bool_t inside = fLeft->Contains(&local[0]);
   if (inside) node->SetSelected(1);
   else return kFALSE;
   fRightMat->MasterToLocal(point, &local[0]);
   inside = !fRight->Contains(&local[0]);
   if (!inside) node->SetSelected(2);
   return inside;
}
//-----------------------------------------------------------------------------
Int_t TGeoSubtraction::DistanceToPrimitive(Int_t /*px*/, Int_t /*py*/)
{
// Compute minimum distance to shape vertices
   return 9999;
}
//-----------------------------------------------------------------------------
Double_t TGeoSubtraction::DistFromInside(Double_t *point, Double_t *dir, Int_t iact,
                              Double_t step, Double_t *safe) const
{
// Compute distance from a given point inside to the shape boundary.
   if (iact<3 && safe) {
      // compute safe distance
      *safe = Safety(point,kTRUE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }
   TGeoBoolNode *node = (TGeoBoolNode*)this;
   Double_t local[3], ldir[3], rdir[3];
   Double_t d1, d2, snxt=0.;
   fLeftMat->MasterToLocal(point, &local[0]);
   fLeftMat->MasterToLocalVect(dir, &ldir[0]);
   fRightMat->MasterToLocalVect(dir, &rdir[0]);
   d1 = fLeft->DistFromInside(&local[0], &ldir[0], iact, step, safe);
   fRightMat->MasterToLocal(point, &local[0]);
   d2 = fRight->DistFromOutside(&local[0], &rdir[0], iact, step, safe);
   if (d1<d2) {
      snxt = d1;
      node->SetSelected(1);
   } else {
      snxt = d2;
      node->SetSelected(2);
   }      
   return snxt;
}   
//-----------------------------------------------------------------------------
Double_t TGeoSubtraction::DistFromOutside(Double_t *point, Double_t *dir, Int_t iact,
                              Double_t step, Double_t *safe) const
{
// Compute distance from a given point outside to the shape.
   if (iact<3 && safe) {
      // compute safe distance
      *safe = Safety(point,kFALSE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }
   TGeoBoolNode *node = (TGeoBoolNode*)this;
   Double_t local[3], master[3], ldir[3], rdir[3];
   memcpy(&master[0], point, 3*sizeof(Double_t));
   Int_t i;
   Double_t d1, d2, snxt=0.;
   fRightMat->MasterToLocal(point, &local[0]);
   fLeftMat->MasterToLocalVect(dir, &ldir[0]);
   fRightMat->MasterToLocalVect(dir, &rdir[0]);
   // check if inside '-'
   Bool_t inside = fRight->Contains(&local[0]);
   Double_t epsil = 0.;
   while (1) {
      if (inside) {
         // propagate to outside of '-'
         node->SetSelected(2);
         d1 = fRight->DistFromInside(&local[0], &rdir[0], iact, step, safe);
         snxt += d1+epsil;
         for (i=0; i<3; i++) master[i] += (d1+1E-8)*dir[i];
         epsil = 1.E-8;
         // now master outside '-'; check if inside '+'
         fLeftMat->MasterToLocal(&master[0], &local[0]);
         if (fLeft->Contains(&local[0])) return snxt;
      } 
      // master outside '-' and outside '+' ;  find distances to both
      node->SetSelected(1);
      fLeftMat->MasterToLocal(&master[0], &local[0]);
      d2 = fLeft->DistFromOutside(&local[0], &ldir[0], iact, step, safe);
      if (d2>1E20) return TGeoShape::Big();
      
      fRightMat->MasterToLocal(&master[0], &local[0]);
      d1 = fRight->DistFromOutside(&local[0], &rdir[0], iact, step, safe);
      if (d2<d1) {
         snxt += d2+epsil;
         return snxt;
      }   
      // propagate to '-'
      snxt += d1+epsil;
      for (i=0; i<3; i++) master[i] += (d1+1E-8)*dir[i];
      epsil = 1.E-8;
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
//_____________________________________________________________________________
void TGeoSubtraction::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
// Save a primitive as a C++ statement(s) on output stream "out".
   TGeoBoolNode::SavePrimitive(out,option);
   out << "   pBoolNode = new TGeoSubtraction(";
   out << fLeft->GetPointerName() << ",";
   out << fRight->GetPointerName() << ",";
   if (!fLeftMat->IsIdentity()) out << fLeftMat->GetPointerName() << ",";
   else                         out << "0,";
   if (!fRightMat->IsIdentity()) out << fRightMat->GetPointerName() << ");" << endl;
   else                         out << "0);" << endl;
}   
//-----------------------------------------------------------------------------
void TGeoSubtraction::SetPoints(Double_t * /*points*/) const
{
// Fill buffer with shape vertices.
}
//-----------------------------------------------------------------------------
void TGeoSubtraction::SetPoints(Float_t * /*points*/) const
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
void TGeoIntersection::Paint(Option_t *option)
{
// Paint method.
   TVirtualViewer3D *viewer = gPad->GetViewer3D();

   if (!viewer) {
      Error("Paint", "gPad->GetViewer3D() returned 0, cannot work with composite!\n");
      return;
   }

   viewer->AddCompositeOp(TBuffer3D::kCSIntersection);

   TGeoBoolNode::Paint(option);
}

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
// Normal computation in POINT. The orientation is chosen so that DIR.dot.NORM>0.
   Double_t local[3], ldir[3], lnorm[3];
   norm[0] = norm[1] = 0.;
   norm[2] = 1.;
   if (fSelected == 1) {
      fLeftMat->MasterToLocal(point, local);
      fLeftMat->MasterToLocalVect(dir, ldir);
      fLeft->ComputeNormal(local,ldir,lnorm);
      fLeftMat->LocalToMasterVect(lnorm, norm);
      return;
   }
   if (fSelected == 2) {
      fRightMat->MasterToLocal(point, local);
      fRightMat->MasterToLocalVect(dir, ldir);
      fRight->ComputeNormal(local,ldir,lnorm);
      fRightMat->LocalToMasterVect(lnorm, norm);
      return;
   }            
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
Double_t TGeoIntersection::DistFromInside(Double_t *point, Double_t *dir, Int_t iact,
                              Double_t step, Double_t *safe) const
{
// Compute distance from a given point inside to the shape boundary.
   if (iact<3 && safe) {
      // compute safe distance
      *safe = Safety(point,kTRUE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }
   TGeoBoolNode *node = (TGeoBoolNode*)this;
   Double_t local[3], ldir[3], rdir[3];
   Double_t d1, d2, snxt=0.;
   fLeftMat->MasterToLocal(point, &local[0]);
   fLeftMat->MasterToLocalVect(dir, &ldir[0]);
   fRightMat->MasterToLocalVect(dir, &rdir[0]);
   d1 = fLeft->DistFromInside(&local[0], &ldir[0], iact, step, safe);
   fRightMat->MasterToLocal(point, &local[0]);
   d2 = fRight->DistFromInside(&local[0], &rdir[0], iact, step, safe);
   if (d1<d2) {
      snxt = d1;
      node->SetSelected(1);
   } else {
      snxt = d2;
      node->SetSelected(2);
   }      
   return snxt;
}   
//-----------------------------------------------------------------------------
Double_t TGeoIntersection::DistFromOutside(Double_t *point, Double_t *dir, Int_t iact,
                              Double_t step, Double_t *safe) const
{
// Compute distance from a given point outside to the shape.
   if (iact<3 && safe) {
      // compute safe distance
      *safe = Safety(point,kFALSE);
      if (iact==0) return TGeoShape::Big();
      if (iact==1 && step<*safe) return TGeoShape::Big();
   }
   TGeoBoolNode *node = (TGeoBoolNode*)this;
   Double_t lpt[3], rpt[3], master[3], ldir[3], rdir[3];
   memcpy(master, point, 3*sizeof(Double_t));
   Int_t i;
   Double_t d1 = 0.;
   Double_t d2 = 0.;
   fLeftMat->MasterToLocal(point, lpt);
   fRightMat->MasterToLocal(point, rpt);
   fLeftMat->MasterToLocalVect(dir, ldir);
   fRightMat->MasterToLocalVect(dir, rdir);
   Bool_t inleft = fLeft->Contains(lpt);
   Bool_t inright = fRight->Contains(rpt);
   node->SetSelected(0);
   Double_t snext = 0.0;
   if (inleft && inright) return snext;

   while (1) {
      d1 = d2 = 0.0;
      if (!inleft)  {
         d1 = fLeft->DistFromOutside(lpt,ldir,3);
         if (d1>1E20) return TGeoShape::Big();
      }
      if (!inright) {  
         d2 = fRight->DistFromOutside(rpt,rdir,3);
         if (d2>1E20) return TGeoShape::Big();
      }
   
      if (d1>d2) {
         // propagate to left shape
         snext += d1;
         node->SetSelected(1);
         inleft = kTRUE;
         for (i=0; i<3; i++) master[i] += d1*dir[i];
         fRightMat->MasterToLocal(master,rpt);
         // check if propagated point is inside right shape
         inright = fRight->Contains(rpt);
         if (inright) return snext;
         // here inleft=true, inright=false         
      } else {
         // propagate to right shape
         snext += d2;
         node->SetSelected(2);
         inright = kTRUE;
         for (i=0; i<3; i++) master[i] += d2*dir[i];
         fLeftMat->MasterToLocal(master,lpt);
         // check if propagated point is inside left shape
         inleft = fLeft->Contains(lpt);
         if (inleft) return snext;
         // here inleft=false, inright=true
      }            
   }   
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
//_____________________________________________________________________________
void TGeoIntersection::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
// Save a primitive as a C++ statement(s) on output stream "out".
   TGeoBoolNode::SavePrimitive(out,option);
   out << "   pBoolNode = new TGeoIntersection(";
   out << fLeft->GetPointerName() << ",";
   out << fRight->GetPointerName() << ",";
   if (!fLeftMat->IsIdentity()) out << fLeftMat->GetPointerName() << ",";
   else                         out << "0,";
   if (!fRightMat->IsIdentity()) out << fRightMat->GetPointerName() << ");" << endl;
   else                         out << "0);" << endl;
}   
//-----------------------------------------------------------------------------
void TGeoIntersection::SetPoints(Double_t * /*points*/) const
{
// Fill buffer with shape vertices.
}
//-----------------------------------------------------------------------------
void TGeoIntersection::SetPoints(Float_t * /*points*/) const
{
// Fill buffer with shape vertices.
}
//-----------------------------------------------------------------------------
void TGeoIntersection::Sizeof3D() const
{
// Register 3D size of this shape.
   TGeoBoolNode::Sizeof3D();
}
