// @(#)root/geom:$Id$
// Author: Andrei Gheata   28/04/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


////////////////////////////////////////////////////////////////////////////////
//   TGeoHelix - class representing a helix curve
//
//  A helix is a curve defined by the following equations:
//     x = (1/c) * COS(q*phi)
//     y = (1/c) * SIN(q*phi)
//     z = s * alfa
// where:
//     c = 1/Rxy  - curvature in XY plane
//     phi        - phi angle 
//     S = 2*PI*s - vertical separation between helix loops
//     q = +/- 1  - (+)=left-handed, (-)=right-handed
//
//   In particular, a helix describes the trajectory of a charged particle in magnetic 
// field. In such case, the helix is right-handed for negative particle charge.
// To define a helix, one must define:
//   - the curvature - positive defined
//   - the Z step made after one full turn of the helix
//   - the particle charge sign
//   - the initial particle position and direction (force normalization to unit)
//   - the magnetic field direction
//
// A helix provides:
//   - propagation to a given Z position (in global frame)
//   Double_t *point = TGeoHelix::PropagateToZ(Double_t z);
//   - propagation to an arbitrary plane, returning also the new point
//   - propagation in a geometry until the next crossed surface
//   - computation of the total track length along a helix

#include "TMath.h"
#include "TGeoShape.h"
#include "TGeoMatrix.h"
#include "TGeoHelix.h"

ClassImp(TGeoHelix)

//_____________________________________________________________________________
TGeoHelix::TGeoHelix()
{ 
// Dummy constructor
   fC    = 0.;
   fS    = 0.;
   fStep = 0.;
   fPhi  = 0.;
   fPointInit[0] = fPointInit[1] = fPointInit[2] = 0.;
   fDirInit[0] = fDirInit[1] = fDirInit[2] = 0.;
   fPoint[0] = fPoint[1] = fPoint[2] = 0.;
   fDir[0] = fDir[1] = fDir[2] = 0.;
   fB[0] = fB[1] = fB[2] = 0.;
   fQ    = 0;
   fMatrix = 0;
   TObject::SetBit(kHelixNeedUpdate, kTRUE);   
   TObject::SetBit(kHelixStraigth, kFALSE);
   TObject::SetBit(kHelixCircle, kFALSE);
}

//_____________________________________________________________________________
TGeoHelix::TGeoHelix(Double_t curvature, Double_t hstep, Int_t charge)
{ 
// Normal constructor
   SetXYcurvature(curvature);
   SetHelixStep(hstep);
   SetCharge(charge);
   fStep = 0.;
   fPhi  = 0.;
   fPointInit[0] = fPointInit[1] = fPointInit[2] = 0.;
   fDirInit[0] = fDirInit[1] = fDirInit[2] = 0.;
   fPoint[0] = fPoint[1] = fPoint[2] = 0.;
   fDir[0] = fDir[1] = fDir[2] = 0.;
   fB[0] = fB[1] = fB[2] = 0.;
   fQ    = 0;
   fMatrix    = new TGeoHMatrix();
   TObject::SetBit(kHelixNeedUpdate, kTRUE);   
   TObject::SetBit(kHelixStraigth, kFALSE);
   TObject::SetBit(kHelixCircle, kFALSE);
}

//_____________________________________________________________________________
TGeoHelix::~TGeoHelix()
{
// Destructor
   if (fMatrix)    delete fMatrix;
}

//_____________________________________________________________________________
Double_t TGeoHelix::ComputeSafeStep(Double_t epsil) const
{
// Compute safe linear step that can be made such that the error
// between linear-helix extrapolation is less than EPSIL.
   if (TestBit(kHelixStraigth) || TMath::Abs(fC)<TGeoShape::Tolerance()) return 1.E30;
   Double_t c = GetTotalCurvature();
   Double_t step = TMath::Sqrt(2.*epsil/c);
   return step;
}   

//_____________________________________________________________________________
void TGeoHelix::InitPoint(Double_t x0, Double_t y0, Double_t z0)
{
// Initialize coordinates of a point on the helix
   fPointInit[0] = x0;
   fPointInit[1] = y0;
   fPointInit[2] = z0;
   TObject::SetBit(kHelixNeedUpdate, kTRUE);   
}   

//_____________________________________________________________________________
void TGeoHelix::InitPoint (Double_t *point)
{
// Set initial point on the helix.
   InitPoint(point[0], point[1], point[2]);
}

//_____________________________________________________________________________
void TGeoHelix::InitDirection(Double_t dirx, Double_t diry, Double_t dirz, Bool_t is_normalized)
{
// Initialize particle direction (tangent on the helix in initial point)   
   fDirInit[0] = dirx;
   fDirInit[1] = diry;
   fDirInit[2] = dirz;
   TObject::SetBit(kHelixNeedUpdate, kTRUE);
   if (is_normalized) return;
   Double_t norm = 1./TMath::Sqrt(dirx*dirx+diry*diry+dirz*dirz);
   for (Int_t i=0; i<3; i++) fDirInit[i] *= norm;   
}   
   
//_____________________________________________________________________________
void TGeoHelix::InitDirection(Double_t *dir, Bool_t is_normalized)
{
// Initialize particle direction (tangent on the helix in initial point)   
   InitDirection(dir[0], dir[1], dir[2], is_normalized);
}     

//_____________________________________________________________________________
Double_t TGeoHelix::GetTotalCurvature() const
{
// Compute helix total curvature
   Double_t k = fC/(1.+fC*fC*fS*fS);
   return k;
}

//_____________________________________________________________________________
void TGeoHelix::SetXYcurvature(Double_t curvature)
{
// Set XY curvature: c = 1/Rxy
   fC    = curvature;
   TObject::SetBit(kHelixNeedUpdate, kTRUE);
   if (fC < 0) {
      Error("SetXYcurvature", "Curvature %f not valid. Must be positive.", fC);
      return;
   } 
   if (TMath::Abs(fC) < TGeoShape::Tolerance()) {
      Warning("SetXYcurvature", "Curvature is zero. Helix is a straigth line.");      
      TObject::SetBit(kHelixStraigth, kTRUE);
   }   
}  

//_____________________________________________________________________________
void TGeoHelix::SetCharge(Int_t charge)
{
// Positive charge means left-handed helix.   
   if (charge==0) {
      Error("ctor", "charge cannot be 0 - define it positive for a left-handed helix, negative otherwise");
      return;
   }   
   Int_t q = TMath::Sign(1, charge);
   if (q == fQ) return;
   fQ = q;
   TObject::SetBit(kHelixNeedUpdate, kTRUE);
}   

//_____________________________________________________________________________
void TGeoHelix::SetField(Double_t bx, Double_t by, Double_t bz, Bool_t is_normalized)
{
// Initialize particle direction (tangent on the helix in initial point)   
   fB[0] = bx;
   fB[1] = by;
   fB[2] = bz;
   TObject::SetBit(kHelixNeedUpdate, kTRUE);
   if (is_normalized) return;
   Double_t norm = 1./TMath::Sqrt(bx*bx+by*by+bz*bz);
   for (Int_t i=0; i<3; i++) fB[i] *= norm;   
}

//_____________________________________________________________________________
void TGeoHelix::SetHelixStep(Double_t step)
{
// Set Z step of the helix on a complete turn. Positive or null.
   if (step < 0) {
      Error("ctor", "Z step %f not valid. Must be positive.", step);
      return;
   }   
   TObject::SetBit(kHelixNeedUpdate, kTRUE);
   fS    = 0.5*step/TMath::Pi();
   if (fS < TGeoShape::Tolerance()) TObject::SetBit(kHelixCircle, kTRUE);
}   

//_____________________________________________________________________________
void TGeoHelix::ResetStep()
{
// Reset current point/direction to initial values
   fStep = 0.;
   memcpy(fPoint, fPointInit, 3*sizeof(Double_t));
   memcpy(fDir, fDirInit, 3*sizeof(Double_t));
}   

//_____________________________________________________________________________
void TGeoHelix::Step(Double_t step)
{
// Make a step from current point along the helix and compute new point, direction and angle
// To reach a plane/ shape boundary, one has to:
//  1. Compute the safety to the plane/boundary
//  2. Define / update a helix according local field and particle state (position, direction, charge)
//  3. Compute the magnetic safety (maximum distance for which the field can be considered constant)
//  4. Call TGeoHelix::Step() having as argument the minimum between 1. and 3.
//  5. Repeat from 1. until the step to be made is small enough.
//  6. Add to the total step the distance along a straigth line from the last point
//     to the plane/shape boundary
   Int_t i;
   fStep += step;
   if (TObject::TestBit(kHelixStraigth)) {
      for (i=0; i<3; i++) {
         fPoint[i] = fPointInit[i]+fStep*fDirInit[i];
         fDir[i] = fDirInit[i];
      }   
      return;
   }
   if (TObject::TestBit(kHelixNeedUpdate)) UpdateHelix();
   Double_t r = 1./fC;
   fPhi = fStep/TMath::Sqrt(r*r+fS*fS);
   Double_t vect[3];
   vect[0] = r * TMath::Cos(fPhi);  
   vect[1] = -fQ * r * TMath::Sin(fPhi);  
   vect[2] = fS * fPhi;
   fMatrix->LocalToMaster(vect, fPoint);

   Double_t ddb = fDirInit[0]*fB[0]+fDirInit[1]*fB[1]+fDirInit[2]*fB[2];
   Double_t f = -TMath::Sqrt(1.-ddb*ddb);
   vect[0] = f*TMath::Sin(fPhi);
   vect[1] = fQ*f*TMath::Cos(fPhi);
   vect[2] = ddb;
   TMath::Normalize(vect);
   fMatrix->LocalToMasterVect(vect, fDir);   
}

//_____________________________________________________________________________
Double_t TGeoHelix::StepToPlane(Double_t *point, Double_t *norm) 
{
// Propagate initial point up to a given Z position in MARS.
   Double_t step = 0.;
   Double_t snext = 1.E30;
   Double_t dx, dy, dz;
   Double_t ddn, pdn;
   if (TObject::TestBit(kHelixNeedUpdate)) UpdateHelix();
   dx = point[0] - fPoint[0];
   dy = point[1] - fPoint[1];
   dz = point[2] - fPoint[2];
   pdn = dx*norm[0]+dy*norm[1]+dz*norm[2];
   ddn = fDir[0]*norm[0]+fDir[1]*norm[1]+fDir[2]*norm[2];
   if (TObject::TestBit(kHelixStraigth)) {
      // propagate straigth line to plane
      if ((pdn*ddn) <= 0) return snext;
      snext = pdn/ddn;
      Step(snext);
      return snext;
   }   
   
   Double_t r = 1./fC;
   Double_t dist;
   Double_t safety = TMath::Abs(pdn);
   Double_t safestep = ComputeSafeStep();
   snext = 1.E30;
   Bool_t approaching = (ddn*pdn>0)?kTRUE:kFALSE;
   if (approaching) snext = pdn/ddn;
   else if (safety > 2.*r) return snext;
   while (snext > safestep) {
      dist = TMath::Max(safety, safestep);
      Step(dist);
      step += dist;
      dx = point[0] - fPoint[0];
      dy = point[1] - fPoint[1];
      dz = point[2] - fPoint[2];
      pdn = dx*norm[0]+dy*norm[1]+dz*norm[2];
      ddn = fDir[0]*norm[0]+fDir[1]*norm[1]+fDir[2]*norm[2];
      safety = TMath::Abs(pdn);
      approaching = (ddn*pdn>0)?kTRUE:kFALSE;
      snext = 1.E30;
      if (approaching) snext = pdn/ddn;
      else if (safety > 2.*r) {
         ResetStep();
         return snext; 
      }   
   }
   step += snext;
   Step(snext);
   return step;
}

//_____________________________________________________________________________
void TGeoHelix::UpdateHelix()
{
// Update the local helix matrix.
   TObject::SetBit(kHelixNeedUpdate, kFALSE);
   fStep = 0.;
   memcpy(fPoint, fPointInit, 3*sizeof(Double_t));
   memcpy(fDir, fDirInit, 3*sizeof(Double_t));
   Double_t rot[9];
   Double_t tr[3];
   Double_t ddb = fDirInit[0]*fB[0]+fDirInit[1]*fB[1]+fDirInit[2]*fB[2];
   if ((1.-TMath::Abs(ddb))<TGeoShape::Tolerance() || TMath::Abs(fC)<TGeoShape::Tolerance()) {
      // helix is just a straigth line
      TObject::SetBit(kHelixStraigth, kTRUE);
      fMatrix->Clear();
      return;
   }   
   rot[2] = fB[0];
   rot[5] = fB[1];
   rot[8] = fB[2];
   if (ddb < 0) fS = -TMath::Abs(fS);
   Double_t fy = - fQ*TMath::Sqrt(1.-ddb*ddb);
   fy = 1./fy;
   rot[1] = fy*(fDirInit[0]-fB[0]*ddb);
   rot[4] = fy*(fDirInit[1]-fB[1]*ddb);
   rot[7] = fy*(fDirInit[2]-fB[2]*ddb);

   rot[0] = rot[4]*rot[8] - rot[7]*rot[5];
   rot[3] = rot[7]*rot[2] - rot[1]*rot[8];
   rot[6] = rot[1]*rot[5] - rot[4]*rot[2];
   
   tr[0] = fPointInit[0] - rot[0]/fC;
   tr[1] = fPointInit[1] - rot[3]/fC;
   tr[2] = fPointInit[2] - rot[6]/fC;
   
   fMatrix->SetTranslation(tr);
   fMatrix->SetRotation(rot);
   
}    
