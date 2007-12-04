// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveTrackPropagator.h"
#include "TEveTrack.h"

#include "TMath.h"

#include <cassert>

//______________________________________________________________________________
void TEveTrackPropagator::Helix::Step(Vertex4D& v, TEveVector& p)
{
   v.x += (p.x*fSin - p.y*(1 - fCos))/fA + fXoff;
   v.y += (p.y*fSin + p.x*(1 - fCos))/fA + fYoff;
   v.z += fLam*TMath::Abs(fR*fPhiStep);
   v.t += fTimeStep;

   Float_t pxt = p.x*fCos  - p.y*fSin;
   Float_t pyt = p.y*fCos  + p.x*fSin;
   p.x = pxt;
   p.y = pyt;
}

//______________________________________________________________________________
void TEveTrackPropagator::Helix::StepVertex(Vertex4D& v, TEveVector& p, Vertex4D& forw)
{
   forw.x = v.x + (p.x*fSin - p.y*(1 - fCos))/fA + fXoff;
   forw.y = v.y + (p.y*fSin + p.x*(1 - fCos))/fA + fYoff;
   forw.z = v.z + fLam*TMath::Abs(fR*fPhiStep);
   forw.t = v.t + fTimeStep;
}


//______________________________________________________________________________
// TEveTrackPropagator
//
// Holding structure for a number of track rendering parameters.
// Calculates path taking into account the parameters.
//
// This is decoupled from TEveTrack/TEveTrackList to allow sharing of the
// Propagator among several instances. Back references are kept so the
// tracks can be recreated when the parameters change.
//
// TEveTrackList has Get/Set methods for RnrStlye. TEveTrackEditor and
// TEveTrackListEditor provide editor access.

ClassImp(TEveTrackPropagator)

Float_t             TEveTrackPropagator::fgDefMagField = 0.5;
const Float_t       TEveTrackPropagator::fgkB2C        = 0.299792458e-2;
TEveTrackPropagator TEveTrackPropagator::fgDefStyle;

//______________________________________________________________________________
TEveTrackPropagator::TEveTrackPropagator() :
   TObject(),
   TEveRefBackPtr(),

   fMagField(fgDefMagField),
   fMaxR  (350),
   fMaxZ  (450),
   fMaxOrbs (0.5),
   fMinAng  (45),
   fDelta   (0.1),

   fEditPathMarks(kFALSE),
   fFitDaughters  (kTRUE),
   fFitReferences (kTRUE),
   fFitDecay      (kTRUE),
   fRnrDaughters  (kTRUE),
   fRnrReferences (kTRUE),
   fRnrDecay      (kTRUE),
   fRnrFV(kFALSE),
   fPMAtt(),
   fFVAtt(),

   fCharge   (0),
   fVelocity (0.0f),
   fV        (),
   fN        (0),
   fNLast    (0),
   fNMax     (4096)
{
   // Default constructor.
}

//______________________________________________________________________________
void TEveTrackPropagator::InitTrack(TEveVector &v, TEveVector &p,
                                    Float_t beta,  Int_t charge)
{ 
   // Initializae internal data-members for given particle parameters.

   fV.x = v.x;
   fV.y = v.y;
   fV.z = v.z;
   fV.t = 0;
   fPoints.push_back(fV);

   fVelocity = TMath::C()*beta;
   fCharge = charge;
   if (fCharge)
   {
      // initialise helix
      using namespace TMath;
      Float_t pT = p.Perp();
      fH.fA      = fgkB2C * fMagField * charge;
      fH.fLam    = p.z / pT;
      fH.fR      = pT  / fH.fA;

      fH.fPhiStep = fMinAng * DegToRad();
      if (fDelta < Abs(fH.fR))
      {
         Float_t ang  = 2*ACos(1 - fDelta/Abs(fH.fR));
         if (ang < fH.fPhiStep) fH.fPhiStep = ang;
      }
      if (fH.fA < 0) fH.fPhiStep *= -1;
      //printf("PHI STEP %f \n", fH.fPhiStep);

      fH.fTimeStep = 0.01* Abs(fH.fR*fH.fPhiStep)*Sqrt(1+(fH.fLam*fH.fLam))/fVelocity;//cm->m
      fH.fSin = Sin(fH.fPhiStep);
      fH.fCos = Cos(fH.fPhiStep);
   }
}

//______________________________________________________________________________
void TEveTrackPropagator::ResetTrack()
{
   // Reset cache holding particle trajectory.

   fPoints.clear();
   fN = 0; 
   fNLast = 0;
}

//______________________________________________________________________________
Bool_t TEveTrackPropagator::GoToVertex(TEveVector& v, TEveVector& p)
{
   // Propagate particle with momentum p to vertex v.

   Bool_t hit;
   if (fCharge != 0 && TMath::Abs(fMagField) > 1e-5 && p.Perp2() > 1e-12)
      hit = HelixToVertex(v, p);
   else
      hit = LineToVertex(v);
   return hit;
}

//______________________________________________________________________________
void TEveTrackPropagator::GoToBounds(TEveVector& p)
{
   // Propagate particle to bounds.

   if(fCharge != 0 && TMath::Abs(fMagField) > 1e-5 && p.Perp2() > 1e-12)
      HelixToBounds(p);
   else
      LineToBounds(p);
}

//______________________________________________________________________________
void TEveTrackPropagator::SetNumOfSteps()
{
   // Calculate number of steps needed to achieve desired precision.

   using namespace TMath;
   // max orbits
   fNLast = Int_t(fMaxOrbs*TwoPi()/Abs(fH.fPhiStep));
   // Z boundaries
   Float_t nz;
   if (fH.fLam > 0) {
      nz = ( fMaxZ - fV.z) / (fH.fLam*Abs(fH.fR*fH.fPhiStep));
   } else {
      nz = (-fMaxZ - fV.z) / (fH.fLam*Abs(fH.fR*fH.fPhiStep));
   }
   if (nz < fNLast) fNLast = Int_t(nz + 1);
   // printf("end steps in helix line %d \n", fNLast);
}


//______________________________________________________________________________
void TEveTrackPropagator::HelixToBounds(TEveVector& p)
{
   // Propagate charged particle with momentum p to bounds.

   // printf("HelixToBounds\n");
   SetNumOfSteps();
   if (fNLast > 0)
   {
      Bool_t crosR = kFALSE;
      if (fV.Perp() < fMaxR + TMath::Abs(fH.fR))
         crosR = true;

      Float_t maxR2 = fMaxR * fMaxR;
      Vertex4D forw;
      while (fN < fNLast)
      {
         fH.StepVertex(fV, p, forw);
         if (crosR && forw.Perp2() > maxR2)
         {
            Float_t t = (fMaxR - fV.R()) / (forw.R() - fV.R());
            assert(t >= 0 && t <= 1);
            fPoints.push_back(fV + (forw-fV)*t);fN++;
            return;
         }
         if (TMath::Abs(forw.z) > fMaxZ)
         {
            Float_t t = (fMaxZ - TMath::Abs(fV.z)) / TMath::Abs((forw.z - fV.z));
            assert(t >= 0 && t <= 1);
            fPoints.push_back(fV + (forw-fV)*t);fN++;
            return;
         }
         fH.Step(fV, p); fPoints.push_back(fV); fN++;
      }
      return;
   }
}

//______________________________________________________________________________
Bool_t TEveTrackPropagator::HelixToVertex(TEveVector& v, TEveVector& p)
{
   // Propagate charged particle with momentum p to vertex v.

   Float_t p0x = p.x, p0y = p.y;
   Float_t zs = fH.fLam*TMath::Abs(fH.fR*fH.fPhiStep);
   Float_t maxrsq  = fMaxR * fMaxR;
   Float_t fnsteps = (v.z - fV.z)/zs;
   Int_t   nsteps  = Int_t((v.z - fV.z)/zs);
   Float_t sinf = TMath::Sin(fnsteps*fH.fPhiStep); // final sin
   Float_t cosf = TMath::Cos(fnsteps*fH.fPhiStep); // final cos

   // check max orbits
   nsteps = TMath::Min(nsteps, fNLast -fN);
   {
      if (nsteps > 0)
      {
         // check offset and distribute it over all steps
         Float_t xf  = fV.x + (p.x*sinf - p.y*(1 - cosf))/fH.fA;
         Float_t yf =  fV.y + (p.y*sinf + p.x*(1 - cosf))/fH.fA;
         fH.fXoff =  (v.x - xf)/fnsteps;
         fH.fYoff =  (v.y - yf)/fnsteps;
         Vertex4D forw;
         for (Int_t l=0; l<nsteps; l++)
         {
            fH.StepVertex(fV, p, forw);
            if (fV.Perp2() > maxrsq || TMath::Abs(fV.z) > fMaxZ)
               return kFALSE;
            fH.Step(fV, p); fPoints.push_back(fV); fN++;
         }
      }
      // set time to the end point
      fV.t += TMath::Sqrt((fV.x-v.x)*(fV.x-v.x)+(fV.y-v.y)*(fV.y-v.y) +(fV.z-v.z)*(fV.z-v.z))/fVelocity;
      fV.x = v.x; fV.y = v.y; fV.z = v.z;
      fPoints.push_back(fV); fN++;
   }
   { // rotate momentum for residuum
      Float_t cosr = TMath::Cos((fnsteps-nsteps)*fH.fPhiStep);
      Float_t sinr = TMath::Sin((fnsteps-nsteps)*fH.fPhiStep);
      Float_t pxt = p.x*cosr - p.y*sinr;
      Float_t pyt = p.y*cosr + p.x*sinr;
      p.x = pxt;
      p.y = pyt;
   }
   { // calculate size of faked p.x,py
      Float_t pxf = (p0x*cosf - p0y*sinf)/TMath::Abs(fH.fA) + fH.fXoff/fH.fPhiStep;
      Float_t pyf = (p0y*cosf + p0x*sinf)/TMath::Abs(fH.fA) + fH.fYoff/fH.fPhiStep;
      Float_t fac = TMath::Sqrt((p0x*p0x + p0y*p0y) / (pxf*pxf + pyf*pyf));
      p.x = fac*pxf;
      p.y = fac*pyf;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TEveTrackPropagator::LineToVertex(TEveVector& v)
{
   // Propagate neutral particle to vertex v.

   fV.t += TMath::Sqrt((fV.x-v.x)*(fV.x-v.x)+(fV.y-v.y)*(fV.y-v.y)+(fV.z-v.z)*(fV.z-v.z))/fVelocity;
   fV.x = v.x;
   fV.y = v.y;
   fV.z = v.z;
   fPoints.push_back(fV);

   return kTRUE;
}

//______________________________________________________________________________
void TEveTrackPropagator::LineToBounds(TEveVector& p)
{
   // Propagatate neutral particle with momentum p to bounds.

   Float_t tZ = 0, Tb = 0;
   // time where particle intersect +/- fMaxZ
   if (p.z > 0) {
      tZ = (fMaxZ - fV.z)/p.z;
   }
   else  if (p.z < 0 ) {
      tZ = (-1)*(fMaxZ + fV.z)/p.z;
   }
   // time where particle intersects cylinder
   Float_t tR = 0;
   Double_t a = p.x*p.x + p.y*p.y;
   Double_t b = 2*(fV.x*p.x + fV.y*p.y);
   Double_t c = fV.x*fV.x + fV.y*fV.y - fMaxR*fMaxR;
   Double_t D = b*b - 4*a*c;
   if (D >= 0) {
      Double_t D_sqrt=TMath::Sqrt(D);
      tR = ( -b - D_sqrt )/(2*a);
      if (tR < 0) {
         tR = ( -b + D_sqrt )/(2*a);
      }
      Tb = tR < tZ ? tR : tZ; // compare the two times
   } else {
      Tb = tZ;
   }
   TEveVector nv(fV.x + p.x*Tb, fV.y + p.y*Tb, fV.z+ p.z*Tb);
   LineToVertex(nv);
}

//______________________________________________________________________________
void TEveTrackPropagator::FillPointSet(TEvePointSet* ps) const
{
   // Reset ps and populate it with points in propagation cache.

   Int_t size = TMath::Min(fNMax, (Int_t)fPoints.size());
   ps->Reset(size);
   for (Int_t i = 0; i < size; ++i)
   {
      const Vertex4D& v = fPoints[i];
      ps->SetNextPoint(v.x, v.y, v.z);
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackPropagator::RebuildTracks()
{
   // Rebuild all tracks using this render-style.

   TEveTrack* track;
   std::list<TEveElement*>::iterator i = fBackRefs.begin();
   while (i != fBackRefs.end())
   {
      track = dynamic_cast<TEveTrack*>(*i);
      track->MakeTrack();
      track->ElementChanged();
      ++i;
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackPropagator::SetMagField(Float_t x)
{
   // Set constant magnetic field and rebuild tracks.

   fMagField = x;
   RebuildTracks();
}

//______________________________________________________________________________
void TEveTrackPropagator::SetMaxR(Float_t x)
{
   // Set maximum radius and rebuild tracks.

   fMaxR = x;
   RebuildTracks();
}

//______________________________________________________________________________
void TEveTrackPropagator::SetMaxZ(Float_t x)
{
   // Set maximum z and rebuild tracks.

   fMaxZ = x;
   RebuildTracks();
}

//______________________________________________________________________________
void TEveTrackPropagator::SetMaxOrbs(Float_t x)
{
   // Set maximum number of orbits and rebuild tracks.

   fMaxOrbs = x;
   RebuildTracks();
}

//______________________________________________________________________________
void TEveTrackPropagator::SetMinAng(Float_t x)
{
   // Set minimum step angle and rebuild tracks.

   fMinAng = x;
   RebuildTracks();
}

//______________________________________________________________________________
void TEveTrackPropagator::SetDelta(Float_t x)
{
   // Set maximum error and rebuild tracks.

   fDelta = x;
   RebuildTracks();
}

//______________________________________________________________________________
void TEveTrackPropagator::SetFitDaughters(Bool_t x)
{
   // Set daughter creation point fitting and rebuild tracks.

   fFitDaughters = x;
   RebuildTracks();
}

//______________________________________________________________________________
void TEveTrackPropagator::SetFitReferences(Bool_t x)
{
   // Set track-reference fitting and rebuild tracks.

   fFitReferences = x;
   RebuildTracks();
}

//______________________________________________________________________________
void TEveTrackPropagator::SetFitDecay(Bool_t x)
{
   // Set decay fitting and rebuild tracks.

   fFitDecay = x;
   RebuildTracks();
}

//______________________________________________________________________________
void TEveTrackPropagator::SetRnrDecay(Bool_t rnr)
{
   // Set decay rendering and rebuild tracks.

   fRnrDecay = rnr;
   RebuildTracks();
}

//______________________________________________________________________________
void TEveTrackPropagator::SetRnrDaughters(Bool_t rnr)
{
   // Set daughter rendering and rebuild tracks.

   fRnrDaughters = rnr;
   RebuildTracks();
}

//______________________________________________________________________________
void TEveTrackPropagator::SetRnrReferences(Bool_t rnr)
{
   // Set track-reference rendering and rebuild tracks.

   fRnrReferences = rnr;
   RebuildTracks();
}

