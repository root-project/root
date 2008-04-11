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
void TEveTrackPropagator::Helix_t::Step(Vertex4D_t& v, TEveVector& p)
{
   // Step from position 'v' with momentum 'p'.
   // Both quantities are input and output.

   v.fX += (p.fX*fSin - p.fY*(1 - fCos))/fA + fXoff;
   v.fY += (p.fY*fSin + p.fX*(1 - fCos))/fA + fYoff;
   v.fZ += fLam*TMath::Abs(fR*fPhiStep);
   v.fT += fTimeStep;

   Float_t pxt = p.fX*fCos  - p.fY*fSin;
   Float_t pyt = p.fY*fCos  + p.fX*fSin;
   p.fX = pxt;
   p.fY = pyt;
}

//______________________________________________________________________________
void TEveTrackPropagator::Helix_t::StepVertex(const Vertex4D_t& v, const TEveVector& p,
                                              Vertex4D_t& forw)
{
   // Step from position 'v' with momentum 'p'.
   // Return end position in 'forw'. Momentum is not changed.
   //
   // This is used for checking if particle goes outside of the
   // boundaries in next step.

   forw.fX = v.fX + (p.fX*fSin - p.fY*(1 - fCos))/fA + fXoff;
   forw.fY = v.fY + (p.fY*fSin + p.fX*(1 - fCos))/fA + fYoff;
   forw.fZ = v.fZ + fLam*TMath::Abs(fR*fPhiStep);
   forw.fT = v.fT + fTimeStep;
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
   fMaxR    (350),
   fMaxZ    (450),
   fMaxOrbs (0.5),
   fMinAng  (45),
   fDelta   (0.1),

   fEditPathMarks (kTRUE),
   fFitDaughters  (kTRUE),
   fFitReferences (kTRUE),
   fFitDecay      (kTRUE),
   fRnrDaughters  (kTRUE),
   fRnrReferences (kTRUE),
   fRnrDecay      (kTRUE),
   fRnrFV         (kFALSE),
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
   // Initialize internal data-members for given particle parameters.

   fV.fX = v.fX;
   fV.fY = v.fY;
   fV.fZ = v.fZ;
   fV.fT = 0;
   fPoints.push_back(fV);

   fVelocity = TMath::C()*beta;
   fCharge = charge;
   InitHelix(p);
}

//______________________________________________________________________________
void TEveTrackPropagator::ResetTrack()
{
   // Reset cache holding particle trajectory.

   fPoints.clear();
   fN     = 0;
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

   if (fCharge != 0 && TMath::Abs(fMagField) > 1e-5 && p.Perp2() > 1e-12)
      HelixToBounds(p);
   else
      LineToBounds(p);
}

//______________________________________________________________________________
void TEveTrackPropagator::InitHelix(const TEveVector& p)
{
   // Initialize helix parameters for given momentum.

   if (fCharge)
   {
      // initialise helix
      using namespace TMath;
      Float_t pT = p.Perp();
      fH.fA      = fgkB2C * fMagField * fCharge;
      fH.fLam    = p.fZ / pT;
      fH.fR      = pT   / fH.fA;

      fH.fPhiStep = fMinAng * DegToRad();
      if (fDelta < Abs(fH.fR))
      {
         Float_t ang  = 2*ACos(1 - fDelta/Abs(fH.fR));
         if (ang < fH.fPhiStep) fH.fPhiStep = ang;
      }
      if (fH.fA < 0) fH.fPhiStep *= -1;
      //printf("PHI STEP %f \n", fH.fPhiStep);

      fH.fTimeStep = 0.01 * Abs(fH.fR*fH.fPhiStep)*Sqrt(1+(fH.fLam*fH.fLam))/fVelocity;//cm->m
      fH.fSin = Sin(fH.fPhiStep);
      fH.fCos = Cos(fH.fPhiStep);
   }
}

//______________________________________________________________________________
void TEveTrackPropagator::SetNumOfSteps()
{
   // Calculate number of steps needed to get to R/Z bounds.

   using namespace TMath;

   // max orbits
   Int_t newCount = Int_t(fMaxOrbs*TwoPi()/Abs(fH.fPhiStep));
   // Z boundaries
   Float_t nz;
   if (fH.fLam > 0) {
      nz = ( fMaxZ - fV.fZ) / (fH.fLam*Abs(fH.fR*fH.fPhiStep));
   } else {
      nz = (-fMaxZ - fV.fZ) / (fH.fLam*Abs(fH.fR*fH.fPhiStep));
   }
   if (nz < newCount) newCount = Int_t(nz + 1);

   fNLast = fN + newCount;
   // printf("end steps in helix line %d \n", fNLast);
}


//______________________________________________________________________________
void TEveTrackPropagator::HelixToBounds(TEveVector& p)
{
   // Propagate charged particle with momentum p to bounds.

   InitHelix(p);
   SetNumOfSteps();

   if (fN < fNLast)
   {
      Bool_t crosR = kFALSE;
      if (fV.Perp() < fMaxR + TMath::Abs(fH.fR))
         crosR = true;

      Float_t maxR2 = fMaxR * fMaxR;
      Vertex4D_t forw;
      while (fN < fNLast)
      {
         fH.StepVertex(fV, p, forw);
         if (crosR && forw.Perp2() > maxR2)
         {
            Float_t t = (fMaxR - fV.R()) / (forw.R() - fV.R());
            assert(t >= 0 && t <= 1);
            fPoints.push_back(fV + (forw-fV)*t);
            ++fN;
            return;
         }
         if (TMath::Abs(forw.fZ) > fMaxZ)
         {
            Float_t t = (fMaxZ - TMath::Abs(fV.fZ)) / TMath::Abs((forw.fZ - fV.fZ));
            assert(t >= 0 && t <= 1);
            fPoints.push_back(fV + (forw-fV)*t);
            ++fN;
            return;
         }
         fH.Step(fV, p); fPoints.push_back(fV); ++fN;
      }
      return;
   }
}

//______________________________________________________________________________
Bool_t TEveTrackPropagator::HelixToVertex(TEveVector& v, TEveVector& p)
{
   // Propagate charged particle with momentum p to vertex v.

   InitHelix(p);
   SetNumOfSteps();

   Float_t p0x = p.fX, p0y = p.fY;
   Float_t zs  = fH.fLam*TMath::Abs(fH.fR*fH.fPhiStep);
   Float_t maxrsq  = fMaxR * fMaxR;
   Float_t fnsteps = (v.fZ - fV.fZ)/zs;
   Int_t   nsteps  = Int_t((v.fZ - fV.fZ)/zs);
   Float_t sinf = TMath::Sin(fnsteps*fH.fPhiStep); // final sin
   Float_t cosf = TMath::Cos(fnsteps*fH.fPhiStep); // final cos

   // check max orbits
   nsteps = TMath::Min(nsteps, fNLast - fN);
   {
      if (nsteps > 0)
      {
         // check offset and distribute it over all steps
         Float_t xf = fV.fX + (p.fX*sinf - p.fY*(1 - cosf)) / fH.fA;
         Float_t yf = fV.fY + (p.fY*sinf + p.fX*(1 - cosf)) / fH.fA;
         fH.fXoff   = (v.fX - xf) / fnsteps;
         fH.fYoff   = (v.fY - yf) / fnsteps;
         Vertex4D_t forw;
         for (Int_t l=0; l<nsteps; l++)
         {
            fH.StepVertex(fV, p, forw);
            if (fV.Perp2() > maxrsq || TMath::Abs(fV.fZ) > fMaxZ)
               return kFALSE;
            fH.Step(fV, p); fPoints.push_back(fV);
            ++fN;
         }
      }
      // set time to the end point
      fV.fT += TMath::Sqrt((fV.fX-v.fX)*(fV.fX-v.fX) +
                           (fV.fY-v.fY)*(fV.fY-v.fY) +
                           (fV.fZ-v.fZ)*(fV.fZ-v.fZ)) / fVelocity;
      fV.fX = v.fX; fV.fY = v.fY; fV.fZ = v.fZ;
      fPoints.push_back(fV);
      ++fN;
   }
   { // rotate momentum for residuum
      Float_t cosr = TMath::Cos((fnsteps-nsteps)*fH.fPhiStep);
      Float_t sinr = TMath::Sin((fnsteps-nsteps)*fH.fPhiStep);
      Float_t pxt = p.fX*cosr - p.fY*sinr;
      Float_t pyt = p.fY*cosr + p.fX*sinr;
      p.fX = pxt;
      p.fY = pyt;
   }
   { // calculate size of faked p.fX,py
      Float_t pxf = (p0x*cosf - p0y*sinf)/TMath::Abs(fH.fA) + fH.fXoff/fH.fPhiStep;
      Float_t pyf = (p0y*cosf + p0x*sinf)/TMath::Abs(fH.fA) + fH.fYoff/fH.fPhiStep;
      Float_t fac = TMath::Sqrt((p0x*p0x + p0y*p0y) / (pxf*pxf + pyf*pyf));
      p.fX = fac*pxf;
      p.fY = fac*pyf;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TEveTrackPropagator::LineToVertex(TEveVector& v)
{
   // Propagate neutral particle to vertex v.

   fV.fT += TMath::Sqrt((fV.fX-v.fX)*(fV.fX-v.fX) +
                        (fV.fY-v.fY)*(fV.fY-v.fY) +
                        (fV.fZ-v.fZ)*(fV.fZ-v.fZ)) / fVelocity;
   fV.fX = v.fX;
   fV.fY = v.fY;
   fV.fZ = v.fZ;
   fPoints.push_back(fV);

   return kTRUE;
}

//______________________________________________________________________________
void TEveTrackPropagator::LineToBounds(TEveVector& p)
{
   // Propagatate neutral particle with momentum p to bounds.

   Float_t tZ = 0, tR = 0, tB = 0;
   // time where particle intersect +/- fMaxZ
   if (p.fZ > 0) {
      tZ = (fMaxZ - fV.fZ)/p.fZ;
   }
   else  if (p.fZ < 0 ) {
      tZ = (-1)*(fMaxZ + fV.fZ)/p.fZ;
   }
   // time where particle intersects cylinder
   Double_t a = p.fX*p.fX + p.fY*p.fY;
   Double_t b = 2*(fV.fX*p.fX + fV.fY*p.fY);
   Double_t c = fV.fX*fV.fX + fV.fY*fV.fY - fMaxR*fMaxR;
   Double_t d = b*b - 4*a*c;
   if (d >= 0) {
      Double_t sqrtD=TMath::Sqrt(d);
      tR = ( -b - sqrtD )/(2*a);
      if (tR < 0) {
         tR = ( -b + sqrtD )/(2*a);
      }
      tB = tR < tZ ? tR : tZ; // compare the two times
   } else {
      tB = tZ;
   }
   TEveVector nv(fV.fX + p.fX*tB, fV.fY + p.fY*tB, fV.fZ+ p.fZ*tB);
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
      const Vertex4D_t& v = fPoints[i];
      ps->SetNextPoint(v.fX, v.fY, v.fZ);
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

