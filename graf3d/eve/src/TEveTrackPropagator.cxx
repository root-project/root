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
TEveTrackPropagator::Helix_t::Helix_t() :
   fCharge(0), fMinAng(45), fDelta(0.1),
   fPhi(0), fValid(kFALSE),
   fLam(-1), fR(-1), fPhiStep(-1), fSin(-1), fCos(-1),
   fPtMag(-1), fPlDir(-1), fTStep(-1)
{
   // Default constructor.
}

//______________________________________________________________________________
void TEveTrackPropagator::Helix_t::Update(const TEveVector& p, const TEveVector& b,
                                          Bool_t fullUpdate, Float_t fraction)
{
   // Update helix parameters.

   fB = b;

   // base vectors
   fE1 = b;
   fE1.Normalize();
   fPlDir = p.Dot(fE1);
   fPl    = fE1*fPlDir;

   fPt    = p - fPl;
   fPtMag = fPt.Mag();
   fE2    = fPt;
   fE2.Normalize();

   // helix parameters
   TMath::Cross(fE1.Arr(), fE2.Arr(), fE3.Arr());
   if (fCharge < 0) fE3.NegateXYZ();

   if (fullUpdate)
   {
      using namespace TMath;

      Float_t a = fgkB2C * b.Mag() * Abs(fCharge);
      if (a > 1e-10 && fPtMag*fPtMag > 1e-12)
      {
         fValid = kTRUE;

         fR   = Abs(fPtMag / a);
         fLam = fPl.Mag() / fPtMag;
         if (fPlDir < 0) fLam = - fLam;
      }
      else
      {
         fValid = kFALSE;
         return;
      }

      // phi steps
      fPhiStep = fMinAng * DegToRad();
      if (fDelta < fR)
      {
         Float_t ang  = 2*ACos(1 - fDelta/fR);
         if (ang < fPhiStep) fPhiStep = ang;
      }
      if (fraction > 0) fPhiStep *= fraction;

      fTStep = fR*fPhiStep;
      fSin   = Sin(fPhiStep);
      fCos   = Cos(fPhiStep);
   }
}

//______________________________________________________________________________
void TEveTrackPropagator::Helix_t::Step(const TEveVector4& v, const TEveVector& p,
                                        TEveVector4& vOut, TEveVector& pOut)
{
   // Step helix for given momentum p from vertex v.

   vOut = v; 

   if (fValid)
   {
      TEveVector d = fE2*(fR*fSin) + fE3*(fR*(1-fCos)) + fE1*(fLam*fTStep);
      vOut    += d;
      vOut.fT += fTStep;

      pOut = fPl + fE2*(fPtMag*fCos) + fE3*(fPtMag*fSin);

      fPhi += fPhiStep;
   }
   else
   {
      // case: pT < 1e-6 or B < 1e-7
      // might happen if field directon changes pT ~ 0 or B becomes zero

      if (fTStep == -1)
      {
         printf("WARNING TEveTrackPropagator::Helix_t::Step step-size not initialised.\n");
      }
      vOut += p * (fTStep / p.Mag());
      pOut  = p;
   }
}


//==============================================================================
// TEveTrackPropagator
//==============================================================================

//______________________________________________________________________________
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

ClassImp(TEveTrackPropagator);

Float_t             TEveTrackPropagator::fgDefMagField = 0.5;
const Float_t       TEveTrackPropagator::fgkB2C        = 0.299792458e-2;
TEveTrackPropagator TEveTrackPropagator::fgDefStyle;

//______________________________________________________________________________
TEveTrackPropagator::TEveTrackPropagator(const Text_t* n, const Text_t* t,
                                         TEveMagField *field) :
   TEveElementList(n, t),
   TEveRefBackPtr(),

   fMagFieldObj(field),
   fMaxR    (350),
   fMaxZ    (450),

   fNMax     (4096),
   fMaxOrbs  (0.5),

   fEditPathMarks (kTRUE),
   fFitDaughters  (kTRUE),
   fFitReferences (kTRUE),
   fFitDecay      (kTRUE),
   fFitCluster2Ds (kTRUE),

   fRnrDaughters  (kFALSE),
   fRnrReferences (kFALSE),
   fRnrDecay      (kFALSE),
   fRnrFV         (kFALSE),

   fPMAtt(),
   fFVAtt(),

   fV()
{
   // Default constructor.

   fPMAtt.SetMarkerColor(kYellow);
   fPMAtt.SetMarkerStyle(2);
   fPMAtt.SetMarkerSize(2);

   fFVAtt.SetMarkerColor(kRed);
   fFVAtt.SetMarkerStyle(4);
   fFVAtt.SetMarkerSize(1.5);


   if (fMagFieldObj == 0)
      fMagFieldObj = new TEveMagFieldConst(0., 0., fgDefMagField);
}

//______________________________________________________________________________
TEveTrackPropagator::~TEveTrackPropagator()
{
   // Destructor.

   delete fMagFieldObj;
}

//______________________________________________________________________________
void TEveTrackPropagator::ElementChanged(Bool_t update_scenes, Bool_t redraw)
{
   // Element-change notification.
   // Stamp all tracks as requiring display-list regeneration.
   // Virtual from TEveElement.

   TEveTrack* track;
   std::list<TEveElement*>::iterator i = fBackRefs.begin();
   while (i != fBackRefs.end())
   {
      track = dynamic_cast<TEveTrack*>(*i);
      track->StampObjProps();
      ++i;
   }
   TEveElementList::ElementChanged(update_scenes, redraw);
}

//==============================================================================

//______________________________________________________________________________
void TEveTrackPropagator::InitTrack(TEveVector &v, TEveVector& /*p*/,
                                    Float_t /*beta*/,  Int_t charge)
{
   // Initialize internal data-members for given particle parameters.

   fV.fX = v.fX;
   fV.fY = v.fY;
   fV.fZ = v.fZ;

   fPoints.push_back(fV);

   // init helix
   fH.fPhi    = 0;
   fH.fCharge = charge;
}

//______________________________________________________________________________
void TEveTrackPropagator::ResetTrack()
{
   // Reset cache holding particle trajectory.

   fPoints.clear();
}

//______________________________________________________________________________
Bool_t TEveTrackPropagator::GoToVertex(TEveVector& v, TEveVector& p)
{
   // Propagate particle with momentum p to vertex v.
  
   fH.Update(p, fMagFieldObj->GetField(fV), kTRUE);

   Bool_t hit;
   if (fH.fValid)
      hit = HelixToVertex(v, p);
   else
      hit = LineToVertex(v);
   return hit;
}

//______________________________________________________________________________
void TEveTrackPropagator::GoToBounds(TEveVector& p)
{
   // Propagate particle to bounds.

   fH.Update(p, fMagFieldObj->GetField(fV), kTRUE);

   if (fH.fValid)
      HelixToBounds(p);
   else
      LineToBounds(p);
}


//______________________________________________________________________________
void TEveTrackPropagator::StepHelix(TEveVector4 &v, TEveVector &p, TEveVector4 &vOut, TEveVector &pOut)
{
   // Wrapper to step helix.

   if (fMagFieldObj->IsConst())
   {
      fH.Update(p, fH.fB, kFALSE);
   }
   else
   {
      fH.Update(p, fMagFieldObj->GetField(v), kTRUE);
   }
   fH.Step(v, p, vOut, pOut);
}

//______________________________________________________________________________
void TEveTrackPropagator::HelixToBounds(TEveVector& p)
{
   // Propagate charged particle with momentum p to bounds.

   TEveVector4 currV(fV);
   TEveVector4 forwV(fV);
   TEveVector  forwP (p);

   Int_t np = fPoints.size();
   Float_t maxRsq = fMaxR*fMaxR;
   Float_t maxPhi = fMaxOrbs*TMath::TwoPi();
   while (fH.fPhi < maxPhi && np<fNMax)
   {
      StepHelix(currV, p, forwV, forwP);

      // cross R
      if (forwV.Perp2() > maxRsq)
      {
         Float_t t = (fMaxR - currV.R()) / (forwV.R() - currV.R());
         if (t < 0 || t > 1)
         {
            Warning("TEveTrackPropagator::HelixToBounds",
                    "In MaxR crossing expected t>=0 && t<=1: t=%f, r1=%f, r2=%f, MaxR=%f.",
                    t, currV.R(), forwV.R(), fMaxR);

            return;
         }
         TEveVector d(forwV);
         d -= currV;
         d *= t;
         d += currV;
         fPoints.push_back(d);
         return;
      }

      // cross Z
      else if (TMath::Abs(forwV.fZ) > fMaxZ)
      {
         Float_t t = (fMaxZ - TMath::Abs(currV.fZ)) / TMath::Abs((forwV.fZ - currV.fZ));
         if (t < 0 || t > 1)
         {
            Warning("TEveTrackPropagator::HelixToBounds",
                    "In MaxZ crossing expected t>=0 && t<=1: t=%f, z1=%f, z2=%f, MaxZ=%f.",
                    t, currV.fZ, forwV.fZ, fMaxZ);
            return;
         }
         TEveVector d(forwV -currV);
         d *= t;
         d +=currV;
         fPoints.push_back(d);
         return;
      }

      currV = forwV;
      p =  forwP;
      fPoints.push_back(currV);
      np++;
   }
}

//______________________________________________________________________________
Bool_t TEveTrackPropagator::HelixToVertex(TEveVector& v, TEveVector& p)
{
   // Propagate charged particle with momentum p to vertex v.

   if (fMagFieldObj->IsConst())
      fH.Update(p, fMagFieldObj->GetField(fV), kTRUE);

   const Float_t maxRsq = fMaxR*fMaxR;

   TEveVector4 currV(fV);
   TEveVector4 forwV(fV);
   TEveVector  forwP(p);

   Int_t new_points  = 0;
   Int_t first_point = fPoints.size();
   Int_t np = fPoints.size();
   Bool_t hitBounds = kFALSE;

   while ( ! PointOverVertex(v, currV) && np < fNMax)
   {
      StepHelix(currV, p, forwV, forwP);

      if (IsOutsideBounds(forwV, maxRsq, fMaxZ)) 
      {
         hitBounds = kTRUE;
         TEveVector d = v-forwV;
         break;
      }
      currV = forwV;
      p = forwP;
      fPoints.push_back(currV);
      ++ np;
      ++new_points;
   }

   /* Debug
   {
      TEveVector d  = v - currV;
      Float_t    af = d.Mag2() / fH.GetStepSize2();
      if (af > 1)
         printf("Helix propagation %d ended with %f of step distance \n", np, af);
   }
   */

   // make the remaining fractional step
   Float_t af = (v - currV).Mag() / fH.fTStep;
   fH.Update(p, fH.fB, kTRUE, af);
   StepHelix(currV, p, forwV, forwP);
   p = forwP;

   // correct for offset
   TEveVector off(v); off -= currV;
   off *= 1.0f / currV.fT;
   for(UInt_t i = first_point; i < fPoints.size(); ++i)
   {
      fPoints[i] += off * fPoints[i].fT;
   }

   fPoints.push_back(v);
   fV = fPoints.back();

   return ! hitBounds;
}

//______________________________________________________________________________
Bool_t TEveTrackPropagator::LineToVertex(TEveVector& v)
{
   // Propagate neutral particle to vertex v.

   TEveVector4 currV = v;

   currV.fX = v.fX;
   currV.fY = v.fY;
   currV.fZ = v.fZ;
   fPoints.push_back(currV);

   fV = v;
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
Bool_t TEveTrackPropagator::HelixIntersectPlane(const TEveVector& p,
                                                const TEveVector& point,
                                                const TEveVector& normal,
                                                TEveVector& itsect)
{

   TEveVector pos(fV);
   TEveVector mom(p);
   if (fMagFieldObj->IsConst())
      fH.Update(mom, fMagFieldObj->GetField(pos), kTRUE);

   TEveVector n(normal);
   TEveVector delta = pos - point;
   Float_t d = delta.Dot(n);
   if (d > 0) {
      n.NegateXYZ(); // Turn normal around so that we approach from negative side of the plane
      d = -d;
   }

   TEveVector4 forwV;
   TEveVector forwP;
   TEveVector4 pos4(pos);
   while (1)
   {
      StepHelix(pos4, mom, forwV , forwP);
      Float_t new_d = (forwV - point).Dot(n);
      if (new_d < d)
      {
         // We are going further away ... fail intersect.
         Warning("TEveTrackPropagator::HelixIntersectPlane", "going away from the plane.");
         return kFALSE;
      }
      if (new_d > 0)
      {
         delta = forwV - pos;
         itsect = pos + delta * (d / (d - new_d));
         return kTRUE;
      }
      pos4 = forwV;
      mom  = forwP;
   }
}

//______________________________________________________________________________
Bool_t TEveTrackPropagator::LineIntersectPlane(const TEveVector& p,
                                               const TEveVector& point,
                                               const TEveVector& normal,
                                                     TEveVector& itsect)
{
   TEveVector pos(fV.fX, fV.fY, fV.fZ);
   TEveVector delta = pos - point;

   Float_t d = delta.Dot(normal);
   if (d == 0) {
      itsect = pos;
      return kTRUE;
   }

   Float_t t = (p.Dot(normal)) / d;
   if (t < 0) {
      return kFALSE;
   } else {
      itsect = pos + p*t;
      return kTRUE;
   }
}

//______________________________________________________________________________
Bool_t TEveTrackPropagator::IntersectPlane(const TEveVector& p,
                                           const TEveVector& point,
                                           const TEveVector& normal,
                                                 TEveVector& itsect)
{
   // Find intersection of currently propagated track with a plane.
   // Current track position is used as starting point.
   //
   // Args:
   //  p        - track momentum to use for extrapolation
   //  point    - a point on a plane
   //  normal   - normal of the plane
   //  itsect   - output, point of intersection
   // Returns:
   //  kFALSE if intersection can not be found, kTRUE otherwise.

   if (fH.fCharge && fMagFieldObj && p.Perp2() > 1e-12)
      return HelixIntersectPlane(p, point, normal, itsect);
   else
      return LineIntersectPlane(p, point, normal, itsect);
}

//______________________________________________________________________________
void TEveTrackPropagator::FillPointSet(TEvePointSet* ps) const
{
   // Reset ps and populate it with points in propagation cache.

   Int_t size = TMath::Min(fNMax, (Int_t)fPoints.size());
   ps->Reset(size);
   for (Int_t i = 0; i < size; ++i)
   {
      const TEveVector4& v = fPoints[i];
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
      track->StampObjProps();
      ++i;
   }
}

//______________________________________________________________________________
void TEveTrackPropagator::SetMagField(Float_t bX, Float_t bY, Float_t bZ)
{
   // Set constant magnetic field and rebuild tracks.

   if (fMagFieldObj) delete fMagFieldObj;

   fMagFieldObj = new TEveMagFieldConst(bX, bY, bZ);
   RebuildTracks();
}

//______________________________________________________________________________
void TEveTrackPropagator::SetMagFieldObj(TEveMagField *mff)
{
   // Set constant magnetic field and rebuild tracks.
  if (fMagFieldObj) delete fMagFieldObj;

   fMagFieldObj = mff;
   RebuildTracks();
}

//______________________________________________________________________________
void TEveTrackPropagator::PrintMagField(Float_t x, Float_t y, Float_t z) const
{
   if (fMagFieldObj) fMagFieldObj->PrintField(x, y, z);
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

   fH.fMinAng = x;
   RebuildTracks();
}

//______________________________________________________________________________
void TEveTrackPropagator::SetDelta(Float_t x)
{
   // Set maximum error and rebuild tracks.

   fH.fDelta = x;
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
void TEveTrackPropagator::SetFitCluster2Ds(Bool_t x)
{
   // Set 2D-cluster fitting and rebuild tracks.

   fFitCluster2Ds = x;
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
void TEveTrackPropagator::SetRnrCluster2Ds(Bool_t rnr)
{
   // Set rendering of 2D-clusters and rebuild tracks.

   fRnrCluster2Ds = rnr;
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
