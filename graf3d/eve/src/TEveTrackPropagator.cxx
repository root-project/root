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
#include "TEveTrans.h"

#include "TMath.h"

#include <cassert>

/** \class TEveMagField
\ingroup TEve
Abstract base-class for interfacing to magnetic field needed by the
TEveTrackPropagator.

To implement your own version, redefine the following virtual functions:
   virtual Double_t    GetMaxFieldMagD() const;
   virtual TEveVectorD GetFieldD(Double_t x, Double_t y, Double_t z) const;

See sub-classes TEveMagFieldConst and TEveMagFieldDuo for two simple implementations.

NOTE: For backward compatibility float versions are the primary
sources of field information in this base-class. The float versions are
not used in TEve and can be ignored in sub-classes.

NOTE: Magnetic field direction convention is inverted.
*/

ClassImp(TEveMagField);

/** \class TEveMagFieldConst
\ingroup TEve
Implements constant magnetic field, given by a vector fB.

NOTE: Magnetic field direction convention is inverted.
*/

ClassImp(TEveMagFieldConst);

/** \class TEveMagFieldDuo
\ingroup TEve
Implements constant magnetic filed that switches on given axial radius fR2
from vector fBIn to fBOut.

NOTE: Magnetic field direction convention is inverted.
*/

ClassImp(TEveMagFieldDuo);

namespace
{
   //const Double_t kBMin     = 1e-6;
   const Double_t kPtMinSqr = 1e-20;
   const Double_t kAMin     = 1e-10;
   const Double_t kStepEps  = 1e-3;
}

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

TEveTrackPropagator::Helix_t::Helix_t() :
   fCharge(0),
   fMaxAng(45), fMaxStep(20.f), fDelta(0.1),
   fPhi(0), fValid(kFALSE),
   fLam(-1), fR(-1), fPhiStep(-1), fSin(-1), fCos(-1),
   fRKStep(20.0),
   fPtMag(-1), fPlMag(-1), fLStep(-1)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Common update code for helix and RK propagation.

void TEveTrackPropagator::Helix_t::UpdateCommon(const TEveVectorD& p, const TEveVectorD& b)
{
   fB = b;

   // base vectors
   fE1 = b;
   fE1.Normalize();
   fPlMag = p.Dot(fE1);
   fPl    = fE1*fPlMag;

   fPt    = p - fPl;
   fPtMag = fPt.Mag();
   fE2    = fPt;
   fE2.Normalize();
}

////////////////////////////////////////////////////////////////////////////////
/// Update helix parameters.

void TEveTrackPropagator::Helix_t::UpdateHelix(const TEveVectorD& p, const TEveVectorD& b,
                                               Bool_t full_update, Bool_t enforce_max_step)
{
   UpdateCommon(p, b);

   // helix parameters
   TMath::Cross(fE1.Arr(), fE2.Arr(), fE3.Arr());
   if (fCharge < 0) fE3.NegateXYZ();

   if (full_update)
   {
      using namespace TMath;
      Double_t a = fgkB2C * b.Mag() * Abs(fCharge);
      if (a > kAMin && fPtMag*fPtMag > kPtMinSqr)
      {
         fValid = kTRUE;

         fR   = Abs(fPtMag / a);
         fLam = fPlMag / fPtMag;

         // get phi step, compare fMaxAng with fDelta
         fPhiStep = fMaxAng * DegToRad();
         if (fR > fDelta)
         {
            Double_t ang  = 2.0 * ACos(1.0f - fDelta/fR);
            if (ang < fPhiStep)
               fPhiStep = ang;
         }

         // check max step size
         Double_t curr_step = fR * fPhiStep * Sqrt(1.0f + fLam*fLam);
         if (curr_step > fMaxStep || enforce_max_step)
            fPhiStep *= fMaxStep / curr_step;

         fLStep = fR * fPhiStep * fLam;
         fSin   = Sin(fPhiStep);
         fCos   = Cos(fPhiStep);
      }
      else
      {
         fValid = kFALSE;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Update helix for stepper RungeKutta.

void TEveTrackPropagator::Helix_t::UpdateRK(const TEveVectorD& p, const TEveVectorD& b)
{
   UpdateCommon(p, b);

   if (fCharge)
   {
      fValid = kTRUE;

      // cached values for propagator
      fB = b;
      fPlMag = p.Dot(fB);
   }
   else
   {
      fValid = kFALSE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Step helix for given momentum p from vertex v.

void TEveTrackPropagator::Helix_t::Step(const TEveVector4D& v, const TEveVectorD& p,
                                        TEveVector4D& vOut, TEveVectorD& pOut)
{
   vOut = v;

   if (fValid)
   {
      TEveVectorD d = fE2*(fR*fSin) + fE3*(fR*(1-fCos)) + fE1*fLStep;
      vOut    += d;
      vOut.fT += TMath::Abs(fLStep);

      pOut = fPl + fE2*(fPtMag*fCos) + fE3*(fPtMag*fSin);

      fPhi += fPhiStep;
   }
   else
   {
      // case: pT < kPtMinSqr or B < kBMin
      // might happen if field directon changes pT ~ 0 or B becomes zero
      vOut    += p * (fMaxStep / p.Mag());
      vOut.fT += fMaxStep;
      pOut  = p;
   }
}

/** \class TEveTrackPropagator
\ingroup TEve
Holding structure for a number of track rendering parameters.
Calculates path taking into account the parameters.

NOTE: Magnetic field direction convention is inverted.

This is decoupled from TEveTrack/TEveTrackList to allow sharing of the
Propagator among several instances. Back references are kept so the tracks
can be recreated when the parameters change.

TEveTrackList has Get/Set methods for RnrStlye. TEveTrackEditor and
TEveTrackListEditor provide editor access.

Enum EProjTrackBreaking_e and member fProjTrackBreaking specify whether 2D
projected tracks get broken into several segments when the projected space
consists of separate domains (like Rho-Z). The track-breaking is enabled by
default.
*/

ClassImp(TEveTrackPropagator);

Double_t             TEveTrackPropagator::fgDefMagField = 0.5;
const Double_t       TEveTrackPropagator::fgkB2C        = 0.299792458e-2;
TEveTrackPropagator  TEveTrackPropagator::fgDefault;

Double_t             TEveTrackPropagator::fgEditorMaxR  = 2000;
Double_t             TEveTrackPropagator::fgEditorMaxZ  = 4000;

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

TEveTrackPropagator::TEveTrackPropagator(const char* n, const char* t,
                                         TEveMagField *field, Bool_t own_field) :
   TEveElementList(n, t),
   TEveRefBackPtr(),

   fStepper(kHelix),
   fMagFieldObj(field),
   fOwnMagFiledObj(own_field),

   fMaxR    (350),   fMaxZ    (450),
   fNMax    (4096),  fMaxOrbs (0.5),

   fEditPathMarks (kTRUE),
   fFitDaughters  (kTRUE),   fFitReferences (kTRUE),
   fFitDecay      (kTRUE),
   fFitCluster2Ds (kTRUE),   fFitLineSegments (kTRUE),
   fRnrDaughters  (kFALSE),  fRnrReferences (kFALSE),
   fRnrDecay      (kFALSE),  fRnrCluster2Ds (kFALSE),
   fRnrFV         (kFALSE),
   fPMAtt(), fFVAtt(),

   fProjTrackBreaking(kPTB_Break), fRnrPTBMarkers(kFALSE), fPTBAtt(),

   fV()
{
   fPMAtt.SetMarkerColor(kYellow);
   fPMAtt.SetMarkerStyle(2);
   fPMAtt.SetMarkerSize(2);

   fFVAtt.SetMarkerColor(kRed);
   fFVAtt.SetMarkerStyle(4);
   fFVAtt.SetMarkerSize(1.5);

   fPTBAtt.SetMarkerColor(kBlue);
   fPTBAtt.SetMarkerStyle(4);
   fPTBAtt.SetMarkerSize(0.8);

   if (fMagFieldObj == 0) {
      fMagFieldObj = new TEveMagFieldConst(0., 0., fgDefMagField);
      fOwnMagFiledObj = kTRUE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TEveTrackPropagator::~TEveTrackPropagator()
{
   if (fOwnMagFiledObj)
   {
      delete fMagFieldObj;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from TEveRefBackPtr - track reference count has reached zero.

void TEveTrackPropagator::OnZeroRefCount()
{
   CheckReferenceCount("TEveTrackPropagator::OnZeroRefCount ");
}

////////////////////////////////////////////////////////////////////////////////
/// Check reference count - virtual from TEveElement.
/// Must also take into account references from TEveRefBackPtr.

void TEveTrackPropagator::CheckReferenceCount(const TEveException& eh)
{
   if (fRefCount <= 0)
   {
      TEveElementList::CheckReferenceCount(eh);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Element-change notification.
/// Stamp all tracks as requiring display-list regeneration.
/// Virtual from TEveElement.

void TEveTrackPropagator::ElementChanged(Bool_t update_scenes, Bool_t redraw)
{
   TEveTrack* track;
   RefMap_i i = fBackRefs.begin();
   while (i != fBackRefs.end())
   {
      track = dynamic_cast<TEveTrack*>(i->first);
      track->StampObjProps();
      ++i;
   }
   TEveElementList::ElementChanged(update_scenes, redraw);
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize internal data-members for given particle parameters.

void TEveTrackPropagator::InitTrack(const TEveVectorD &v, Int_t charge)
{
   fV = v;
   fPoints.push_back(fV);

   // init helix
   fH.fPhi    = 0;
   fH.fCharge = charge;
}

////////////////////////////////////////////////////////////////////////////////
/// TEveVectorF wrapper.

void TEveTrackPropagator::InitTrack(const TEveVectorF& v, Int_t charge)
{
   TEveVectorD vd(v);
   InitTrack(vd, charge);
}

////////////////////////////////////////////////////////////////////////////////
/// Reset cache holding particle trajectory.

void TEveTrackPropagator::ResetTrack()
{
   fLastPoints.clear();
   fPoints.swap(fLastPoints);

   // reset helix
   fH.fPhi = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get index of current point on track.

Int_t TEveTrackPropagator::GetCurrentPoint() const
{
   return fPoints.size() - 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate track length from start_point to end_point.
/// If end_point is less than 0, distance to the end is returned.

Double_t TEveTrackPropagator::GetTrackLength(Int_t start_point, Int_t end_point) const
{
   if (end_point < 0) end_point = fPoints.size() - 1;

   Double_t sum = 0;
   for (Int_t i = start_point; i < end_point; ++i)
   {
      sum += (fPoints[i+1] - fPoints[i]).Mag();
   }
   return sum;
}

////////////////////////////////////////////////////////////////////////////////
/// Propagate particle with momentum p to vertex v.

Bool_t TEveTrackPropagator::GoToVertex(TEveVectorD& v, TEveVectorD& p)
{
   Update(fV, p, kTRUE);

   if ((v-fV).Mag() < kStepEps)
   {
      fPoints.push_back(v);
      return kTRUE;
   }

   return fH.fValid ? LoopToVertex(v, p) : LineToVertex(v);
}

////////////////////////////////////////////////////////////////////////////////
/// Propagate particle with momentum p to line with start point s and vector r
/// to the second point.

Bool_t TEveTrackPropagator::GoToLineSegment(const TEveVectorD& s, const TEveVectorD& r, TEveVectorD& p)
{
   Update(fV, p, kTRUE);

   if (!fH.fValid)
   {
      TEveVectorD v;
      ClosestPointBetweenLines(s, r, fV, p, v);
      LineToVertex(v);
      return kTRUE;
   }
   else
   {
      return LoopToLineSegment(s, r, p);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// TEveVectorF wrapper.

Bool_t TEveTrackPropagator::GoToVertex(TEveVectorF& v, TEveVectorF& p)
{
   TEveVectorD vd(v), pd(p);
   Bool_t result = GoToVertex(vd, pd);
   v = vd; p = pd;
   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// TEveVectorF wrapper.

Bool_t TEveTrackPropagator::GoToLineSegment(const TEveVectorF& s, const TEveVectorF& r, TEveVectorF& p)
{
   TEveVectorD sd(s), rd(r), pd(p);
   Bool_t result = GoToLineSegment(sd, rd, pd);
   p = pd;
   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Propagate particle to bounds.
/// Return TRUE if hit bounds.

void TEveTrackPropagator::GoToBounds(TEveVectorD& p)
{
   Update(fV, p, kTRUE);

   fH.fValid ? LoopToBounds(p): LineToBounds(p);
}

////////////////////////////////////////////////////////////////////////////////
/// TEveVectorF wrapper.

void TEveTrackPropagator::GoToBounds(TEveVectorF& p)
{
   TEveVectorD pd(p);
   GoToBounds(pd);
   p = pd;
}

////////////////////////////////////////////////////////////////////////////////
/// Update helix / B-field projection state.

void TEveTrackPropagator::Update(const TEveVector4D& v, const TEveVectorD& p,
                                 Bool_t full_update, Bool_t enforce_max_step)
{
   if (fStepper == kHelix)
   {
      fH.UpdateHelix(p, fMagFieldObj->GetFieldD(v), !fMagFieldObj->IsConst() || full_update, enforce_max_step);
   }
   else
   {
      fH.UpdateRK(p, fMagFieldObj->GetFieldD(v));

      if (full_update)
      {
         using namespace TMath;

         Float_t a = fgkB2C * fMagFieldObj->GetMaxFieldMagD() * Abs(fH.fCharge);
         if (a > kAMin)
         {
            fH.fR = p.Mag() / a;

            // get phi step, compare fDelta with MaxAng
            fH.fPhiStep = fH.fMaxAng * DegToRad();
            if (fH.fR > fH.fDelta )
            {
               Double_t ang  = 2.0 * ACos(1.0f - fH.fDelta/fH.fR);
               if (ang < fH.fPhiStep)
                  fH.fPhiStep = ang;
            }

            // check against maximum step-size
            fH.fRKStep = fH.fR * fH.fPhiStep * Sqrt(1 + fH.fLam*fH.fLam);
            if (fH.fRKStep > fH.fMaxStep || enforce_max_step)
            {
               fH.fPhiStep *= fH.fMaxStep / fH.fRKStep;
               fH.fRKStep   = fH.fMaxStep;
            }
         }
         else
         {
            fH.fRKStep = fH.fMaxStep;
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Wrapper to step helix.

void TEveTrackPropagator::Step(const TEveVector4D &v, const TEveVectorD &p, TEveVector4D &vOut, TEveVectorD &pOut)
{
   if (fStepper == kHelix)
   {
      fH.Step(v, p, vOut, pOut);
   }
   else
   {
      Double_t vecRKIn[7];
      vecRKIn[0] = v.fX;
      vecRKIn[1] = v.fY;
      vecRKIn[2] = v.fZ;
      Double_t pm = p.Mag();
      Double_t nm = 1.0 / pm;
      vecRKIn[3] = p.fX*nm;
      vecRKIn[4] = p.fY*nm;
      vecRKIn[5] = p.fZ*nm;
      vecRKIn[6] = p.Mag();

      Double_t vecRKOut[7];
      StepRungeKutta(fH.fRKStep, vecRKIn, vecRKOut);

      vOut.fX = vecRKOut[0];
      vOut.fY = vecRKOut[1];
      vOut.fZ = vecRKOut[2];
      vOut.fT = v.fT + fH.fRKStep;
      pm = vecRKOut[6];
      pOut.fX = vecRKOut[3]*pm;
      pOut.fY = vecRKOut[4]*pm;
      pOut.fZ = vecRKOut[5]*pm;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Propagate charged particle with momentum p to bounds.
/// It is expected that Update() with full-update was called before.

void TEveTrackPropagator::LoopToBounds(TEveVectorD& p)
{
   const Double_t maxRsq = fMaxR*fMaxR;

   TEveVector4D currV(fV);
   TEveVector4D forwV(fV);
   TEveVectorD  forwP (p);

   Int_t np = fPoints.size();
   Double_t maxPhi = fMaxOrbs*TMath::TwoPi();

   while (fH.fPhi < maxPhi && np<fNMax)
   {
      Step(currV, p, forwV, forwP);

      // cross R
      if (forwV.Perp2() > maxRsq)
      {
         Float_t t = (fMaxR - currV.R()) / (forwV.R() - currV.R());
         if (t < 0 || t > 1)
         {
            Warning("HelixToBounds", "In MaxR crossing expected t>=0 && t<=1: t=%f, r1=%f, r2=%f, MaxR=%f.",
                    t, currV.R(), forwV.R(), fMaxR);
            return;
         }
         TEveVectorD d(forwV);
         d -= currV;
         d *= t;
         d += currV;
         fPoints.push_back(d);
         return;
      }

      // cross Z
      else if (TMath::Abs(forwV.fZ) > fMaxZ)
      {
         Double_t t = (fMaxZ - TMath::Abs(currV.fZ)) / TMath::Abs((forwV.fZ - currV.fZ));
         if (t < 0 || t > 1)
         {
            Warning("HelixToBounds", "In MaxZ crossing expected t>=0 && t<=1: t=%f, z1=%f, z2=%f, MaxZ=%f.",
                    t, currV.fZ, forwV.fZ, fMaxZ);
            return;
         }
         TEveVectorD d(forwV -currV);
         d *= t;
         d += currV;
         fPoints.push_back(d);
         return;
      }

      currV = forwV;
      p     = forwP;
      Update(currV, p);

      fPoints.push_back(currV);
      ++np;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Propagate charged particle with momentum p to vertex v.
/// It is expected that Update() with full-update was called before.

Bool_t TEveTrackPropagator::LoopToVertex(TEveVectorD& v, TEveVectorD& p)
{
   const Double_t maxRsq = fMaxR * fMaxR;

   TEveVector4D currV(fV);
   TEveVector4D forwV(fV);
   TEveVectorD  forwP(p);

   Int_t first_point = fPoints.size();
   Int_t np          = first_point;

   Double_t prod0=0, prod1;

   do
   {
      Step(currV, p, forwV, forwP);
      Update(forwV, forwP);

      if (PointOverVertex(v, forwV, &prod1))
      {
         break;
      }

      if (IsOutsideBounds(forwV, maxRsq, fMaxZ))
      {
         fV = currV;
         return kFALSE;
      }

      fPoints.push_back(forwV);
      currV = forwV;
      p     = forwP;
      prod0 = prod1;
      ++np;
   } while (np < fNMax);

   // make the remaining fractional step
   if (np > first_point)
   {
      if ((v - currV).Mag() > kStepEps)
      {
         Double_t step_frac = prod0 / (prod0 - prod1);
         if (step_frac > 0)
         {
            // Step for fraction of previous step size.
            // We pass 'enforce_max_step' flag to Update().
            Float_t orig_max_step = fH.fMaxStep;
            fH.fMaxStep = step_frac * (forwV - currV).Mag();
            Update(currV, p, kTRUE, kTRUE);
            Step(currV, p, forwV, forwP);
            p     = forwP;
            currV = forwV;
            fPoints.push_back(currV);
            ++np;
            fH.fMaxStep = orig_max_step;
         }

         // Distribute offset to desired crossing point over all segment.

         TEveVectorD off(v - currV);
         off *= 1.0f / currV.fT;
         DistributeOffset(off,  first_point, np, p);
         fV = v;
         return kTRUE;
      }
   }

   fPoints.push_back(v);
   fV = v;
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Propagate charged particle with momentum p to line segment with point s and
/// vector r to the second point. It is expected that Update() with full-update
/// was called before. Returns kFALSE if hits bounds.

Bool_t TEveTrackPropagator::LoopToLineSegment(const TEveVectorD& s, const TEveVectorD& r, TEveVectorD& p)
{
   const Double_t maxRsq = fMaxR * fMaxR;
   const Double_t rMagInv = 1./r.Mag();

   TEveVector4D currV(fV);
   TEveVector4D forwV(fV);
   TEveVectorD  forwP(p);

   Int_t first_point = fPoints.size();
   Int_t np          = first_point;

   TEveVectorD forwC;
   TEveVectorD currC;
   do
   {
      Step(currV, p, forwV, forwP);
      Update(forwV, forwP);

      ClosestPointFromVertexToLineSegment(forwV, s, r, rMagInv, forwC);

      // check forwV is over segment with orthogonal component of
      // momentum to vector r
      TEveVectorD b = r; b.Normalize();
      Double_t    x = forwP.Dot(b);
      TEveVectorD pTPM = forwP - x*b;
      if (pTPM.Dot(forwC - forwV) < 0)
      {
         break;
      }

      if (IsOutsideBounds(forwV, maxRsq, fMaxZ))
      {
         fV = currV;
         return kFALSE;
      }

      fPoints.push_back(forwV);
      currV = forwV;
      p     = forwP;
      currC = forwC;
      ++np;
   } while (np < fNMax);

   // Get closest point on segment relative to line with forw and currV points.
   TEveVectorD v;
   ClosestPointBetweenLines(s, r, currV, forwV - currV, v);

   // make the remaining fractional step
   if (np > first_point)
   {
      if ((v - currV).Mag() > kStepEps)
      {
         TEveVector last_step = forwV - currV;
         TEveVector delta     = v - currV;
         Double_t  step_frac  = last_step.Dot(delta) / last_step.Mag2();
         if (step_frac > 0)
         {
            // Step for fraction of previous step size.
            // We pass 'enforce_max_step' flag to Update().
            Float_t orig_max_step = fH.fMaxStep;
            fH.fMaxStep = step_frac * (forwV - currV).Mag();
            Update(currV, p, kTRUE, kTRUE);
            Step(currV, p, forwV, forwP);
            p     = forwP;
            currV = forwV;
            fPoints.push_back(currV);
            ++np;
            fH.fMaxStep = orig_max_step;
         }

         // Distribute offset to desired crossing point over all segment.

         TEveVectorD off(v - currV);
         off *= 1.0f / currV.fT;
         DistributeOffset(off, first_point, np, p);
         fV = v;
         return kTRUE;
      }
   }

   fPoints.push_back(v);
   fV = v;
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Distribute offset between first and last point index and rotate
/// momentum.

void TEveTrackPropagator::DistributeOffset(const TEveVectorD& off, Int_t first_point, Int_t np, TEveVectorD& p)
{
   // Calculate the required momentum rotation.
   // lpd - last-points-delta
   TEveVectorD lpd0(fPoints[np-1]);
   lpd0 -= fPoints[np-2];
   lpd0.Normalize();

   for (Int_t i = first_point; i < np; ++i)
   {
      fPoints[i] += off * fPoints[i].fT;
   }

   TEveVectorD lpd1(fPoints[np-1]);
   lpd1 -= fPoints[np-2];
   lpd1.Normalize();

   TEveTrans tt;
   tt.SetupFromToVec(lpd0, lpd1);

   // TEveVectorD pb4(p);
   // printf("Rotating momentum: p0 = "); p.Dump();
   tt.RotateIP(p);
   // printf("                   p1 = "); p.Dump();
   // printf("  n1=%f, n2=%f, dp = %f deg\n", pb4.Mag(), p.Mag(),
   //        TMath::RadToDeg()*TMath::ACos(p.Dot(pb4)/(pb4.Mag()*p.Mag())));
}

////////////////////////////////////////////////////////////////////////////////
/// Propagate neutral particle to vertex v.

Bool_t TEveTrackPropagator::LineToVertex(TEveVectorD& v)
{
   TEveVector4D currV = v;

   currV.fX = v.fX;
   currV.fY = v.fY;
   currV.fZ = v.fZ;
   fPoints.push_back(currV);

   fV = v;
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Propagate neutral particle with momentum p to bounds.

void TEveTrackPropagator::LineToBounds(TEveVectorD& p)
{
   Double_t tZ = 0, tR = 0, tB = 0;

   // time where particle intersect +/- fMaxZ
   if (p.fZ > 0)
      tZ = (fMaxZ - fV.fZ) / p.fZ;
   else if (p.fZ < 0)
      tZ = - (fMaxZ + fV.fZ) / p.fZ;
   else
      tZ = 1e99;

   // time where particle intersects cylinder
   Double_t a = p.fX*p.fX + p.fY*p.fY;
   Double_t b = 2.0 * (fV.fX*p.fX + fV.fY*p.fY);
   Double_t c = fV.fX*fV.fX + fV.fY*fV.fY - fMaxR*fMaxR;
   Double_t d = b*b - 4.0*a*c;
   if (d >= 0) {
      Double_t sqrtD = TMath::Sqrt(d);
      tR = (-b - sqrtD) / (2.0 * a);
      if (tR < 0) {
         tR = (-b + sqrtD) / (2.0 * a);
      }
      tB = tR < tZ ? tR : tZ; // compare the two times
   } else {
      tB = tZ;
   }
   TEveVectorD nv(fV.fX + p.fX*tB, fV.fY + p.fY*tB, fV.fZ + p.fZ*tB);
   LineToVertex(nv);
}

////////////////////////////////////////////////////////////////////////////////
/// Intersect helix with a plane. Current position and argument p define
/// the helix.

Bool_t TEveTrackPropagator::HelixIntersectPlane(const TEveVectorD& p,
                                                const TEveVectorD& point,
                                                const TEveVectorD& normal,
                                                TEveVectorD& itsect)
{
   TEveVectorD pos(fV);
   TEveVectorD mom(p);
   if (fMagFieldObj->IsConst())
      fH.UpdateHelix(mom, fMagFieldObj->GetFieldD(pos), kFALSE, kFALSE);

   TEveVectorD n(normal);
   TEveVectorD delta = pos - point;
   Double_t d = delta.Dot(n);
   if (d > 0) {
      n.NegateXYZ(); // Turn normal around so that we approach from negative side of the plane
      d = -d;
   }

   TEveVector4D forwV;
   TEveVectorD  forwP;
   TEveVector4D pos4(pos);
   while (kTRUE)
   {
      Update(pos4, mom);
      Step(pos4, mom, forwV , forwP);
      Double_t new_d = (forwV - point).Dot(n);
      if (new_d < d)
      {
         // We are going further away ... fail intersect.
         Warning("HelixIntersectPlane", "going away from the plane.");
         return kFALSE;
      }
      if (new_d > 0)
      {
         delta  = forwV - pos4;
         itsect = pos4 + delta * ((point - pos4).Dot(n) / delta.Dot(n));

         return kTRUE;
      }
      pos4 = forwV;
      mom  = forwP;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Intersect line with a plane. Current position and argument p define
/// the line.

Bool_t TEveTrackPropagator::LineIntersectPlane(const TEveVectorD& p,
                                               const TEveVectorD& point,
                                               const TEveVectorD& normal,
                                                     TEveVectorD& itsect)
{
   TEveVectorD pos(fV.fX, fV.fY, fV.fZ);
   TEveVectorD delta = point - pos;

   Double_t pn = p.Dot(normal);
   if (pn == 0)
   {
     return kFALSE;
   }
   Double_t t = delta.Dot(normal) / pn;
   if (t < 0) {
      return kFALSE;
   } else {
      itsect = pos + p*t;
      return kTRUE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Find intersection of currently propagated track with a plane.
/// Current track position is used as starting point.
///
/// Args:
///  - p        - track momentum to use for extrapolation
///  - point    - a point on a plane
///  - normal   - normal of the plane
///  - itsect   - output, point of intersection
/// Returns:
///  - kFALSE if intersection can not be found, kTRUE otherwise.

Bool_t TEveTrackPropagator::IntersectPlane(const TEveVectorD& p,
                                           const TEveVectorD& point,
                                           const TEveVectorD& normal,
                                                 TEveVectorD& itsect)
{
   if (fH.fCharge && fMagFieldObj)
      return HelixIntersectPlane(p, point, normal, itsect);
   else
      return LineIntersectPlane(p, point, normal, itsect);
}

////////////////////////////////////////////////////////////////////////////////
/// Get closest point from given vertex v to line segment defined with s and r.
/// Argument rMagInv is cached. rMagInv= 1./rMag()

void TEveTrackPropagator::ClosestPointFromVertexToLineSegment(const TEveVectorD& v,
                                                              const TEveVectorD& s,
                                                              const TEveVectorD& r,
                                                              Double_t rMagInv,
                                                              TEveVectorD& c)
{
   TEveVectorD dir = v - s;
   TEveVectorD b1  = r * rMagInv;

   // parallel distance
   Double_t dot     = dir.Dot(b1);
   TEveVectorD dirI = dot * b1;

   Double_t facX = dot * rMagInv;

   if (facX <= 0)
      c = s;
   else if (facX >= 1)
      c = s + r;
   else
      c = s + dirI;
}

////////////////////////////////////////////////////////////////////////////////
/// Get closest point on line defined with vector p0 and u.
/// Return false if the point is forced on the line segment.

Bool_t TEveTrackPropagator::ClosestPointBetweenLines(const TEveVectorD& p0,
                                                     const TEveVectorD& u,
                                                     const TEveVectorD& q0,
                                                     const TEveVectorD& v,
                                                     TEveVectorD& out)
{
   TEveVectorD w0 = p0 -q0;
   Double_t a = u.Mag2();
   Double_t b = u.Dot(v);
   Double_t c = v.Mag2();
   Double_t d = u.Dot(w0);
   Double_t e = v.Dot(w0);

   Double_t x = (b*e - c*d)/(a*c -b*b);
   Bool_t force = (x < 0 || x > 1);
   out = p0 + TMath::Range(0., 1., x) * u;
   return force;
}

////////////////////////////////////////////////////////////////////////////////
/// Reset ps and populate it with points in propagation cache.

void TEveTrackPropagator::FillPointSet(TEvePointSet* ps) const
{
   Int_t size = TMath::Min(fNMax, (Int_t)fPoints.size());
   ps->Reset(size);
   for (Int_t i = 0; i < size; ++i)
   {
      const TEveVector4D& v = fPoints[i];
      ps->SetNextPoint(v.fX, v.fY, v.fZ);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Rebuild all tracks using this render-style.

void TEveTrackPropagator::RebuildTracks()
{
   TEveTrack* track;
   RefMap_i i = fBackRefs.begin();
   while (i != fBackRefs.end())
   {
      track = dynamic_cast<TEveTrack*>(i->first);
      track->MakeTrack();
      track->StampObjProps();
      ++i;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set constant magnetic field and rebuild tracks.

void TEveTrackPropagator::SetMagField(Double_t bX, Double_t bY, Double_t bZ)
{
   SetMagFieldObj(new TEveMagFieldConst(bX, bY, bZ));
}

////////////////////////////////////////////////////////////////////////////////
/// Set constant magnetic field and rebuild tracks.

void TEveTrackPropagator::SetMagFieldObj(TEveMagField* field, Bool_t own_field)
{
   if (fMagFieldObj && fOwnMagFiledObj) delete fMagFieldObj;

   fMagFieldObj    = field;
   fOwnMagFiledObj = own_field;

   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////

void TEveTrackPropagator::PrintMagField(Double_t x, Double_t y, Double_t z) const
{
   if (fMagFieldObj) fMagFieldObj->PrintField(x, y, z);
}

////////////////////////////////////////////////////////////////////////////////
/// Set maximum radius and rebuild tracks.

void TEveTrackPropagator::SetMaxR(Double_t x)
{
   fMaxR = x;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set maximum z and rebuild tracks.

void TEveTrackPropagator::SetMaxZ(Double_t x)
{
   fMaxZ = x;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set maximum number of orbits and rebuild tracks.

void TEveTrackPropagator::SetMaxOrbs(Double_t x)
{
   fMaxOrbs = x;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set maximum step angle and rebuild tracks.
/// WARNING -- this method / variable was mis-named.

void TEveTrackPropagator::SetMinAng(Double_t x)
{
   Warning("SetMinAng", "This method was mis-named, use SetMaxAng() instead!");
   SetMaxAng(x);
}
////////////////////////////////////////////////////////////////////////////////
/// Get maximum step angle.
/// WARNING -- this method / variable was mis-named.

Double_t TEveTrackPropagator::GetMinAng() const
{
   Warning("GetMinAng", "This method was mis-named, use GetMaxAng() instead!");
   return GetMaxAng();
}

////////////////////////////////////////////////////////////////////////////////
/// Set maximum step angle and rebuild tracks.

void TEveTrackPropagator::SetMaxAng(Double_t x)
{
   fH.fMaxAng = x;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set maximum step-size and rebuild tracks.

void TEveTrackPropagator::SetMaxStep(Double_t x)
{
   fH.fMaxStep = x;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set maximum error and rebuild tracks.

void TEveTrackPropagator::SetDelta(Double_t x)
{
   fH.fDelta = x;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set daughter creation point fitting and rebuild tracks.

void TEveTrackPropagator::SetFitDaughters(Bool_t x)
{
   fFitDaughters = x;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set track-reference fitting and rebuild tracks.

void TEveTrackPropagator::SetFitReferences(Bool_t x)
{
   fFitReferences = x;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set decay fitting and rebuild tracks.

void TEveTrackPropagator::SetFitDecay(Bool_t x)
{
   fFitDecay = x;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set line segment fitting and rebuild tracks.

void TEveTrackPropagator::SetFitLineSegments(Bool_t x)
{
   fFitLineSegments = x;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set 2D-cluster fitting and rebuild tracks.

void TEveTrackPropagator::SetFitCluster2Ds(Bool_t x)
{
   fFitCluster2Ds = x;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set decay rendering and rebuild tracks.

void TEveTrackPropagator::SetRnrDecay(Bool_t rnr)
{
   fRnrDecay = rnr;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set rendering of 2D-clusters and rebuild tracks.

void TEveTrackPropagator::SetRnrCluster2Ds(Bool_t rnr)
{
   fRnrCluster2Ds = rnr;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set daughter rendering and rebuild tracks.

void TEveTrackPropagator::SetRnrDaughters(Bool_t rnr)
{
   fRnrDaughters = rnr;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set track-reference rendering and rebuild tracks.

void TEveTrackPropagator::SetRnrReferences(Bool_t rnr)
{
   fRnrReferences = rnr;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set first-vertex rendering and rebuild tracks.

void TEveTrackPropagator::SetRnrFV(Bool_t x)
{
   fRnrFV = x;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set projection break-point mode and rebuild tracks.

void TEveTrackPropagator::SetProjTrackBreaking(UChar_t x)
{
   fProjTrackBreaking = x;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Set projection break-point rendering and rebuild tracks.

void TEveTrackPropagator::SetRnrPTBMarkers(Bool_t x)
{
   fRnrPTBMarkers = x;
   RebuildTracks();
}

////////////////////////////////////////////////////////////////////////////////
/// Wrapper to step with method RungeKutta.

void TEveTrackPropagator::StepRungeKutta(Double_t step,
                                         Double_t* vect, Double_t* vout)
{
  /// ******************************************************************
  /// *                                                                *
  /// *  Runge-Kutta method for tracking a particle through a magnetic *
  /// *  field. Uses Nystroem algorithm (See Handbook Nat. Bur. of     *
  /// *  Standards, procedure 25.5.20)                                 *
  /// *                                                                *
  /// *  Input parameters                                              *
  /// *         CHARGE    Particle charge                              *
  /// *         STEP      Step size                                    *
  /// *         VECT      Initial co-ords,direction cosines,momentum   *
  /// *  Output parameters                                             *
  /// *         VOUT      Output co-ords,direction cosines,momentum    *
  /// *  User routine called                                           *
  /// *         CALL GUFLD(X,F)                                        *
  /// *                                                                *
  /// *    ==>Called by : <USER>, GUSWIM                               *
  /// *         Authors    R.Brun, M.Hansroul  *********               *
  /// *                     V.Perevoztchikov (CUT STEP implementation) *
  /// *                                                                *
  /// *                                                                *
  /// ******************************************************************

  Double_t h2, h4, f[4];
  Double_t /* xyzt[3], */ a, b, c, ph,ph2;
  Double_t secxs[4],secys[4],seczs[4],hxp[3];
  Double_t g1, g2, g3, g4, g5, g6, ang2, dxt, dyt, dzt;
  Double_t est, at, bt, ct, cba;
  Double_t f1, f2, f3, f4, rho, tet, hnorm, hp, rho1, sint, cost;

  Double_t x;
  Double_t y;
  Double_t z;

  Double_t xt;
  Double_t yt;
  Double_t zt;

  // const Int_t maxit = 1992;
  const Int_t maxit  = 500;
  const Int_t maxcut = 11;

  const Double_t hmin   = 1e-4; // !!! MT ADD,  should be member
  const Double_t kdlt   = 1e-3; // !!! MT CHANGE from 1e-4, should be member
  const Double_t kdlt32 = kdlt/32.;
  const Double_t kthird = 1./3.;
  const Double_t khalf  = 0.5;
  const Double_t kec    = 2.9979251e-3;

  const Double_t kpisqua = 9.86960440109;
  const Int_t kix  = 0;
  const Int_t kiy  = 1;
  const Int_t kiz  = 2;
  const Int_t kipx = 3;
  const Int_t kipy = 4;
  const Int_t kipz = 5;

  // *.
  // *.    ------------------------------------------------------------------
  // *.
  // *             this constant is for units cm,gev/c and kgauss
  // *
  Int_t iter = 0;
  Int_t ncut = 0;
  for(Int_t j = 0; j < 7; j++)
    vout[j] = vect[j];

  Double_t pinv   = kec * fH.fCharge / vect[6];
  Double_t tl     = 0.;
  Double_t h      = step;
  Double_t rest;

  do {
    rest  = step - tl;
    if (TMath::Abs(h) > TMath::Abs(rest))
       h = rest;

    f[0] = -fH.fB.fX;
    f[1] = -fH.fB.fY;
    f[2] = -fH.fB.fZ;

    // * start of integration
    x      = vout[0];
    y      = vout[1];
    z      = vout[2];
    a      = vout[3];
    b      = vout[4];
    c      = vout[5];

    h2     = khalf * h;
    h4     = khalf * h2;
    ph     = pinv * h;
    ph2    = khalf * ph;
    secxs[0] = (b * f[2] - c * f[1]) * ph2;
    secys[0] = (c * f[0] - a * f[2]) * ph2;
    seczs[0] = (a * f[1] - b * f[0]) * ph2;
    ang2 = (secxs[0]*secxs[0] + secys[0]*secys[0] + seczs[0]*seczs[0]);
    if (ang2 > kpisqua) break;

    dxt    = h2 * a + h4 * secxs[0];
    dyt    = h2 * b + h4 * secys[0];
    dzt    = h2 * c + h4 * seczs[0];
    xt     = x + dxt;
    yt     = y + dyt;
    zt     = z + dzt;

    // * second intermediate point
    est = TMath::Abs(dxt) + TMath::Abs(dyt) + TMath::Abs(dzt);
    if (est > h) {
      if (ncut++ > maxcut) break;
      h *= khalf;
      continue;
    }

    // xyzt[0] = xt;
    // xyzt[1] = yt;
    // xyzt[2] = zt;

    fH.fB = fMagFieldObj->GetFieldD(xt, yt, zt);
    f[0] = -fH.fB.fX;
    f[1] = -fH.fB.fY;
    f[2] = -fH.fB.fZ;

    at     = a + secxs[0];
    bt     = b + secys[0];
    ct     = c + seczs[0];

    secxs[1] = (bt * f[2] - ct * f[1]) * ph2;
    secys[1] = (ct * f[0] - at * f[2]) * ph2;
    seczs[1] = (at * f[1] - bt * f[0]) * ph2;
    at     = a + secxs[1];
    bt     = b + secys[1];
    ct     = c + seczs[1];
    secxs[2] = (bt * f[2] - ct * f[1]) * ph2;
    secys[2] = (ct * f[0] - at * f[2]) * ph2;
    seczs[2] = (at * f[1] - bt * f[0]) * ph2;
    dxt    = h * (a + secxs[2]);
    dyt    = h * (b + secys[2]);
    dzt    = h * (c + seczs[2]);
    xt     = x + dxt;
    yt     = y + dyt;
    zt     = z + dzt;
    at     = a + 2.*secxs[2];
    bt     = b + 2.*secys[2];
    ct     = c + 2.*seczs[2];

    est = TMath::Abs(dxt)+TMath::Abs(dyt)+TMath::Abs(dzt);
    if (est > 2.*TMath::Abs(h)) {
      if (ncut++ > maxcut) break;
      h *= khalf;
      continue;
    }

    // xyzt[0] = xt;
    // xyzt[1] = yt;
    // xyzt[2] = zt;

    fH.fB = fMagFieldObj->GetFieldD(xt, yt, zt);
    f[0] = -fH.fB.fX;
    f[1] = -fH.fB.fY;
    f[2] = -fH.fB.fZ;

    z      = z + (c + (seczs[0] + seczs[1] + seczs[2]) * kthird) * h;
    y      = y + (b + (secys[0] + secys[1] + secys[2]) * kthird) * h;
    x      = x + (a + (secxs[0] + secxs[1] + secxs[2]) * kthird) * h;

    secxs[3] = (bt*f[2] - ct*f[1])* ph2;
    secys[3] = (ct*f[0] - at*f[2])* ph2;
    seczs[3] = (at*f[1] - bt*f[0])* ph2;
    a      = a+(secxs[0]+secxs[3]+2. * (secxs[1]+secxs[2])) * kthird;
    b      = b+(secys[0]+secys[3]+2. * (secys[1]+secys[2])) * kthird;
    c      = c+(seczs[0]+seczs[3]+2. * (seczs[1]+seczs[2])) * kthird;

    est    = TMath::Abs(secxs[0]+secxs[3] - (secxs[1]+secxs[2]))
      + TMath::Abs(secys[0]+secys[3] - (secys[1]+secys[2]))
      + TMath::Abs(seczs[0]+seczs[3] - (seczs[1]+seczs[2]));

    if (est > kdlt && TMath::Abs(h) > hmin) {
      if (ncut++ > maxcut) break;
      h *= khalf;
      continue;
    }

    ncut = 0;
    // * if too many iterations, go to helix
    if (iter++ > maxit) break;

    tl += h;
    if (est < kdlt32)
      h *= 2.;
    cba    = 1./ TMath::Sqrt(a*a + b*b + c*c);
    vout[0] = x;
    vout[1] = y;
    vout[2] = z;
    vout[3] = cba*a;
    vout[4] = cba*b;
    vout[5] = cba*c;
    rest = step - tl;
    if (step < 0.) rest = -rest;
    if (rest < 1.e-5*TMath::Abs(step))
    {
       Float_t dot = (vout[3]*vect[3] + vout[4]*vect[4] + vout[5]*vect[5]);
       fH.fPhi += TMath::ACos(dot);
       return;
    }

  } while(1);

  // angle too big, use helix

  f1  = f[0];
  f2  = f[1];
  f3  = f[2];
  f4  = TMath::Sqrt(f1*f1+f2*f2+f3*f3);
  rho = -f4*pinv;
  tet = rho * step;

  hnorm = 1./f4;
  f1 = f1*hnorm;
  f2 = f2*hnorm;
  f3 = f3*hnorm;

  hxp[0] = f2*vect[kipz] - f3*vect[kipy];
  hxp[1] = f3*vect[kipx] - f1*vect[kipz];
  hxp[2] = f1*vect[kipy] - f2*vect[kipx];

  hp = f1*vect[kipx] + f2*vect[kipy] + f3*vect[kipz];

  rho1 = 1./rho;
  sint = TMath::Sin(tet);
  cost = 2.*TMath::Sin(khalf*tet)*TMath::Sin(khalf*tet);

  g1 = sint*rho1;
  g2 = cost*rho1;
  g3 = (tet-sint) * hp*rho1;
  g4 = -cost;
  g5 = sint;
  g6 = cost * hp;

  vout[kix] = vect[kix] + g1*vect[kipx] + g2*hxp[0] + g3*f1;
  vout[kiy] = vect[kiy] + g1*vect[kipy] + g2*hxp[1] + g3*f2;
  vout[kiz] = vect[kiz] + g1*vect[kipz] + g2*hxp[2] + g3*f3;

  vout[kipx] = vect[kipx] + g4*vect[kipx] + g5*hxp[0] + g6*f1;
  vout[kipy] = vect[kipy] + g4*vect[kipy] + g5*hxp[1] + g6*f2;
  vout[kipz] = vect[kipz] + g4*vect[kipz] + g5*hxp[2] + g6*f3;

  fH.fPhi += tet;
}
