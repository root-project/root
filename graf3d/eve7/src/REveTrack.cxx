// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveTrack.hxx>
#include <ROOT/REveTrackPropagator.hxx>
#include <ROOT/REvePointSet.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveTrackProjected.hxx>

#include "TParticle.h"
#include "TPolyLine3D.h"
#include "TMarker.h"
#include "TPolyMarker3D.h"
#include "TColor.h"
#include "TParticlePDG.h"
#include "TClass.h"
#include "Riostream.h"

#include <vector>
#include <algorithm>
#include <functional>

#include "json.hpp"


using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

/** \class REveTrack
\ingroup REve
Visual representation of a track.

If member fDpDs is set, the momentum is reduced on all path-marks that do
not fix the momentum according to the distance travelled from the previous
pathmark.
*/

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

REveTrack::REveTrack() :
   REveLine(),

   fV(),
   fP(),
   fPEnd(),
   fBeta(0),
   fDpDs(0),
   fPdg(0),
   fCharge(0),
   fLabel(kMinInt),
   fIndex(kMinInt),
   fStatus(0),
   fLockPoints(kFALSE),
   fPathMarks(),
   fLastPMIdx(0),
   fPropagator(nullptr)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor from TParticle.

REveTrack::REveTrack(TParticle *t, Int_t label, REveTrackPropagator *prop):
   REveLine(),

   fV(t->Vx(), t->Vy(), t->Vz()),
   fP(t->Px(), t->Py(), t->Pz()),
   fPEnd(),
   fBeta(t->P()/t->Energy()),
   fDpDs(0),
   fPdg(0),
   fCharge(0),
   fLabel(label),
   fIndex(kMinInt),
   fStatus(t->GetStatusCode()),
   fLockPoints(kFALSE),
   fPathMarks(),
   fLastPMIdx(0),
   fPropagator(nullptr)
{
   SetPropagator(prop);
   SetMainColorPtr(&fLineColor);

   TParticlePDG *pdgp = t->GetPDG();
   if (pdgp) {
      fPdg    = pdgp->PdgCode();
      fCharge = (Int_t) TMath::Nint(pdgp->Charge()/3);
   }

   SetName(t->GetName());
}

/*
////////////////////////////////////////////////////////////////////////////////

REveTrack::REveTrack(REveMCTrack* t, REveTrackPropagator* prop):
   REveLine(),

   fV(t->Vx(), t->Vy(), t->Vz()),
   fP(t->Px(), t->Py(), t->Pz()),
   fPEnd(),
   fBeta(t->P()/t->Energy()),
   fDpDs(0),
   fPdg(0),
   fCharge(0),
   fLabel(t->fLabel),
   fIndex(t->fIndex),
   fStatus(t->GetStatusCode()),
   fLockPoints(kFALSE),
   fPathMarks(),
   fLastPMIdx(0),
   fPropagator(0)
{
   // Constructor from REveUtil Monte Carlo track.

   SetPropagator(prop);
   fMainColorPtr = &fLineColor;

   TParticlePDG* pdgp = t->GetPDG();
   if (pdgp) {
      fCharge = (Int_t) TMath::Nint(pdgp->Charge()/3);
   }

   SetName(t->GetName());
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor from REveRecTrack<double> reconstructed track.

REveTrack::REveTrack(REveRecTrackD* t, REveTrackPropagator* prop) :
   REveLine(),

   fV(t->fV),
   fP(t->fP),
   fPEnd(),
   fBeta(t->fBeta),
   fDpDs(0),
   fPdg(0),
   fCharge(t->fSign),
   fLabel(t->fLabel),
   fIndex(t->fIndex),
   fStatus(t->fStatus),
   fLockPoints(kFALSE),
   fPathMarks(),
   fLastPMIdx(0),
   fPropagator(0)
{
   SetPropagator(prop);
   fMainColorPtr = &fLineColor;

   SetName(t->GetName());
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor from REveRecTrack<float> reconstructed track.
/// It is recommended to use constructor with  REveRecTrack<double> since
/// REveTrackPropagator operates with double type.

REveTrack::REveTrack(REveRecTrack* t, REveTrackPropagator* prop) :
   REveLine(),

   fV(t->fV),
   fP(t->fP),
   fPEnd(),
   fBeta(t->fBeta),
   fDpDs(0),
   fPdg(0),
   fCharge(t->fSign),
   fLabel(t->fLabel),
   fIndex(t->fIndex),
   fStatus(t->fStatus),
   fLockPoints(kFALSE),
   fPathMarks(),
   fLastPMIdx(0),
   fPropagator(0)
{
   SetPropagator(prop);
   fMainColorPtr = &fLineColor;

   SetName(t->GetName());
}
*/

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor. Track parameters are copied but the
/// extrapolation is not performed so you should still call
/// MakeTrack() to do that.
/// If points of 't' are locked, they are cloned.

REveTrack::REveTrack(const REveTrack& t) :
   REveLine(t),
   fV(t.fV),
   fP(t.fP),
   fPEnd(),
   fBeta(t.fBeta),
   fDpDs(t.fDpDs),
   fPdg(t.fPdg),
   fCharge(t.fCharge),
   fLabel(t.fLabel),
   fIndex(t.fIndex),
   fStatus(t.fStatus),
   fLockPoints(t.fLockPoints),
   fPathMarks(),
   fLastPMIdx(t.fLastPMIdx),
   fPropagator(nullptr)
{
   if (fLockPoints)
      ClonePoints(t);

   SetPathMarks(t);
   SetPropagator (t.fPropagator);

   CopyVizParams(&t);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

REveTrack::~REveTrack()
{
   SetPropagator(nullptr);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the bounding box of the track.

void REveTrack::ComputeBBox()
{
   if (fSize > 0 || ! fPathMarks.empty())
   {
      BBoxInit();
      Int_t    n = fSize;
      Float_t *p = & fPoints[0].fX;
      for (Int_t i = 0; i < n; ++i, p += 3)
      {
         BBoxCheckPoint(p);
      }
      for (vPathMark_ci i = fPathMarks.begin(); i != fPathMarks.end(); ++i)
      {
         BBoxCheckPoint(i->fV.fX, i->fV.fY, i->fV.fZ);
      }
   }
   else
   {
      BBoxZero();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set standard track title based on most data-member values.

void REveTrack::SetStdTitle()
{
   TString idx(fIndex == kMinInt ? "<undef>" : Form("%d", fIndex));
   TString lbl(fLabel == kMinInt ? "<undef>" : Form("%d", fLabel));
   SetTitle(Form("Index=%s, Label=%s\nChg=%d, Pdg=%d\n"
                 "pT=%.3f, pZ=%.3f\nV=(%.3f, %.3f, %.3f)",
                 idx.Data(), lbl.Data(), fCharge, fPdg,
                 fP.Perp(), fP.fZ, fV.fX, fV.fY, fV.fZ));
}

////////////////////////////////////////////////////////////////////////////////
/// Copy track parameters from t. Track-propagator is set, too.
/// PathMarks are cleared - you can copy them via SetPathMarks(t).
/// If track 't' is locked, you should probably clone its points
/// over - use REvePointSet::ClonePoints(t);

void REveTrack::SetTrackParams(const REveTrack& t)
{
   fV          = t.fV;
   fP          = t.fP;
   fBeta       = t.fBeta;
   fPdg        = t.fPdg;
   fCharge     = t.fCharge;
   fLabel      = t.fLabel;
   fIndex      = t.fIndex;

   fPathMarks.clear();
   SetPropagator(t.fPropagator);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy path-marks from t.

void REveTrack::SetPathMarks(const REveTrack& t)
{
   std::copy(t.RefPathMarks().begin(), t.RefPathMarks().end(),
             std::back_insert_iterator<vPathMark_t>(fPathMarks));
}

////////////////////////////////////////////////////////////////////////////////
/// Set track's propagator.
/// Reference counts of old and new propagator are updated.

void REveTrack::SetPropagator(REveTrackPropagator *prop)
{
   if (fPropagator == prop) return;
   if (fPropagator) fPropagator->DecRefCount(this);
   fPropagator = prop;
   if (fPropagator) fPropagator->IncRefCount(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Set line and marker attributes from REveTrackList.

void REveTrack::SetAttLineAttMarker(REveTrackList* tl)
{
   SetRnrLine(tl->GetRnrLine());
   SetLineColor(tl->GetLineColor());
   SetLineStyle(tl->GetLineStyle());
   SetLineWidth(tl->GetLineWidth());

   SetRnrPoints(tl->GetRnrPoints());
   SetMarkerColor(tl->GetMarkerColor());
   SetMarkerStyle(tl->GetMarkerStyle());
   SetMarkerSize(tl->GetMarkerSize());
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate track representation based on track data and current
/// settings of the propagator.
/// If recurse is true, descend into children.

void REveTrack::MakeTrack(Bool_t recurse)
{
   if (!fLockPoints)
   {
      Reset(0);
      fLastPMIdx = 0;

      REveTrackPropagator& rTP((fPropagator != 0) ? *fPropagator : REveTrackPropagator::fgDefault);

      const Double_t maxRsq = rTP.GetMaxR() * rTP.GetMaxR();
      const Double_t maxZ   = rTP.GetMaxZ();

      if ( ! REveTrackPropagator::IsOutsideBounds(fV, maxRsq, maxZ))
      {
         REveVectorD currP = fP;
         Bool_t decay = kFALSE;
         rTP.InitTrack(fV, fCharge);
         for (vPathMark_i pm = fPathMarks.begin(); pm != fPathMarks.end(); ++pm, ++fLastPMIdx)
         {
            Int_t start_point = rTP.GetCurrentPoint();

            if (rTP.GetFitReferences() && pm->fType == REvePathMarkD::kReference)
            {
               if (REveTrackPropagator::IsOutsideBounds(pm->fV, maxRsq, maxZ))
                  break;
               if (rTP.GoToVertex(pm->fV, currP))
               {
                  currP.fX = pm->fP.fX; currP.fY = pm->fP.fY; currP.fZ = pm->fP.fZ;
               }
               else
               {
                  break;
               }
            }
            else if (rTP.GetFitDaughters() && pm->fType == REvePathMarkD::kDaughter)
            {
               if (REveTrackPropagator::IsOutsideBounds(pm->fV, maxRsq, maxZ))
                  break;
               if (rTP.GoToVertex(pm->fV, currP))
               {
                  currP.fX -= pm->fP.fX; currP.fY -= pm->fP.fY; currP.fZ -= pm->fP.fZ;
                  if (fDpDs != 0)
                  {
                     Double_t dp = fDpDs * rTP.GetTrackLength(start_point);
                     Double_t p  = currP.Mag();
                     if (p > dp)   currP *= 1.0 - dp / p;
                  }
               }
               else
               {
                  break;
               }
            }
            else if (rTP.GetFitDecay() && pm->fType == REvePathMarkD::kDecay)
            {
               if (REveTrackPropagator::IsOutsideBounds(pm->fV, maxRsq, maxZ))
                  break;
               rTP.GoToVertex(pm->fV, currP);
               decay = kTRUE;
               ++fLastPMIdx;
               break;
            }
            else if (rTP.GetFitCluster2Ds() && pm->fType == REvePathMarkD::kCluster2D)
            {
               REveVectorD itsect;
               if (rTP.IntersectPlane(currP, pm->fV, pm->fP, itsect))
               {
                  REveVectorD delta   = itsect - pm->fV;
                  REveVectorD vtopass = pm->fV + pm->fE*(pm->fE.Dot(delta));
                  if (REveTrackPropagator::IsOutsideBounds(vtopass, maxRsq, maxZ))
                     break;
                  if ( ! rTP.GoToVertex(vtopass, currP))
                     break;

                  if (fDpDs != 0)
                  {
                     Double_t dp = fDpDs * rTP.GetTrackLength(start_point);
                     Double_t p  = currP.Mag();
                     if (p > dp)   currP *= 1.0 - dp / p;
                  }
               }
               else
               {
                  Warning("REveTrack::MakeTrack", "Failed to intersect plane for Cluster2D. Ignoring path-mark.");
               }
            }
            else if (rTP.GetFitLineSegments() && pm->fType == REvePathMarkD::kLineSegment)
            {
               if (REveTrackPropagator::IsOutsideBounds(pm->fV, maxRsq, maxZ))
                  break;

               if (rTP.GoToLineSegment(pm->fV, pm->fE, currP))
               {
                  if (fDpDs != 0)
                  {
                     Double_t dp = fDpDs * rTP.GetTrackLength(start_point);
                     Double_t p  = currP.Mag();
                     if (p > dp)   currP *= 1.0 - dp / p;
                  }
               }
               else
               {
                  break;
               }
            }
            else
            {
               if (REveTrackPropagator::IsOutsideBounds(pm->fV, maxRsq, maxZ))
                  break;
            }
         } // loop path-marks

         if (!decay)
         {
            // printf("%s loop to bounds  \n",fName.Data() );
            rTP.GoToBounds(currP);
         }
         fPEnd = currP;
         //  make_polyline:
         rTP.FillPointSet(this);
         rTP.ResetTrack();
      }
   }

   if (recurse)
   {
      for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
      {
         REveTrack* t = dynamic_cast<REveTrack*>(*i);
         if (t) t->MakeTrack(recurse);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Copy visualization parameters from element el.

void REveTrack::CopyVizParams(const REveElement* el)
{
   // No local parameters.

   // const REveTrack* t = dynamic_cast<const REveTrack*>(el);
   // if (t)
   // {}

   REveLine::CopyVizParams(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Write visualization parameters.

void REveTrack::WriteVizParams(std::ostream& out, const TString& var)
{
   REveLine::WriteVizParams(out, var);

   // TString t = "   " + var + "->";
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from REveProjectable, return REveTrackProjected class.

TClass* REveTrack::ProjectedClass(const REveProjection*) const
{
   return TClass::GetClass<REveTrackProjected>();
}

namespace
{
   struct Cmp_pathmark_t
   {
      bool operator()(REvePathMarkD const & a, REvePathMarkD const & b)
      { return a.fTime < b.fTime; }
   };
}

////////////////////////////////////////////////////////////////////////////////
/// Sort registered pat-marks by time.

void REveTrack::SortPathMarksByTime()
{
   std::sort(fPathMarks.begin(), fPathMarks.end(), Cmp_pathmark_t());
}

////////////////////////////////////////////////////////////////////////////////
/// Print registered path-marks.

void REveTrack::PrintPathMarks()
{
   static const REveException eh("REveTrack::PrintPathMarks ");

   printf("REveTrack '%s', number of path marks %d, label %d\n",
          GetCName(), (Int_t)fPathMarks.size(), fLabel);

   for (vPathMark_i pm = fPathMarks.begin(); pm != fPathMarks.end(); ++pm)
   {
      printf("  %-9s  p: %8f %8f %8f Vertex: %8e %8e %8e %g Extra:%8f %8f %8f\n",
             pm->TypeName(),
             pm->fP.fX,  pm->fP.fY, pm->fP.fZ,
             pm->fV.fX,  pm->fV.fY, pm->fV.fZ,
             pm->fE.fX,  pm->fE.fY, pm->fE.fZ,
             pm->fTime);
   }
}

//------------------------------------------------------------------------------

Int_t REveTrack::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   // TODO: missing streaming of fitting points
   Int_t ret = REveLine::WriteCoreJson(j, rnr_offset);
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Crates 3D point array for rendering.

void REveTrack::BuildRenderData()
{
   // TODO: missing streaming o fitting points
   REveLine::BuildRenderData();
}

////////////////////////////////////////////////////////////////////////////////
/// Emits "SecSelected(REveTrack*)" signal.

void REveTrack::SecSelected(REveTrack* /*track*/)
{
   // Emit("SecSelected(REveTrack*)", (Long_t)track);
}


//==============================================================================
//==============================================================================

/** \class REveTrackList
\ingroup REve
A list of tracks supporting change of common attributes and
selection based on track parameters.
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor. If track-propagator argument is 0, a new default
/// one is created.

REveTrackList::REveTrackList(REveTrackPropagator* prop) :
   REveElement(),
   TAttMarker(1, 20, 1),
   TAttLine(1,1,1),

   fPropagator(0),
   fRecurse(kTRUE),
   fRnrLine(kTRUE),
   fRnrPoints(kFALSE),

   fMinPt (0), fMaxPt (0), fLimPt (0),
   fMinP  (0), fMaxP  (0), fLimP  (0)
{

   fChildClass = TClass::GetClass<REveTrack>(); // override member from base REveElementList

   fMainColorPtr = &fLineColor;

   if (!prop) prop = new REveTrackPropagator;
   SetPropagator(prop);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor. If track-propagator argument is 0, a new default
/// one is created.

REveTrackList::REveTrackList(const std::string& name, REveTrackPropagator* prop) :
   REveElement(name),
   TAttMarker(1, 20, 1),
   TAttLine(1,1,1),

   fPropagator(0),
   fRecurse(kTRUE),
   fRnrLine(kTRUE),
   fRnrPoints(kFALSE),

   fMinPt (0), fMaxPt (0), fLimPt (0),
   fMinP  (0), fMaxP  (0), fLimP  (0)
{
   fChildClass = TClass::GetClass<REveTrack>(); // override member from base REveElementList

   fMainColorPtr = &fLineColor;

   if (!prop) prop = new REveTrackPropagator;
   SetPropagator(prop);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

REveTrackList::~REveTrackList()
{
   SetPropagator(0);
}

////////////////////////////////////////////////////////////////////////////////
/// Set default propagator for tracks.
/// This is not enforced onto the tracks themselves but this is the
/// propagator that is shown in the GUI.

void REveTrackList::SetPropagator(REveTrackPropagator* prop)
{
   if (fPropagator == prop) return;
   if (fPropagator) fPropagator->DecRefCount();
   fPropagator = prop;
   if (fPropagator) fPropagator->IncRefCount();
}

////////////////////////////////////////////////////////////////////////////////
/// Regenerate the visual representations of tracks.
/// The momentum limits are rescanned during the same traversal.

void REveTrackList::MakeTracks(Bool_t recurse)
{
   fLimPt = fLimP = 0;

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      REveTrack* track = dynamic_cast<REveTrack*>(*i);
      if (track)
      {
         track->MakeTrack(recurse);

         fLimPt = TMath::Max(fLimPt, track->fP.Perp());
         fLimP  = TMath::Max(fLimP,  track->fP.Mag());
      }
      if (recurse)
         FindMomentumLimits(*i, recurse);
   }

   fLimPt = RoundMomentumLimit(fLimPt);
   fLimP  = RoundMomentumLimit(fLimP);

   SanitizeMinMaxCuts();
}

////////////////////////////////////////////////////////////////////////////////
/// Loop over children and find highest pT and p of contained EveTracks.
/// These are stored in members fLimPt and fLimP.

void REveTrackList::FindMomentumLimits(Bool_t recurse)
{
   fLimPt = fLimP = 0;

   if (HasChildren())
   {
      for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
      {
         REveTrack* track = dynamic_cast<REveTrack*>(*i);
         if (track)
         {
            fLimPt = TMath::Max(fLimPt, track->fP.Perp());
            fLimP  = TMath::Max(fLimP,  track->fP.Mag());
         }
         if (recurse)
            FindMomentumLimits(*i, recurse);
      }

      fLimPt = RoundMomentumLimit(fLimPt);
      fLimP  = RoundMomentumLimit(fLimP);
   }

   SanitizeMinMaxCuts();
}

////////////////////////////////////////////////////////////////////////////////
/// Loop over track elements of argument el and find highest pT and p.
/// These are stored in members fLimPt and fLimP.

void REveTrackList::FindMomentumLimits(REveElement* el, Bool_t recurse)
{
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      REveTrack* track = dynamic_cast<REveTrack*>(*i);
      if (track)
      {
         fLimPt = TMath::Max(fLimPt, track->fP.Perp());
         fLimP  = TMath::Max(fLimP,  track->fP.Mag());
      }
      if (recurse)
         FindMomentumLimits(*i, recurse);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Round the momentum limit up to a nice value.

Double_t REveTrackList::RoundMomentumLimit(Double_t x)
{
   using namespace TMath;

   if (x < 1e-3) return 1e-3;

   Double_t fac = Power(10, 1 - Floor(Log10(x)));
   return Ceil(fac*x) / fac;
}

////////////////////////////////////////////////////////////////////////////////
/// Set Min/Max cuts so that they are within detected limits.

void REveTrackList::SanitizeMinMaxCuts()
{
   using namespace TMath;

   fMinPt = Min(fMinPt, fLimPt);
   fMaxPt = fMaxPt == 0 ? fLimPt : Min(fMaxPt, fLimPt);
   fMinP  = Min(fMinP,  fLimP);
   fMaxP  = fMaxP  == 0 ? fLimP  : Min(fMaxP,  fLimP);
}

////////////////////////////////////////////////////////////////////////////////
/// Set rendering of track as line for the list and the elements.

void REveTrackList::SetRnrLine(Bool_t rnr)
{
   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      REveTrack* track = (REveTrack*)(*i);
      if (track->GetRnrLine() == fRnrLine)
         track->SetRnrLine(rnr);
      if (fRecurse)
         SetRnrLine(rnr, *i);
   }
   fRnrLine = rnr;
}

////////////////////////////////////////////////////////////////////////////////
/// Set rendering of track as line for children of el.

void REveTrackList::SetRnrLine(Bool_t rnr, REveElement* el)
{
   REveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<REveTrack*>(*i);
      if (track && (track->GetRnrLine() == fRnrLine))
         track->SetRnrLine(rnr);
      if (fRecurse)
         SetRnrLine(rnr, *i);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set rendering of track as points for the list and the elements.

void REveTrackList::SetRnrPoints(Bool_t rnr)
{
   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      REveTrack* track = (REveTrack*)(*i);
      if (track->GetRnrPoints() == fRnrPoints)
         track->SetRnrPoints(rnr);
      if (fRecurse)
         SetRnrPoints(rnr, *i);
   }
   fRnrPoints = rnr;
}

////////////////////////////////////////////////////////////////////////////////
/// Set rendering of track as points for children of el.

void REveTrackList::SetRnrPoints(Bool_t rnr, REveElement* el)
{
   REveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<REveTrack*>(*i);
      if (track)
         if (track->GetRnrPoints() == fRnrPoints)
            track->SetRnrPoints(rnr);
      if (fRecurse)
         SetRnrPoints(rnr, *i);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set main (line) color for the list and the elements.

void REveTrackList::SetMainColor(Color_t col)
{
   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      REveTrack* track = (REveTrack*)(*i);
      if (track->GetLineColor() == fLineColor)
         track->SetLineColor(col);
      if (fRecurse)
         SetLineColor(col, *i);
   }
   REveElement::SetMainColor(col);
}

////////////////////////////////////////////////////////////////////////////////
/// Set line color for children of el.

void REveTrackList::SetLineColor(Color_t col, REveElement* el)
{
   REveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<REveTrack*>(*i);
      if (track && track->GetLineColor() == fLineColor)
         track->SetLineColor(col);
      if (fRecurse)
         SetLineColor(col, *i);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set line width for the list and the elements.

void REveTrackList::SetLineWidth(Width_t width)
{
   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      REveTrack* track = (REveTrack*)(*i);
      if (track->GetLineWidth() == fLineWidth)
         track->SetLineWidth(width);
      if (fRecurse)
         SetLineWidth(width, *i);
   }
   fLineWidth = width;
}

////////////////////////////////////////////////////////////////////////////////
/// Set line width for children of el.

void REveTrackList::SetLineWidth(Width_t width, REveElement* el)
{
   REveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<REveTrack*>(*i);
      if (track && track->GetLineWidth() == fLineWidth)
         track->SetLineWidth(width);
      if (fRecurse)
         SetLineWidth(width, *i);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set line style for the list and the elements.

void REveTrackList::SetLineStyle(Style_t style)
{
   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      REveTrack* track = (REveTrack*)(*i);
      if (track->GetLineStyle() == fLineStyle)
         track->SetLineStyle(style);
      if (fRecurse)
         SetLineStyle(style, *i);
   }
   fLineStyle = style;
}

////////////////////////////////////////////////////////////////////////////////
/// Set line style for children of el.

void REveTrackList::SetLineStyle(Style_t style, REveElement* el)
{
   REveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<REveTrack*>(*i);
      if (track && track->GetLineStyle() == fLineStyle)
         track->SetLineStyle(style);
      if (fRecurse)
         SetLineStyle(style, *i);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set marker style for the list and the elements.

void REveTrackList::SetMarkerStyle(Style_t style)
{
   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      REveTrack* track = (REveTrack*)(*i);
      if (track->GetMarkerStyle() == fMarkerStyle)
         track->SetMarkerStyle(style);
      if (fRecurse)
         SetMarkerStyle(style, *i);
   }
   fMarkerStyle = style;
}

////////////////////////////////////////////////////////////////////////////////
/// Set marker style for children of el.

void REveTrackList::SetMarkerStyle(Style_t style, REveElement* el)
{
   REveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<REveTrack*>(*i);
      if (track && track->GetMarkerStyle() == fMarkerStyle)
         track->SetMarkerStyle(style);
      if (fRecurse)
         SetMarkerStyle(style, *i);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set marker color for the list and the elements.

void REveTrackList::SetMarkerColor(Color_t col)
{
   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      REveTrack* track = (REveTrack*)(*i);
      if (track->GetMarkerColor() == fMarkerColor)
         track->SetMarkerColor(col);
      if (fRecurse)
         SetMarkerColor(col, *i);
   }
   fMarkerColor = col;
}

////////////////////////////////////////////////////////////////////////////////
/// Set marker color for children of el.

void REveTrackList::SetMarkerColor(Color_t col, REveElement* el)
{
   REveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<REveTrack*>(*i);
      if (track && track->GetMarkerColor() == fMarkerColor)
         track->SetMarkerColor(col);
      if (fRecurse)
         SetMarkerColor(col, *i);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set marker size for the list and the elements.

void REveTrackList::SetMarkerSize(Size_t size)
{
   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      REveTrack* track = (REveTrack*)(*i);
      if (track->GetMarkerSize() == fMarkerSize)
         track->SetMarkerSize(size);
      if (fRecurse)
         SetMarkerSize(size, *i);
   }
   fMarkerSize = size;
}

////////////////////////////////////////////////////////////////////////////////
/// Set marker size for children of el.

void REveTrackList::SetMarkerSize(Size_t size, REveElement* el)
{
   REveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<REveTrack*>(*i);
      if (track && track->GetMarkerSize() == fMarkerSize)
         track->SetMarkerSize(size);
      if (fRecurse)
         SetMarkerSize(size, *i);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Select visibility of tracks by transverse momentum.
/// If data-member fRecurse is set, the selection is applied
/// recursively to all children.

void REveTrackList::SelectByPt(Double_t min_pt, Double_t max_pt)
{
   fMinPt = min_pt;
   fMaxPt = max_pt;

   const Double_t minptsq = min_pt*min_pt;
   const Double_t maxptsq = max_pt*max_pt;

   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      const Double_t ptsq = ((REveTrack*)(*i))->fP.Perp2();
      Bool_t on = ptsq >= minptsq && ptsq <= maxptsq;
      (*i)->SetRnrState(on);
      if (on && fRecurse)
         SelectByPt(min_pt, max_pt, *i);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Select visibility of el's children tracks by transverse momentum.

void REveTrackList::SelectByPt(Double_t min_pt, Double_t max_pt, REveElement* el)
{
   const Double_t minptsq = min_pt*min_pt;
   const Double_t maxptsq = max_pt*max_pt;

   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      REveTrack* track = dynamic_cast<REveTrack*>(*i);
      if (track)
      {
         const Double_t ptsq = track->fP.Perp2();
         Bool_t on = ptsq >= minptsq && ptsq <= maxptsq;
         track->SetRnrState(on);
         if (on && fRecurse)
            SelectByPt(min_pt, max_pt, *i);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Select visibility of tracks by momentum.
/// If data-member fRecurse is set, the selection is applied
/// recursively to all children.

void REveTrackList::SelectByP(Double_t min_p, Double_t max_p)
{
   fMinP = min_p;
   fMaxP = max_p;

   const Double_t minpsq = min_p*min_p;
   const Double_t maxpsq = max_p*max_p;

   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      const Double_t psq  = ((REveTrack*)(*i))->fP.Mag2();
      Bool_t on = psq >= minpsq && psq <= maxpsq;
      (*i)->SetRnrState(psq >= minpsq && psq <= maxpsq);
      if (on && fRecurse)
         SelectByP(min_p, max_p, *i);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Select visibility of el's children tracks by momentum.

void REveTrackList::SelectByP(Double_t min_p, Double_t max_p, REveElement* el)
{
   const Double_t minpsq = min_p*min_p;
   const Double_t maxpsq = max_p*max_p;

   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      REveTrack* track = dynamic_cast<REveTrack*>(*i);
      if (track)
      {
         const Double_t psq  = ((REveTrack*)(*i))->fP.Mag2();
         Bool_t on = psq >= minpsq && psq <= maxpsq;
         track->SetRnrState(on);
         if (on && fRecurse)
            SelectByP(min_p, max_p, *i);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Find track by label, select it and display it in the editor.

REveTrack* REveTrackList::FindTrackByLabel(Int_t label)
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      if (((REveTrack*)(*i))->GetLabel() == label)
      {
         // TGListTree     *lt   = REX::gEve->GetLTEFrame()->GetListTree();
         // TGListTreeItem *mlti = lt->GetSelected();
         // if (mlti->GetUserData() != this)
         //    mlti = FindListTreeItem(lt);
         // TGListTreeItem *tlti = (*i)->FindListTreeItem(lt, mlti);
         // lt->HighlightItem(tlti);
         // lt->SetSelected(tlti);
         // REX::gEve->EditElement(*i);
         return (REveTrack*) *i;
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Find track by index, select it and display it in the editor.

REveTrack* REveTrackList::FindTrackByIndex(Int_t index)
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      if (((REveTrack*)(*i))->GetIndex() == index)
      {
         // TGListTree     *lt   = REX::gEve->GetLTEFrame()->GetListTree();
         // TGListTreeItem *mlti = lt->GetSelected();
         // if (mlti->GetUserData() != this)
         //    mlti = FindListTreeItem(lt);
         // TGListTreeItem *tlti = (*i)->FindListTreeItem(lt, mlti);
         // lt->HighlightItem(tlti);
         // lt->SetSelected(tlti);
         // REX::gEve->EditElement(*i);
         return (REveTrack*) *i;
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy visualization parameters from element el.

void REveTrackList::CopyVizParams(const REveElement* el)
{
   const REveTrackList* m = dynamic_cast<const REveTrackList*>(el);
   if (m)
   {
      TAttMarker::operator=(*m);
      TAttLine::operator=(*m);
      fRecurse = m->fRecurse;
      fRnrLine = m->fRnrLine;
      fRnrPoints = m->fRnrPoints;
      fMinPt   = m->fMinPt;
      fMaxPt   = m->fMaxPt;
      fLimPt   = m->fLimPt;
      fMinP    = m->fMinP;
      fMaxP    = m->fMaxP;
      fLimP    = m->fLimP;
   }

   REveElement::CopyVizParams(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Write visualization parameters.

void REveTrackList::WriteVizParams(std::ostream& out, const TString& var)
{
   REveElement::WriteVizParams(out, var);

   TString t = "   " + var + "->";
   TAttMarker::SaveMarkerAttributes(out, var);
   TAttLine  ::SaveLineAttributes  (out, var);
   out << t << "SetRecurse("   << ToString(fRecurse)   << ");\n";
   out << t << "SetRnrLine("   << ToString(fRnrLine)   << ");\n";
   out << t << "SetRnrPoints(" << ToString(fRnrPoints) << ");\n";
   // These setters are not available -- need proper AND/OR mode.
   // out << t << "SetMinPt(" << fMinPt << ");\n";
   // out << t << "SetMaxPt(" << fMaxPt << ");\n";
   // out << t << "SetLimPt(" << fLimPt << ");\n";
   // out << t << "SetMinP("  << fMinP  << ");\n";
   // out << t << "SetMaxP("  << fMaxP  << ");\n";
   // out << t << "SetLimP("  << fLimP  << ");\n";
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from REveProjectable, returns REveTrackListProjected class.

TClass* REveTrackList::ProjectedClass(const REveProjection*) const
{
   return REveTrackListProjected::Class();
}
