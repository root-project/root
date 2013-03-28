// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEvePointSet.h"

#include "TPolyLine3D.h"
#include "TMarker.h"
#include "TPolyMarker3D.h"
#include "TColor.h"
#include "TParticlePDG.h"

// Updates
#include "TEveManager.h"
#include "TEveBrowser.h"
#include "TEveTrackProjected.h"

#include "Riostream.h"

#include <vector>
#include <algorithm>
#include <functional>

//==============================================================================
//==============================================================================
// TEveTrack
//==============================================================================

//______________________________________________________________________________
//
// Visual representation of a track.
//
// If member fDpDs is set, the momentum is reduced on all path-marks that do
// not fix the momentum according to the distance travelled from the previous
// pathmark.

ClassImp(TEveTrack);

//______________________________________________________________________________
TEveTrack::TEveTrack() :
   TEveLine(),

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
   fPropagator(0)
{
   // Default constructor.
}

//______________________________________________________________________________
TEveTrack::TEveTrack(TParticle* t, Int_t label, TEveTrackPropagator* prop):
   TEveLine(),

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
   fPropagator(0)
{
   // Constructor from TParticle.

   SetPropagator(prop);
   fMainColorPtr = &fLineColor;

   TParticlePDG* pdgp = t->GetPDG();
   if (pdgp) {
      fPdg    = pdgp->PdgCode();
      fCharge = (Int_t) TMath::Nint(pdgp->Charge()/3);
   }

   SetName(t->GetName());
}

//______________________________________________________________________________
TEveTrack::TEveTrack(TEveMCTrack* t, TEveTrackPropagator* prop):
   TEveLine(),

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
   // Constructor from TEveUtil Monte Carlo track.

   SetPropagator(prop);
   fMainColorPtr = &fLineColor;

   TParticlePDG* pdgp = t->GetPDG();
   if (pdgp) {
      fCharge = (Int_t) TMath::Nint(pdgp->Charge()/3);
   }

   SetName(t->GetName());
}

//______________________________________________________________________________
TEveTrack::TEveTrack(TEveRecTrackD* t, TEveTrackPropagator* prop) :
   TEveLine(),

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
   // Constructor from TEveRecTrack<double> reconstructed track.

   SetPropagator(prop);
   fMainColorPtr = &fLineColor;

   SetName(t->GetName());
}

//______________________________________________________________________________
TEveTrack::TEveTrack(TEveRecTrack* t, TEveTrackPropagator* prop) :
   TEveLine(),

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
   // Constructor from TEveRecTrack<float> reconstructed track.
   // It is recomended to use constructor with  TEveRecTrack<double> since
   // TEveTrackPropagator operates with double type.

   SetPropagator(prop);
   fMainColorPtr = &fLineColor;

   SetName(t->GetName());
}

//______________________________________________________________________________
TEveTrack::TEveTrack(const TEveTrack& t) :
   TEveLine(),
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
   fPropagator(0)
{
   // Copy constructor. Track paremeters are copied but the
   // extrapolation is not perfermed so you should still call
   // MakeTrack() to do that.
   // If points of 't' are locked, they are cloned.

   if (fLockPoints)
      ClonePoints(t);

   SetPathMarks(t);
   SetPropagator (t.fPropagator);

   CopyVizParams(&t);
}

//______________________________________________________________________________
TEveTrack::~TEveTrack()
{
   // Destructor.

   SetPropagator(0);
}


//______________________________________________________________________________
const TGPicture* TEveTrack::GetListTreeIcon(Bool_t)
{
   // Returns list-tree icon for TEveTrack.

   return fgListTreeIcons[4];
}

//______________________________________________________________________________
void TEveTrack::ComputeBBox()
{
   // Compute the bounding box of the track.

   if (Size() > 0 || ! fPathMarks.empty())
   {
      BBoxInit();
      Int_t    n = Size();
      Float_t* p = TPolyMarker3D::fP;
      for (Int_t i = 0; i < n; ++i, p += 3)
      {
         BBoxCheckPoint(p);
      }
      for (vPathMark_ci i = fPathMarks.begin(); i != fPathMarks.end(); ++i)
      {
         BBoxCheckPoint(i->fV.fX, i->fV.fY,i->fV.fZ);
      }
   }
   else
   {
      BBoxZero();
   }
}

//==============================================================================

//______________________________________________________________________________
void TEveTrack::SetStdTitle()
{
   // Set standard track title based on most data-member values.

   TString idx(fIndex == kMinInt ? "<undef>" : Form("%d", fIndex));
   TString lbl(fLabel == kMinInt ? "<undef>" : Form("%d", fLabel));
   SetTitle(Form("Index=%s, Label=%s\nChg=%d, Pdg=%d\n"
                 "pT=%.3f, pZ=%.3f\nV=(%.3f, %.3f, %.3f)",
                 idx.Data(), lbl.Data(), fCharge, fPdg,
                 fP.Perp(), fP.fZ, fV.fX, fV.fY, fV.fZ));
}

//______________________________________________________________________________
void TEveTrack::SetTrackParams(const TEveTrack& t)
{
   // Copy track parameters from t. Track-propagator is set, too.
   // PathMarks are cleared - you can copy them via SetPathMarks(t).
   // If track 't' is locked, you should probably clone its points
   // over - use TEvePointSet::ClonePoints(t);

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

//______________________________________________________________________________
void TEveTrack::SetPathMarks(const TEveTrack& t)
{
   // Copy path-marks from t.

   std::copy(t.RefPathMarks().begin(), t.RefPathMarks().end(),
             std::back_insert_iterator<vPathMark_t>(fPathMarks));
}

//==============================================================================

//______________________________________________________________________________
void TEveTrack::SetPropagator(TEveTrackPropagator* prop)
{
   // Set track's render style.
   // Reference counts of old and new propagator are updated.

   if (fPropagator == prop) return;
   if (fPropagator) fPropagator->DecRefCount(this);
   fPropagator = prop;
   if (fPropagator) fPropagator->IncRefCount(this);
}

//==============================================================================

//______________________________________________________________________________
void TEveTrack::SetAttLineAttMarker(TEveTrackList* tl)
{
   // Set line and marker attributes from TEveTrackList.

   SetRnrLine(tl->GetRnrLine());
   SetLineColor(tl->GetLineColor());
   SetLineStyle(tl->GetLineStyle());
   SetLineWidth(tl->GetLineWidth());

   SetRnrPoints(tl->GetRnrPoints());
   SetMarkerColor(tl->GetMarkerColor());
   SetMarkerStyle(tl->GetMarkerStyle());
   SetMarkerSize(tl->GetMarkerSize());
}

//==============================================================================

//______________________________________________________________________________
void TEveTrack::MakeTrack(Bool_t recurse)
{
   // Calculate track representation based on track data and current
   // settings of the propagator.
   // If recurse is true, descend into children.

   if (!fLockPoints)
   {
      Reset(0);
      fLastPMIdx = 0;

      TEveTrackPropagator& rTP((fPropagator != 0) ? *fPropagator : TEveTrackPropagator::fgDefault);

      const Double_t maxRsq = rTP.GetMaxR() * rTP.GetMaxR();
      const Double_t maxZ   = rTP.GetMaxZ();

      if ( ! TEveTrackPropagator::IsOutsideBounds(fV, maxRsq, maxZ))
      {
         TEveVectorD currP = fP;
         Bool_t decay = kFALSE;
         rTP.InitTrack(fV, fCharge);
         for (vPathMark_i pm = fPathMarks.begin(); pm != fPathMarks.end(); ++pm, ++fLastPMIdx)
         {
            Int_t start_point = rTP.GetCurrentPoint();

            if (rTP.GetFitReferences() && pm->fType == TEvePathMarkD::kReference)
            {
               if (TEveTrackPropagator::IsOutsideBounds(pm->fV, maxRsq, maxZ))
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
            else if (rTP.GetFitDaughters() && pm->fType == TEvePathMarkD::kDaughter)
            {
               if (TEveTrackPropagator::IsOutsideBounds(pm->fV, maxRsq, maxZ))
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
            else if (rTP.GetFitDecay() && pm->fType == TEvePathMarkD::kDecay)
            {
               if (TEveTrackPropagator::IsOutsideBounds(pm->fV, maxRsq, maxZ))
                  break;
               rTP.GoToVertex(pm->fV, currP);
               decay = kTRUE;
               ++fLastPMIdx;
               break;
            }
            else if (rTP.GetFitCluster2Ds() && pm->fType == TEvePathMarkD::kCluster2D)
            {
               TEveVectorD itsect;
               if (rTP.IntersectPlane(currP, pm->fV, pm->fP, itsect))
               {
                  TEveVectorD delta   = itsect - pm->fV;
                  TEveVectorD vtopass = pm->fV + pm->fE*(pm->fE.Dot(delta));
                  if (TEveTrackPropagator::IsOutsideBounds(vtopass, maxRsq, maxZ))
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
                  Warning("TEveTrack::MakeTrack", "Failed to intersect plane for Cluster2D. Ignoring path-mark.");
               }
            }
            else if (rTP.GetFitLineSegments() && pm->fType == TEvePathMarkD::kLineSegment)
            {
               if (TEveTrackPropagator::IsOutsideBounds(pm->fV, maxRsq, maxZ))
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
               if (TEveTrackPropagator::IsOutsideBounds(pm->fV, maxRsq, maxZ))
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
         TEveTrack* t = dynamic_cast<TEveTrack*>(*i);
         if (t) t->MakeTrack(recurse);
      }
   }
}

//==============================================================================

//______________________________________________________________________________
void TEveTrack::CopyVizParams(const TEveElement* el)
{
   // Copy visualization parameters from element el.

   // No local parameters.

   // const TEveTrack* t = dynamic_cast<const TEveTrack*>(el);
   // if (t)
   // {}

   TEveLine::CopyVizParams(el);
}

//______________________________________________________________________________
void TEveTrack::WriteVizParams(std::ostream& out, const TString& var)
{
   // Write visualization parameters.

   TEveLine::WriteVizParams(out, var);

   // TString t = "   " + var + "->";
}

//______________________________________________________________________________
TClass* TEveTrack::ProjectedClass(const TEveProjection*) const
{
   // Virtual from TEveProjectable, return TEveTrackProjected class.

   return TEveTrackProjected::Class();
}

//==============================================================================

namespace
{
   struct Cmp_pathmark_t
   {
      bool operator()(TEvePathMarkD const & a, TEvePathMarkD const & b)
      { return a.fTime < b.fTime; }
   };
}

//______________________________________________________________________________
void TEveTrack::SortPathMarksByTime()
{
   // Sort registered pat-marks by time.

   std::sort(fPathMarks.begin(), fPathMarks.end(), Cmp_pathmark_t());
}

//______________________________________________________________________________
void TEveTrack::PrintPathMarks()
{
   // Print registered path-marks.

   static const TEveException eh("TEveTrack::PrintPathMarks ");

   printf("TEveTrack '%s', number of path marks %d, label %d\n",
          GetName(), (Int_t)fPathMarks.size(), fLabel);

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

//______________________________________________________________________________
void TEveTrack::SecSelected(TEveTrack* track)
{
   // Emits "SecSelected(TEveTrack*)" signal.
   // Called from TEveTrackGL on secondary-selection.

   Emit("SecSelected(TEveTrack*)", (Long_t)track);
}


//==============================================================================
//==============================================================================
// TEveTrackList
//==============================================================================

//______________________________________________________________________________
//
// A list of tracks supporting change of common attributes and
// selection based on track parameters.

ClassImp(TEveTrackList);

//______________________________________________________________________________
TEveTrackList::TEveTrackList(TEveTrackPropagator* prop) :
   TEveElementList(),
   TAttMarker(1, 20, 1),
   TAttLine(1,1,1),

   fPropagator(0),
   fRecurse(kTRUE),
   fRnrLine(kTRUE),
   fRnrPoints(kFALSE),

   fMinPt (0), fMaxPt (0), fLimPt (0),
   fMinP  (0), fMaxP  (0), fLimP  (0)
{
   // Constructor. If track-propagator argument is 0, a new default
   // one is created.

   fChildClass = TEveTrack::Class(); // override member from base TEveElementList

   fMainColorPtr = &fLineColor;

   if (prop == 0) prop = new TEveTrackPropagator;
   SetPropagator(prop);
}

//______________________________________________________________________________
TEveTrackList::TEveTrackList(const char* name, TEveTrackPropagator* prop) :
   TEveElementList(name),
   TAttMarker(1, 20, 1),
   TAttLine(1,1,1),

   fPropagator(0),
   fRecurse(kTRUE),
   fRnrLine(kTRUE),
   fRnrPoints(kFALSE),

   fMinPt (0), fMaxPt (0), fLimPt (0),
   fMinP  (0), fMaxP  (0), fLimP  (0)
{
   // Constructor. If track-propagator argument is 0, a new default
   // one is created.

   fChildClass = TEveTrack::Class(); // override member from base TEveElementList

   fMainColorPtr = &fLineColor;

   if (prop == 0) prop = new TEveTrackPropagator;
   SetPropagator(prop);
}

//______________________________________________________________________________
TEveTrackList::~TEveTrackList()
{
   // Destructor.

   SetPropagator(0);
}

//==============================================================================

//______________________________________________________________________________
void TEveTrackList::SetPropagator(TEveTrackPropagator* prop)
{
   // Set default propagator for tracks.
   // This is not enforced onto the tracks themselves but this is the
   // propagator that is shown in the TEveTrackListEditor.

   if (fPropagator == prop) return;
   if (fPropagator) fPropagator->DecRefCount();
   fPropagator = prop;
   if (fPropagator) fPropagator->IncRefCount();
}

//==============================================================================

//______________________________________________________________________________
void TEveTrackList::MakeTracks(Bool_t recurse)
{
   // Regenerate the visual representations of tracks.
   // The momentum limits are rescanned during the same traversal.

   fLimPt = fLimP = 0;

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TEveTrack* track = dynamic_cast<TEveTrack*>(*i);
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

//______________________________________________________________________________
void TEveTrackList::FindMomentumLimits(Bool_t recurse)
{
   // Loop over children and find highest pT and p of contained TEveTracks.
   // These are stored in members fLimPt and fLimP.

   fLimPt = fLimP = 0;

   if (HasChildren())
   {
      for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
      {
         TEveTrack* track = dynamic_cast<TEveTrack*>(*i);
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

//______________________________________________________________________________
void TEveTrackList::FindMomentumLimits(TEveElement* el, Bool_t recurse)
{
   // Loop over track elements of argument el and find highest pT and p.
   // These are stored in members fLimPt and fLimP.

   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      TEveTrack* track = dynamic_cast<TEveTrack*>(*i);
      if (track)
      {
         fLimPt = TMath::Max(fLimPt, track->fP.Perp());
         fLimP  = TMath::Max(fLimP,  track->fP.Mag());
      }
      if (recurse)
         FindMomentumLimits(*i, recurse);
   }
}

//______________________________________________________________________________
Double_t TEveTrackList::RoundMomentumLimit(Double_t x)
{
   // Round the momentum limit up to a nice value.

   using namespace TMath;

   if (x < 1e-3) return 1e-3;

   Double_t fac = Power(10, 1 - Floor(Log10(x)));
   return Ceil(fac*x) / fac;
}

//______________________________________________________________________________
void TEveTrackList::SanitizeMinMaxCuts()
{
   // Set Min/Max cuts so that they are within detected limits.

   using namespace TMath;

   fMinPt = Min(fMinPt, fLimPt);
   fMaxPt = fMaxPt == 0 ? fLimPt : Min(fMaxPt, fLimPt);
   fMinP  = Min(fMinP,  fLimP);
   fMaxP  = fMaxP  == 0 ? fLimP  : Min(fMaxP,  fLimP);
}

//==============================================================================

//______________________________________________________________________________
void TEveTrackList::SetRnrLine(Bool_t rnr)
{
   // Set rendering of track as line for the list and the elements.

   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      TEveTrack* track = (TEveTrack*)(*i);
      if (track->GetRnrLine() == fRnrLine)
         track->SetRnrLine(rnr);
      if (fRecurse)
         SetRnrLine(rnr, *i);
   }
   fRnrLine = rnr;
}

//______________________________________________________________________________
void TEveTrackList::SetRnrLine(Bool_t rnr, TEveElement* el)
{
   // Set rendering of track as line for children of el.

   TEveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<TEveTrack*>(*i);
      if (track && (track->GetRnrLine() == fRnrLine))
         track->SetRnrLine(rnr);
      if (fRecurse)
         SetRnrLine(rnr, *i);
   }
}

//==============================================================================

//______________________________________________________________________________
void TEveTrackList::SetRnrPoints(Bool_t rnr)
{
   // Set rendering of track as points for the list and the elements.

   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      TEveTrack* track = (TEveTrack*)(*i);
      if (track->GetRnrPoints() == fRnrPoints)
         track->SetRnrPoints(rnr);
      if (fRecurse)
         SetRnrPoints(rnr, *i);
   }
   fRnrPoints = rnr;
}

//______________________________________________________________________________
void TEveTrackList::SetRnrPoints(Bool_t rnr, TEveElement* el)
{
   // Set rendering of track as points for children of el.

   TEveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<TEveTrack*>(*i);
      if (track)
         if (track->GetRnrPoints() == fRnrPoints)
            track->SetRnrPoints(rnr);
      if (fRecurse)
         SetRnrPoints(rnr, *i);
   }
}

//==============================================================================

//______________________________________________________________________________
void TEveTrackList::SetMainColor(Color_t col)
{
   // Set main (line) color for the list and the elements.

   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      TEveTrack* track = (TEveTrack*)(*i);
      if (track->GetLineColor() == fLineColor)
         track->SetLineColor(col);
      if (fRecurse)
         SetLineColor(col, *i);
   }
   TEveElement::SetMainColor(col);
}

//______________________________________________________________________________
void TEveTrackList::SetLineColor(Color_t col, TEveElement* el)
{
   // Set line color for children of el.

   TEveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<TEveTrack*>(*i);
      if (track && track->GetLineColor() == fLineColor)
         track->SetLineColor(col);
      if (fRecurse)
         SetLineColor(col, *i);
   }
}

//==============================================================================

//______________________________________________________________________________
void TEveTrackList::SetLineWidth(Width_t width)
{
   // Set line width for the list and the elements.

   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      TEveTrack* track = (TEveTrack*)(*i);
      if (track->GetLineWidth() == fLineWidth)
         track->SetLineWidth(width);
      if (fRecurse)
         SetLineWidth(width, *i);
   }
   fLineWidth = width;
}

//______________________________________________________________________________
void TEveTrackList::SetLineWidth(Width_t width, TEveElement* el)
{
   // Set line width for children of el.

   TEveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<TEveTrack*>(*i);
      if (track && track->GetLineWidth() == fLineWidth)
         track->SetLineWidth(width);
      if (fRecurse)
         SetLineWidth(width, *i);
   }
}

//==============================================================================

//______________________________________________________________________________
void TEveTrackList::SetLineStyle(Style_t style)
{
   // Set line style for the list and the elements.

   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      TEveTrack* track = (TEveTrack*)(*i);
      if (track->GetLineStyle() == fLineStyle)
         track->SetLineStyle(style);
      if (fRecurse)
         SetLineStyle(style, *i);
   }
   fLineStyle = style;
}

//______________________________________________________________________________
void TEveTrackList::SetLineStyle(Style_t style, TEveElement* el)
{
   // Set line style for children of el.

   TEveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<TEveTrack*>(*i);
      if (track && track->GetLineStyle() == fLineStyle)
         track->SetLineStyle(style);
      if (fRecurse)
         SetLineStyle(style, *i);
   }
}

//==============================================================================

//______________________________________________________________________________
void TEveTrackList::SetMarkerStyle(Style_t style)
{
   // Set marker style for the list and the elements.

   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      TEveTrack* track = (TEveTrack*)(*i);
      if (track->GetMarkerStyle() == fMarkerStyle)
         track->SetMarkerStyle(style);
      if (fRecurse)
         SetMarkerStyle(style, *i);
   }
   fMarkerStyle = style;
}

//______________________________________________________________________________
void TEveTrackList::SetMarkerStyle(Style_t style, TEveElement* el)
{
   // Set marker style for children of el.

   TEveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<TEveTrack*>(*i);
      if (track && track->GetMarkerStyle() == fMarkerStyle)
         track->SetMarkerStyle(style);
      if (fRecurse)
         SetMarkerStyle(style, *i);
   }
}

//==============================================================================

//______________________________________________________________________________
void TEveTrackList::SetMarkerColor(Color_t col)
{
   // Set marker color for the list and the elements.

   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      TEveTrack* track = (TEveTrack*)(*i);
      if (track->GetMarkerColor() == fMarkerColor)
         track->SetMarkerColor(col);
      if (fRecurse)
         SetMarkerColor(col, *i);
   }
   fMarkerColor = col;
}

//______________________________________________________________________________
void TEveTrackList::SetMarkerColor(Color_t col, TEveElement* el)
{
   // Set marker color for children of el.

   TEveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<TEveTrack*>(*i);
      if (track && track->GetMarkerColor() == fMarkerColor)
         track->SetMarkerColor(col);
      if (fRecurse)
         SetMarkerColor(col, *i);
   }
}

//==============================================================================

//______________________________________________________________________________
void TEveTrackList::SetMarkerSize(Size_t size)
{
   // Set marker size for the list and the elements.

   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      TEveTrack* track = (TEveTrack*)(*i);
      if (track->GetMarkerSize() == fMarkerSize)
         track->SetMarkerSize(size);
      if (fRecurse)
         SetMarkerSize(size, *i);
   }
   fMarkerSize = size;
}

//______________________________________________________________________________
void TEveTrackList::SetMarkerSize(Size_t size, TEveElement* el)
{
   // Set marker size for children of el.

   TEveTrack* track;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      track = dynamic_cast<TEveTrack*>(*i);
      if (track && track->GetMarkerSize() == fMarkerSize)
         track->SetMarkerSize(size);
      if (fRecurse)
         SetMarkerSize(size, *i);
   }
}

//==============================================================================

//______________________________________________________________________________
void TEveTrackList::SelectByPt(Double_t min_pt, Double_t max_pt)
{
   // Select visibility of tracks by transverse momentum.
   // If data-member fRecurse is set, the selection is applied
   // recursively to all children.

   fMinPt = min_pt;
   fMaxPt = max_pt;

   const Double_t minptsq = min_pt*min_pt;
   const Double_t maxptsq = max_pt*max_pt;

   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      const Double_t ptsq = ((TEveTrack*)(*i))->fP.Perp2();
      Bool_t on = ptsq >= minptsq && ptsq <= maxptsq;
      (*i)->SetRnrState(on);
      if (on && fRecurse)
         SelectByPt(min_pt, max_pt, *i);
   }
}

//______________________________________________________________________________
void TEveTrackList::SelectByPt(Double_t min_pt, Double_t max_pt, TEveElement* el)
{
   // Select visibility of el's children tracks by transverse momentum.

   const Double_t minptsq = min_pt*min_pt;
   const Double_t maxptsq = max_pt*max_pt;

   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      TEveTrack* track = dynamic_cast<TEveTrack*>(*i);
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

//_________ ___________________________________________________________________
void TEveTrackList::SelectByP(Double_t min_p, Double_t max_p)
{
   // Select visibility of tracks by momentum.
   // If data-member fRecurse is set, the selection is applied
   // recursively to all children.

   fMinP = min_p;
   fMaxP = max_p;

   const Double_t minpsq = min_p*min_p;
   const Double_t maxpsq = max_p*max_p;

   for (List_i i=BeginChildren(); i!=EndChildren(); ++i)
   {
      const Double_t psq  = ((TEveTrack*)(*i))->fP.Mag2();
      Bool_t on = psq >= minpsq && psq <= maxpsq;
      (*i)->SetRnrState(psq >= minpsq && psq <= maxpsq);
      if (on && fRecurse)
         SelectByP(min_p, max_p, *i);
   }
}

//______________________________________________________________________________
void TEveTrackList::SelectByP(Double_t min_p, Double_t max_p, TEveElement* el)
{
   // Select visibility of el's children tracks by momentum.

   const Double_t minpsq = min_p*min_p;
   const Double_t maxpsq = max_p*max_p;

   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      TEveTrack* track = dynamic_cast<TEveTrack*>(*i);
      if (track)
      {
         const Double_t psq  = ((TEveTrack*)(*i))->fP.Mag2();
         Bool_t on = psq >= minpsq && psq <= maxpsq;
         track->SetRnrState(on);
         if (on && fRecurse)
            SelectByP(min_p, max_p, *i);
      }
   }
}

//==============================================================================

//______________________________________________________________________________
TEveTrack* TEveTrackList::FindTrackByLabel(Int_t label)
{
   // Find track by label, select it and display it in the editor.

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      if (((TEveTrack*)(*i))->GetLabel() == label)
      {
         TGListTree     *lt   = gEve->GetLTEFrame()->GetListTree();
         TGListTreeItem *mlti = lt->GetSelected();
         if (mlti->GetUserData() != this)
            mlti = FindListTreeItem(lt);
         TGListTreeItem *tlti = (*i)->FindListTreeItem(lt, mlti);
         lt->HighlightItem(tlti);
         lt->SetSelected(tlti);
         gEve->EditElement(*i);
         return (TEveTrack*) *i;
      }
   }
   return 0;
}

//______________________________________________________________________________
TEveTrack* TEveTrackList::FindTrackByIndex(Int_t index)
{
   // Find track by index, select it and display it in the editor.

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      if (((TEveTrack*)(*i))->GetIndex() == index)
      {
         TGListTree     *lt   = gEve->GetLTEFrame()->GetListTree();
         TGListTreeItem *mlti = lt->GetSelected();
         if (mlti->GetUserData() != this)
            mlti = FindListTreeItem(lt);
         TGListTreeItem *tlti = (*i)->FindListTreeItem(lt, mlti);
         lt->HighlightItem(tlti);
         lt->SetSelected(tlti);
         gEve->EditElement(*i);
         return (TEveTrack*) *i;
      }
   }
   return 0;
}

//==============================================================================

//______________________________________________________________________________
void TEveTrackList::CopyVizParams(const TEveElement* el)
{
   // Copy visualization parameters from element el.

   const TEveTrackList* m = dynamic_cast<const TEveTrackList*>(el);
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

   TEveElement::CopyVizParams(el);
}

//______________________________________________________________________________
void TEveTrackList::WriteVizParams(std::ostream& out, const TString& var)
{
   // Write visualization parameters.

   TEveElement::WriteVizParams(out, var);

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

//______________________________________________________________________________
TClass* TEveTrackList::ProjectedClass(const TEveProjection*) const
{
   // Virtual from TEveProjectable, returns TEveTrackListProjected class.

   return TEveTrackListProjected::Class();
}
