// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveProjectionAxis.hxx>
#include <ROOT/REveProjectionManager.hxx>
#include <ROOT/REveProjections.hxx>
#include <ROOT/REveRenderData.hxx>

#include "THLimitsFinder.h"
#include "TMath.h"
#include "TString.h"

#include <cmath>

using namespace ROOT::Experimental;

/** \class REveProjectionAxis
\ingroup REve
Scales and tick labels for a projected view, the REve counterpart of
TEveProjectionAxes.

In kValue mode the ticks sit at round numbers in the original space and are
placed at their projected positions, so under a non-linear projection their
screen spacing is uneven. The projection exists only on the server, so the
mapping from original to projected coordinates is done here and the ticks are
streamed in projected coordinates. The client maps them to the screen, which
for the orthographic camera of a projected view is affine.

The tick set covers fRangeFactor times the projection manager's extent. The
client discards ticks outside the view and labels that overlap, so zooming and
panning need no server round trip. Ticks are recomputed only when the
projection or the scene extent changes.

The projection manager is held as an aunt and is not owned.

Inherits REveText for its style: font, size, hinting, weight and text colour
apply to the labels, and the line colour to the ticks. fText, fPosition,
fMode, fResizable, the alignment and the frame are not used.
*/

namespace {

/// Format a tick value with enough decimals to separate ticks `step` apart;
/// exponent form outside [1e-4, 1e5).
std::string FormatTickLabel(Double_t v, Double_t step)
{
   if (std::fabs(v) < 1e-12) return "0";

   Int_t nd = 0;
   if (step > 0.0 && step < 1.0)
      nd = TMath::Min(6, (Int_t)std::ceil(-std::log10(step)));

   // Very large or very small numbers read better in exponent form.
   Double_t a = std::fabs(v);
   if (a >= 1e5 || a < 1e-4)
      return TString::Format("%.1e", v).Data();

   return TString::Format("%.*f", nd, v).Data();
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// Constructor. Registers as a niece of `m`. ProjectChildren() visits the
/// nieces but reprojects only REveProjected ones, so the axis is skipped.

REveProjectionAxis::REveProjectionAxis(REveProjectionManager *m, const Text_t *n, const Text_t *t)
   : REveText(n, t), fManager(m)
{
   // Style defaults for tick labels rather than a free-standing text box.
   SetMode(1);          // not read by makeProjectionAxis
   // Larger and slightly bolder than the REveText default, for labels a few
   // pixels tall in a small projected pane.
   SetFontSize(0.028f);
   SetFontWeight(0.06f);
   SetDrawFrame(kFALSE);
   SetTextAlign(kCenterH, kTop);

   if (fManager)
      fManager->AddNiece(this);

   UpdateTicks();
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor. The aunt link is dropped by REveElement's own cleanup, which
/// calls RemoveNieceInternal() on every aunt, so nothing to do here.

REveProjectionAxis::~REveProjectionAxis()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Recompute both tick sets. Call after the projection or the scene extent
/// changes; camera motion needs no update.

void REveProjectionAxis::UpdateTicks()
{
   BuildTicks(0);
   BuildTicks(1);
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Build the tick set for one screen axis, 0 horizontal and 1 vertical. The
/// range is the projection manager's bounding box widened by fRangeFactor.

void REveProjectionAxis::BuildTicks(Int_t ax)
{
   fTicks[ax].clear();

   if (!fManager) return;
   REveProjection *proj = fManager->GetProjection();
   if (!proj) return;

   fManager->AssertBBox();
   Float_t *bb = fManager->GetBBox();
   if (!bb) return;

   Float_t pmin_t = bb[ax * 2], pmax_t = bb[ax * 2 + 1];
   if (pmax_t <= pmin_t) return;

   // The tick step comes from the true range and only the extent from the
   // widened one. A step derived from the widened range can jump to a coarser
   // round number and leave fewer labels inside the view.
   Float_t center = 0.5f * (pmin_t + pmax_t);
   Float_t half = 0.5f * (pmax_t - pmin_t) * fRangeFactor;
   Float_t pmin = center - half;
   Float_t pmax = center + half;

   // Decoded as in TEveProjectionAxes: primary = n / 100, secondary = n % 100.
   Int_t n1a = TMath::FloorNint(fNdivisions / 100);
   Int_t n2a = fNdivisions - n1a * 100;
   Int_t bn1, bn2;
   Double_t bw1, bw2;
   Double_t bl1 = 0, bh1 = 0, bl2 = 0, bh2 = 0;

   if (fLabMode == kValue) {
      // Round numbers in the original space, placed where the projection puts
      // them. Their screen spacing is uneven under a non-linear projection.
      Float_t v1 = proj->GetValForScreenPos(ax, pmin_t);
      Float_t v2 = proj->GetValForScreenPos(ax, pmax_t);
      if (v2 <= v1) return;

      // Step from the true range (see above), extent from the widened one.
      THLimitsFinder::Optimize(v1, v2, n1a, bl1, bh1, bn1, bw1);
      THLimitsFinder::Optimize(bl1, bl1 + bw1, n2a, bl2, bh2, bn2, bw2);
      if (bw1 <= 0) return;

      Double_t vc = 0.5 * (v1 + v2), vh = 0.5 * (v2 - v1) * fRangeFactor;
      Int_t k1 = TMath::FloorNint((vc - vh - bl1) / bw1);
      Int_t k2 = TMath::CeilNint((vc + vh - bl1) / bw1);

      // Cached for the cheap per-tick form of GetScreenVal().
      REveVector dirVec;
      proj->SetDirectionalVector(ax, dirVec);
      REveVector oCenter;
      proj->GetOrthogonalCenter(ax, oCenter);

      for (Int_t l = k1; l <= k2; ++l) {
         Double_t v = bl1 + l * bw1;
         Tick_t major;
         major.fPos = proj->GetScreenVal(ax, v);
         major.fLabel = FormatTickLabel(v, bw1);
         major.fMajor = kTRUE;
         fTicks[ax].push_back(major);

         for (Int_t k = 1; k < bn2; ++k) {
            Tick_t minor;
            minor.fPos = proj->GetScreenVal(ax, v + k * bw2, dirVec, oCenter);
            minor.fMajor = kFALSE;
            fTicks[ax].push_back(minor);
         }
      }
   } else {
      // Even spacing in projected space; the labels are then irregular values.
      // Step from the true range here too, for the same reason.
      THLimitsFinder::Optimize(pmin_t, pmax_t, n1a, bl1, bh1, bn1, bw1);
      THLimitsFinder::Optimize(bl1, bl1 + bw1, n2a, bl2, bh2, bn2, bw2);
      if (bw1 <= 0) return;

      Int_t k1 = TMath::CeilNint(pmin / bw1);
      Int_t k2 = TMath::FloorNint(pmax / bw1);

      for (Int_t l = k1; l <= k2; ++l) {
         Double_t p = l * bw1;

         Tick_t major;
         major.fPos = p;
         major.fLabel = FormatTickLabel(proj->GetValForScreenPos(ax, p), bw1);
         major.fMajor = kTRUE;
         fTicks[ax].push_back(major);

         for (Int_t k = 1; k < bn2; ++k) {
            Double_t pm = p + k * bw2;
            if (pm > pmax) break;
            Tick_t minor;
            minor.fPos = pm;
            minor.fMajor = kFALSE;
            fTicks[ax].push_back(minor);
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Stream style, mode and the tick sets. Positions are in projected
/// coordinates; the client maps them to screen with its own camera.

Int_t REveProjectionAxis::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   Int_t ret = REveText::WriteCoreJson(j, rnr_offset);

   j["fLabMode"] = (int)fLabMode;
   j["fAxesMode"] = (int)fAxesMode;
   j["fUseFgColor"] = fUseFgColor;
   j["fDrawCenter"] = fDrawCenter;
   j["fDrawOrigin"] = fDrawOrigin;

   for (int ax = 0; ax < 2; ++ax) {
      nlohmann::json pos = nlohmann::json::array();
      nlohmann::json lab = nlohmann::json::array();
      nlohmann::json maj = nlohmann::json::array();
      for (auto &t : fTicks[ax]) {
         pos.push_back(t.fPos);
         lab.push_back(t.fLabel);
         maj.push_back(t.fMajor);
      }
      std::string key = (ax == 0) ? "H" : "V";
      j["fTickPos" + key] = pos;
      j["fTickLab" + key] = lab;
      j["fTickMaj" + key] = maj;
   }

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Render data. The ticks travel as JSON above, since the labels are strings;
/// this only names the client-side factory.

void REveProjectionAxis::BuildRenderData()
{
   fRenderData = std::make_unique<REveRenderData>("makeProjectionAxis");
   REveElement::BuildRenderData();
   fRenderData->PushV(0.f, 0.f, 0.f); // keep the buffer non-empty
}

////////////////////////////////////////////////////////////////////////////////
/// Attach a text element that shows the current distortion. Only its text is
/// rewritten, so it can sit in any scene. The label is not owned and its
/// lifetime is not tracked.

void REveProjectionAxis::SetDistortionLabel(REveText *t)
{
   fDistLabel = t;
   UpdateDistortionLabel();
}

////////////////////////////////////////////////////////////////////////////////
/// Forget the projection manager when it is destroyed, then clear any click
/// target as REveText does.

void REveProjectionAxis::RemoveAunt(REveAunt *au)
{
   if (fManager && au == fManager)
      fManager = nullptr;
   REveText::RemoveAunt(au);
}

////////////////////////////////////////////////////////////////////////////////
/// Rewrite the read-out. Shown as distortion * 1000, the same scaling
/// REveProjectionManager uses in its own name.

void REveProjectionAxis::UpdateDistortionLabel()
{
   if (!fDistLabel || !fManager) return;
   REveProjection *proj = fManager->GetProjection();
   if (!proj) return;

   fDistLabel->SetText(TString::Format("%.1f", proj->GetDistortion() * 1000).Data());
}
