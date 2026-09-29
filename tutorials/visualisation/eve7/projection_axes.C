/// \file
/// \ingroup tutorial_eve_7
/// Scales and tick labels for projected views, using REveProjectionAxis.
///
/// A barrel and a few jets are projected in RhoPhi and RhoZ, and each projected
/// view gets an axis. Ticks sit at round numbers in the original space and are
/// drawn where the projection puts them, so with a non-zero distortion their
/// spacing on screen is uneven. Change the distortion with the <<< and >>>
/// buttons in each view, or from the ROOT prompt:
///
///     pa_distortion(0.001)   // then 0.005, and back to 0
///
/// The browser relayouts the labels on zoom and pan. Only a change of the
/// projection goes back to the server. The axes and buttons are in overlay
/// scenes, which are drawn in front of the geometry.
///
/// \macro_code
///
/// \author Matevz Tadel

#include <ROOT/REveElement.hxx>
#include <ROOT/REveGeoShape.hxx>
#include <ROOT/REveJetCone.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveProjectionAxis.hxx>
#include <ROOT/REveProjectionManager.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveText.hxx>
#include <ROOT/REveViewer.hxx>

#include "TROOT.h"
#include "TColor.h"
#include "TGeoTube.h"
#include "TMath.h"
#include "TRandom.h"

using namespace ROOT::Experimental;

REveProjectionManager *gPaRPhi = nullptr;
REveProjectionManager *gPaRhoZ = nullptr;
/// Ships with ROOT in TROOT::GetDataDir()/fonts.
static const char *kAxisFont = "LiberationSerif-Regular";

REveProjectionAxis *gPaAxisRPhi = nullptr;
REveProjectionAxis *gPaAxisRhoZ = nullptr;

const Double_t kR_min = 240;
const Double_t kR_max = 250;
const Double_t kZ_d = 300;

//------------------------------------------------------------------------------
/// Change the distortion of both projections and rebuild the tick sets.
/// Callable from the ROOT prompt while the macro runs.

void pa_distortion(float d)
{
   REveManager::ChangeGuard ch;
   for (auto mng : {gPaRPhi, gPaRhoZ}) {
      if (!mng) continue;
      mng->GetProjection()->SetDistortion(d);
      mng->UpdateName();
      mng->ProjectChildren();
   }
   // Update the ticks and the read-out, as REveProjectionManager::BumpDistortion()
   // does for the overlay buttons.
   for (auto ax : {gPaAxisRPhi, gPaAxisRhoZ}) {
      if (!ax) continue;
      ax->UpdateTicks();
      ax->UpdateDistortionLabel();
   }
}

//------------------------------------------------------------------------------

static REveElement *makeSceneContent(REveManager *eveMng)
{
   auto holder = new REveElement("Content");

   auto b = new REveGeoShape("Barrel");
   b->SetShape(new TGeoTube(kR_min, kR_max, kZ_d));
   b->SetMainColor(kCyan);
   b->SetMainTransparency(60);
   b->SetNSegments(64);
   holder->AddElement(b);

   TRandom &r = *gRandom;
   for (int i = 0; i < 4; ++i) {
      auto jet = new REveJetCone(Form("Jet_%d", i));
      jet->SetCylinder(2 * kR_max, 2 * kZ_d);
      jet->AddEllipticCone(r.Uniform(-2.5, 2.5), r.Uniform(0, TMath::TwoPi()), r.Uniform(0.05, 0.2),
                           r.Uniform(0.05, 0.25));
      jet->SetFillColor(kPink - 8);
      jet->SetLineColor(kBlack);
      holder->AddElement(jet);
   }

   eveMng->GetEventScene()->AddElement(holder);
   return holder;
}

/// One projected view: a scene for the projected geometry, an overlay scene for
/// its axis, and an orthographic viewer showing both.
static void makeProjectedView(REveManager *eveMng, REveElement *content, REveProjection::EPType_e type,
                              const char *name, REveProjectionManager *&mng, REveProjectionAxis *&axis)
{
   auto scene = eveMng->SpawnNewScene(Form("%s Scene", name), name);
   mng = new REveProjectionManager(type);
   mng->ImportElements(content, scene);

   // Add the manager to the element tree. This gives it an element id, which the
   // overlay buttons need as their MIR target, and streams its name, which shows
   // the distortion, e.g. "RhoPhi (5.0)".
   eveMng->GetWorld()->AddElement(mng);

   auto ovl = eveMng->SpawnNewScene(Form("%s Axis", name), name);
   ovl->SetIsOverlay(true);

   axis = new REveProjectionAxis(mng, Form("%s Axis", name));
   axis->SetFontSize(0.022);
   axis->SetFont(kAxisFont);
   ovl->AddElement(axis);

   // Distortion controls: three overlay texts laid out as [ <<< | value | >>> ].
   // SetClickAction() makes a text a button that sends a MIR to the projection
   // manager. BumpDistortion() reprojects and updates the axes and the read-out.
   {
      auto mkbtn = [&](const char *label, Float_t x, const char *mir) {
         auto b = new REveText(Form("%s %s", name, label), label);
         b->SetText(label);
         b->SetFont(kAxisFont);
         // These views are short, so the controls need a larger font than the
         // tick labels to stay legible, and enough clearance from the bottom
         // edge that the frame is not clipped by the pane.
         b->SetFontSize(0.034);
         b->SetMode(1);                 // relative screen coordinates
         b->SetPosition(REveVector(x, 0.10f, 0.f));
         b->SetTextAlign(REveText::kCenterH, REveText::kBottom);
         b->SetTextColor(TColor::GetColor("#1f2d36"));
         b->SetDrawFrame(true);
         b->SetFillColor(TColor::GetColor("#dfe6ea"));
         b->SetFillAlpha(210);
         b->SetLineColor(TColor::GetColor("#6b8290"));
         b->SetLineAlpha(255);
         b->SetLineWidth(0.06);         // in units of line height
         b->SetExtraBorder(0.18);       // padding, in font-size units
         b->SetResizable(false);        // a control is not a resizable annotation
         if (mir) b->SetClickAction(mir, mng);
         ovl->AddElement(b);
         return b;
      };

      // The x positions are set by hand, since only the client knows the glyph
      // metrics.
      mkbtn("<<<", 0.26f, "BumpDistortion(-1)");
      auto val = mkbtn("0.0", 0.50f, nullptr);   // read-out, not a button
      mkbtn(">>>", 0.74f, "BumpDistortion(1)");

      axis->SetDistortionLabel(val);
   }

   auto view = eveMng->SpawnNewViewer(Form("%s View", name), "");
   view->SetCameraType(REveViewer::kCameraOrthoXOY);
   view->AddScene(scene);
   view->AddScene(ovl);
}

void projection_axes()
{
   auto eveMng = REveManager::Create();
   eveMng->AllowMultipleRemoteConnections(false, false);

   // Generate the SDF atlas from the font file that ships with ROOT.
   std::string rf = std::string(TROOT::GetDataDir().Data()) + "/fonts/";
   REveText::AssertSdfFont(kAxisFont, rf + kAxisFont + ".ttf");

   auto content = makeSceneContent(eveMng);

   makeProjectedView(eveMng, content, REveProjection::kPT_RPhi, "RPhi", gPaRPhi, gPaAxisRPhi);
   makeProjectedView(eveMng, content, REveProjection::kPT_RhoZ, "RhoZ", gPaRhoZ, gPaAxisRhoZ);

   eveMng->Show();
}
