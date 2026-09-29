/// \file
/// \ingroup tutorial_eve_7
/// Interactive overlay annotations: framed REveText boxes floating in front of
/// the 3D scene, which can be dragged and resized in the browser.
///
/// An overlay scene is a normal REveScene flagged with SetIsOverlay(true). Its
/// contents are drawn by a dedicated orthographic camera into a fixed screen
/// box, after and in front of the 3D scene.
///
/// Coordinates in the overlay box are fractions of the viewport:
///
///     (0,0) = bottom-left corner,  (1,1) = top-right corner
///
/// Font size is a fraction of viewport height. A client with a differently
/// shaped window sees the boxes at the same relative position but covering a
/// different fraction of its width.
///
/// Interaction, once SetPickable(true) is set:
///
///     hover over a box             -> it lightens, and if it is resizable a small
///                                     square grip appears in its bottom-right corner
///     drag the box                 -> move it
///     drag the corner grip         -> resize it (font size and frame together)
///
/// SetPickable() governs moving, SetResizable() governs resizing, so a header can
/// be repositioned while keeping its size. The logo, an REveLogo, moves and
/// resizes the same way.
///
/// Moving and resizing are client-local. Nothing is sent to the server, so two
/// browsers on the same session arrange their annotations independently.
///
/// \macro_code
///
/// \author Matevz Tadel

#include <ROOT/REveElement.hxx>
#include <ROOT/REveGeoShape.hxx>
#include <ROOT/REveJetCone.hxx>
#include <ROOT/REveLogo.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveText.hxx>
#include <ROOT/REveViewer.hxx>

#include "TColor.h"
#include "TGeoTube.h"
#include "TROOT.h"
#include "TMath.h"
#include "TRandom.h"

using namespace ROOT::Experimental;

/// Ships with ROOT in TROOT::GetDataDir()/fonts.
static const char *kOvlFont = "LiberationSerif-Regular";

const Double_t kR_min = 240;
const Double_t kR_max = 250;
const Double_t kZ_d = 300;

//------------------------------------------------------------------------------

/// One framed annotation box. `x`, `y` and `size` are all fractions of the
/// viewport, per the coordinate note above.
REveText *makeAnnotation(REveElement *holder, const char *name, const char *text, float x, float y, float size,
                         Color_t text_color, Color_t frame_color, Color_t fill_color, bool resizable = true)
{
   auto t = new REveText(name);
   t->SetText(text);
   t->SetFont(kOvlFont);
   t->SetFontWeight(0.05f);   // stands in for the bold face ROOT does not ship
   t->SetMode(1); // 1 = screen mode: position is in the (0,1) overlay box
   t->SetFontSize(size);
   t->SetPosition(REveVector(x, y, 0.0));
   t->SetTextColor(text_color);

   // Frame: a filled, outlined plate behind the glyphs. The whole plate is the
   // pick target, so framed text is easier to grab.
   t->SetDrawFrame(true);
   t->SetFillColor(fill_color);
   t->SetFillAlpha(170); // 0..255; translucent so the scene stays visible
   t->SetLineColor(frame_color);
   t->SetLineAlpha(255);
   t->SetLineWidth(0.06);   // in units of line height
   t->SetExtraBorder(0.25); // padding around the text, in font-size units

   // Required for drag and resize: without it the element is not in the picking
   // pass at all and mouse events pass straight through to the camera controls.
   t->SetPickable(true);
   // A non-resizable element can still be moved. The corner grip is drawn only
   // on resizable elements, and only while hovered.
   t->SetResizable(resizable);

   holder->AddElement(t);
   return t;
}

/// A little 3D content to annotate, and to prove the overlay stays in front of it.
void makeSceneContent(REveManager *eveMng)
{
   auto b = new REveGeoShape("Barrel");
   b->SetShape(new TGeoTube(kR_min, kR_max, kZ_d));
   b->SetMainColor(kCyan);
   b->SetMainTransparency(60);
   b->SetNSegments(64);
   eveMng->GetGlobalScene()->AddElement(b);

   TRandom &r = *gRandom;
   auto jets = new REveElement("Jets");
   for (int i = 0; i < 4; ++i) {
      auto jet = new REveJetCone(Form("Jet_%d", i));
      jet->SetCylinder(2 * kR_max, 2 * kZ_d);
      jet->AddEllipticCone(r.Uniform(-2.5, 2.5), r.Uniform(0, TMath::TwoPi()), r.Uniform(0.05, 0.2),
                           r.Uniform(0.05, 0.25));
      jet->SetFillColor(kPink - 8);
      jet->SetLineColor(kBlack);
      jets->AddElement(jet);
   }
   eveMng->GetEventScene()->AddElement(jets);
}

//------------------------------------------------------------------------------

void overlay_drag()
{
   auto eveMng = REveManager::Create();

   // Make the printed URL reusable, so it can be opened in two browser windows.
   // Dragging an annotation in one does not move it in the other. This also
   // accepts non-local clients without a connection key, so use it only for
   // local development.
   eveMng->AllowMultipleRemoteConnections(false, false);

   // Generate the SDF atlas from the font in TROOT::GetDataDir()/fonts. This does
   // nothing once the .png and .js.gz exist. ROOT ships no bold Liberation face,
   // so makeAnnotation() uses SetFontWeight() instead.
   std::string rf = std::string(TROOT::GetDataDir().Data()) + "/fonts/";
   REveText::AssertSdfFont(kOvlFont, rf + kOvlFont + ".ttf");

   makeSceneContent(eveMng);

   // An overlay scene is an ordinary scene, added to a viewer like any other and
   // then flagged as an overlay.
   REveScene *os = eveMng->SpawnNewScene("Overlay scene", "Draggable annotations");
   ((REveViewer *)(eveMng->GetViewers()->FirstChild()))->AddScene(os);
   os->SetIsOverlay(true);

   // The browser fetches images over HTTP, so the directory holding them is
   // registered with SetImageDir(). Any readable directory works. The icons that
   // ship with ROOT are found through TROOT::GetIconPath(), in a build tree and in
   // an installation. See REveLogo::SetFile() for the other image sources.
   REveLogo::SetImageDir(TROOT::GetIconPath().Data());

   auto logo = new REveLogo("Root6Icon.png", "Logo");
   logo->SetPosition(0.88, 0.86);   // (0,1) overlay box; the image is centred here
   logo->SetSize(110);              // height in CSS pixels; width follows the image
   logo->SetOpacity(0.75);          // a watermark: it brightens when hovered
   os->AddElement(logo);

   auto holder = new REveElement("annotations");

   // Muted colours, with a saturated blue frame marking the interactive boxes.
   const Color_t kInk   = TColor::GetColor("#1f2d36"); // deep slate text
   const Color_t kIce   = TColor::GetColor("#f2f7f9"); // plate, a cool white
   const Color_t kQuiet = TColor::GetColor("#8fa6b2"); // frame, static label
   const Color_t kLive  = TColor::GetColor("#3f7d96"); // frame, interactive

   makeAnnotation(holder, "Title", "Run 123456 / Event 42", 0.03, 0.95, 0.035, kInk, kQuiet, kIce, false);
   makeAnnotation(holder, "DragMe", "drag me anywhere", 0.03, 0.85, 0.030, kInk, kLive, kIce);
   makeAnnotation(holder, "ResizeMe", "hover me, then grab the corner", 0.03, 0.76, 0.030, kInk, kLive, kIce);

   // Multi-line: embed newlines in the text. The box grows downwards, one line
   // height per line. The resize grip keeps its single-line size.
   makeAnnotation(holder, "Legend", "legend\n  cyan  barrel\n  pink  jet cones", 0.03, 0.67, 0.030, kInk, kLive,
                  kIce);

   makeAnnotation(holder, "InFront", "always in front of the geometry", 0.34, 0.28, 0.026, kInk, kQuiet, kIce, false);

   os->AddElement(holder);

   eveMng->Show();
}
