/// \file
/// \ingroup tutorial_eve_7
/// Demonstrates usage of TEveBox class.
///
/// \image html eve_box.png
/// \macro_code
///
/// \author Matevz Tadel
#include <ROOT/REveBox.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveTrackPropagator.hxx>
#include <ROOT/REveTrack.hxx>
#include <ROOT/REveJetCone.hxx>
#include <ROOT/REveText.hxx>

using namespace ROOT::Experimental;
const Double_t kR_min = 240;
const Double_t kR_max = 250;
const Double_t kZ_d = 300;

void makeTracks(int N_Tracks, REveElement *trackHolder)
{
   TRandom &r = *gRandom;
   auto prop = new REveTrackPropagator();
   prop->SetMagFieldObj(new REveMagFieldDuo(350, 3.5, -2.0));
   prop->SetMaxR(300);
   prop->SetMaxZ(600);
   prop->SetMaxOrbs(6);
   // Default is kHelix propagator.
   // prop->SetStepper(REX::REveTrackPropagator::kRungeKutta);

   double v = 0.5;
   double m = 5;

   for (int i = 0; i < N_Tracks; i++) {
      auto p = new TParticle();

      int pdg = 11 * (r.Integer(2) > 0 ? 1 : -1);
      p->SetPdgCode(pdg);

      p->SetProductionVertex(r.Uniform(-v, v), r.Uniform(-v, v), r.Uniform(-v, v), 1);
      p->SetMomentum(r.Uniform(-m, m), r.Uniform(-m, m), r.Uniform(-m, m) * r.Uniform(1, 3), 1);
      auto track = new REveTrack(p, 1, prop);
      track->MakeTrack();
      track->SetMainColor(kBlue);
      track->SetName(Form("RandomTrack_%d", i));
      track->SetLineWidth(3);
      trackHolder->AddElement(track);
   }
}

REveElement *makeBox(Float_t a = 10, Float_t d = 5, Float_t x = 0, Float_t y = 0, Float_t z = 0)
{

   TRandom &r = *gRandom;
   auto b = new REveBox;
   b->SetMainColor(kCyan);
   b->SetMainTransparency(0);

#define RND_BOX(x) r.Uniform(-(x), (x))
   b->SetVertex(0, x - a + RND_BOX(d), y - a + RND_BOX(d), z - a + RND_BOX(d));
   b->SetVertex(1, x - a + RND_BOX(d), y + a + RND_BOX(d), z - a + RND_BOX(d));
   b->SetVertex(2, x + a + RND_BOX(d), y + a + RND_BOX(d), z - a + RND_BOX(d));
   b->SetVertex(3, x + a + RND_BOX(d), y - a + RND_BOX(d), z - a + RND_BOX(d));
   b->SetVertex(4, x - a + RND_BOX(d), y - a + RND_BOX(d), z + a + RND_BOX(d));
   b->SetVertex(5, x - a + RND_BOX(d), y + a + RND_BOX(d), z + a + RND_BOX(d));
   b->SetVertex(6, x + a + RND_BOX(d), y + a + RND_BOX(d), z + a + RND_BOX(d));
   b->SetVertex(7, x + a + RND_BOX(d), y - a + RND_BOX(d), z + a + RND_BOX(d));
#undef RND_BOX

   return b;
}

void makeJets(int N_Jets, REveElement *jetHolder)
{
   TRandom &r = *gRandom;

   for (int i = 0; i < N_Jets; i++) {
      auto jet = new REveJetCone(Form("Jet_%d", i));
      jet->SetCylinder(2 * kR_max, 2 * kZ_d);
      jet->AddEllipticCone(r.Uniform(-0.5, 0.5), r.Uniform(0, TMath::TwoPi()), 0.1, 0.2);
      jet->SetFillColor(kRed);
      jet->SetLineColor(kRed);

      jetHolder->AddElement(jet);
   }
}

void makeTexts(REveElement *textHolder)
{
   {
      auto text = new REveText(Form("Text_0"));
      text->SetMainColor(kViolet);
      REveVector pos(0.5, 0.5, 0.2);
      text->SetPosition(pos);
      text->SetFontSize(0.1);
      text->SetFont(2);
      text->SetText(text->GetCName());
      textHolder->AddElement(text);
   }
}

void overlay_test()
{
   auto gEve = REveManager::Create();

   TRandom &r = *gRandom;

   // create an overlay scene
   REveScene *os = gEve->SpawnNewScene("Overly scene", "OverlayTitle");
   ((REveViewer *)(gEve->GetViewers()->FirstChild()))->AddScene(os);
   os->SetIsOverlay(true);

   makeTexts(os);

   auto jetHolder = new REveElement("jets");
   makeJets(2, jetHolder);
   gEve->GetEventScene()->AddElement(jetHolder);

   auto trackHolder = new REveElement("Tracks");
   gEve->GetEventScene()->AddElement(trackHolder);
   makeTracks(10, trackHolder);

   gEve->Show();
}
