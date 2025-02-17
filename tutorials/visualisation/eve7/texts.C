/// \file
/// \ingroup tutorial_eve_7
///  This example display only texts in web browser
///
/// \macro_code
///
/// \author Waad Fahad

#include "TRandom.h"
#include <ROOT/REveElement.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveText.hxx>
#include <ROOT/REveJetCone.hxx>

namespace REX = ROOT::Experimental;

using namespace ROOT::Experimental;

// 1. Basic fonts
// Text bluprs to choose from:
const char *blurbs[] = {"Love", "Peace", "ROOT", "Code", "Courage", "Quiche"};
const int n_blurbs = sizeof(blurbs) / sizeof(char *);

// Some ROOT fonts are supper old and will error out (arial, times, cour).
const char *fonts[] = {"comic", "comicbd", "verdana", "BlackChancery", "georgia", "georgiai"};
const int n_fonts = sizeof(fonts) / sizeof(char *);

// 2. Fonts with diacritcis and most greek letter available through unicode.
// Not all fonts have them -- most that ship with ROOT don't.
const char *blurbs2[] = {"Čüšék! Šèžëçàgïlá", "Αβρασαξ", "πφηθωμβτ"};
const int n_blurbs2 = sizeof(blurbs2) / sizeof(char *);

const char *fonts2[] = {"LiberationMono-Regular", "LiberationSerif-Regular"};
const int n_fonts2 = sizeof(fonts2) / sizeof(char *);

void makeTexts(int N_Texts, REX::REveElement *textHolder)
{
   const double pi = TMath::Pi();
   const double lim = 300;

   TRandom &r = *gRandom;

   for (int i = 0; i < N_Texts; i++) {
      std::string word, font;
      if (r.Integer(2)) {
         word = blurbs[r.Integer(n_blurbs)];
         font = fonts[r.Integer(n_fonts)];
      } else {
         word = blurbs2[r.Integer(n_blurbs2)];
         font = fonts2[r.Integer(n_fonts2)];
      }

      auto name_text = Form("%s_%d", word.data(), i);
      auto text = new REX::REveText(name_text);
      text->SetText(name_text);

      text->SetFont(font); // Set by name of file in $ROOTSYS/ui5/eve7/fonts/

      int mode = r.Integer(2);
      text->SetMode(mode);
      if (mode == 0) { // world
         auto &t = text->RefMainTrans();
         t.SetRotByAngles(r.Uniform(-pi, pi), r.Uniform(-pi, pi), r.Uniform(-pi, pi));
         t.SetPos(r.Uniform(-lim, lim), r.Uniform(-lim, lim), r.Uniform(-lim, lim));
         text->SetFontSize(r.Uniform(0.01 * lim, 0.2 * lim));
      } else { // screen [0, 0] bottom left, [1, 1] top-right corner, font-size in y-units, x scaled with the window
               // aspect ratio.
         text->SetPosition(REX::REveVector(r.Uniform(-0.1, 0.9), r.Uniform(0.1, 1.1), r.Uniform(0.0, 1.0)));
         text->SetFontSize(r.Uniform(0.001, 0.05));
      }
      text->SetTextColor(
         TColor::GetColor((float)r.Uniform(0, 0.5), (float)r.Uniform(0, 0.5), (float)r.Uniform(0, 0.5)));
      // text->SetMainTransparency();
      // text->SetLineColor(text->GetTextColor());
      text->SetLineColor(
         TColor::GetColor((float)r.Uniform(0, 0.2), (float)r.Uniform(0, 0.2), (float)r.Uniform(0, 0.2)));
      text->SetLineAlpha(192);
      text->SetFillColor(
         TColor::GetColor((float)r.Uniform(0.7, 1.0), (float)r.Uniform(0.7, 1.0), (float)r.Uniform(0.7, 1.0)));
      text->SetFillAlpha(128);
      text->SetDrawFrame(true);
      textHolder->AddElement(text);
   }
}
void makeJets(int N_Jets, REveElement *jetHolder)
{
   TRandom &r = *gRandom;

   const Double_t kR_min = 240;
   const Double_t kR_max = 250;
   const Double_t kZ_d = 300;
   for (int i = 0; i < N_Jets; i++) {
      auto jet = new REveJetCone(Form("Jet_%d", i));
      jet->SetCylinder(2 * kR_max, 2 * kZ_d);
      jet->AddEllipticCone(r.Uniform(-0.5, 0.5), r.Uniform(0, TMath::TwoPi()), 0.1, 0.2);
      jet->SetFillColor(kRed);
      jet->SetLineColor(kRed);

      jetHolder->AddElement(jet);
   }
}

void texts()
{
   auto eveMng = REX::REveManager::Create();
   eveMng->AllowMultipleRemoteConnections(false, false);

   // Initialize SDF fonts.
   // REveManager needs to be already created as location redirect needs to be set up.
   // a) When REveText::AssertSdfFont() is called one of the two default locations
   //    will be chosen, if it is writable by the current user:
   //    - $ROOTSYS/ui5/eve7/sdf-fonts/
   //    - sdf-fonts/ in the current working directory.
   //    If neither location is writable, an error will be issued.
   // b) Alternatively, REveText::SetSdfFontDir(std::string_view dir, bool require_write_access)
   //    can be called to set this directory manually. If the directory is already pre-populated
   //    with fonts one can set the `require_write_access` argument to false to avoid the
   //    requirement of having write access to that directory.

   std::string rf_dir = gSystem->ExpandPathName("${ROOTSYS}/fonts/");
   for (int i = 0; i < n_fonts; ++i) {
      REX::REveText::AssertSdfFont(fonts[i], rf_dir + fonts[i] + ".ttf");
   }
   for (int i = 0; i < n_fonts2; ++i) {
      REX::REveText::AssertSdfFont(fonts2[i], rf_dir + fonts2[i] + ".ttf");
   }

   // add box to overlay
   REX::REveScene *os = eveMng->SpawnNewScene("OverlyScene", "OverlayTitle");
   ((REveViewer *)(eveMng->GetViewers()->FirstChild()))->AddScene(os);
   os->SetIsOverlay(true);

   REX::REveElement *textHolder = new REX::REveElement("texts");
   makeTexts(100, textHolder);
   // os->AddElement(textHolder);
   eveMng->GetEventScene()->AddElement(textHolder);

   auto jetHolder = new REveElement("jets");
   makeJets(2, jetHolder);
   eveMng->GetEventScene()->AddElement(jetHolder);

   eveMng->Show();
}
