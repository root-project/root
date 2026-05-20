

#include "TRandom.h"
#include <ROOT/REveElement.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveJetCone.hxx>
#include <ROOT/RWebDisplayArgs.hxx>
#include <ROOT/RWebWindow.hxx>


const Double_t kR_min = 240;
const Double_t kR_max = 250;
const Double_t kZ_d = 300;

using namespace ROOT::Experimental;

void makeJets(int N_Jets, REveElement *jetHolder)
{
   TRandom &r = *gRandom;

   for (int i = 0; i < N_Jets; i++) {
      auto jet = new REveJetCone(Form("Jet_%d", i));
      jet->SetCylinder(2 * kR_max, 2 * kZ_d);
      jet->AddEllipticCone(r.Uniform(-0.5, 0.5), r.Uniform(0, TMath::TwoPi()), 0.1, 0.2);
      jet->SetFillColor(kPink - 8);
      jet->SetLineColor(kViolet - 7);

      jetHolder->AddElement(jet);
   }
}

void single_dark()
{
   auto eveMng = REveManager::Create();
   eveMng->AllowMultipleRemoteConnections(false, false);

   // openui5 theme
   gEnv->SetValue("WebGui.DarkMode", "yes");

   // default viewer, event scene
   eveMng->GetDefaultViewer()->SetBlackBackground(true);
   REveElement *jetHolder = new REveElement("Jets");
   eveMng->GetEventScene()->AddElement(jetHolder);
   makeJets(7, jetHolder);

   // projected view
   auto view = eveMng->SpawnNewViewer("RPhiView", "");
   view->SetBlackBackground(true);
   view->SetCameraType(REveViewer::kCameraOrthoXOY);
   auto eventScene = eveMng->SpawnNewScene("RPZScene");
   view->AddScene(eventScene);
   view->SetMandatory(false);

   auto mngRhoZ = new REveProjectionManager(REveProjection::kPT_RhoZ);
   mngRhoZ->ImportElements(jetHolder, eventScene);

   auto text = new REveText();
   text->SetText("Single View");
   text->SetTextColor(kWhite);
   text->SetPosition(REveVector(0.02, 0.9, 0.2));
   text->SetFontSize(0.05);
   text->SetFont("LiberationSerif-Regular");
   text->SetFillAlpha(228);
   std::string rf_dir = gSystem->ExpandPathName("${ROOTSYS}/fonts/");
   text->AssertSdfFont("LiberationSerif-Regular", rf_dir + "LiberationSerif-Regular.ttf");
   eventScene->AddElement(text);

   // append ?Single=RPhiView"
   std::string url = eveMng->GetWebWindow()->GetUrl();
   url += "?Single=RPhiView";
   std::cout << "Single view URL" << url << "\n";
   ROOT::RWebDisplayArgs args;
   args.SetUrlOpt("Single=RPhiView");

   eveMng->GetWebWindow()->Show();
   eveMng->GetWebWindow()->Show(args);
}
