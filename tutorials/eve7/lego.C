/// \file
/// \ingroup tutorial_eve7
///  This example display only points in web browser
///
/// \macro_code
///

#include "TRandom.h"
#include <ROOT/REveElement.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REvePointSet.hxx>

namespace REX = ROOT::Experimental;

REX::REvePointSet *createPointSet(int npoints = 2, float s = 2, int color = 28)
{
   TRandom &r = *gRandom;

   REX::REvePointSet *ps = new REX::REvePointSet("MyTestPoints", "list of eve points", npoints);

   for (Int_t i=0; i < npoints; ++i)
      ps->SetNextPoint(r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));

   ps->SetMarkerColor(color);
   ps->SetMarkerSize(3+r.Uniform(1, 2));
   ps->SetMarkerStyle(4);
   return ps;
}

void lego()
{
   auto eveMng = REX::REveManager::Create();

   // disable default view
   eveMng->GetViewers()->FirstChild()->SetRnrSelf(false);

   auto scene = eveMng->SpawnNewScene("Lego", "LegoView");
   auto view = eveMng->SpawnNewViewer("Lego", "");
   view->AddScene(scene);

   auto ps = createPointSet(100);
   scene->AddElement(ps);

   {
      // gROOT->SetBatch();

      // TPad *p = new TPad("LegoPad", "Lego Pad Tit", 0, 0, 1, 1);
      TPad *p = new TCanvas("LegoPad", "Lego Pad Tit", 800,400);
      p->SetMargin(0, 0, 0, 0);

      // *** Simple TH2
      /*
      TH2F* h = new TH2F("Booboo","exampul",128,-5,5,64,-2.5,2.5);
      TRandom r;
      for(int i=0;i<1000000;++i) {
         h->Fill(r.Gaus() - 2, r.Gaus());
         h->Fill(r.Gaus() + 2, r.Gaus());
      }
      for(int i=0;i<6000;++i) {
         h->Fill(0.1*r.Gaus() - 2, 0.1*r.Gaus());
         h->Fill(0.1*r.Gaus() + 2, 0.1*r.Gaus());
      }
      // h->Draw("LEGO2");
      p->GetListOfPrimitives()->Add(h, "LEGO2");
      p->Modified(kTRUE);
      */

      // *** Load std CMS calo demo
      const char* histFile = "http://amraktad.web.cern.ch/amraktad/cms_calo_hist.root";
      TFile::SetCacheFileDir(".");
      auto hf = TFile::Open(histFile, "CACHEREAD");
      auto ecalHist = (TH2F*)hf->Get("ecalLego");
      auto hcalHist = (TH2F*)hf->Get("hcalLego");

      THStack *s = new THStack("LegoStack", ""); // "ECal undt HCal");
      ecalHist->SetFillColor(kRed);
      ecalHist->GetXaxis()->SetLabelSize(1);
      ecalHist->GetXaxis()->SetTitle(reinterpret_cast<const char *>(u8"\u03B7")); // "#eta");
      ecalHist->GetYaxis()->SetLabelSize(1);
      ecalHist->GetYaxis()->SetTitle(reinterpret_cast<const char *>(u8"\u03C6")); // "#varphi");
      ecalHist->GetZaxis()->SetLabelSize(1);
      s->Add(ecalHist);
      hcalHist->SetFillColor(kBlue);
      s->Add(hcalHist);
      p->GetListOfPrimitives()->Add(s);

      TGraph2D *line = new TGraph2D(200);
      for (int i = 0; i < 200; ++i)
         line->SetPoint(i, std::cos(i*0.1), std::sin(i*0.1), i*0.25);
      line->SetLineWidth(5);
      line->SetLineColor(kCyan - 2);
      p->GetListOfPrimitives()->Add(line, "LINE");

      p->Modified(kTRUE);

      TString json( TBufferJSON::ToJSON(p) );

      ps->SetTitle(TBase64::Encode(json).Data());

      s->Draw();
   }

   eveMng->Show();
}
