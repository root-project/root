#include "ROOT/TEveManager.hxx"
#include "ROOT/TEveDataClasses.hxx"
#include <ROOT/TEveScene.hxx>
#include <ROOT/TEveViewer.hxx>
#include <ROOT/TEveGeoShape.hxx>


#include <ROOT/TEveManager.hxx>
#include "TParticle.h"
#include "TRandom.h"
#include "TSystem.h"

namespace REX = ROOT::Experimental;
REX::TEveManager* eveMng = 0;
std::vector<TParticle> ext_col;



void fill_ext_col(int N)
{
   ext_col.clear();
   ext_col.reserve(N);

   TRandom &r = * gRandom;
   r.SetSeed(0);

   for (int i = 1; i <= N; ++i)
   {
      double pt  = r.Uniform(0.5, 10);
      double eta = r.Uniform(-2.55, 2.55);
      double phi = r.Uniform(0, TMath::TwoPi());

      double px = pt * std::cos(phi);
      double py = pt * std::sin(phi);
      double pz = pt * (1. / (std::tan(2*std::atan(std::exp(-eta)))));

      // printf("%2d: pt=%.2f, eta=%.2f, phi=%.2f\n", i, pt, eta, phi);

      ext_col.push_back
         (TParticle(0, 0, 0, 0, 0, 0,
                    px, py, pz, std::sqrt(px*px + py*py + pz*pz + 80*80),
                    0, 0, 0, 0 ));

      int pdg = 11 * (r.Integer(2) > 0 ? 1 : -1);
      ext_col.back().SetPdgCode(pdg);
   }
}

void makeTableScene( REX::TEveDataCollection* col)
{

   auto tbl = new REX::TEveDataTable();

   tbl->SetCollection(col);

   {
      auto c = new REX::TEveDataColumn("pt");
      tbl->AddElement(c);
      c->SetExpressionAndType("std::abs(i.Pt())", REX::TEveDataColumn::FT_Double);
   }
   {
      auto c = new REX::TEveDataColumn("phi");
      tbl->AddElement(c);
      c->SetExpressionAndType("i.Phi()", REX::TEveDataColumn::FT_Double);
      c->SetPrecision(3);
   }

   {
      auto c = new REX::TEveDataColumn("eta");
      tbl->AddElement(c);
      c->SetExpressionAndType("i.Eta()", REX::TEveDataColumn::FT_Double);
      c->SetPrecision(3);
   }
   
   auto scene  = eveMng->SpawnNewScene("Table","Table");
   scene->AddElement(tbl);
   auto view   = eveMng->SpawnNewViewer("Table", "Table");
   view->AddScene(scene);
}

void table()
{
   namespace REX = ROOT::Experimental;

   gSystem->Load("libROOTEve.so");
   eveMng = REX::TEveManager::Create();
   
   REX::TEveElement* defaultViewer = (*eveMng->GetViewers()->BeginChildren());
   defaultViewer->SetRnrSelf(false);

   fill_ext_col(100);
   
   auto col = new REX::TEveDataCollection();

   col->SetItemClass(TParticle::Class());

   {
      int i = 1;
      for (auto &p : ext_col)
      {
         TString pname; pname.Form("Particle %2d", i++);

         col->AddItem(&p, pname.Data(), "");
      }
   }
   col->SetFilterExpr("i.Pt() > 1 && std::abs(i.Eta()) < 1");
   col->ApplyFilter();
   eveMng->GetWorld()->AddElement(col);

   makeTableScene(col);
}
