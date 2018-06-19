#include "ROOT/TEveManager.hxx"
#include "ROOT/TEveDataClasses.hxx"
#include <ROOT/TEveTrack.hxx>
#include <ROOT/TEveTrackPropagator.hxx>
#include <ROOT/TEveScene.hxx>
#include <ROOT/TEveViewer.hxx>
#include <ROOT/TEveGeoShape.hxx>


#include "TParticle.h"
#include "TRandom.h"
#include "TSystem.h"
#include "TGeoTube.h"

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
      double pt  = r.Uniform(0.5, 20);
      double eta = r.Uniform(-2.55, 2.55);
      double phi = r.Uniform(0, TMath::TwoPi());

      double px = pt * std::cos(phi);
      double py = pt * std::sin(phi);
      double pz = pt * (1. / (std::tan(2*std::atan(std::exp(-eta)))));

      printf("%2d: pt=%.2f, eta=%.2f, phi=%.2f\n", i, pt, eta, phi);

      ext_col.push_back
         (TParticle(0, 0, 0, 0, 0, 0,
                    px, py, pz, std::sqrt(px*px + py*py + pz*pz + 80*80),
                    0, 0, 0, 0 ));
   }
}

void makeGeometryScene()
{
   auto b1 = new REX::TEveGeoShape("Barrel 1");
   b1->SetShape(new TGeoTube(100, 100, 100));
   b1->SetMainColor(kCyan);
   eveMng->GetGlobalScene()->AddElement(b1);
}


void makeEventScene(REX::TEveDataCollection* col)
{
   REX::TEveElement* event = eveMng->GetEventScene();
   
   auto prop = new REX::TEveTrackPropagator();
   prop->SetMagFieldObj(new REX::TEveMagFieldDuo(350, -3.5, 2.0));
   prop->SetMaxR(300);
   prop->SetMaxZ(600);
   prop->SetMaxOrbs(6);
   REX::TEveElement* trackHolder = new REX::TEveElementList("Tracks");

      int i = 0;
      for (auto &p : ext_col)
      {
         TString pname; pname.Form("Particle %2d", i);

         auto track = new REX::TEveTrack(&p, 1, prop);
          track->SetMainColor(kBlue);
          track->MakeTrack();
         track->SetElementName(Form("RandomTrack_%d",i ));
         track->SetRnrSelf(col->GetDataItem(i)->GetFiltered());
         trackHolder->AddElement(track);
         i++;
      }
   event->AddElement(trackHolder);
}

void makeTableScene( REX::TEveDataCollection* col)
{
   // --------------------------------

   auto tbl = new REX::TEveDataTable();

   tbl->SetCollection(col);

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
   {
   auto c2 = new REX::TEveDataColumn("is_central");
   tbl->AddElement(c2);
   c2->SetExpressionAndType("std::abs(i.Eta()) < 1.0", REX::TEveDataColumn::FT_Bool);
   }
   // tbl->PrintTable();

   auto scene  = eveMng->SpawnNewScene("Table","Table");
   scene->AddElement(tbl);
   auto view   = eveMng->SpawnNewViewer("Table", "Table");
   view->AddScene(scene);
}

void test_data_classes()
{
   namespace REX = ROOT::Experimental;

   gSystem->Load("libROOTEve.so");
   eveMng = REX::TEveManager::Create();
   

   fill_ext_col(10);

   
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

   // --------------------------------
   
   makeGeometryScene();
   makeEventScene(col);
   makeTableScene(col);
}
