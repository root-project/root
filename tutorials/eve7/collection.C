#include "ROOT/TEveManager.hxx"
#include "ROOT/TEveDataClasses.hxx"
#include <ROOT/TEveTrack.hxx>
#include <ROOT/TEveTrackPropagator.hxx>
#include <ROOT/TEveScene.hxx>
#include <ROOT/TEveViewer.hxx>
#include <ROOT/TEveGeoShape.hxx>
#include <ROOT/TEveJetCone.hxx>


#include "TParticle.h"
#include "TRandom.h"
#include "TSystem.h"
#include "TGeoTube.h"

const Double_t kR_min = 240;
const Double_t kR_max = 250;
const Double_t kZ_d   = 300;
namespace REX = ROOT::Experimental;
REX::TEveManager* eveMng = 0;
std::vector<TParticle> ext_col;


REX::TEvePointSet* getPointSet(int npoints = 2, float s=2, int color=28)
{
   TRandom &r = *gRandom;

   REX::TEvePointSet* ps = new REX::TEvePointSet("fu", npoints);

   for (Int_t i=0; i<npoints; ++i)
      ps->SetNextPoint(r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));
   
   ps->SetMarkerColor(color);
   ps->SetMarkerSize(3+r.Uniform(1, 2));
   ps->SetMarkerStyle(4);
   return ps;
}

void addJets()
{
   TRandom &r = *gRandom;

   REX::TEveElement* event = eveMng->GetEventScene();
   auto jetHolder = new REX::TEveElementList("Jets");
   int  N_Jets = 5;
   for (int i = 0; i < N_Jets; i++)
   {
      auto jet = new REX::TEveJetCone("Jet_1");
      jet->SetCylinder(2*kR_max, 2*kZ_d);
      jet->AddEllipticCone(r.Uniform(-3.5, 3.5), r.Uniform(0, TMath::TwoPi()),
                           r.Uniform(0.02, 0.2), r.Uniform(0.02, 0.3));
      jet->SetFillColor(kYellow);
      jet->SetLineColor(kYellow - 7);

      jetHolder->AddElement(jet);
   }
   event->AddElement(jetHolder);
}

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

void makeGeometryScene()
{
   auto b1 = new REX::TEveGeoShape("Barrel 1");
   float s = 0.2;
   b1->SetShape(new TGeoTube(kR_min*s, kR_max*s, kZ_d*s));
   b1->SetFillColor(kGray);
   eveMng->GetGlobalScene()->AddElement(b1);
   
   auto b2 = new REX::TEveGeoShape("Barell 2");
   b2->SetShape(new TGeoTube(kR_min, kR_max, kZ_d));
   b2->SetFillColor(kGray);
   b2->SetMainTransparency(80);
   eveMng->GetGlobalScene()->AddElement(b2);
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
      track->SetMainColor(kBlue+2);
      track->MakeTrack();

          
      track->SetElementName(Form("RandomTrack_%d",i ));
      track->SetRnrSelf(!col->GetDataItem(i)->GetFiltered());
      printf("track %s [filtered %d]: eta= %.2f, pt=%.2f\n",  track->GetElementName(),col->GetDataItem(i)->GetFiltered(), p.Eta(), p.Phi());
      trackHolder->AddElement(track);
      i++;
   }
   event->AddElement(trackHolder);
   
   // points
   {
      auto ps1 = getPointSet(20, 40);
      ps1->SetElementName("Points_1");
      ps1->SetMainColor(kRed);
      event->AddElement(ps1);
   }
   {
      auto ps1 = getPointSet(100, 10);
      ps1->SetElementName("Points_2");
      ps1->SetMainColor(kCyan);
      event->AddElement(ps1);

   }
   addJets();
}

void makeTableScene( REX::TEveDataCollection* col)
{
   // --------------------------------

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
   }/*
      {
      auto c2 = new REX::TEveDataColumn("is_central");
      tbl->AddElement(c2);
      c2->SetExpressionAndType("std::abs(i.Eta()) < 1.0", REX::TEveDataColumn::FT_Bool);
      }*/
   // tbl->PrintTable();

   auto scene  = eveMng->SpawnNewScene("Table","Table");
   scene->AddElement(tbl);
   auto view   = eveMng->SpawnNewViewer("Table", "Table");
   view->AddScene(scene);
}

void collection()
{
   namespace REX = ROOT::Experimental;

   gSystem->Load("libROOTEve.so");
   eveMng = REX::TEveManager::Create();
   

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

   // --------------------------------
   
   makeGeometryScene();
   makeEventScene(col);
   makeTableScene(col);
}
