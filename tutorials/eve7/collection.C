/// \file
/// \ingroup tutorial_eve7
///  This example display collection of ??? in web browser
///
/// \macro_code
///

#include <ROOT/REvePointSet.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveJetCone.hxx>
#include <ROOT/REveGeoShape.hxx>
#include "ROOT/REveDataClasses.hxx"
#include <ROOT/REveTrack.hxx>
#include <ROOT/REveTrackPropagator.hxx>
#include <ROOT/REveViewer.hxx>
#include <ROOT/REveProjectionManager.hxx>
#include "ROOT/REveManager.hxx"

#include "TParticle.h"
#include "TRandom.h"
#include "TGeoTube.h"

namespace REX = ROOT::Experimental;

const Double_t kR_min = 240;
const Double_t kR_max = 250;
const Double_t kZ_d   = 300;
REX::REveManager *eveMng = nullptr;
std::vector<TParticle> ext_col;

REX::REveProjectionManager *mngRhoPhi = nullptr;
REX::REveProjectionManager *mngRhoZ   = nullptr;
REX::REveScene  *rPhiGeomScene = nullptr, *rPhiEventScene = nullptr;
REX::REveScene  *rhoZGeomScene = nullptr, *rhoZEventScene = nullptr;
REX::REveViewer *rphiView = nullptr;
REX::REveViewer *rhoZView = nullptr;

REX::REvePointSet *getPointSet(int npoints = 2, float s=2, int color=28)
{
   TRandom &r = *gRandom;

   REX::REvePointSet* ps = new REX::REvePointSet("fu", npoints);

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

   REX::REveElement* event = eveMng->GetEventScene();
   auto jetHolder = new REX::REveElementList("Jets");
   int  N_Jets = 5;
   for (int i = 0; i < N_Jets; i++)
   {
      auto jet = new REX::REveJetCone("Jet_1");
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
   auto b1 = new REX::REveGeoShape("Barrel 1");
   float s = 0.2;
   b1->SetShape(new TGeoTube(kR_min*s, kR_max*s, kZ_d*s));
   b1->SetFillColor(kGray);
   eveMng->GetGlobalScene()->AddElement(b1);

   auto b2 = new REX::REveGeoShape("Barell 2");
   b2->SetShape(new TGeoTube(kR_min, kR_max, kZ_d));
   b2->SetFillColor(kGray);
   b2->SetMainTransparency(80);
   eveMng->GetGlobalScene()->AddElement(b2);
}


void makeEventScene(REX::REveDataCollection* col)
{
   REX::REveElement* event = eveMng->GetEventScene();

   auto prop = new REX::REveTrackPropagator();
   prop->SetMagFieldObj(new REX::REveMagFieldDuo(350, -3.5, 2.0));
   prop->SetMaxR(300);
   prop->SetMaxZ(600);
   prop->SetMaxOrbs(0.5);
   REX::REveElement* trackHolder = new REX::REveElementList("Tracks");

   int i = 0;
   for (auto &p : ext_col)
   {
      TString pname; pname.Form("Particle %2d", i+1);

      auto track = new REX::REveTrack(&p, 1, prop);
      track->SetMainColor(kBlue+2);
      track->MakeTrack();


      track->SetElementName(pname.Data());
      track->SetRnrSelf(!col->GetDataItem(i)->GetFiltered());
      printf("track %s [filtered %d/%d]: eta= %.2f, pt=%.2f \n",  track->GetElementName(),col->GetDataItem(i)->GetFiltered(), track->GetRnrSelf(), p.Eta(), p.Phi());
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

void makeTableScene( REX::REveDataCollection* col)
{
   // --------------------------------

   auto tbl = new REX::REveDataTable();

   tbl->SetCollection(col);

   {
      auto c = new REX::REveDataColumn("pt");
      tbl->AddElement(c);
      c->SetExpressionAndType("std::abs(i.Pt())", REX::REveDataColumn::FT_Double);
   }

   {
      auto c = new REX::REveDataColumn("phi");
      tbl->AddElement(c);
      c->SetExpressionAndType("i.Phi()", REX::REveDataColumn::FT_Double);
      c->SetPrecision(3);
   }

   {
      auto c = new REX::REveDataColumn("eta");
      tbl->AddElement(c);
      c->SetExpressionAndType("i.Eta()", REX::REveDataColumn::FT_Double);
      c->SetPrecision(3);
   }/*
      {
      auto c2 = new REX::REveDataColumn("is_central");
      tbl->AddElement(c2);
      c2->SetExpressionAndType("std::abs(i.Eta()) < 1.0", REX::REveDataColumn::FT_Bool);
      }*/
   // tbl->PrintTable();

   auto scene  = eveMng->SpawnNewScene("Table","Table");
   scene->AddElement(tbl);
   auto view   = eveMng->SpawnNewViewer("Table", "Table");
   view->AddScene(scene);
}


void createProjectionStuff()
{
   /*
   // project RhoPhi
   rPhiGeomScene  = eveMng->SpawnNewScene("RPhi Geometry","RPhi");
   rPhiEventScene = eveMng->SpawnNewScene("RPhi Event Data","RPhi");

   mngRhoPhi = new REX::REveProjectionManager(REX::REveProjection::kPT_RPhi);

   rphiView = eveMng->SpawnNewViewer("RPhi View", "");
   rphiView->AddScene(rPhiGeomScene);
   rphiView->AddScene(rPhiEventScene);

   // ----------------------------------------------------------------
   */
   rhoZGeomScene  = eveMng->SpawnNewScene("RhoZ Geometry", "RhoZ");
   rhoZEventScene = eveMng->SpawnNewScene("RhoZ Event Data","RhoZ");

   mngRhoZ = new REX::REveProjectionManager(REX::REveProjection::kPT_RhoZ);

   rhoZView = eveMng->SpawnNewViewer("RhoZ View", "");
   rhoZView->AddScene(rhoZGeomScene);
   rhoZView->AddScene(rhoZEventScene);
}

void projectScenes(bool geomp, bool eventp)
{
   if (geomp)
   {
      for (auto & ie : eveMng->GetGlobalScene()->RefChildren())
      {
         //  mngRhoPhi->ImportElements(ie, rPhiGeomScene);
         mngRhoZ  ->ImportElements(ie, rhoZGeomScene);
      }
   }
   if (eventp)
   {
      for (auto & ie : eveMng->GetEventScene()->RefChildren())
      {
         //  mngRhoPhi->ImportElements(ie, rPhiEventScene);
         mngRhoZ  ->ImportElements(ie, rhoZEventScene);
      }
   }
}

void collection()
{
   eveMng = REX::REveManager::Create();

   fill_ext_col(100);

   auto col = new REX::REveDataCollection();

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
   createProjectionStuff();
   projectScenes(true, true);
   makeTableScene(col);

   eveMng->Show();
}
