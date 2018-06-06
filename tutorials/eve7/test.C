/// \file
/// \ingroup tutorial_http
///  This program creates and fills one and two dimensional histogram
///  Macro used to demonstrate usage of custom HTML page in custom.htm
///  One can use plain JavaScript to assign different actions with HTML buttons
///
/// \macro_code
///



#include <vector>
#include <string>
#include <iostream>
#include <sstream>

#include "TROOT.h"
#include "TSystem.h"
#include "TRandom.h"
#include "TFile.h"
#include "TGeoBBox.h"
#include "TGeoTube.h"
#include "TParticle.h"

#include <ROOT/TEveGeoShape.hxx>
#include <ROOT/TEveScene.hxx>
#include <ROOT/TEveViewer.hxx>
#include <ROOT/TEveElement.hxx>
#include <ROOT/TEveManager.hxx>
#include <ROOT/TEveProjectionManager.hxx>
#include <ROOT/TEveProjectionBases.hxx>
#include <ROOT/TEvePointSet.hxx>
#include <ROOT/TEveJetCone.hxx>

#include <ROOT/TEveTrack.hxx>
#include <ROOT/TEveTrackPropagator.hxx>

namespace REX = ROOT::Experimental;

using RenderData = REX::RenderData;

// globals
REX::TEveManager* eveMng = 0;
REX::TEveProjectionManager* mngRhoPhi = 0;
REX::TEveProjectionManager* mngRhoZ = 0;

const Double_t kR_min = 240;
const Double_t kR_max = 250;
const Double_t kZ_d   = 300;

REX::TEvePointSet* getPointSet(int npoints = 2, float s=2, int color=28)
{
   TRandom r(0);
   REX::TEvePointSet* ps = new REX::TEvePointSet("fu", npoints);

   for (Int_t i=0; i<npoints; ++i)
         ps->SetNextPoint(r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));
   
   ps->SetMarkerColor(color);
   ps->SetMarkerSize(3+r.Uniform(1, 2));
   ps->SetMarkerStyle(4);
   return ps;
}

void addPoints()
{
   REX::TEveElement* event = eveMng->GetEventScene();
   REX::TEveElement* pntHolder = new REX::TEveElementList("Hits");
   auto ps1 = getPointSet(20, 100);
   ps1->SetElementName("Points_1");
   pntHolder->AddElement(ps1);
   /*
   auto ps2 = getPointSet(10, 200, 4);
   ps2->SetElementName("Points_2");
   pntHolder->AddElement(ps2);
   */
   event->AddElement(pntHolder);
}

void addTracks()
{
   REX::TEveElement* event = eveMng->GetEventScene();
   auto prop = new REX::TEveTrackPropagator();
   prop->SetMagFieldObj(new REX::TEveMagFieldDuo(350, -3.5, 2.0));
   prop->SetMaxR(300);
   prop->SetMaxZ(600);
   prop->SetMaxOrbs(6);
   REX::TEveElement* trackHolder = new REX::TEveElementList("Tracks");
   if (1)   {
      TParticle* p = new TParticle();p->SetPdgCode(11);
      p->SetProductionVertex(0.068, 0.2401, -0.07629, 1);
      p->SetMomentum(4.82895, 2.35083, -0.611757, 1);
      auto track = new REX::TEveTrack(p, 1, prop);
      track->MakeTrack();
      track->SetMainColor(kBlue);
      track->SetElementName("TestTrack_1");
      trackHolder->AddElement(track);
   }

   if (1) {
      TParticle* p = new TParticle(); p->SetPdgCode(11);
      p->SetProductionVertex(0.068, 0.2401, -0.07629, 1);
       p->SetMomentum(-0.82895, 0.83, -1.1757, 1);
      auto track = new REX::TEveTrack(p, 1, prop);
      track->MakeTrack();
      track->SetMainColor(kBlue);
      track->SetElementName("TestTrack_2");
      trackHolder->AddElement(track);
   }
   {
      double v = 0.5;
      double m = 5;
      TRandom r(0);
      for (int i = 0; i < 10; i++)
      {
         TParticle* p = new TParticle(); p->SetPdgCode(11);

         p->SetProductionVertex(r.Uniform(-v,v), r.Uniform(-v,v), r.Uniform(-v,v), 1);
         p->SetMomentum(r.Uniform(-m,m), r.Uniform(-m,m), r.Uniform(-m,m)*r.Uniform(1, 3), 1);
         auto track = new REX::TEveTrack(p, 1, prop);
         track->MakeTrack();
         track->SetMainColor(kBlue);
         track->SetElementName(Form("RandomTrack_%d",i ));
         trackHolder->AddElement(track);
      }
   }
   event->AddElement(trackHolder);
}

void addJets()
{
   REX::TEveElement* event = eveMng->GetEventScene();
   auto jetHolder = new REX::TEveElementList("Jets");
   {
      auto jet = new REX::TEveJetCone("Jet_1");
      jet->SetCylinder(2*kR_max, 2*kZ_d);
      jet->AddEllipticCone(0.7, 1, 0.1, 0.3);

      jetHolder->AddElement(jet);
   }
   event->AddElement(jetHolder);
}

void makeEventScene()
{
   addPoints();
   addTracks();
   addJets();
}

void makeGeometryScene()
{
   auto b1 = new REX::TEveGeoShape("Barrel 1");
   b1->SetShape(new TGeoTube(kR_min, kR_max, kZ_d));
   b1->SetMainColor(kCyan);
   eveMng->GetGlobalScene()->AddElement(b1);
}


void projectScenes()
{
   // project RhoPhi
   auto rPhiGeomScene  = eveMng->SpawnNewScene("RPhi Geometry","RPhi");
   auto rPhiEventScene = eveMng->SpawnNewScene("RPhi Event Data","RPhi");

   mngRhoPhi = new REX::TEveProjectionManager(REX::TEveProjection::kPT_RPhi);      
   mngRhoPhi->ImportElements(eveMng->GetGlobalScene(),rPhiGeomScene );    
   mngRhoPhi->ImportElements(eveMng->GetEventScene(), rPhiEventScene);
   
   auto rphiView = eveMng->SpawnNewViewer("RPhi View", "");
   rphiView->AddScene(rPhiGeomScene);
   rphiView->AddScene(rPhiEventScene);

   return;
   // project rhoZ
   auto rhoZGeomScene  = eveMng->SpawnNewScene("RhoZ Geometry", "RhoZ");
   auto rhoZEventScene = eveMng->SpawnNewScene("RhoZ Event Data","RhoZ");

   mngRhoZ = new REX::TEveProjectionManager(REX::TEveProjection::kPT_RhoZ); 
   mngRhoZ->ImportElements(REX::gEve->GetGlobalScene(),rhoZGeomScene);
   mngRhoZ->ImportElements(REX::gEve->GetEventScene(), rhoZEventScene);
   
   auto rhoZView = eveMng->SpawnNewViewer("RhoZ View", "");
   rhoZView->AddScene(rhoZGeomScene);
   rhoZView->AddScene(rhoZEventScene);
}

void test()
{
   gSystem->Load("libROOTEve");
   eveMng = REX::TEveManager::Create();

   makeGeometryScene();
   makeEventScene();

   projectScenes();
}
