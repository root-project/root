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

#include "TClass.h"
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
REX::TEveProjectionManager* mngRhoZ   = 0;
REX::TEveScene  *rPhiGeomScene = 0, *rPhiEventScene = 0;
REX::TEveScene  *rhoZGeomScene = 0, *rhoZEventScene = 0;
REX::TEveViewer *rphiView = 0;
REX::TEveViewer *rhoZView = 0;

const Double_t kR_min = 240;
const Double_t kR_max = 250;
const Double_t kZ_d   = 300;

const Int_t N_Tracks =   40;
const Int_t N_Jets   =   20;


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
   TRandom &r = *gRandom;

   REX::TEveElement* event = eveMng->GetEventScene();
   auto prop = new REX::TEveTrackPropagator();
   prop->SetMagFieldObj(new REX::TEveMagFieldDuo(350, -3.5, 2.0));
   prop->SetMaxR(300);
   prop->SetMaxZ(600);
   prop->SetMaxOrbs(6);
   REX::TEveElement* trackHolder = new REX::TEveElementList("Tracks");

   double v = 0.5;
   double m = 5;

   for (int i = 0; i < N_Tracks; i++)
   {
      TParticle* p = new TParticle();

      int pdg = 11* (r.Integer(2) -1);
      p->SetPdgCode(pdg);

      p->SetProductionVertex(r.Uniform(-v,v), r.Uniform(-v,v), r.Uniform(-v,v), 1);
      p->SetMomentum(r.Uniform(-m,m), r.Uniform(-m,m), r.Uniform(-m,m)*r.Uniform(1, 3), 1);
      auto track = new REX::TEveTrack(p, 1, prop);
      track->MakeTrack();
      track->SetMainColor(kBlue);
      track->SetElementName(Form("RandomTrack_%d",i ));
      trackHolder->AddElement(track);
   }

   event->AddElement(trackHolder);
}

void addJets()
{
   TRandom &r = *gRandom;

   REX::TEveElement* event = eveMng->GetEventScene();
   auto jetHolder = new REX::TEveElementList("Jets");

   for (int i = 0; i < N_Jets; i++)
   {
      auto jet = new REX::TEveJetCone("Jet_1");
      jet->SetCylinder(2*kR_max, 2*kZ_d);
      jet->AddEllipticCone(r.Uniform(-3.5, 3.5), r.Uniform(0, TMath::TwoPi()),
                           r.Uniform(0.02, 0.2), r.Uniform(0.02, 0.3));
      jet->SetFillColor(kPink - 8);
      jet->SetLineColor(kViolet - 7);

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

   // Debug of surface fill in RPhi (index buffer screwed).
   // b1->SetNSegments(3);
   b1->SetNSegments(40);
}


void createProjectionStuff()
{
   // project RhoPhi
   rPhiGeomScene  = eveMng->SpawnNewScene("RPhi Geometry","RPhi");
   rPhiEventScene = eveMng->SpawnNewScene("RPhi Event Data","RPhi");

   mngRhoPhi = new REX::TEveProjectionManager(REX::TEveProjection::kPT_RPhi);

   rphiView = eveMng->SpawnNewViewer("RPhi View", "");
   rphiView->AddScene(rPhiGeomScene);
   rphiView->AddScene(rPhiEventScene);

   // ----------------------------------------------------------------

   rhoZGeomScene  = eveMng->SpawnNewScene("RhoZ Geometry", "RhoZ");
   rhoZEventScene = eveMng->SpawnNewScene("RhoZ Event Data","RhoZ");

   mngRhoZ = new REX::TEveProjectionManager(REX::TEveProjection::kPT_RhoZ); 

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
         mngRhoPhi->ImportElements(ie, rPhiGeomScene);
         mngRhoZ  ->ImportElements(ie, rhoZGeomScene);
      }
   }
   if (eventp)
   {
      for (auto & ie : eveMng->GetEventScene()->RefChildren())
      {
         mngRhoPhi->ImportElements(ie, rPhiEventScene);
         mngRhoZ  ->ImportElements(ie, rhoZEventScene);
      }
   }

   // auto t0 = eveMng->GetEventScene()->FindChild("Tracks")->FirstChild();
   // printf("t0=%p, %s %s\n", t0, t0->GetElementName(), t0->IsA()->GetName());
   // dynamic_cast<REX::TEveTrack*>(t0)->Print("all");

   // auto t1 = rPhiEventScene->FindChild("Tracks [P]")->FirstChild();
   // printf("t1=%p, %s %s\n", t1, t1->GetElementName(), t1->IsA()->GetName());
   // dynamic_cast<REX::TEveTrack*>(t1)->Print("all");
}

//==============================================================================

#pragma link C++ class EventManager+;

class EventManager : public REX::TEveElementList
{
public:
   EventManager(){ }
   void NextEvent()
   {
      printf("NEXT EVENT \n");

      TEveElement::List_t ev_scenes;
      ev_scenes.push_back(rPhiEventScene);
      ev_scenes.push_back(rhoZEventScene);
      ev_scenes.push_back(eveMng->GetEventScene());

      eveMng->DestroyElementsOf(ev_scenes);

      makeEventScene();

      projectScenes(false, true);

      eveMng->BroadcastElementsOf(ev_scenes);
   }
   ClassDef(EventManager, 1);
};

void test()
{
   gRandom->SetSeed(0);

   gSystem->Load("libROOTEve");
   eveMng = REX::TEveManager::Create();

   auto eventMng = new EventManager();
   eventMng->SetElementName("EventManager");
   eveMng->GetWorld()->AddElement(eventMng);
   
   makeGeometryScene();
   makeEventScene();

   createProjectionStuff();

   projectScenes(true, true);
}
