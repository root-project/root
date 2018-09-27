#include "TRandom.h"
#include "TSystem.h"
#include <ROOT/TEveElement.hxx>
#include <ROOT/TEveScene.hxx>
#include <ROOT/TEveManager.hxx>
#include <ROOT/TEvePointSet.hxx>

namespace REX = ROOT::Experimental;

REX::TEvePointSet* getPointSet(int npoints = 2, float s=2, int color=28)
{
   TRandom &r = *gRandom;

   REX::TEvePointSet* ps = new REX::TEvePointSet("MyTestPoints", npoints);

   for (Int_t i=0; i<npoints; ++i) {
      ps->SetNextPoint(r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));
   }
   
   ps->SetMarkerColor(color);
   ps->SetMarkerSize(3+r.Uniform(1, 2));
   ps->SetMarkerStyle(4);
   return ps;
}

void points(bool mapNewWindow = true)
{
   gSystem->Load("libROOTEve");
   auto eveMng = REX::TEveManager::Create();
   
   REX::TEveElement* event = eveMng->GetEventScene();
   auto ps = getPointSet(100);
   event->AddElement(ps);
}
