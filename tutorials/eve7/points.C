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

void points()
{
   auto eveMng = REX::REveManager::Create();

   REX::REveElement *event = eveMng->GetEventScene();
   auto ps = createPointSet(100);
   event->AddElement(ps);

   eveMng->Show();
}
