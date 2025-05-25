/// \file
/// \ingroup tutorial_eve_7
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

   for (Int_t i = 0; i < npoints; ++i)
      ps->SetNextPoint(r.Uniform(-s, s), r.Uniform(-s, s), r.Uniform(-s, s));

   ps->SetMarkerColor(color);
   ps->SetMarkerSize(5 + r.Uniform(1, 15));
   ps->SetMarkerStyle(4);
   return ps;
}

void points()
{
   auto eveMng = REX::REveManager::Create();

   REX::REveElement *event = eveMng->GetEventScene();

   auto psDot = createPointSet(100, 300);
   psDot->SetMarkerStyle(1);
   event->AddElement(psDot);

   auto psSquare = createPointSet(100, 300);
   psSquare->SetMarkerStyle(2);
   event->AddElement(psSquare);

   auto psStar = createPointSet(10, 300);
   psStar->SetMarkerStyle(3);
   event->AddElement(psStar);

   eveMng->Show();
}
