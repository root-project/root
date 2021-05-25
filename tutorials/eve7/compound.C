#include "TRandom.h"
#include <ROOT/REveManager.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveLine.hxx>
#include <ROOT/REveCompound.hxx>

namespace REX = ROOT::Experimental;

REX::REveLine* random_line(TRandom& rnd, Int_t n, Float_t delta)
{
   auto line = new REX::REveLine;
   line->SetMainColor(kGreen);

   Float_t x = 0, y = 0, z = 0;
   for (Int_t i=0; i<n; ++i) {
      line->SetNextPoint(x, y, z);
      x += rnd.Uniform(0, delta);
      y += rnd.Uniform(0, delta);
      z += rnd.Uniform(0, delta);
   }

   return line;
}

void compound()
{
   // disable browser cache - all scripts and html files will be loaded every time, useful for development
   // gEnv->SetValue("WebGui.HttpMaxAge", 0);

   auto eveMng = REX::REveManager::Create();
   TRandom rnd(0);
   /*
   auto* ml = random_line(rnd, 20, 10);
   ml->SetMainColor(kRed);
   ml->SetLineStyle(2);
   ml->SetLineWidth(3);
   eveMng->InsertVizDBEntry("BigLine", ml);
   */
   auto cmp = new REX::REveCompound;
   cmp->SetMainColor(kGreen);


   cmp->OpenCompound();
   cmp->AddElement(random_line(rnd, 20, 10));
   cmp->AddElement(random_line(rnd, 20, 10));

   auto line = random_line(rnd, 20, 12);
   line->SetMainColor(kRed);
   line->SetLineStyle(2);
   line->SetLineWidth(3);
   //  line->ApplyVizTag("BigLine");
   cmp->AddElement(line);


   cmp->CloseCompound();

   eveMng->GetEventScene()->AddElement(cmp);

   eveMng->Show();
}
