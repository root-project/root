/// \file
/// \ingroup tutorial_eve
/// Demonstrates usage of class REveStraightLineSet.
///
/// \macro_code
///

#include "TRandom.h"

#include <ROOT/REveElement.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveStraightLineSet.hxx>

//#include <ROOT/REveJetCone.hxx>
namespace REX = ROOT::Experimental;


REX::REveStraightLineSet* makeLineSet(Int_t nlines = 40, Int_t nmarkers = 4)
{
   TRandom r(0);
   Float_t s = 100;

   auto ls = new REX::REveStraightLineSet();

   for (Int_t i = 0; i<nlines; i++) {
      ls->AddLine( r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s),
                   r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));
      // add random number of markers
      Int_t nm = Int_t(nmarkers* r.Rndm());
      for (Int_t m = 0; m < nm; m++) ls->AddMarker(i, r.Rndm());
   }

   ls->SetMarkerSize(0.5);
   ls->SetMarkerStyle(1);
   REX::gEve->GetEventScene()->AddElement(ls);

   return ls;
}

void lineset()
{
   auto eveMng = REX::REveManager::Create();

   auto ls1 = makeLineSet(10, 50);
   ls1->SetMainColor(kViolet);
   ls1->SetName("LineSet_1");

   auto ls2 = makeLineSet(3, 4);
   ls2->SetMainColor(kBlue);
   ls2->SetName("LineSet_2");
   //ls2->InitMainTrans();
   //   ls2->RefMainTrans().Move3LF(40, 100, 100);


   eveMng->Show();
}
