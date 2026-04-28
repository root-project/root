/// \file
/// \ingroup tutorial_eve_7
/// Demonstrates usage of class REveStraightLineSet. The elements in the set can be individually picked when enable
/// secondary select. The REveStraightLineSet is a projectable class. It can be visible in RhoZ and RhoPhi projected
/// views.
///
/// \macro_code
///

#include "TRandom.h"

#include <ROOT/REveElement.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveStraightLineSet.hxx>

namespace REX = ROOT::Experimental;

REX::REveStraightLineSet *makeLineSet(Int_t nlines = 40, Int_t nmarkers = 4, bool sc = true)
{
   TRandom r(0);
   Float_t s = 100;

   auto ls = new REX::REveStraightLineSet();

   for (Int_t i = 0; i < nlines; i++) {
      ls->AddLine(r.Uniform(-s, s), r.Uniform(-s, s), r.Uniform(-s, s), r.Uniform(-s, s), r.Uniform(-s, s),
                  r.Uniform(-s, s));
      // add random number of markers
      Int_t nm = 1 + Int_t(nmarkers * r.Rndm());
      for (Int_t m = 0; m < nm; m++)
         ls->AddMarker(i, r.Rndm());
   }

   ls->SetMarkerSize(5);
   ls->SetMarkerStyle(1);
   ls->SetAlwaysSecSelect(sc);
   REX::gEve->GetEventScene()->AddElement(ls);

   return ls;
}

void lineset(bool secondarySelect = true)
{
   auto eveMng = REX::REveManager::Create();
   // eveMng->AllowMultipleRemoteConnections(false, false);

   auto ls1 = makeLineSet(50, 10, secondarySelect);
   ls1->SetLineWidth(2);
   ls1->SetMarkerSize(8);
   ls1->SetMainColor(kViolet);
   ls1->SetMarkerColor(kMagenta);
   ls1->SetName("LineSet Violet");

   auto ls2 = makeLineSet(10, 10, secondarySelect);
   ls2->SetLineWidth(2);
   ls2->SetMarkerSize(8);
   ls2->SetMainColor(kBlue);
   ls2->SetMarkerColor(kCyan + 2);
   ls2->SetName("LineSet Blue");
   ls2->InitMainTrans();
   ls2->RefMainTrans().Move3LF(40, 200, 200);

   eveMng->Show();
}