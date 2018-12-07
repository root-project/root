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



void lineset(Int_t nlines = 40, Int_t nmarkers = 4)
{
   auto eveMng = REX::REveManager::Create();

   TRandom r(0);
   Float_t s = 100;

   auto ls = new REX::REveStraightLineSet();
   ls->SetMainColor(kBlue);
   ls->SetMarkerColor(kRed);

   for (Int_t i = 0; i<nlines; i++) {
      ls->AddLine( r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s),
                   r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));
      // add random number of markers
      Int_t nm = Int_t(nmarkers* r.Rndm());
      for (Int_t m = 0; m < nm; m++) ls->AddMarker(i, r.Rndm());
   }

   ls->SetMarkerSize(1.5);
   ls->SetMarkerStyle(4);
   eveMng->GetEventScene()->AddElement(ls);
   
   
   eveMng->Show();
}
