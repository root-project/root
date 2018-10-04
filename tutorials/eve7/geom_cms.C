R__LOAD_LIBRARY(libROOTEve);

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoMatrix.h"
#include "TSystem.h"
#include "TFile.h"
#include <ROOT/REveGeoShape.hxx>
#include <ROOT/REveTrans.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveViewer.hxx>
#include <ROOT/REveElement.hxx>
#include <ROOT/REveManager.hxx>


namespace REX = ROOT::Experimental;
void geom_cms()
{
   gSystem->Load("libROOTEve");
   gSystem->Load("libGeom");
   auto eveMng = REX::REveManager::Create();

   TFile::SetCacheFileDir(".");
   
   auto geoManager = eveMng->GetGeometry("http://root.cern.ch/files/cms.root");
   TGeoVolume* top = geoManager->GetTopVolume()->FindNode("CMSE_1")->GetVolume();
   auto node1 = top->FindNode("MUON_1");
   auto muonvol = node1->GetVolume();
   
   if (0) {
      auto gs = node1->GetVolume()->GetShape();
      auto b1 = new REX::REveGeoShape("Muon");
      b1->SetShape(gs);
      b1->SetMainColor(kBlue);
      eveMng->GetGlobalScene()->AddElement(b1);
   }

   {
      auto n = muonvol->FindNode("MB_1");
      auto gs = n->GetVolume()->GetShape();
      auto b1 = new REX::REveGeoShape("MuonBarrel");
      b1->SetShape(gs);
      b1->SetMainColor(kBlue);
      eveMng->GetGlobalScene()->AddElement(b1);


      auto bv =  n->GetVolume();
      for (int i = 1; i < 5; ++i ) {
         auto n = bv->FindNode(Form("MBXC_%d",i));
         auto gss = n->GetVolume()->GetShape();
         auto b1s = new REX::REveGeoShape(Form("Arc %d", i));
         b1s->InitMainTrans();
         const double* move = n->GetMatrix()->GetTranslation();
         b1s->RefMainTrans().SetFrom( *(n->GetMatrix()));
         b1s->SetShape(gss);
         b1s->SetMainColor(kCyan);
         eveMng->GetGlobalScene()->AddElement(b1s);
      }
   }
}
