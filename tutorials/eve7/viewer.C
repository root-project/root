/// \file
/// \ingroup tutorial_eve7
///  Web-based geometry viewer for CMS geometry
///
/// \macro_code
///
/// \author Sergey Linev

#include <ROOT/REveGeomViewer.hxx>
#include <ROOT/RDirectory.hxx>

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TFile.h"

void viewer()
{
   TFile::SetCacheFileDir(".");

   TGeoManager::Import("https://root.cern/files/cms.root");

   gGeoManager->DefaultColors();
   gGeoManager->SetVisLevel(4);
   gGeoManager->GetVolume("TRAK")->InvisibleAll();
   gGeoManager->GetVolume("HVP2")->SetTransparency(20);
   gGeoManager->GetVolume("HVEQ")->SetTransparency(20);
   gGeoManager->GetVolume("YE4")->SetTransparency(10);
   gGeoManager->GetVolume("YE3")->SetTransparency(20);
   gGeoManager->GetVolume("RB2")->SetTransparency(99);
   gGeoManager->GetVolume("RB3")->SetTransparency(99);
   gGeoManager->GetVolume("COCF")->SetTransparency(99);
   gGeoManager->GetVolume("HEC1")->SetLineColor(7);
   gGeoManager->GetVolume("EAP1")->SetLineColor(7);
   gGeoManager->GetVolume("EAP2")->SetLineColor(7);
   gGeoManager->GetVolume("EAP3")->SetLineColor(7);
   gGeoManager->GetVolume("EAP4")->SetLineColor(7);
   gGeoManager->GetVolume("HTC1")->SetLineColor(2);

   using namespace ROOT::Experimental;

   auto viewer = std::make_shared<REveGeomViewer>(gGeoManager);

   // select volume to draw
   viewer->SelectVolume("CMSE");

   // specify JSROOT draw options - here clipping on X,Y,Z axes
   viewer->SetDrawOptions("clipxyz");

   // set default limits for number of visible nodes and faces
   // when viewer created, initial values exported from TGeoManager
   viewer->SetLimits();

   // start browser
   viewer->Show();

   // add to global heap to avoid immediate destroy of RGeomViewer
   RDirectory::Heap().Add("geom_viewer", viewer);
}
