/// \file
/// \ingroup tutorial_webgui
///  Web-based geometry viewer for CMS geometry
///
/// \macro_code
///
/// \author Sergey Linev

#include <ROOT/RGeomViewer.hxx>

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TFile.h"

void web_cms(bool split = false)
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

   auto viewer = std::make_shared<ROOT::RGeomViewer>(gGeoManager);

   // select volume to draw
   viewer->SelectVolume("CMSE");

   // specify JSROOT draw options - here clipping on X,Y,Z axes
   viewer->SetDrawOptions("clipxyz");

   // set default limits for number of visible nodes and faces
   // when viewer created, initial values exported from TGeoManager
   viewer->SetLimits();

   viewer->SetShowHierarchy(!split);

   // start web browser
   viewer->Show();

   // destroy viewer only when connection to client is closed
   viewer->ClearOnClose(viewer);

   if (split) {
      // create separate widget with geometry hierarchy only
      auto hier = std::make_shared<ROOT::RGeomHierarchy>(viewer->Description());

      // start web browser with hierarchy
      hier->Show();

      // destroy widget only when connection to client is closed
      hier->ClearOnClose(hier);
   }
}
