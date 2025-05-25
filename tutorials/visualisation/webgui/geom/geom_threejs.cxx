/// \file
/// \ingroup tutorial_webgui
/// \ingroup webwidgets
/// The tutorial demonstrates how three.js model for geometry can be created and displayed.
///
/// In geom_threejs.cxx one uses RGeomDescription class from geometry viewer, which produces
/// JSON data with all necessary information. Then RWebWindow is started and this information provided.
/// In client.html one uses **build** function to create Object3D with geometry
/// Then such object placed in three.js scene and rendered. Also simple animation is implemented
///
/// \macro_code
///
/// \author Sergey Linev

#include <ROOT/RGeomData.hxx>

#include <ROOT/RWebWindow.hxx>

std::shared_ptr<ROOT::RWebWindow> window;

TString base64;

void ProcessData(unsigned connid, const std::string &arg)
{
   if (arg == "get") {
      // send arbitrary text message
      window->Send(connid, base64.Data());
   } else if (arg == "halt") {
      // terminate ROOT
      window->TerminateROOT();
   }
}

void geom_threejs()
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

   ROOT::RGeomDescription data;

   data.Build(gGeoManager, "CMSE");

   std::string json = data.ProduceJson();

   base64 = TBase64::Encode(json.c_str());

   // create window
   window = ROOT::RWebWindow::Create();

   // configure default html page
   // either HTML code can be specified or just name of file after 'file:' prefix
   std::string fdir = __FILE__;
   auto pos = fdir.find("geom_threejs.cxx");
   if (pos > 0)
      fdir.resize(pos);
   else
      fdir = gROOT->GetTutorialsDir() + std::string("/visualisation/webgui/geom/");
   window->SetDefaultPage("file:" + fdir + "geom_threejs.html");

   // this is call-back, invoked when message received from client
   window->SetDataCallBack(ProcessData);

   window->SetGeometry(800, 600); // configure predefined geometry

   window->Show();
}
