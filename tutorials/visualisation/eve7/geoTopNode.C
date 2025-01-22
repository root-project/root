/// \file
/// \ingroup tutorial_eve_7
///
/// \macro_code
///
#include <ROOT/REveGeoTopNode.hxx>
#include <ROOT/REveManager.hxx>

namespace REX = ROOT::Experimental;

void setDetColors()
{
   gGeoManager->DefaultColors();
   // gGeoManager->GetVolume("TRAK")->InvisibleAll();
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
}

TGeoNode *getNodeFromPath(TGeoNode *top, std::string path)
{
   TGeoNode *node = top;
   std::istringstream f(path);
   std::string s;
   while (getline(f, s, '/'))
      node = node->GetVolume()->FindNode(s.c_str());

   return node;
}

void geoTopNode()
{
   auto eveMng = REX::REveManager::Create();
   // eveMng->AllowMultipleRemoteConnections(false, false);

   TFile::SetCacheFileDir(".");

   TGeoManager::Import("https://root.cern/files/cms.root");
   TGeoNode *top = gGeoManager->GetTopVolume()->FindNode("CMSE_1");
   setDetColors();

   // tracker barrel
   {
      // Init RGeomDescription object
      auto data = new REX::REveGeoTopNodeData("TRACK GeoNode-Data ");
      data->SetTNode(getNodeFromPath(top, "TRAK_1/SVTX_1"));
      data->RefDescription().SetVisLevel(2);
      eveMng->GetWorld()->AddElement(data); // data can be added to any scene

      // 3D GL representation
      auto geoViz = new REX::REveGeoTopNodeViz("CaloTopNode");
      geoViz->SetGeoData(data);
      geoViz->SetPickable(true);
      data->AddNiece(geoViz);
      eveMng->GetEventScene()->AddElement(geoViz);
   }

   // muon barrel
   {
      // Init RGeomDescription object
      auto data = new REX::REveGeoTopNodeData("MUON GeoNode-Data");
      data->SetTNode(getNodeFromPath(top, "MUON_1/MB_1"));
      data->RefDescription().SetVisLevel(2);
      eveMng->GetWorld()->AddElement(data);

      // 3D GL representation
      auto geoViz = new REX::REveGeoTopNodeViz("MUON TopNode");
      geoViz->SetGeoData(data);
      geoViz->SetPickable(true);
      data->AddNiece(geoViz);
      eveMng->GetEventScene()->AddElement(geoViz);
   }
   eveMng->Show();
}
