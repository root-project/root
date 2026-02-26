/// \file
/// \ingroup tutorial_eve
/// Shows ATLAS geometry.
///
/// \image html eve_geom_atlas.png
/// \macro_code
///
/// \author Matevz Tadel

void geom_atlas()
{
   TEveManager::Create();

   TFile::SetCacheFileDir(".");
   gGeoManager = gEve->GetGeometry("http://root.cern/files/atlas.root");
   gGeoManager->DefaultColors();

   auto node1 = gGeoManager->GetTopVolume()->FindNode("INNE_1");
   TEveGeoTopNode *inn = new TEveGeoTopNode(gGeoManager, node1);
   gEve->AddGlobalElement(inn);

   auto node2 = gGeoManager->GetTopVolume()->FindNode("CENT_1");
   TEveGeoTopNode *cnt = new TEveGeoTopNode(gGeoManager, node2);
   gEve->AddGlobalElement(cnt);

   auto node3 = gGeoManager->GetTopVolume()->FindNode("OUTE_1");
   TEveGeoTopNode *out = new TEveGeoTopNode(gGeoManager, node3);
   gEve->AddGlobalElement(out);

   gEve->FullRedraw3D(kTRUE);

   auto v = gEve->GetDefaultGLViewer();
   v->GetClipSet()->SetClipType(TGLClip::EType::kClipPlane);
   v->RefreshPadEditor(v);

   v->CurrentCamera().RotateRad(-.7, 0.5);
   v->DoDraw();
}
