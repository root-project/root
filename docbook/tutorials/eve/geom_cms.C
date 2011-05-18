// @(#)root/eve:$Id$
// Author: Matevz Tadel

// Shows CMS geometry.

void geom_cms()
{
   TEveManager::Create();

   TFile::SetCacheFileDir(".");
   gGeoManager = gEve->GetGeometry("http://root.cern.ch/files/cms.root");
   gGeoManager->DefaultColors();

   TGeoVolume* top = gGeoManager->GetTopVolume()->FindNode("CMSE_1")->GetVolume();

   TEveGeoTopNode* trk = new TEveGeoTopNode(gGeoManager, top->FindNode("TRAK_1"));
   trk->SetVisLevel(6);
   gEve->AddGlobalElement(trk);

   TEveGeoTopNode* calo = new TEveGeoTopNode(gGeoManager, top->FindNode("CALO_1"));
   calo->SetVisLevel(3);
   gEve->AddGlobalElement(calo);

   TEveGeoTopNode* muon = new TEveGeoTopNode(gGeoManager, top->FindNode("MUON_1"));
   muon->SetVisLevel(4);
   gEve->AddGlobalElement(muon);

   gEve->FullRedraw3D(kTRUE);

   // EClipType not exported to CINT (see TGLUtil.h):
   // 0 - no clip, 1 - clip plane, 2 - clip box
   TGLViewer *v = gEve->GetDefaultGLViewer();
   v->GetClipSet()->SetClipType(1);
   v->ColorSet().Background().SetColor(kMagenta+4);
   v->SetGuideState(TGLUtil::kAxesEdge, kTRUE, kFALSE, 0);
   v->RefreshPadEditor(v);

   v->CurrentCamera().RotateRad(-1.2, 0.5);
   v->DoDraw();
}
