// @(#)root/eve:$Id$
// Author: Matevz Tadel

// Demonstates usage of geometry aliases - merge ALICE ITS with ATLAS MUON.

void geom_alias()
{
   TEveManager::Create();

   gEve->RegisterGeometryAlias("ALICE", "http://root.cern.ch/files/alice.root");
   gEve->RegisterGeometryAlias("ATLAS", "http://root.cern.ch/files/atlas.root");


   gGeoManager = gEve->GetGeometryByAlias("ALICE");

   TGeoNode* node1 = gGeoManager->GetTopVolume()->FindNode("ITSV_1");
   TEveGeoTopNode* its = new TEveGeoTopNode(gGeoManager, node1);
   gEve->AddGlobalElement(its);


   gGeoManager = gEve->GetGeometryByAlias("ATLAS");

   TGeoNode* node2 = gGeoManager->GetTopVolume()->FindNode("OUTE_1");
   TEveGeoTopNode* atlas = new TEveGeoTopNode(gGeoManager, node2);
   gEve->AddGlobalElement(atlas);


   gEve->FullRedraw3D(kTRUE);

   // EClipType not exported to CINT (see TGLUtil.h):
   // 0 - no clip, 1 - clip plane, 2 - clip box
   TGLViewer *v = gEve->GetDefaultGLViewer();
   v->GetClipSet()->SetClipType(2);
   v->RefreshPadEditor(v);

   v->CurrentCamera().RotateRad(-0.5, -2.4);
   v->DoDraw();
}
