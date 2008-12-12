// @(#)root/eve:$Id$
// Author: Matevz Tadel

// Demonstrates usage of "Default" geometry alias.

void geom_default()
{
   TEveManager::Create();

   gEve->RegisterGeometryAlias("Default", "http://root.cern.ch/files/alice.root");

   gGeoManager = gEve->GetDefaultGeometry();

   TGeoNode* node1 = gGeoManager->GetTopVolume()->FindNode("ITSV_1");
   TEveGeoTopNode* its = new TEveGeoTopNode(gGeoManager, node1);
   gEve->AddGlobalElement(its);

   TGeoNode* node2 = gGeoManager->GetTopVolume()->FindNode("TPC_M_1");
   TEveGeoTopNode* tpc = new TEveGeoTopNode(gGeoManager, node2);
   gEve->AddGlobalElement(tpc);

   gEve->Redraw3D(kTRUE);
}
