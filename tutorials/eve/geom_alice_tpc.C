// @(#)root/eve:$Id$
// Author: Matevz Tadel

// Shows geometry of ALICE TPC.

void geom_alice_tpc()
{
   TEveManager::Create();

   gGeoManager = gEve->GetGeometry("http://root.cern.ch/files/alice.root");

   TGeoNode* node = gGeoManager->GetTopVolume()->FindNode("TPC_M_1");
   TEveGeoTopNode* tpc = new TEveGeoTopNode(gGeoManager, node);
   gEve->AddGlobalElement(tpc);

   gEve->Redraw3D(kTRUE);
}
