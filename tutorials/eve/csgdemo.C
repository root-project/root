/// \file
/// \ingroup tutorial_eve
/// Combinatorial Solid Geometry example
///
/// Stripped down to demonstrate EVE shape-extracts.
/// 1. `Run root csgdemo.C`
///    This will produce csg.root containing the extract.
/// 2. Display the assembly as:
///    `root show_extract.C("csg.root")`
///
/// \image html eve_csgdemo.png
/// \macro_code
///
/// \author Andrei Gheata

#include "TGeoManager.h"

//____________________________________________________________________________
void csgdemo ()
{
   gSystem->Load("libGeom");

   auto c = new TCanvas("composite shape", "A * B - C");
   c->Iconify();

   if (gGeoManager) delete gGeoManager;

   new TGeoManager("xtru", "poza12");
   auto mat = new TGeoMaterial("Al", 26.98,13,2.7);
   auto med = new TGeoMedium("MED",1,mat);
   auto top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);

   // define shape components with names
   auto box  = new TGeoBBox("box", 20., 20., 20.);
   auto box1 = new TGeoBBox("box1", 5., 5., 5.);
   auto sph  = new TGeoSphere("sph", 5., 25.);
   auto sph1 = new TGeoSphere("sph1", 1., 15.);
   // create the composite shape based on a Boolean expression
   auto tr  = new TGeoTranslation(0., 30., 0.);
   auto tr1 = new TGeoTranslation(0., 40., 0.);
   auto tr2 = new TGeoTranslation(0., 30., 0.);
   auto tr3 = new TGeoTranslation(0., 30., 0.);
   tr->SetName("tr");
   tr1->SetName("tr1");
   tr2->SetName("tr2");
   tr3->SetName("tr3");
   // register all used transformations
   tr->RegisterYourself();
   tr1->RegisterYourself();
   tr2->RegisterYourself();
   tr3->RegisterYourself();

   TGeoCompositeShape *cs = new TGeoCompositeShape
      ("mir", "(sph * box) + (sph1:tr - box1:tr1)");

   auto vol = new TGeoVolume("COMP4", cs);
   vol->SetLineColor(kMagenta);
   top->AddNode(vol,1);
   gGeoManager->CloseGeometry();
   top->Draw();

   gGeoManager->SetNsegments(40);
   TEveGeoNode::SetCSGExportNSeg(40);

   TGLFaceSet::SetEnforceTriangles(kTRUE);
   TEveManager::Create();

   auto node = gGeoManager->GetTopNode();
   auto en = new TEveGeoTopNode(gGeoManager, node);
   en->SetVisLevel(4);
   en->GetNode()->GetVolume()->SetVisibility(kFALSE);

   gEve->AddGlobalElement(en);

   gEve->Redraw3D(kTRUE);

   en->ExpandIntoListTreesRecursively();
   en->SaveExtract("csg.root", "CSG Demo", kFALSE);
}
