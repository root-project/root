/// \file
/// \ingroup tutorial_eve
/// Combinatorial Solid Geometry example
///
/// Stripped down to demonstrate EVE shape-extracts.
/// 1. `Run root csgdemo.C`
///    This will produce csg.root containing the extract.
/// 2. Display the assebly as:
///    `root show_extract.C("csg.root")`
///
/// \image html eve_csgdemo.png
/// \macro_code
///
/// \author Andrei Gheata

#include "TSystem.h"

#include "TGeoManager.h"
#include "TGeoCompositeShape.h"
#include "TGeoSphere.h"

#include <ROOT/REveManager.hxx>
#include <ROOT/REveGeoShapeExtract.hxx>
#include <ROOT/REveGeoShape.hxx>
#include <ROOT/REveGeoPolyShape.hxx>

R__LOAD_LIBRARY(libGeom);

namespace REX = ROOT::Experimental;

REX::REveGeoPolyShape *eve_pshape = nullptr;
REX::REveGeoShape     *eve_shape  = nullptr;

//____________________________________________________________________________
void csgdemo ()
{
   //TCanvas *c = new TCanvas("composite shape", "A * B - C");
   // c->Iconify();

   if (gGeoManager) delete gGeoManager;

   new TGeoManager("xtru", "poza12");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium   *med = new TGeoMedium("MED",1,mat);
   TGeoVolume   *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);

   // define shape components with names
   TGeoBBox   *box  = new TGeoBBox("box", 20., 20., 20.);
   TGeoBBox   *box1 = new TGeoBBox("box1", 5., 5., 5.);
   TGeoSphere *sph  = new TGeoSphere("sph", 5., 25.);
   TGeoSphere *sph1 = new TGeoSphere("sph1", 1., 15.);
   // create the composite shape based on a Boolean expression
   TGeoTranslation *tr  = new TGeoTranslation(0., 30., 0.);
   TGeoTranslation *tr1 = new TGeoTranslation(0., 40., 0.);
   TGeoTranslation *tr2 = new TGeoTranslation(0., 30., 0.);
   TGeoTranslation *tr3 = new TGeoTranslation(0., 30., 0.);
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

   TGeoVolume *vol = new TGeoVolume("COMP4", cs);
   vol->SetLineColor(kMagenta);
   top->AddNode(vol,1);
   gGeoManager->CloseGeometry();

   // To draw original
   // gGeoManager->SetNsegments(80);
   // top->Draw("ogl");

   REX::REveManager::Create();

   REX::REveGeoPolyShape::SetAutoEnforceTriangles(true);

   auto node = gGeoManager->GetTopNode();
   auto geo_cshape = dynamic_cast<TGeoCompositeShape*>(node->GetDaughter(0)->GetVolume()->GetShape());

   if ( ! geo_cshape) throw std::runtime_error("The first vshape is not a CSG shape.");

   bool poly_first = false;
   if (poly_first)
   {
      eve_pshape = new REX::REveGeoPolyShape;
      eve_pshape.BuildFromComposite(geo_cshape, 40);

      eve_shape = new REX::REveGeoShape("CSG_Result");
      eve_shape->SetShape(eve_pshape);
   }
   else
   {
      eve_shape = new REX::REveGeoShape("CSG_Result");
      eve_shape->SetNSegments(40);
      eve_shape->SetShape(geo_cshape);

      eve_pshape = dynamic_cast<REX::REveGeoPolyShape*>(eve_shape->GetShape());
   }
   eve_shape->SetMainColor(kMagenta);

   // If one doesn't enable triangles globally, one can do it on per shape basis:
   // eve_pshape->EnforceTriangles();

   eve_pshape->Draw("ogl");

   eve_shape->SaveExtract("csg.root", "CSG Demo");
}
