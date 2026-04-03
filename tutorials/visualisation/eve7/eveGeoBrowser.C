/// \file
/// \ingroup tutorial_eve_7
///
/// \macro_code
///

#include <ROOT/REveGeoTopNode.hxx>
#include <ROOT/REveGeoPolyShape.hxx>
#include <ROOT/REveManager.hxx>

#include <set>
#include <vector>
#include <iostream>

using namespace ROOT::Experimental;
const Double_t kR_min = 240;
const Double_t kR_max = 250;
const Double_t kZ_d = 300;

void makeJets(int N_Jets, REveElement *jetHolder)
{
   TRandom &r = *gRandom;

   for (int i = 0; i < N_Jets; i++) {
      auto jet = new REveJetCone(Form("Jet_%d", i));
      jet->SetCylinder(2 * kR_max, 2 * kZ_d);
      jet->AddEllipticCone(r.Uniform(-0.5, 0.5), r.Uniform(0, TMath::TwoPi()), 0.1, 0.2);
      jet->SetFillColor(kPink - 8);
      jet->SetLineColor(kViolet - 7);

      jetHolder->AddElement(jet);
   }
}

void eveGeoBrowser()
{
   auto eveMng = REveManager::Create();
   eveMng->AllowMultipleRemoteConnections(false, false);

   TFile::SetCacheFileDir(".");

   // initialize RGeomDesc from TGeoNode
   auto data = new REveGeoTopNodeData("http://xrd-cache-1.t2.ucsd.edu/alja/mail/geo/cmsSimGeo2026.root");
   data->InitPath("/tracker:Tracker_1");
   data->InitPath("/");
   data->RefDescription().SetVisLevel(2);

   // make geoTable
   auto scene = eveMng->SpawnNewScene("GeoSceneTable");
   auto view = eveMng->SpawnNewViewer("GeoTable");
   view->AddScene(scene);
   scene->AddElement(data);

   // 3D EveViz representation
   auto geoViz = new REveGeoTopNodeViz();
   geoViz->SetGeoData(data);
   data->AddNiece(geoViz);
   geoViz->SetPickable(true);

   // add jets for BBox issues
   eveMng->GetEventScene()->AddElement(geoViz);
   REveElement *jetHolder = new REveElement("Jets");
   eveMng->GetEventScene()->AddElement(jetHolder);
   makeJets(7, jetHolder);

   eveMng->Show();
}
