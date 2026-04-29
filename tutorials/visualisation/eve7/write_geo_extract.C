
#include <ROOT/REveGeoTopNode.hxx>
#include <ROOT/REveGeoPolyShape.hxx>
#include <ROOT/REveManager.hxx>
#include "TFile.h"

#include <set>
#include <vector>
#include <iostream>


// \file
/// \ingroup tutorial_eve7
/// Helper script to create REveGeoShapeExtract fro TGeo geometry
/// One can rely on GeoTable to access paths
/// \macro_code
///
/// \author Alja Mrak Tadel

using namespace ROOT::Experimental;

REveGeoShape *makeShape(const std::string &targetPath, const char* name)
{
   // for the moment top node is tracker at the startup

   TGeoIterator next(gGeoManager->GetTopVolume());
   TGeoNode *currentNode;

   while ((currentNode = next())) {
      TString currentPath;
      next.GetPath(currentPath); // Retrieves the full hierarchy path
      // std::cout << "compare" << currentPath << "\n";
      if (currentPath == targetPath) {
         // Found the node via path matching
         printf("Node found: %s\n", currentNode->GetName());
         break;
      }
   }

   const TGeoMatrix *mat = next.GetCurrentMatrix();
   const Double_t *t = mat->GetTranslation();    // size 3
   const Double_t *r = mat->GetRotationMatrix(); // size 9 (3x3)

   Double_t m[16];
   if (mat->IsScale()) {
      const Double_t *s = mat->GetScale();
      m[0] = r[0] * s[0];
      m[1] = r[3] * s[0];
      m[2] = r[6] * s[0];
      m[3] = 0;
      m[4] = r[1] * s[1];
      m[5] = r[4] * s[1];
      m[6] = r[7] * s[1];
      m[7] = 0;
      m[8] = r[2] * s[2];
      m[9] = r[5] * s[2];
      m[10] = r[8] * s[2];
      m[11] = 0;
      m[12] = t[0];
      m[13] = t[1];
      m[14] = t[2];
      m[15] = 1;
   } else {
      m[0] = r[0];
      m[1] = r[3];
      m[2] = r[6];
      m[3] = 0;
      m[4] = r[1];
      m[5] = r[4];
      m[6] = r[7];
      m[7] = 0;
      m[8] = r[2];
      m[9] = r[5];
      m[10] = r[8];
      m[11] = 0;
      m[12] = t[0];
      m[13] = t[1];
      m[14] = t[2];
      m[15] = 1;
   }

   TGeoShape *shape = currentNode->GetVolume()->GetShape();
   shape->SetName(name);
   REveGeoShape* rgs = new REveGeoShape(name);
   rgs->SetShape(shape);

   rgs->InitMainTrans();
   rgs->RefMainTrans().SetFrom(m);
   return rgs;
}

void write_geo_extract()
{
   TFile::SetCacheFileDir(".");

   auto eveMng = REveManager::Create();
   eveMng->AllowMultipleRemoteConnections(false, false);

   auto s_geoManager = TGeoManager::Import("http://xrd-cache-1.t2.ucsd.edu/alja/mail/geo/cmsSimGeo2026.root");

   // tracker wrapper
   //REveElement *holder = new REveElement();
   REveGeoShape *tracker = makeShape("cms:OCMS/tracker:Tracker_1/otst:supportR1190Z1450_1", "tracker");
   tracker->SetMainColor(kRed);
   tracker->SetMainTransparency(50);
  // holder->AddElement(tracker);

   // tracker barrel
   REveGeoShape *barrel =
      makeShape("cms:OCMS/tracker:Tracker_1/pixbar:Phase2OTBarrel_1/tracker:supportR212Z604_1", "barrel");
   tracker->AddElement(barrel);

   // front and back endcaps
   REveGeoShape *e1 = makeShape("cms:OCMS/tracker:Tracker_1/pixfwd:Phase2OTForward_1", "endcap_1");
   tracker->AddElement(e1);
   REveGeoShape *e2 = makeShape("cms:OCMS/tracker:Tracker_1/pixfwd:Phase2OTForward_2", "endcap_2");
   tracker->AddElement(e2);

   // save extract on top element
   tracker->SaveExtract("testShapeExtract.root", "VSDGeo");

   eveMng->GetEventScene()->AddElement(tracker);
   eveMng->Show();
}